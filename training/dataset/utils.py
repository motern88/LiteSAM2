# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Some wrapping utilities extended from pytorch's to support repeat factor sampling in particular"""

from typing import Iterable

import torch
from torch.utils.data import (
    ConcatDataset as TorchConcatDataset,
    Dataset,
    Subset as TorchSubset,
)


class ConcatDataset(TorchConcatDataset):
    def __init__(self, datasets: Iterable[Dataset]) -> None:
        super(ConcatDataset, self).__init__(datasets)
        # 合并各个数据集的重复因子
        self.repeat_factors = torch.cat([d.repeat_factors for d in datasets])

    def set_epoch(self, epoch: int):
        for dataset in self.datasets:
            if hasattr(dataset, "epoch"):
                dataset.epoch = epoch
            if hasattr(dataset, "set_epoch"):
                dataset.set_epoch(epoch)

# 用于数据集的子集类
class Subset(TorchSubset):
    def __init__(self, dataset, indices) -> None:
        super(Subset, self).__init__(dataset, indices)
        # 获取子集对应的重复因子
        self.repeat_factors = dataset.repeat_factors[indices]
        assert len(indices) == len(self.repeat_factors)  # 确保索引和重复因子的长度一致


# 来自 Detectron2 的修改版 / Adapted from Detectron2
class RepeatFactorWrapper(Dataset):
    """
    Thin wrapper around a dataset to implement repeat factor sampling.
    The underlying dataset must have a repeat_factors member to indicate the per-image factor.
    Set it to uniformly ones to disable repeat factor sampling
    /
    用于实现重复因子采样的包装类。
    底层数据集必须具有 `repeat_factors` 属性来指示每个图像的重复因子。
    如果设置为统一的1，则禁用重复因子采样。
    """

    def __init__(self, dataset, seed: int = 0):
        self.dataset = dataset  # 存储底层数据集
        self.epoch_ids = None # 初始化 epoch_ids 为 None
        self._seed = seed

        # Split into whole number (_int_part) and fractional (_frac_part) parts.
        # 将重复因子拆分为整数部分和小数部分
        self._int_part = torch.trunc(dataset.repeat_factors)  # 获取整数部分
        self._frac_part = dataset.repeat_factors - self._int_part  # 获取小数部分

    def _get_epoch_indices(self, generator):
        """
        Create a list of dataset indices (with repeats) to use for one epoch.
        创建用于一个 epoch 的数据集索引列表（包括重复）。

        Args:
            generator (torch.Generator): pseudo random number generator used for
                stochastic rounding.
        参数:
            generator (torch.Generator): 用于伪随机数生成的生成器，执行随机四舍五入。


        Returns:
            torch.Tensor: list of dataset indices to use in one epoch. Each index
                is repeated based on its calculated repeat factor.
        返回:
            torch.Tensor: 一个 epoch 使用的数据集索引列表。每个索引根据其计算的重复因子重复
        """
        # Since repeat factors are fractional, we use stochastic rounding so
        # that the target repeat factor is achieved in expectation over the
        # course of training
        # /
        # 由于重复因子是小数，我们使用随机四舍五入，以便在训练过程中期望地实现目标重复因子。
        rands = torch.rand(len(self._frac_part), generator=generator)  # 生成随机数
        rep_factors = self._int_part + (rands < self._frac_part).float()  # 根据小数部分执行随机四舍五入
        # Construct a list of indices in which we repeat images as specified
        # 构建重复索引列表，每个索引根据其重复因子进行重复
        indices = []
        for dataset_index, rep_factor in enumerate(rep_factors):
            indices.extend([dataset_index] * int(rep_factor.item()))  # 对每个索引重复添加
        return torch.tensor(indices, dtype=torch.int64)  # 返回重复索引的 tensor

    def __len__(self):
        if self.epoch_ids is None:
            # Here we raise an error instead of returning original len(self.dataset) avoid
            # accidentally using unwrapped length. Otherwise it's error-prone since the
            # length changes to `len(self.epoch_ids)`changes after set_epoch is called.
            # /
            # 这里我们抛出一个错误，而不是返回原始的 len(self.dataset)，以避免意外使用未包装的长度。
            # 否则，在调用 set_epoch 后，长度会变成 len(self.epoch_ids)，这可能会导致错误。
            raise RuntimeError("please call set_epoch first to get wrapped length")
            # return len(self.dataset)

        return len(self.epoch_ids)

    def set_epoch(self, epoch: int):
        g = torch.Generator()  # 创建随机数生成器
        g.manual_seed(self._seed + epoch)  # 设置种子，使得每个 epoch 都能生成不同的随机数
        self.epoch_ids = self._get_epoch_indices(g)  # 获取当前 epoch 的重复索引列表
        if hasattr(self.dataset, "set_epoch"):  # 如果数据集有 `set_epoch` 方法
            self.dataset.set_epoch(epoch)  # 调用子数据集的 `set_epoch` 方法

    def __getitem__(self, idx):
        if self.epoch_ids is None:
            raise RuntimeError(
                "Repeat ids haven't been computed. Did you forget to call set_epoch?"
            )
        # 根据计算后的 epoch_ids 返回数据集中的元素
        return self.dataset[self.epoch_ids[idx]]
