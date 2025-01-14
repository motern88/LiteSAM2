# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import logging
import math
from typing import Callable, Iterable, List, Optional, Sequence

import torch

from torch.utils.data import BatchSampler, DataLoader, Dataset, IterableDataset, Subset

from torch.utils.data.distributed import DistributedSampler


class MixedDataLoader:
    def __init__(self, dataloaders: List[DataLoader], mixing_prob: torch.FloatTensor):
        """
        初始化混合数据加载器

        Args:
            dataloaders (List[DataLoader]): List of DataLoaders to be mixed.
            mixing_prob (torch.FloatTensor): Probability of each dataloader to be sampled from
        /
        参数:
            dataloaders (List[DataLoader]): 需要混合的数据加载器列表。
            mixing_prob (torch.FloatTensor): 每个数据加载器被采样的概率。

        """
        # 确保数据加载器数量与概率的数量一致
        assert len(dataloaders) == mixing_prob.shape[0]
        self.dataloaders = dataloaders  # 存储数据加载器列表
        self.mixing_prob = mixing_prob  # 存储每个数据加载器的采样概率
        # 初始化迭代器状态 / Iterator state
        self._iter_dls = None
        self._iter_mixing_prob = None
        self.random_generator = torch.Generator()  # 随机数生成器

    def __len__(self):
        # 返回所有数据加载器中数据的总数量
        return sum([len(d) for d in self.dataloaders])

    def __iter__(self):
        # 同步数据加载器的随机种子 / Synchronize dataloader seeds
        self.random_generator.manual_seed(42)
        # 为每个数据加载器创建迭代器
        self._iter_dls = [iter(loader) for loader in self.dataloaders]
        self._iter_mixing_prob = self.mixing_prob.clone()  # 克隆混合概率
        return self

    def __next__(self):
        """
        Sample a dataloader to sample from based on mixing probabilities. If one of the dataloaders is exhausted, we continue sampling from the other loaders until all are exhausted.
        根据混合概率从一个数据加载器中采样。如果某个数据加载器已经耗尽，我们会继续从其他加载器中采样直到全部加载器都耗尽。
        """
        # 如果没有初始化迭代器，抛出异常
        if self._iter_dls is None:
            raise TypeError(f"{type(self).__name__} object is not an iterator")

        while self._iter_mixing_prob.any():    # 至少有一个数据加载器的概率非零 / at least one D-Loader with non-zero prob.
            # 根据混合概率选择数据加载器
            dataset_idx = self._iter_mixing_prob.multinomial(
                1, generator=self.random_generator
            ).item()
            try:
                # 从选中的数据加载器中获取下一批数据
                item = next(self._iter_dls[dataset_idx])
                return item
            except StopIteration:
                # No more iterations for this dataset, set it's mixing probability to zero and try again.
                # 当前数据加载器已经耗尽，将其概率设为0并继续尝试
                self._iter_mixing_prob[dataset_idx] = 0
            except Exception as e:
                # 记录并抛出其他异常 / log and raise any other unexpected error.
                logging.error(e)
                raise e

        # 如果所有的数据加载器都已耗尽，抛出停止迭代异常 / Exhausted all iterators
        raise StopIteration


class TorchTrainMixedDataset:
    def __init__(
        self,
        datasets: List[Dataset],
        batch_sizes: List[int],
        num_workers: int,
        shuffle: bool,
        pin_memory: bool,
        drop_last: bool,
        collate_fn: Optional[Callable] = None,
        worker_init_fn: Optional[Callable] = None,
        phases_per_epoch: int = 1,
        dataset_prob: Optional[List[float]] = None,
    ) -> None:
        """
        初始化混合数据集训练

        Args:
            datasets (List[Dataset]): List of Datasets to be mixed.
            batch_sizes (List[int]): Batch sizes for each dataset in the list.
            num_workers (int): Number of workers per dataloader.
            shuffle (bool): Whether or not to shuffle data.
            pin_memory (bool): If True, use pinned memory when loading tensors from disk.
            drop_last (bool): Whether or not to drop the last batch of data.
            collate_fn (Callable): Function to merge a list of samples into a mini-batch.
            worker_init_fn (Callable): Function to init each dataloader worker.
            phases_per_epoch (int): Number of phases per epoch.
            dataset_prob (List[float]): Probability of choosing the dataloader to sample from. Should sum to 1.0

        参数:
            datasets (List[Dataset]): 数据集列表。
            batch_sizes (List[int]): 每个数据集对应的批大小。
            num_workers (int): 每个数据加载器使用的工作线程数。
            shuffle (bool): 是否打乱数据。
            pin_memory (bool): 如果为True，则在加载数据时将其固定到内存。
            drop_last (bool): 是否丢弃最后一批数据。
            collate_fn (Callable): 用于将样本合并成小批次的函数。
            worker_init_fn (Callable): 初始化每个数据加载器工作线程的函数。
            phases_per_epoch (int): 每个epoch的阶段数。
            dataset_prob (List[float]): 每个数据集的采样概率，应当总和为1.0。
        """

        self.datasets = datasets
        self.batch_sizes = batch_sizes
        self.num_workers = num_workers
        self.shuffle = shuffle
        self.pin_memory = pin_memory
        self.drop_last = drop_last
        self.collate_fn = collate_fn
        self.worker_init_fn = worker_init_fn
        assert len(self.datasets) > 0  # 确保有数据集
        for dataset in self.datasets:
            assert not isinstance(dataset, IterableDataset), "Not supported"
            # `RepeatFactorWrapper` requires calling set_epoch first to get its length
            # `RepeatFactorWrapper`需要调用set_epoch才能获取数据集长度
            self._set_dataset_epoch(dataset, 0)
        self.phases_per_epoch = phases_per_epoch
        self.chunks = [None] * len(datasets)  # 存储每个数据集的分块
        if dataset_prob is None:
            # If not provided, assign each dataset a probability proportional to its length.
            # 如果未提供采样概率，则根据数据集长度比例分配概率
            dataset_lens = [
                (math.floor(len(d) / bs) if drop_last else math.ceil(len(d) / bs))
                for d, bs in zip(datasets, batch_sizes)
            ]
            total_len = sum(dataset_lens)  # 总数据量
            dataset_prob = torch.tensor([d_len / total_len for d_len in dataset_lens])
        else:
            assert len(dataset_prob) == len(datasets)  # 确保提供的概率数量与数据集数量一致
            dataset_prob = torch.tensor(dataset_prob)

        logging.info(f"Dataset mixing probabilities: {dataset_prob.tolist()}")
        assert dataset_prob.sum().item() == 1.0, "Probabilities should sum to 1.0"
        self.dataset_prob = dataset_prob  # 存储数据集的采样概率

    # 设置数据集的epoch
    def _set_dataset_epoch(self, dataset, epoch: int) -> None:
        if hasattr(dataset, "epoch"):
            dataset.epoch = epoch
        if hasattr(dataset, "set_epoch"):
            dataset.set_epoch(epoch)

    # 获取混合数据加载器
    def get_loader(self, epoch) -> Iterable:
        dataloaders = []
        for d_idx, (dataset, batch_size) in enumerate(
            zip(self.datasets, self.batch_sizes)
        ):
            if self.phases_per_epoch > 1:
                # Major epoch that looops over entire dataset
                # 每个epoch包含多个阶段，计算主epoch
                # len(main_epoch) == phases_per_epoch * len(epoch)
                main_epoch = epoch // self.phases_per_epoch

                # 计算本阶段 / Phase with in the main epoch
                local_phase = epoch % self.phases_per_epoch

                # Start of new data-epoch or job is resumed after preemtion.
                # 开始新数据阶段或任务从中断处恢复
                if local_phase == 0 or self.chunks[d_idx] is None:
                    # set seed for dataset epoch
                    # If using RepeatFactorWrapper, this step currectly re-samples indices before chunking.
                    # 为数据集设置 epoch
                    # 如果使用 RepeatFactorWrapper，这一步会在分块之前重新采样索引。
                    self._set_dataset_epoch(dataset, main_epoch)

                    # Separate random generator for subset sampling
                    # 为子集采样使用不同的随机生成器
                    g = torch.Generator()
                    g.manual_seed(main_epoch)
                    self.chunks[d_idx] = torch.chunk(
                        torch.randperm(len(dataset), generator=g),
                        self.phases_per_epoch,
                    )

                # 使用子集来创建数据加载器
                dataset = Subset(dataset, self.chunks[d_idx][local_phase])
            else:
                self._set_dataset_epoch(dataset, epoch)

            sampler = DistributedSampler(dataset, shuffle=self.shuffle)  # 使用分布式采样器
            sampler.set_epoch(epoch)  # 设置epoch

            batch_sampler = BatchSampler(sampler, batch_size, drop_last=self.drop_last)  # 批采样器
            dataloaders.append(
                DataLoader(
                    dataset,
                    num_workers=self.num_workers,
                    pin_memory=self.pin_memory,
                    batch_sampler=batch_sampler,
                    collate_fn=self.collate_fn,
                    worker_init_fn=self.worker_init_fn,
                )
            )
        return MixedDataLoader(dataloaders, self.dataset_prob)  # 返回混合数据加载器
