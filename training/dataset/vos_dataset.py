# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import logging
import random
from copy import deepcopy

import numpy as np

import torch
from iopath.common.file_io import g_pathmgr
from PIL import Image as PILImage
from torchvision.datasets.vision import VisionDataset

from training.dataset.vos_raw_dataset import VOSRawDataset
from training.dataset.vos_sampler import VOSSampler
from training.dataset.vos_segment_loader import JSONSegmentLoader

from training.utils.data_utils import Frame, Object, VideoDatapoint

MAX_RETRIES = 100  # 最大重试次数


class VOSDataset(VisionDataset):
    def __init__(
        self,
        transforms,
        training: bool,
        video_dataset: VOSRawDataset,
        sampler: VOSSampler,
        multiplier: int,
        always_target=True,
        target_segments_available=True,
    ):
        # 初始化数据集的一些基本参数
        self._transforms = transforms  # 数据预处理方法
        self.training = training  # 是否为训练模式
        self.video_dataset = video_dataset  # 视频数据集
        self.sampler = sampler  # 采样器，用于选择样本

        # 初始化repeat_factors，用于指定每个样本的重复次数
        self.repeat_factors = torch.ones(len(self.video_dataset), dtype=torch.float32)
        self.repeat_factors *= multiplier  # 每个样本的重复次数由multiplier控制
        print(f"Raw dataset length = {len(self.video_dataset)}")

        self.curr_epoch = 0  # 用于处理在不同的训练轮次（epoch）中，数据加载器的行为可能会发生变化的情况 / Used in case data loader behavior changes across epochs
        self.always_target = always_target  # 是否总是为每帧提供目标
        self.target_segments_available = target_segments_available  # 是否每帧都有目标分割

    def _get_datapoint(self, idx):
        # 获取指定索引的视频数据点
        for retry in range(MAX_RETRIES):
            try:
                if isinstance(idx, torch.Tensor):
                    idx = idx.item()
                # 从视频数据集中加载视频 / sample a video
                video, segment_loader = self.video_dataset.get_video(idx)
                # 根据当前视频，使用sampler采样帧和目标对象 /  sample frames and object indices to be used in a datapoint
                sampled_frms_and_objs = self.sampler.sample(
                    video, segment_loader, epoch=self.curr_epoch
                )
                break  # 成功加载视频 / Succesfully loaded video
            except Exception as e:
                if self.training:
                    # 如果加载失败且是训练模式，尝试重新加载
                    logging.warning(
                        f"Loading failed (id={idx}); Retry {retry} with exception: {e}"
                    )
                    idx = random.randrange(0, len(self.video_dataset))  # 随机选择一个视频
                else:
                    # 在验证模式下，加载失败应该直接抛出异常 / Shouldn't fail to load a val video
                    raise e

        # 构建数据点
        datapoint = self.construct(video, sampled_frms_and_objs, segment_loader)
        for transform in self._transforms:
            datapoint = transform(datapoint, epoch=self.curr_epoch)  # 应用所有转换操作
        return datapoint

    def construct(self, video, sampled_frms_and_objs, segment_loader):
        """
        Constructs a VideoDatapoint sample to pass to transforms
        构建一个VideoDatapoint样本并传递给转换操作
        """
        sampled_frames = sampled_frms_and_objs.frames  # 采样的帧
        sampled_object_ids = sampled_frms_and_objs.object_ids  # 采样的对象ID

        images = []  # 存储每帧的图像数据
        rgb_images = load_images(sampled_frames)  # 加载每一帧的RGB图像数据
        # Iterate over the sampled frames and store their rgb data and object data (bbox, segment)
        # 遍历所有采样帧，提取每帧的图像数据和目标数据（边界框和分割掩膜）
        for frame_idx, frame in enumerate(sampled_frames):
            w, h = rgb_images[frame_idx].size  # 获取图像宽高
            images.append(
                Frame(
                    data=rgb_images[frame_idx],  # 当前帧的图像数据
                    objects=[],  # 当前帧的目标列表（待填充）
                )
            )
            # We load the gt segments associated with the current frame
            # 加载当前帧的ground truth分割数据
            if isinstance(segment_loader, JSONSegmentLoader):
                segments = segment_loader.load(
                    frame.frame_idx, obj_ids=sampled_object_ids  # 加载指定对象的分割数据
                )
            else:
                segments = segment_loader.load(frame.frame_idx)  # 加载当前帧的分割数据
            for obj_id in sampled_object_ids:
                # 提取目标的分割数据 / Extract the segment
                if obj_id in segments:
                    assert (
                        segments[obj_id] is not None
                    ), "None targets are not supported"
                    # segment is uint8 and remains uint8 throughout the transforms
                    # 确保分割数据是uint8类型，并在后续转换中保持一致
                    segment = segments[obj_id].to(torch.uint8)
                else:
                    # There is no target, we either use a zero mask target or drop this object
                    # 如果没有目标分割数据，则使用全零掩膜或跳过该目标
                    if not self.always_target:
                        continue
                    segment = torch.zeros(h, w, dtype=torch.uint8)

                # 将目标信息添加到当前帧的objects列表中
                images[frame_idx].objects.append(
                    Object(
                        object_id=obj_id,
                        frame_index=frame.frame_idx,
                        segment=segment,
                    )
                )
        # 返回构建的VideoDatapoint
        return VideoDatapoint(
            frames=images,  # 帧的列表
            video_id=video.video_id,  # 视频ID
            size=(h, w),  # 视频帧的尺寸
        )

    def __getitem__(self, idx):
        # 获取指定索引的数据点
        return self._get_datapoint(idx)

    def __len__(self):
        # 返回数据集的大小（即视频的数量）
        return len(self.video_dataset)


def load_images(frames):
    all_images = []
    cache = {}
    for frame in frames:
        if frame.data is None:
            # 如果当前帧的数据尚未加载，则从文件加载 / Load the frame rgb data from file
            path = frame.image_path
            if path in cache:
                all_images.append(deepcopy(all_images[cache[path]]))
                continue
            with g_pathmgr.open(path, "rb") as fopen:
                all_images.append(PILImage.open(fopen).convert("RGB"))
            cache[path] = len(all_images) - 1
        else:
            # The frame rgb data has already been loaded Convert it to a PILImage
            # 如果当前帧的数据已经加载，则将Tensor转换为PIL图像
            all_images.append(tensor_2_PIL(frame.data))

    return all_images


def tensor_2_PIL(data: torch.Tensor) -> PILImage.Image:
    # 将Tensor数据转换为PIL图像
    data = data.cpu().numpy().transpose((1, 2, 0)) * 255.0  # 转换为Numpy数组并归一化到[0, 255]
    data = data.astype(np.uint8)  # 转换为uint8类型
    return PILImage.fromarray(data)
