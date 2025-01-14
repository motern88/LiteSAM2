# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import random
from dataclasses import dataclass
from typing import List

from training.dataset.vos_segment_loader import LazySegments

MAX_RETRIES = 1000  # 最大重试次数，用于在采样过程中确保可用的帧和对象数量

# 存储采样的帧及其对应的对象ID
@dataclass
class SampledFramesAndObjects:
    frames: List[int]
    object_ids: List[int]

# 视频目标分割采样器的基类
class VOSSampler:
    def __init__(self, sort_frames=True):
        # frames are ordered by frame id when sort_frames is True
        # 当sort_frames为True时，帧将根据帧ID进行排序
        self.sort_frames = sort_frames

    def sample(self, video):
        # 抽象方法，子类需要实现采样逻辑
        raise NotImplementedError()


# 随机均匀采样器：从视频中随机采样固定数量的帧，并选择其中的对象。
class RandomUniformSampler(VOSSampler):
    def __init__(
        self,
        num_frames,
        max_num_objects,
        reverse_time_prob=0.0,
    ):
        self.num_frames = num_frames  # 每次采样的帧数
        self.max_num_objects = max_num_objects  # 每次采样的最大对象数
        self.reverse_time_prob = reverse_time_prob  # 翻转时间的概率

    def sample(self, video, segment_loader, epoch=None):
        '''从视频中采样指定数量的帧，并返回相应的对象ID'''
        for retry in range(MAX_RETRIES):
            if len(video.frames) < self.num_frames:  # 重试次数
                raise Exception(
                    f"Cannot sample {self.num_frames} frames from video {video.video_name} as it only has {len(video.frames)} annotated frames."
                )
            # 随机选择一个起始帧索引，保证能够采样到指定数量的帧
            start = random.randrange(0, len(video.frames) - self.num_frames + 1)
            frames = [video.frames[start + step] for step in range(self.num_frames)]
            if random.uniform(0, 1) < self.reverse_time_prob:
                # 有一定概率翻转帧的顺序 / Reverse time
                frames = frames[::-1]

            # 获取第一帧的对象ID / Get first frame object ids
            visible_object_ids = []
            loaded_segms = segment_loader.load(frames[0].frame_idx)
            if isinstance(loaded_segms, LazySegments):
                # 如果是LazySegments（针对SA1BRawDataset），获取所有可见对象的ID / LazySegments for SA1BRawDataset
                visible_object_ids = list(loaded_segms.keys())
            else:
                # 否则，遍历所有对象，只有包含像素的对象才被视为可见
                for object_id, segment in segment_loader.load(
                    frames[0].frame_idx
                ).items():
                    if segment.sum():
                        visible_object_ids.append(object_id)

            # 如果第一帧没有可见的对象，重试 / First frame needs to have at least a target to track
            if len(visible_object_ids) > 0:
                break
            if retry >= MAX_RETRIES - 1:
                raise Exception("No visible objects")

        # 随机选择对象ID（最多选择max_num_objects个）
        object_ids = random.sample(
            visible_object_ids,
            min(len(visible_object_ids), self.max_num_objects),
        )
        return SampledFramesAndObjects(frames=frames, object_ids=object_ids)


class EvalSampler(VOSSampler):
    """
    VOS Sampler for evaluation: sampling all the frames and all the objects in a video
    用于评估的视频目标分割采样器：采样视频中的所有帧和所有对象
    """

    def __init__(
        self,
    ):
        super().__init__()

    def sample(self, video, segment_loader, epoch=None):
        """
        Sampling all the frames and all the objects
        采样所有帧及其对应的对象
        """
        if self.sort_frames:
            # 如果需要排序，则按帧ID排序 / ordered by frame id
            frames = sorted(video.frames, key=lambda x: x.frame_idx)
        else:
            # 否则使用原始顺序 / use the original order
            frames = video.frames
        object_ids = segment_loader.load(frames[0].frame_idx).keys()
        if len(object_ids) == 0:
            raise Exception("First frame of the video has no objects")

        return SampledFramesAndObjects(frames=frames, object_ids=object_ids)
