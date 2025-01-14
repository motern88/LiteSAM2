# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Misc functions, including distributed helpers.

Mostly copy-paste from torchvision references.
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import torch

from PIL import Image as PILImage
from tensordict import tensorclass


@tensorclass
class BatchedVideoMetaData:
    """
    This class represents metadata about a batch of videos.
    该类表示一批视频的元数据。
    Attributes:
        unique_objects_identifier: A tensor of shape Bx3 containing unique identifiers for each object in the batch. Index consists of (video_id, obj_id, frame_id)
            形状为 Bx3 的张量，包含批次中每个对象的唯一标识符。索引由 (video_id, obj_id, frame_id) 组成。
        frame_orig_size: A tensor of shape Bx2 containing the original size of each frame in the batch.
            形状为 Bx2 的张量，包含批次中每个帧的原始大小。
    """

    unique_objects_identifier: torch.LongTensor
    frame_orig_size: torch.LongTensor


@tensorclass
class BatchedVideoDatapoint:
    """
    This class represents a batch of videos with associated annotations and metadata.
    该类表示一批视频及其相关注释和元数据。
    Attributes:
        img_batch: A [TxBxCxHxW] tensor containing the image data for each frame in the batch, where T is the number of frames per video, and B is the number of videos in the batch.
            一个形状为 [TxBxCxHxW] 的张量，包含批次中每帧的视频图像数据，其中 T 为每个视频的帧数，B 为批次中的视频数量。
        obj_to_frame_idx: A [TxOx2] tensor containing the image_batch index which the object belongs to. O is the number of objects in the batch.
            一个形状为 [TxOx2] 的张量，包含每个对象所属的图像批次索引，其中 O 为批次中的对象数量。
        masks: A [TxOxHxW] tensor containing binary masks for each object in the batch.
            一个形状为 [TxOxHxW] 的张量，包含批次中每个对象的二进制掩码。
        metadata: An instance of BatchedVideoMetaData containing metadata about the batch.
            一个 BatchedVideoMetaData 实例，包含批次的元数据。
        dict_key: A string key used to identify the batch.
            一个字符串键，用于标识该批次。
    """

    img_batch: torch.FloatTensor
    obj_to_frame_idx: torch.IntTensor
    masks: torch.BoolTensor
    metadata: BatchedVideoMetaData

    dict_key: str

    # 将张量移到固定内存，适用于 GPU
    def pin_memory(self, device=None):
        return self.apply(torch.Tensor.pin_memory, device=device)

    @property
    def num_frames(self) -> int:
        """
        Returns the number of frames per video.
        返回每个视频的帧数。
        """
        return self.batch_size[0]

    @property
    def num_videos(self) -> int:
        """
        Returns the number of videos in the batch.
        返回批次中的视频数量。
        """
        return self.img_batch.shape[1]

    @property
    def flat_obj_to_img_idx(self) -> torch.IntTensor:
        """
        Returns a flattened tensor containing the object to img index.
        The flat index can be used to access a flattened img_batch of shape [(T*B)xCxHxW]
        返回一个展平的张量，包含对象到图像的索引。
        扁平的索引可用于访问展平后的 img_batch，形状为 [(T*B)xCxHxW]
        """
        frame_idx, video_idx = self.obj_to_frame_idx.unbind(dim=-1)
        flat_idx = video_idx * self.num_frames + frame_idx
        return flat_idx

    @property
    def flat_img_batch(self) -> torch.FloatTensor:
        """
        Returns a flattened img_batch_tensor of shape [(B*T)xCxHxW]
        返回展平后的 img_batch 张量，形状为 [(B*T)xCxHxW]
        """

        return self.img_batch.transpose(0, 1).flatten(0, 1)


@dataclass
class Object:
    # 对象在媒体中的唯一标识符 / Id of the object in the media
    object_id: int
    # 对象所在帧的索引（如果是单张图像，则为 0） / Index of the frame in the media (0 if single image)
    frame_index: int
    segment: Union[torch.Tensor, dict]  # RLE 字典或二值掩码 / RLE dict or binary mask


@dataclass
class Frame:
    data: Union[torch.Tensor, PILImage.Image]
    objects: List[Object]


@dataclass
class VideoDatapoint:
    """
    Refers to an image/video and all its annotations
    表示一段视频或图像及其所有的注释信息。
    """

    frames: List[Frame]
    video_id: int
    size: Tuple[int, int]


def collate_fn(
    batch: List[VideoDatapoint],
    dict_key,
) -> BatchedVideoDatapoint:
    """
    将一批 VideoDatapoint 数据点聚合为 BatchedVideoDatapoint 批次

    Args:
        batch: A list of VideoDatapoint instances.
            一个包含多个 VideoDatapoint 实例的列表。
        dict_key (str): A string key used to identify the batch.
            标识该批次的字符串键。
    """
    img_batch = []
    for video in batch:
        # 提取每个视频中的所有帧数据，并堆叠为张量。
        img_batch += [torch.stack([frame.data for frame in video.frames], dim=0)]

    # 将视频的帧数据堆叠并转置为形状 [T, B, C, H, W]。
    img_batch = torch.stack(img_batch, dim=0).permute((1, 0, 2, 3, 4))
    T = img_batch.shape[0]  # 每个视频的帧数

    # Prepare data structures for sequential processing. Per-frame processing but batched across videos.
    # 准备数据结构用于顺序处理。逐帧处理，但在视频间进行批量处理。
    step_t_objects_identifier = [[] for _ in range(T)]
    step_t_frame_orig_size = [[] for _ in range(T)]
    step_t_masks = [[] for _ in range(T)]
    step_t_obj_to_frame_idx = [[] for _ in range(T)]  # 存储每时间步的帧索引 / List to store frame indices for each time step

    for video_idx, video in enumerate(batch):
        orig_video_id = video.video_id
        orig_frame_size = video.size
        for t, frame in enumerate(video.frames):
            objects = frame.objects
            for obj in objects:
                orig_obj_id = obj.object_id
                orig_frame_idx = obj.frame_index
                # 存储对象到帧的索引
                step_t_obj_to_frame_idx[t].append(
                    torch.tensor([t, video_idx], dtype=torch.int)
                )
                # 存储对象的二值掩码
                step_t_masks[t].append(obj.segment.to(torch.bool))
                # 存储对象标识符（视频 ID、对象 ID、帧 ID）
                step_t_objects_identifier[t].append(
                    torch.tensor([orig_video_id, orig_obj_id, orig_frame_idx])
                )
                # 存储帧的原始尺寸
                step_t_frame_orig_size[t].append(torch.tensor(orig_frame_size))

    # 将对象到帧的索引堆叠为张量
    obj_to_frame_idx = torch.stack(
        [
            torch.stack(obj_to_frame_idx, dim=0)
            for obj_to_frame_idx in step_t_obj_to_frame_idx
        ],
        dim=0,
    )
    # 将掩码、对象标识符和帧尺寸堆叠为张量
    masks = torch.stack([torch.stack(masks, dim=0) for masks in step_t_masks], dim=0)
    objects_identifier = torch.stack(
        [torch.stack(id, dim=0) for id in step_t_objects_identifier], dim=0
    )
    frame_orig_size = torch.stack(
        [torch.stack(id, dim=0) for id in step_t_frame_orig_size], dim=0
    )
    # 返回 BatchedVideoDatapoint 实例
    return BatchedVideoDatapoint(
        img_batch=img_batch,
        obj_to_frame_idx=obj_to_frame_idx,
        masks=masks,
        metadata=BatchedVideoMetaData(
            unique_objects_identifier=objects_identifier,
            frame_orig_size=frame_orig_size,
        ),
        dict_key=dict_key,
        batch_size=[T],
    )
