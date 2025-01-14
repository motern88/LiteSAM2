# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import glob
import logging
import os
from dataclasses import dataclass

from typing import List, Optional

import pandas as pd

import torch

from iopath.common.file_io import g_pathmgr

from omegaconf.listconfig import ListConfig

from training.dataset.vos_segment_loader import (
    JSONSegmentLoader,
    MultiplePNGSegmentLoader,
    PalettisedPNGSegmentLoader,
    SA1BSegmentLoader,
)


@dataclass
class VOSFrame:
    # 视频帧类，包含帧的索引、图像路径以及数据
    frame_idx: int
    image_path: str
    data: Optional[torch.Tensor] = None
    is_conditioning_only: Optional[bool] = False  # 是否仅用于条件输入，默认是 False


@dataclass
class VOSVideo:
    # 视频类，包含视频名称、视频 ID 和帧数据
    video_name: str
    video_id: int
    frames: List[VOSFrame]

    def __len__(self):
        return len(self.frames)


class VOSRawDataset:
    # VOS 原始数据集基类
    def __init__(self):
        pass

    def get_video(self, idx):
        # 抽象方法：获取指定索引的 VOS 视频
        raise NotImplementedError()


class PNGRawDataset(VOSRawDataset):
    # PNG 格式的原始数据集
    def __init__(
        self,
        img_folder,
        gt_folder,
        file_list_txt=None,
        excluded_videos_list_txt=None,
        sample_rate=1,
        is_palette=True,
        single_object_mode=False,
        truncate_video=-1,
        frames_sampling_mult=False,
    ):
        # 初始化 PNG 数据集
        self.img_folder = img_folder  # 图像文件夹路径
        self.gt_folder = gt_folder  # 标注文件夹路径
        self.sample_rate = sample_rate  # 采样率
        self.is_palette = is_palette  # 是否使用调色板
        self.single_object_mode = single_object_mode  # 是否为单一物体模式
        self.truncate_video = truncate_video  # 截断视频帧数

        # 如果提供了文件列表，读取文件 / Read the subset defined in file_list_txt
        if file_list_txt is not None:
            with g_pathmgr.open(file_list_txt, "r") as f:
                subset = [os.path.splitext(line.strip())[0] for line in f]
        else:
            subset = os.listdir(self.img_folder)

        # 如果提供了排除视频列表，读取排除的视频 / Read and process excluded files if provided
        if excluded_videos_list_txt is not None:
            with g_pathmgr.open(excluded_videos_list_txt, "r") as f:
                excluded_files = [os.path.splitext(line.strip())[0] for line in f]
        else:
            excluded_files = []

        # 过滤掉排除的视频 / Check if it's not in excluded_files
        self.video_names = sorted(
            [video_name for video_name in subset if video_name not in excluded_files]
        )

        if self.single_object_mode:
            # 单一物体模式 / single object mode
            self.video_names = sorted(
                [
                    os.path.join(video_name, obj)
                    for video_name in self.video_names
                    for obj in os.listdir(os.path.join(self.gt_folder, video_name))
                ]
            )

        if frames_sampling_mult:
            # 如果启用了帧采样多倍数
            video_names_mult = []
            for video_name in self.video_names:
                num_frames = len(os.listdir(os.path.join(self.img_folder, video_name)))
                video_names_mult.extend([video_name] * num_frames)
            self.video_names = video_names_mult

    def get_video(self, idx):
        """
        Given a VOSVideo object, return the mask tensors.
        给定一个视频 ID，返回视频帧和对应的掩码（mask）张量
        """
        video_name = self.video_names[idx]

        if self.single_object_mode:
            video_frame_root = os.path.join(
                self.img_folder, os.path.dirname(video_name)
            )
        else:
            video_frame_root = os.path.join(self.img_folder, video_name)

        video_mask_root = os.path.join(self.gt_folder, video_name)

        # 根据是否使用调色板选择掩码加载器
        if self.is_palette:
            segment_loader = PalettisedPNGSegmentLoader(video_mask_root)
        else:
            segment_loader = MultiplePNGSegmentLoader(
                video_mask_root, self.single_object_mode
            )

        all_frames = sorted(glob.glob(os.path.join(video_frame_root, "*.jpg")))
        if self.truncate_video > 0:
            all_frames = all_frames[: self.truncate_video]
        frames = []
        for _, fpath in enumerate(all_frames[:: self.sample_rate]):
            fid = int(os.path.basename(fpath).split(".")[0])
            frames.append(VOSFrame(fid, image_path=fpath))
        video = VOSVideo(video_name, idx, frames)
        return video, segment_loader

    def __len__(self):
        # 返回视频数据集的长度（视频数量）
        return len(self.video_names)


class SA1BRawDataset(VOSRawDataset):
    # SA1B 格式的原始数据集
    def __init__(
        self,
        img_folder,
        gt_folder,
        file_list_txt=None,
        excluded_videos_list_txt=None,
        num_frames=1,
        mask_area_frac_thresh=1.1,  # 默认不进行过滤 / no filtering by default
        uncertain_iou=-1,  # 默认不进行过滤 / no filtering by default
    ):
        # 初始化 SA1B 数据集
        self.img_folder = img_folder  # 图像文件夹路径
        self.gt_folder = gt_folder  # 标注文件夹路径
        self.num_frames = num_frames  # 使用的帧数
        self.mask_area_frac_thresh = mask_area_frac_thresh  # 掩码区域阈值
        self.uncertain_iou = uncertain_iou  # 稳定性评分 / stability score

        # 如果提供了文件列表，读取文件 / Read the subset defined in file_list_txt
        if file_list_txt is not None:
            with g_pathmgr.open(file_list_txt, "r") as f:
                subset = [os.path.splitext(line.strip())[0] for line in f]
        else:
            subset = os.listdir(self.img_folder)
            subset = [
                path.split(".")[0] for path in subset if path.endswith(".jpg")
            ]  # 去掉扩展名 / remove extension

        # Read and process excluded files if provided
        # 如果提供了排除视频列表，读取排除的视频
        if excluded_videos_list_txt is not None:
            with g_pathmgr.open(excluded_videos_list_txt, "r") as f:
                excluded_files = [os.path.splitext(line.strip())[0] for line in f]
        else:
            excluded_files = []

        # 过滤掉排除的视频 / Check if it's not in excluded_files and it exists
        self.video_names = [
            video_name for video_name in subset if video_name not in excluded_files
        ]

    def get_video(self, idx):
        """
        Given a VOSVideo object, return the mask tensors.
        给定一个视频 ID，返回视频帧和对应的掩码（mask）张量
        """
        video_name = self.video_names[idx]

        video_frame_path = os.path.join(self.img_folder, video_name + ".jpg")
        video_mask_path = os.path.join(self.gt_folder, video_name + ".json")

        # 加载视频的掩码数据
        segment_loader = SA1BSegmentLoader(
            video_mask_path,
            mask_area_frac_thresh=self.mask_area_frac_thresh,
            video_frame_path=video_frame_path,
            uncertain_iou=self.uncertain_iou,
        )

        frames = []
        for frame_idx in range(self.num_frames):
            frames.append(VOSFrame(frame_idx, image_path=video_frame_path))
        video_name = video_name.split("_")[-1]   # 文件名格式为 sa_{int} / filename is sa_{int}
        # video id needs to be image_id to be able to load correct annotation file during eval
        # 视频 ID 使用图像 ID，以便在评估时加载正确的标注文件
        video = VOSVideo(video_name, int(video_name), frames)
        return video, segment_loader

    def __len__(self):
        return len(self.video_names)


class JSONRawDataset(VOSRawDataset):
    """
    Dataset where the annotation in the format of SA-V json files
    数据集，标注格式为 SA-V 的 JSON 文件。
    """

    def __init__(
        self,
        img_folder,
        gt_folder,
        file_list_txt=None,
        excluded_videos_list_txt=None,
        sample_rate=1,
        rm_unannotated=True,
        ann_every=1,
        frames_fps=24,
    ):
        # 初始化 JSON 格式的原始数据集
        self.gt_folder = gt_folder  # 标注文件夹路径
        self.img_folder = img_folder  # 图像文件夹路径
        self.sample_rate = sample_rate  # 帧采样率
        self.rm_unannotated = rm_unannotated  # 是否移除未标注的帧
        self.ann_every = ann_every  # 每几帧标注一次
        self.frames_fps = frames_fps  # 视频帧率

        # 读取并处理排除的视频列表（如果提供） / Read and process excluded files if provided
        excluded_files = []
        if excluded_videos_list_txt is not None:
            if isinstance(excluded_videos_list_txt, str):
                excluded_videos_lists = [excluded_videos_list_txt]
            elif isinstance(excluded_videos_list_txt, ListConfig):
                excluded_videos_lists = list(excluded_videos_list_txt)
            else:
                raise NotImplementedError

            for excluded_videos_list_txt in excluded_videos_lists:
                with open(excluded_videos_list_txt, "r") as f:
                    excluded_files.extend(
                        [os.path.splitext(line.strip())[0] for line in f]
                    )
        excluded_files = set(excluded_files)

        # 读取文件列表（如果提供） / Read the subset defined in file_list_txt
        if file_list_txt is not None:
            with g_pathmgr.open(file_list_txt, "r") as f:
                subset = [os.path.splitext(line.strip())[0] for line in f]
        else:
            subset = os.listdir(self.img_folder)

        # 排除不需要的文件
        self.video_names = sorted(
            [video_name for video_name in subset if video_name not in excluded_files]
        )

    def get_video(self, video_idx):
        """
        Given a VOSVideo object, return the mask tensors.
        给定视频索引，返回视频帧和标注掩码数据。
        """
        video_name = self.video_names[video_idx]
        video_json_path = os.path.join(self.gt_folder, video_name + "_manual.json")
        # 使用 JSON 格式的标注加载器
        segment_loader = JSONSegmentLoader(
            video_json_path=video_json_path,
            ann_every=self.ann_every,  # 每隔多少帧做一次标注
            frames_fps=self.frames_fps,  # 帧率
        )

        # 获取视频中所有帧的 ID
        frame_ids = [
            int(os.path.splitext(frame_name)[0])
            for frame_name in sorted(
                os.listdir(os.path.join(self.img_folder, video_name))
            )
        ]

        # 根据帧 ID 获取帧的图像路径
        frames = [
            VOSFrame(
                frame_id,
                image_path=os.path.join(
                    self.img_folder, f"{video_name}/%05d.jpg" % (frame_id)
                ),
            )
            for frame_id in frame_ids[:: self.sample_rate]  # 根据采样率采样帧
        ]

        if self.rm_unannotated:
            # Eliminate the frames that have not been annotated
            # 如果设置了移除未标注帧，则排除掉没有标注的帧
            valid_frame_ids = [
                i * segment_loader.ann_every
                for i, annot in enumerate(segment_loader.frame_annots)
                if annot is not None and None not in annot  # 有效标注帧
            ]
            frames = [f for f in frames if f.frame_idx in valid_frame_ids]

        # 创建并返回 VOSVideo 对象
        video = VOSVideo(video_name, video_idx, frames)
        return video, segment_loader

    def __len__(self):
        return len(self.video_names)
