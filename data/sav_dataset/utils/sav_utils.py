# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the sav_dataset directory of this source tree.
import json
import os
from typing import Dict, List, Optional, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pycocotools.mask as mask_util


def decode_video(video_path: str) -> List[np.ndarray]:
    """
    Decode the video and return the RGB frames / 解码视频并返回RGB帧
    """
    video = cv2.VideoCapture(video_path)
    video_frames = []
    while video.isOpened():
        ret, frame = video.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            video_frames.append(frame)
        else:
            break
    return video_frames


def show_anns(masks, colors: List, borders=True) -> None:
    """
    show the annotations / 显示标注
    """
    # return if no masks / 如果没有标注，直接返回
    if len(masks) == 0:
        return

    # sort masks by size / 按大小对标注进行排序
    sorted_annot_and_color = sorted(
        zip(masks, colors), key=(lambda x: x[0].sum()), reverse=True
    )
    H, W = sorted_annot_and_color[0][0].shape[0], sorted_annot_and_color[0][0].shape[1]

    canvas = np.ones((H, W, 4))
    canvas[:, :, 3] = 0  # set the alpha channel / 设置alpha通道为0
    contour_thickness = max(1, int(min(5, 0.01 * min(H, W))))
    for mask, color in sorted_annot_and_color:
        canvas[mask] = np.concatenate([color, [0.55]])
        if borders:
            contours, _ = cv2.findContours(
                np.array(mask, dtype=np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE
            )
            cv2.drawContours(
                canvas, contours, -1, (0.05, 0.05, 0.05, 1), thickness=contour_thickness
            )

    ax = plt.gca()
    ax.imshow(canvas)


class SAVDataset:
    """
    SAVDataset is a class to load the SAV dataset and visualize the annotations.
    /
    SAVDataset 类用于加载 SAV 数据集并可视化标注
    """

    def __init__(self, sav_dir, annot_sample_rate=4):
        """
        Args:
            sav_dir: the directory of the SAV dataset
            annot_sample_rate: the sampling rate of the annotations.
                The annotations are aligned with the videos at 6 fps.
            /
            sav_dir: SAV 数据集的目录路径
            annot_sample_rate: 标注的采样率。标注是以6fps对齐视频的。
        """
        self.sav_dir = sav_dir
        self.annot_sample_rate = annot_sample_rate
        self.manual_mask_colors = np.random.random((256, 3))
        self.auto_mask_colors = np.random.random((256, 3))

    def read_frames(self, mp4_path: str) -> None:
        """
        Read the frames and downsample them to align with the annotations.
        /
        读取视频帧并进行降采样，以与标注对齐。
        """
        if not os.path.exists(mp4_path):
            print(f"{mp4_path} doesn't exist.")
            return None
        else:
            # decode the video / 解码视频
            frames = decode_video(mp4_path)
            print(f"There are {len(frames)} frames decoded from {mp4_path} (24fps).")

            # downsample the frames to align with the annotations / 根据标注对帧进行降采样
            frames = frames[:: self.annot_sample_rate]
            print(
                f"Videos are annotated every {self.annot_sample_rate} frames. "
                "To align with the annotations, "
                f"downsample the video to {len(frames)} frames."
            )
            return frames

    def get_frames_and_annotations(
        self, video_id: str
    ) -> Tuple[List | None, Dict | None, Dict | None]:
        """
        Get the frames and annotations for video.
        /
        获取视频的帧和标注
        """
        # load the video / 加载视频
        mp4_path = os.path.join(self.sav_dir, video_id + ".mp4")
        frames = self.read_frames(mp4_path)
        if frames is None:
            return None, None, None

        # load the manual annotations / 加载手动标注
        manual_annot_path = os.path.join(self.sav_dir, video_id + "_manual.json")
        if not os.path.exists(manual_annot_path):
            print(f"{manual_annot_path} doesn't exist. Something might be wrong.")
            manual_annot = None
        else:
            manual_annot = json.load(open(manual_annot_path))

        # load the auto annotations / 加载自动标注
        auto_annot_path = os.path.join(self.sav_dir, video_id + "_auto.json")
        if not os.path.exists(auto_annot_path):
            print(f"{auto_annot_path} doesn't exist.")
            auto_annot = None
        else:
            auto_annot = json.load(open(auto_annot_path))

        return frames, manual_annot, auto_annot

    def visualize_annotation(
        self,
        frames: List[np.ndarray],
        auto_annot: Optional[Dict],
        manual_annot: Optional[Dict],
        annotated_frame_id: int,
        show_auto=True,
        show_manual=True,
    ) -> None:
        """
        Visualize the annotations on the annotated_frame_id.
        If show_manual is True, show the manual annotations.
        If show_auto is True, show the auto annotations.
        By default, show both auto and manual annotations.
        /
        可视化在 annotated_frame_id 帧上的标注
        如果 show_manual 为 True，显示手动标注
        如果 show_auto 为 True，显示自动标注
        默认同时显示自动和手动标注
        """

        if annotated_frame_id >= len(frames):
            print("invalid annotated_frame_id / 无效的 annotated_frame_id")
            return

        rles = []
        colors = []
        if show_manual and manual_annot is not None:
            rles.extend(manual_annot["masklet"][annotated_frame_id])
            colors.extend(
                self.manual_mask_colors[
                    : len(manual_annot["masklet"][annotated_frame_id])
                ]
            )
        if show_auto and auto_annot is not None:
            rles.extend(auto_annot["masklet"][annotated_frame_id])
            colors.extend(
                self.auto_mask_colors[: len(auto_annot["masklet"][annotated_frame_id])]
            )

        plt.imshow(frames[annotated_frame_id])

        if len(rles) > 0:
            masks = [mask_util.decode(rle) > 0 for rle in rles]
            show_anns(masks, colors)
        else:
            print("No annotation will be shown / 没有标注将被显示")

        plt.axis("off")
        plt.show()
