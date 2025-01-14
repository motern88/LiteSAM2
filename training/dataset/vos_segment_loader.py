# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import glob
import json
import os

import numpy as np
import pandas as pd
import torch

from PIL import Image as PILImage

try:
    from pycocotools import mask as mask_utils  # 尝试导入pycocotools的mask工具，用于处理RLE编码的分割掩膜
except:
    pass  # 如果导入失败，则忽略


class JSONSegmentLoader:
    '''用于加载和处理视频分割标注的类，注释以JSON格式存储，每隔ann_every帧提供一次标注'''
    def __init__(self, video_json_path, ann_every=1, frames_fps=24, valid_obj_ids=None):
        # 每隔ann_every帧提供一次标注 / Annotations in the json are provided every ann_every th frame
        self.ann_every = ann_every
        # 只考虑这些有效对象ID进行采样 / Ids of the objects to consider when sampling this video
        self.valid_obj_ids = valid_obj_ids

        # 读取视频的标注文件（JSON格式）
        with open(video_json_path, "r") as f:
            data = json.load(f)
            if isinstance(data, list):
                self.frame_annots = data  # 如果数据是列表形式，直接将其作为帧标注
            elif isinstance(data, dict):
                masklet_field_name = "masklet" if "masklet" in data else "masks"
                self.frame_annots = data[masklet_field_name]  # 从"masklet"或"masks"字段获取标注
                if "fps" in data:
                    if isinstance(data["fps"], list):
                        annotations_fps = int(data["fps"][0])  # 如果fps是列表形式，取第一个值
                    else:
                        annotations_fps = int(data["fps"])  # 如果fps是单个值
                    assert frames_fps % annotations_fps == 0  # 确保视频帧率与标注帧率匹配
                    self.ann_every = frames_fps // annotations_fps  # 计算每隔多少帧标注一次
            else:
                raise NotImplementedError

    def load(self, frame_id, obj_ids=None):
        '''加载指定帧及对象的分割掩膜（RLE格式）'''
        assert frame_id % self.ann_every == 0  # 确保该帧是标注的帧
        rle_mask = self.frame_annots[frame_id // self.ann_every]  # 获取该帧的RLE掩膜

        valid_objs_ids = set(range(len(rle_mask)))  # 初始化有效对象ID的集合
        if self.valid_obj_ids is not None:
            # 如果提供了有效对象ID，则筛选出这些对象 / Remove the masklets that have been filtered out for this video
            valid_objs_ids &= set(self.valid_obj_ids)
        if obj_ids is not None:
            # 如果提供了采样的对象ID，则仅保留这些对象 / Only keep the objects that have been sampled
            valid_objs_ids &= set(obj_ids)
        valid_objs_ids = sorted(list(valid_objs_ids))  # 将有效对象ID按升序排列

        # Construct rle_masks_filtered that only contains the rle masks we are interested in
        # 构建仅包含感兴趣的RLE掩膜的列表
        id_2_idx = {}
        rle_mask_filtered = []
        for obj_id in valid_objs_ids:
            if rle_mask[obj_id] is not None:
                id_2_idx[obj_id] = len(rle_mask_filtered)
                rle_mask_filtered.append(rle_mask[obj_id])
            else:
                id_2_idx[obj_id] = None

        # 解码RLE掩膜 / Decode the masks
        raw_segments = torch.from_numpy(mask_utils.decode(rle_mask_filtered)).permute(
            2, 0, 1
        )  # （num_obj, h, w）
        segments = {}
        for obj_id in valid_objs_ids:
            if id_2_idx[obj_id] is None:
                segments[obj_id] = None
            else:
                idx = id_2_idx[obj_id]
                segments[obj_id] = raw_segments[idx]  # 获取该对象的分割掩膜
        return segments

    # 获取每个对象在多少帧中有有效的掩膜（即非None的掩膜）
    def get_valid_obj_frames_ids(self, num_frames_min=None):
        # For each object, find all the frames with a valid (not None) mask
        # 获取每帧的对象数量（假设每帧对象数目一致）
        num_objects = len(self.frame_annots[0])

        # The result dict associates each obj_id with the id of its valid frames
        # 初始化一个字典，键为对象ID，值为该对象有效掩膜所在的帧ID列表
        res = {obj_id: [] for obj_id in range(num_objects)}

        # 遍历每帧标注，找到每个对象的有效帧
        for annot_idx, annot in enumerate(self.frame_annots):
            for obj_id in range(num_objects):
                if annot[obj_id] is not None:
                    res[obj_id].append(int(annot_idx * self.ann_every))  # 记录有效帧的帧ID

        if num_frames_min is not None:
            # Remove masklets that have less than num_frames_min valid masks
            # 如果指定了最小有效帧数，则移除有效帧数少于该数目的对象
            for obj_id, valid_frames in list(res.items()):
                if len(valid_frames) < num_frames_min:
                    res.pop(obj_id)

        return res


class PalettisedPNGSegmentLoader:
    '''用于加载存储为调色板PNG格式的分割掩膜的SegmentLoader类'''
    def __init__(self, video_png_root):
        """
        SegmentLoader for datasets with masks stored as palettised PNGs.
        初始化时传入包含所有PNG掩膜的文件夹路径

        video_png_root: the folder contains all the masks stored in png
        video_png_root: 存储PNG掩膜的根目录
        """
        self.video_png_root = video_png_root
        # build a mapping from frame id to their PNG mask path
        # note that in some datasets, the PNG paths could have more
        # than 5 digits, e.g. "00000000.png" instead of "00000.png"
        # /
        # 构建从帧ID到PNG文件名的映射
        png_filenames = os.listdir(self.video_png_root)
        self.frame_id_to_png_filename = {}
        for filename in png_filenames:
            # 提取文件名前的数字部分作为帧ID
            frame_id, _ = os.path.splitext(filename)
            # 处理帧ID，确保其是整数并映射到PNG文件名
            self.frame_id_to_png_filename[int(frame_id)] = filename

    def load(self, frame_id):
        """
        load the single palettised mask from the disk (path: f'{self.video_png_root}/{frame_id:05d}.png')
        加载指定帧的单个掩膜
        Args:
            frame_id: int, define the mask path
            frame_id: int, 定义了要加载的掩膜路径
        Return:
            binary_segments: dict
            binary_segments: dict，包含该帧所有对象的二进制掩膜
        """
        # 构建该帧的PNG掩膜路径 / check the path
        mask_path = os.path.join(
            self.video_png_root, self.frame_id_to_png_filename[frame_id]
        )

        # 加载PNG掩膜 / load the mask
        masks = PILImage.open(mask_path).convert("P")
        masks = np.array(masks)

        # 获取所有对象ID，去除背景（0）
        object_id = pd.unique(masks.flatten())
        object_id = object_id[object_id != 0]  # remove background (0)

        # 为每个对象创建一个二进制掩膜 / convert into N binary segmentation masks
        binary_segments = {}
        for i in object_id:
            bs = masks == i
            binary_segments[i] = torch.from_numpy(bs)

        return binary_segments

    def __len__(self):
        return


class MultiplePNGSegmentLoader:
    def __init__(self, video_png_root, single_object_mode=False):
        """
        video_png_root: the folder contains all the masks stored in png
        single_object_mode: whether to load only a single object at a time
        /
        video_png_root: 包含所有PNG掩膜的文件夹
        single_object_mode: 是否每次仅加载单个对象的掩膜
        """
        self.video_png_root = video_png_root
        self.single_object_mode = single_object_mode
        # 读取一张掩膜来获取视频的分辨率 / read a mask to know the resolution of the video
        if self.single_object_mode:
            tmp_mask_path = glob.glob(os.path.join(video_png_root, "*.png"))[0]
        else:
            tmp_mask_path = glob.glob(os.path.join(video_png_root, "*", "*.png"))[0]
        tmp_mask = np.array(PILImage.open(tmp_mask_path))
        self.H = tmp_mask.shape[0]  # 高度
        self.W = tmp_mask.shape[1]  # 宽度
        if self.single_object_mode:
            # 单个对象模式，根据文件夹名计算对象ID
            self.obj_id = (
                int(video_png_root.split("/")[-1]) + 1
            )  # 偏移1，因为背景是0 / offset by 1 as bg is 0
        else:
            self.obj_id = None

    def load(self, frame_id):
        if self.single_object_mode:
            return self._load_single_png(frame_id)
        else:
            return self._load_multiple_pngs(frame_id)

    def _load_single_png(self, frame_id):
        """
        load single png from the disk (path: f'{self.obj_id}/{frame_id:05d}.png')
        从磁盘加载单个PNG掩膜 (路径: f'{self.obj_id}/{frame_id:05d}.png')

        Args:
            frame_id: int, define the mask path
            frame_id: int, 定义了掩膜的路径

        Return:
            binary_segments: dict
            binary_segments: dict, 包含该帧的二进制掩膜
        """
        mask_path = os.path.join(self.video_png_root, f"{frame_id:05d}.png")
        binary_segments = {}

        if os.path.exists(mask_path):
            mask = np.array(PILImage.open(mask_path))
        else:
            # 如果PNG文件不存在，返回空掩膜 / if png doesn't exist, empty mask
            mask = np.zeros((self.H, self.W), dtype=bool)
        binary_segments[self.obj_id] = torch.from_numpy(mask > 0)
        return binary_segments

    def _load_multiple_pngs(self, frame_id):
        """
        load multiple png masks from the disk (path: f'{obj_id}/{frame_id:05d}.png')
        从磁盘加载多个PNG掩膜 (路径: f'{obj_id}/{frame_id:05d}.png')

        Args:
            frame_id: int, define the mask path
            frame_id: int, 定义了掩膜的路径

        Return:
            binary_segments: dict
            binary_segments: dict, 包含该帧的二进制掩膜
        """
        # 获取所有对象的路径 / get the path
        all_objects = sorted(glob.glob(os.path.join(self.video_png_root, "*")))
        num_objects = len(all_objects)
        assert num_objects > 0

        # 加载掩膜 / load the masks
        binary_segments = {}
        for obj_folder in all_objects:
            # obj_folder is {video_name}/{obj_id}, obj_id is specified by the name of the folder
            # obj_folder是 {视频名称}/{对象ID}，对象ID由文件夹名指定
            obj_id = int(obj_folder.split("/")[-1])
            obj_id = obj_id + 1  # 偏移1，因为背景是0 / offset 1 as bg is 0
            mask_path = os.path.join(obj_folder, f"{frame_id:05d}.png")
            if os.path.exists(mask_path):
                mask = np.array(PILImage.open(mask_path))
            else:
                mask = np.zeros((self.H, self.W), dtype=bool)
            binary_segments[obj_id] = torch.from_numpy(mask > 0)

        return binary_segments

    def __len__(self):
        return


class LazySegments:
    """
    Only decodes segments that are actually used.
    仅解码实际使用的分割掩膜。
    """

    def __init__(self):
        self.segments = {}
        self.cache = {}

    def __setitem__(self, key, item):
        self.segments[key] = item

    def __getitem__(self, key):
        if key in self.cache:
            return self.cache[key]
        rle = self.segments[key]
        mask = torch.from_numpy(mask_utils.decode([rle])).permute(2, 0, 1)[0]
        self.cache[key] = mask
        return mask

    def __contains__(self, key):
        return key in self.segments

    def __len__(self):
        return len(self.segments)

    def keys(self):
        return self.segments.keys()


# SA1B 分割加载器
class SA1BSegmentLoader:
    def __init__(
        self,
        video_mask_path,  # 掩膜的 JSON 文件路径
        mask_area_frac_thresh=1.1,  # 掩膜的最小面积比例阈值
        video_frame_path=None,  # 视频帧路径（可选）
        uncertain_iou=-1,  # 不确定性 IOU 阈值
    ):
        with open(video_mask_path, "r") as f:
            self.frame_annots = json.load(f)

        if mask_area_frac_thresh <= 1.0:
            # 懒加载帧数据 / Lazily read frame
            orig_w, orig_h = PILImage.open(video_frame_path).size
            area = orig_w * orig_h

        self.frame_annots = self.frame_annots["annotations"]

        rle_masks = []
        for frame_annot in self.frame_annots:
            if not frame_annot["area"] > 0:
                continue
            if ("uncertain_iou" in frame_annot) and (
                frame_annot["uncertain_iou"] < uncertain_iou
            ):
                # 不确定性 IOU 小于阈值 / uncertain_iou is stability score
                continue
            if (
                mask_area_frac_thresh <= 1.0
                and (frame_annot["area"] / area) >= mask_area_frac_thresh
            ):
                continue
            rle_masks.append(frame_annot["segmentation"])

        self.segments = LazySegments()
        for i, rle in enumerate(rle_masks):
            self.segments[i] = rle

    def load(self, frame_idx):
        '''加载指定帧的分割掩膜'''
        return self.segments
