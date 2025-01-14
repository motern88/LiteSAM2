# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Transforms and data augmentation for both image + bbox.
"""

import logging

import random
from typing import Iterable

import torch
import torchvision.transforms as T
import torchvision.transforms.functional as F
import torchvision.transforms.v2.functional as Fv2
from PIL import Image as PILImage

from torchvision.transforms import InterpolationMode

from training.utils.data_utils import VideoDatapoint

# 水平翻转图像和相应的对象分割
def hflip(datapoint, index):

    datapoint.frames[index].data = F.hflip(datapoint.frames[index].data)  # 翻转图像数据
    for obj in datapoint.frames[index].objects:
        if obj.segment is not None:
            obj.segment = F.hflip(obj.segment)  # 翻转对象分割

    return datapoint

# 根据目标大小和原始宽高比计算新的图像尺寸
def get_size_with_aspect_ratio(image_size, size, max_size=None):
    w, h = image_size
    if max_size is not None:
        min_original_size = float(min((w, h)))
        max_original_size = float(max((w, h)))
        if max_original_size / min_original_size * size > max_size:
            size = max_size * min_original_size / max_original_size

    # 保证尺寸不会改变图像的宽高比
    if (w <= h and w == size) or (h <= w and h == size):
        return (h, w)

    if w < h:
        ow = int(round(size))
        oh = int(round(size * h / w))
    else:
        oh = int(round(size))
        ow = int(round(size * w / h))

    return (oh, ow)

# 调整图像大小并进行适当的填充或裁剪
def resize(datapoint, index, size, max_size=None, square=False, v2=False):
    # size can be min_size (scalar) or (w, h) tuple
    # size 可以是最小尺寸（标量）或（宽，高）元组
    def get_size(image_size, size, max_size=None):
        if isinstance(size, (list, tuple)):
            return size[::-1]  # 交换宽高
        else:
            return get_size_with_aspect_ratio(image_size, size, max_size)

    if square:
        size = size, size  # 如果是正方形，设置为宽高相等
    else:
        cur_size = (
            datapoint.frames[index].data.size()[-2:][::-1]
            if v2
            else datapoint.frames[index].data.size
        )
        size = get_size(cur_size, size, max_size)

    old_size = (
        datapoint.frames[index].data.size()[-2:][::-1]
        if v2
        else datapoint.frames[index].data.size
    )
    if v2:
        datapoint.frames[index].data = Fv2.resize(
            datapoint.frames[index].data, size, antialias=True  # 使用Fv2进行调整大小
        )
    else:
        datapoint.frames[index].data = F.resize(datapoint.frames[index].data, size)  # 使用F进行调整大小

    new_size = (
        datapoint.frames[index].data.size()[-2:][::-1]
        if v2
        else datapoint.frames[index].data.size
    )

    for obj in datapoint.frames[index].objects:
        if obj.segment is not None:
            obj.segment = F.resize(obj.segment[None, None], size).squeeze()  # 调整对象分割大小

    h, w = size
    datapoint.frames[index].size = (h, w)
    return datapoint

# 填充图像和对象分割
def pad(datapoint, index, padding, v2=False):
    old_h, old_w = datapoint.frames[index].size
    h, w = old_h, old_w
    if len(padding) == 2:
        # 假设只在右下角进行填充 / assumes that we only pad on the bottom right corners
        datapoint.frames[index].data = F.pad(
            datapoint.frames[index].data, (0, 0, padding[0], padding[1])
        )
        h += padding[1]
        w += padding[0]
    else:
        # 左、上、右、下四个方向的填充 / left, top, right, bottom
        datapoint.frames[index].data = F.pad(
            datapoint.frames[index].data,
            (padding[0], padding[1], padding[2], padding[3]),
        )
        h += padding[1] + padding[3]
        w += padding[0] + padding[2]

    datapoint.frames[index].size = (h, w)

    for obj in datapoint.frames[index].objects:
        if obj.segment is not None:
            if v2:
                if len(padding) == 2:
                    obj.segment = Fv2.pad(obj.segment, (0, 0, padding[0], padding[1]))
                else:
                    obj.segment = Fv2.pad(obj.segment, tuple(padding))
            else:
                if len(padding) == 2:
                    obj.segment = F.pad(obj.segment, (0, 0, padding[0], padding[1]))
                else:
                    obj.segment = F.pad(obj.segment, tuple(padding))
    return datapoint

# 随机水平翻转图像和对象
class RandomHorizontalFlip:
    def __init__(self, consistent_transform, p=0.5):
        self.p = p  # 翻转的概率
        self.consistent_transform = consistent_transform  # 是否一致地对所有帧进行翻转

    def __call__(self, datapoint, **kwargs):
        if self.consistent_transform:
            if random.random() < self.p:
                for i in range(len(datapoint.frames)):
                    datapoint = hflip(datapoint, i)  # 对所有帧进行翻转
            return datapoint
        for i in range(len(datapoint.frames)):
            if random.random() < self.p:
                datapoint = hflip(datapoint, i)  # 对每一帧独立进行翻转
        return datapoint

# 随机调整图像大小，支持统一或独立调整每帧的大小
class RandomResizeAPI:
    def __init__(
        self, sizes, consistent_transform, max_size=None, square=False, v2=False
    ):
        # 如果传入的是单一的整数，则将其转为元组形式
        if isinstance(sizes, int):
            sizes = (sizes,)
        assert isinstance(sizes, Iterable)
        self.sizes = list(sizes)
        self.max_size = max_size
        self.square = square
        self.consistent_transform = consistent_transform  # 是否对所有帧应用一致的变换
        self.v2 = v2

    def __call__(self, datapoint, **kwargs):
        # 如果是保持一致的变换（对所有帧进行相同处理）
        if self.consistent_transform:
            size = random.choice(self.sizes)  # 随机选择一个尺寸
            for i in range(len(datapoint.frames)):
                datapoint = resize(
                    datapoint, i, size, self.max_size, square=self.square, v2=self.v2
                )
            return datapoint
        # 如果是独立的变换（每帧可能不同）
        for i in range(len(datapoint.frames)):
            size = random.choice(self.sizes)  # 随机选择一个尺寸
            datapoint = resize(
                datapoint, i, size, self.max_size, square=self.square, v2=self.v2
            )
        return datapoint

# 转换图像为张量（Tensor）
class ToTensorAPI:
    def __init__(self, v2=False):
        self.v2 = v2  # 是否使用v2版本的转换方法

    def __call__(self, datapoint: VideoDatapoint, **kwargs):
        # 对视频数据点中的每一帧进行转化
        for img in datapoint.frames:
            if self.v2:
                img.data = Fv2.to_image_tensor(img.data)  # 使用v2版本的to_tensor
            else:
                img.data = F.to_tensor(img.data)  # 使用默认版本的to_tensor
        return datapoint

# 标准化图像数据
class NormalizeAPI:
    def __init__(self, mean, std, v2=False):
        self.mean = mean
        self.std = std
        self.v2 = v2      # 是否使用v2版本

    def __call__(self, datapoint: VideoDatapoint, **kwargs):
        # 对每一帧图像进行标准化处理
        for img in datapoint.frames:
            if self.v2:
                img.data = Fv2.convert_image_dtype(img.data, torch.float32)  # 转为float32类型
                img.data = Fv2.normalize(img.data, mean=self.mean, std=self.std)  # 标准化
            else:
                img.data = F.normalize(img.data, mean=self.mean, std=self.std)  # 使用标准的normalize方法

        return datapoint

# 组合多个数据预处理操作
class ComposeAPI:
    def __init__(self, transforms):
        self.transforms = transforms  # 包含多个预处理操作

    def __call__(self, datapoint, **kwargs):
        # 依次应用每个预处理操作
        for t in self.transforms:
            datapoint = t(datapoint, **kwargs)
        return datapoint

    def __repr__(self):
        # 打印操作序列
        format_string = self.__class__.__name__ + "("
        for t in self.transforms:
            format_string += "\n"
            format_string += "    {0}".format(t)
        format_string += "\n)"
        return format_string

# 随机将图像转为灰度图
class RandomGrayscale:
    def __init__(self, consistent_transform, p=0.5):
        self.p = p  # 转为灰度图的概率
        self.consistent_transform = consistent_transform  # 是否对所有帧进行一致的转换
        self.Grayscale = T.Grayscale(num_output_channels=3)  # 保证输出为3通道灰度图

    def __call__(self, datapoint: VideoDatapoint, **kwargs):
        # 如果是保持一致的变换（对所有帧进行相同处理）
        if self.consistent_transform:
            if random.random() < self.p:  # 随机决定是否进行灰度化处理
                for img in datapoint.frames:
                    img.data = self.Grayscale(img.data)  # 转为灰度图
            return datapoint
        # 如果是独立的变换（每帧可能不同）
        for img in datapoint.frames:
            if random.random() < self.p:  # 随机决定是否进行灰度化处理
                img.data = self.Grayscale(img.data)  # 转为灰度图
        return datapoint

# 随机调整图像的亮度、对比度、饱和度和色调
class ColorJitter:
    def __init__(self, consistent_transform, brightness, contrast, saturation, hue):
        self.consistent_transform = consistent_transform
        # 将亮度、对比度、饱和度、色调参数转为范围列表
        self.brightness = (
            brightness
            if isinstance(brightness, list)
            else [max(0, 1 - brightness), 1 + brightness]
        )
        self.contrast = (
            contrast
            if isinstance(contrast, list)
            else [max(0, 1 - contrast), 1 + contrast]
        )
        self.saturation = (
            saturation
            if isinstance(saturation, list)
            else [max(0, 1 - saturation), 1 + saturation]
        )
        self.hue = hue if isinstance(hue, list) or hue is None else ([-hue, hue])

    def __call__(self, datapoint: VideoDatapoint, **kwargs):
        # 如果是保持一致的变换（对所有帧进行相同处理）
        if self.consistent_transform:
            # 获取颜色抖动的参数 / Create a color jitter transformation params
            (
                fn_idx,
                brightness_factor,
                contrast_factor,
                saturation_factor,
                hue_factor,
            ) = T.ColorJitter.get_params(
                self.brightness, self.contrast, self.saturation, self.hue
            )
        for img in datapoint.frames:
            if not self.consistent_transform:
                # 获取颜色抖动的参数
                (
                    fn_idx,
                    brightness_factor,
                    contrast_factor,
                    saturation_factor,
                    hue_factor,
                ) = T.ColorJitter.get_params(
                    self.brightness, self.contrast, self.saturation, self.hue
                )
            for fn_id in fn_idx:
                if fn_id == 0 and brightness_factor is not None:
                    img.data = F.adjust_brightness(img.data, brightness_factor)  # 调整亮度
                elif fn_id == 1 and contrast_factor is not None:
                    img.data = F.adjust_contrast(img.data, contrast_factor)  # 调整对比度
                elif fn_id == 2 and saturation_factor is not None:
                    img.data = F.adjust_saturation(img.data, saturation_factor)  # 调整饱和度
                elif fn_id == 3 and hue_factor is not None:
                    img.data = F.adjust_hue(img.data, hue_factor)  # 调整色调
        return datapoint


class RandomAffine:
    def __init__(
        self,
        degrees,
        consistent_transform,
        scale=None,
        translate=None,
        shear=None,
        image_mean=(123, 116, 103),
        log_warning=True,
        num_tentatives=1,
        image_interpolation="bicubic",
    ):
        """
        随机仿射变换类。

        The mask is required for this transform.
        if consistent_transform if True, then the same random affine is applied to all frames and masks.
        /
        这次变换需要掩膜（mask）。
        如果consistent_transform为True，则相同的随机仿射变换将应用于所有帧和掩膜。
        """
        self.degrees = degrees if isinstance(degrees, list) else ([-degrees, degrees])
        self.scale = scale
        self.shear = (
            shear if isinstance(shear, list) else ([-shear, shear] if shear else None)
        )
        self.translate = translate
        self.fill_img = image_mean
        self.consistent_transform = consistent_transform
        self.log_warning = log_warning
        self.num_tentatives = num_tentatives

        if image_interpolation == "bicubic":
            self.image_interpolation = InterpolationMode.BICUBIC
        elif image_interpolation == "bilinear":
            self.image_interpolation = InterpolationMode.BILINEAR
        else:
            raise NotImplementedError

    def __call__(self, datapoint: VideoDatapoint, **kwargs):
        # 多次尝试进行仿射变换
        for _tentative in range(self.num_tentatives):
            res = self.transform_datapoint(datapoint)
            if res is not None:
                return res

        if self.log_warning:
            logging.warning(
                f"Skip RandomAffine for zero-area mask in first frame after {self.num_tentatives} tentatives"
            )
        return datapoint

    def transform_datapoint(self, datapoint: VideoDatapoint):
        # 获取图像的尺寸
        _, height, width = F.get_dimensions(datapoint.frames[0].data)
        img_size = [width, height]

        if self.consistent_transform:
            # 生成一个随机的仿射变换参数 / Create a random affine transformation
            affine_params = T.RandomAffine.get_params(
                degrees=self.degrees,
                translate=self.translate,
                scale_ranges=self.scale,
                shears=self.shear,
                img_size=img_size,
            )

        for img_idx, img in enumerate(datapoint.frames):
            # 获取每个图像的mask
            this_masks = [
                obj.segment.unsqueeze(0) if obj.segment is not None else None
                for obj in img.objects
            ]
            if not self.consistent_transform:
                # if not consistent we create a new affine params for every frame&mask pair Create a random affine transformation
                # 如果不一致，生成每一帧的仿射变换参数
                affine_params = T.RandomAffine.get_params(
                    degrees=self.degrees,
                    translate=self.translate,
                    scale_ranges=self.scale,
                    shears=self.shear,
                    img_size=img_size,
                )

            transformed_bboxes, transformed_masks = [], []
            for i in range(len(img.objects)):
                if this_masks[i] is None:
                    transformed_masks.append(None)
                    # 对于没有目标的mask，使用默认bbox / Dummy bbox for a dummy target
                    transformed_bboxes.append(torch.tensor([[0, 0, 1, 1]]))
                else:
                    transformed_mask = F.affine(
                        this_masks[i],
                        *affine_params,
                        interpolation=InterpolationMode.NEAREST,
                        fill=0.0,
                    )
                    if img_idx == 0 and transformed_mask.max() == 0:
                        # We are dealing with a video and the object is not visible in the first frame
                        # Return the datapoint without transformation
                        # /
                        # 我们正在处理一个视频，但是对象在第一帧中不可见
                        # 返回未经转换的数据点
                        return None
                    transformed_masks.append(transformed_mask.squeeze())

            # 更新每一帧的对象segment
            for i in range(len(img.objects)):
                img.objects[i].segment = transformed_masks[i]

            # 对图像进行仿射变换
            img.data = F.affine(
                img.data,
                *affine_params,
                interpolation=self.image_interpolation,
                fill=self.fill_img,
            )
        return datapoint


def random_mosaic_frame(
    datapoint,
    index,
    grid_h,
    grid_w,
    target_grid_y,
    target_grid_x,
    should_hflip,
):
    '''
    随机马赛克帧的生成函数。
    将当前帧拆分成小网格，并按要求对每个网格进行处理和拼接。
    '''
    # Step 1: downsize the images and paste them into a mosaic
    # 步骤1: 将图像下采样并粘贴到马赛克中
    image_data = datapoint.frames[index].data
    is_pil = isinstance(image_data, PILImage.Image)
    if is_pil:
        H_im = image_data.height
        W_im = image_data.width
        image_data_output = PILImage.new("RGB", (W_im, H_im))
    else:
        H_im = image_data.size(-2)
        W_im = image_data.size(-1)
        image_data_output = torch.zeros_like(image_data)

    downsize_cache = {}
    for grid_y in range(grid_h):
        for grid_x in range(grid_w):
            # 计算每个网格的起始和结束坐标
            y_offset_b = grid_y * H_im // grid_h
            x_offset_b = grid_x * W_im // grid_w
            y_offset_e = (grid_y + 1) * H_im // grid_h
            x_offset_e = (grid_x + 1) * W_im // grid_w
            H_im_downsize = y_offset_e - y_offset_b
            W_im_downsize = x_offset_e - x_offset_b

            # 检查是否已经缓存过该尺寸的下采样结果
            if (H_im_downsize, W_im_downsize) in downsize_cache:
                image_data_downsize = downsize_cache[(H_im_downsize, W_im_downsize)]
            else:
                image_data_downsize = F.resize(
                    image_data,
                    size=(H_im_downsize, W_im_downsize),
                    interpolation=InterpolationMode.BILINEAR,
                    antialias=True,    # 下采样时使用抗锯齿 / antialiasing for downsizing
                )
                downsize_cache[(H_im_downsize, W_im_downsize)] = image_data_downsize
            if should_hflip[grid_y, grid_x].item():
                image_data_downsize = F.hflip(image_data_downsize)

            # 将处理后的图像粘贴到输出图像中
            if is_pil:
                image_data_output.paste(image_data_downsize, (x_offset_b, y_offset_b))
            else:
                image_data_output[:, y_offset_b:y_offset_e, x_offset_b:x_offset_e] = (
                    image_data_downsize
                )

    datapoint.frames[index].data = image_data_output

    # Step 2: downsize the masks and paste them into the target grid of the mosaic
    # 步骤2: 对掩膜进行下采样并粘贴到马赛克中的目标网格
    for obj in datapoint.frames[index].objects:
        if obj.segment is None:
            continue
        assert obj.segment.shape == (H_im, W_im) and obj.segment.dtype == torch.uint8
        segment_output = torch.zeros_like(obj.segment)

        target_y_offset_b = target_grid_y * H_im // grid_h
        target_x_offset_b = target_grid_x * W_im // grid_w
        target_y_offset_e = (target_grid_y + 1) * H_im // grid_h
        target_x_offset_e = (target_grid_x + 1) * W_im // grid_w
        target_H_im_downsize = target_y_offset_e - target_y_offset_b
        target_W_im_downsize = target_x_offset_e - target_x_offset_b

        segment_downsize = F.resize(
            obj.segment[None, None],
            size=(target_H_im_downsize, target_W_im_downsize),
            interpolation=InterpolationMode.BILINEAR,
            antialias=True,  # 下采样时使用抗锯齿 / antialiasing for downsizing
        )[0, 0]
        if should_hflip[target_grid_y, target_grid_x].item():
            segment_downsize = F.hflip(segment_downsize[None, None])[0, 0]

        segment_output[
            target_y_offset_b:target_y_offset_e, target_x_offset_b:target_x_offset_e
        ] = segment_downsize
        obj.segment = segment_output

    return datapoint


class RandomMosaicVideoAPI:
    def __init__(self, prob=0.15, grid_h=2, grid_w=2, use_random_hflip=False):
        self.prob = prob  # 随机拼接的概率
        self.grid_h = grid_h  # 网格的高度
        self.grid_w = grid_w  # 网格的宽度
        self.use_random_hflip = use_random_hflip  # 是否随机水平翻转

    def __call__(self, datapoint, **kwargs):
        if random.random() > self.prob:
            return datapoint  # 如果随机数大于设置的概率，则返回原始数据

        # select a random location to place the target mask in the mosaic
        # 随机选择一个位置，将目标掩码放置在拼接网格中的位置
        target_grid_y = random.randint(0, self.grid_h - 1)
        target_grid_x = random.randint(0, self.grid_w - 1)

        # whether to flip each grid in the mosaic horizontally
        # 决定是否对拼接网格中的每个格子进行水平翻转
        if self.use_random_hflip:
            should_hflip = torch.rand(self.grid_h, self.grid_w) < 0.5
        else:
            should_hflip = torch.zeros(self.grid_h, self.grid_w, dtype=torch.bool)

        # 对每一帧应用随机拼接变换
        for i in range(len(datapoint.frames)):
            datapoint = random_mosaic_frame(
                datapoint,
                i,
                grid_h=self.grid_h,
                grid_w=self.grid_w,
                target_grid_y=target_grid_y,
                target_grid_x=target_grid_x,
                should_hflip=should_hflip,
            )

        return datapoint  # 返回拼接后的数据点
