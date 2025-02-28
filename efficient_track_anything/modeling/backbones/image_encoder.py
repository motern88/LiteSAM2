# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from efficient_track_anything.modeling.efficienttam_utils import LayerNorm2d


class ImageEncoder(nn.Module):
    def __init__(
        self,
        trunk: nn.Module,  # 主干网络
        neck: nn.Module,  # 颈部网络
        scalp: int = 0,  # 用于控制是否丢弃最低分辨率的特征
    ):
        super().__init__()
        self.trunk = trunk
        self.neck = neck
        self.scalp = scalp
        assert (
            self.trunk.channel_list == self.neck.backbone_channel_list
        ), f"Channel dims of trunk and neck do not match. Trunk: {self.trunk.channel_list}, neck: {self.neck.backbone_channel_list}"
        # 断言主干网络和颈部网络的通道维度是否匹配，如果不匹配则抛出错误

    def forward(self, sample: torch.Tensor):
        # 通过主干网络前向传播 / Forward through backbone
        features, pos = self.neck(self.trunk(sample))
        if self.scalp > 0:
            # Discard the lowest resolution features
            # /
            # 如果 scalp 参数大于 0，则丢弃最低分辨率的特征
            features, pos = features[: -self.scalp], pos[: -self.scalp]

        # 获取最高分辨率的特征（即最后一个特征）
        src = features[-1]
        # 构造输出字典，包含视觉特征、视觉位置编码和主干网络的 FPN 特征
        output = {
            "vision_features": src,
            "vision_pos_enc": pos,
            "backbone_fpn": features,
        }
        return output


class ViTDetNeck(nn.Module):
    def __init__(
        self,
        position_encoding: nn.Module,  # 位置编码模块
        d_model: int,  # 模型的维度
        backbone_channel_list: List[int],  # 主干网络各层输出的通道数列表
        kernel_size: int = 1,
        stride: int = 1,
        padding: int = 0,
        neck_norm=None,  # 归一化层，默认为 None
    ):
        """Initialize the neck

        :param trunk: the backbone
        :param position_encoding: the positional encoding to use
        :param d_model: the dimension of the model
        :param neck_norm: the normalization to use
        """
        super().__init__()
        self.backbone_channel_list = backbone_channel_list
        self.position_encoding = position_encoding
        self.convs = nn.ModuleList()  # 用于存储卷积层的 ModuleList
        self.d_model = d_model
        use_bias = neck_norm is None  # 如果未使用归一化层，则卷积层使用偏置
        # backbone_channel_list中所有的通道数都会被映射到d_model维度
        for dim in self.backbone_channel_list:
            current = nn.Sequential()  # 创建一个顺序容器
            current.add_module(
                "conv_1x1",
                nn.Conv2d(
                    in_channels=dim,
                    out_channels=d_model,
                    kernel_size=1,
                    bias=use_bias,
                ),
            )
            if neck_norm is not None:  # 如果使用了归一化层
                current.add_module("norm_0", LayerNorm2d(d_model))  # 添加归一化层
            # 添加 3x3 卷积层，用于进一步提取特征
            current.add_module(
                "conv_3x3",
                nn.Conv2d(
                    in_channels=d_model,
                    out_channels=d_model,
                    kernel_size=3,
                    padding=1,
                    bias=use_bias,
                ),
            )
            if neck_norm is not None:  # 如果使用了归一化层
                current.add_module("norm_1", LayerNorm2d(d_model))  # 添加归一化层
            self.convs.append(current)  # 将当前顺序容器添加到 ModuleList 中

    def forward(self, xs: List[torch.Tensor]):

        out = [None] * len(self.convs)  # 创建一个与卷积层数量相同的列表，用于存储每层的输出
        pos = [None] * len(self.convs)  # 创建一个与卷积层数量相同的列表，用于存储每层的位置编码

        assert len(xs) == len(self.convs)  # 确保输入的张量列表 xs 和卷积层的数量一致

        # 将第一个输入张量通过第一个卷积层进行处理，得到输出
        x = xs[0]
        x_out = self.convs[0](x)

        out[0] = x_out  # 将第一个卷积层的输出存储到 out 列表中
        pos[0] = self.position_encoding(x_out).to(x_out.dtype)  # 为第一个卷积层的输出计算位置编码

        return out, pos
