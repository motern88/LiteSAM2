from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from lite_segment_anything_2.modeling.litesam2_utils import LayerNorm2d

class ImageEncoder(nn.Module):
    def __init__(
        self,
        trunk: nn.Module,  # 主干网络
        neck: nn.Module,  # 颈部网络
        scalp: int = 0,  # 头皮层数
    ):
        super().__init__()  # 初始化
        self.trunk = trunk
        self.neck = neck
        self.scalp = scalp
        # 验证主干网络和颈部网络的通道数是否匹配
        assert (
            self.trunk.channel_list == self.neck.backbone_channel_list
        ), f"Channel dims of trunk and neck do not match. Trunk: {self.trunk.channel_list}, neck: {self.neck.backbone_channel_list}"

    def forward(self, sample: torch.Tensor):
        # 通过主干网络进行前向传播
        features, pos = self.neck(self.trunk(sample))
        if self.scalp > 0:
            # 如果头皮层数大于0，丢弃分辨率最低的特征
            features, pos = features[: -self.scalp], pos[: -self.scalp]

        src = features[-1]  # 获取最高分辨率的特征
        output = {
            "vision_features": src,  # 视觉特征
            "vision_pos_enc": pos,  # 视觉位置编码
            "backbone_fpn": features,  # 主干网络的特征金字塔网络(FPN)
        }
        return output


class ViTDetNeck(nn.Module):
    def __init__(
        self,
        position_encoding: nn.Module,
        d_model: int,
        backbone_channel_list: List[int],
        kernel_size: int = 1,
        stride: int = 1,
        padding: int = 0,
        neck_norm=None,
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
        self.convs = nn.ModuleList()
        self.d_model = d_model
        use_bias = neck_norm is None
        for dim in self.backbone_channel_list:
            current = nn.Sequential()
            current.add_module(
                "conv_1x1",
                nn.Conv2d(
                    in_channels=dim,
                    out_channels=d_model,
                    kernel_size=1,
                    bias=use_bias,
                ),
            )
            if neck_norm is not None:
                current.add_module("norm_0", LayerNorm2d(d_model))
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
            if neck_norm is not None:
                current.add_module("norm_1", LayerNorm2d(d_model))
            self.convs.append(current)

    def forward(self, xs: List[torch.Tensor]):
        out = [None] * len(self.convs)
        pos = [None] * len(self.convs)
        assert len(xs) == len(self.convs)

        x = xs[0]
        x_out = self.convs[0](x)
        out[0] = x_out
        pos[0] = self.position_encoding(x_out).to(x_out.dtype)

        return out, pos
