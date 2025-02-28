"""ViTDet backbone adapted from Detectron2"""

from functools import partial
from typing import List, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from efficient_track_anything.modeling.backbones.utils import (
    get_abs_pos,
    PatchEmbed,
    window_partition,
    window_unpartition,
)

from efficient_track_anything.modeling.efficienttam_utils import (
    DropPath,
    LayerScale,
    MLP,
)


class Attention(nn.Module):
    """
    Multi-head Attention block with relative position embeddings.
    /
    多头注意力机制块与相对位置编码
    """

    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=True,
        use_rel_pos=False,
        rel_pos_zero_init=True,
        input_size=None,
    ):
        """
        Args:
            dim (int): Number of input channels.
            num_heads (int): Number of attention heads.
            qkv_bias (bool:  If True, add a learnable bias to query, key, value.
            rel_pos (bool): If True, add relative positional embeddings to the attention map.
            rel_pos_zero_init (bool): If True, zero initialize relative positional parameters.
            input_size (int or None): Input resolution for calculating the relative positional
                parameter size.
            attn_type: Type of attention operation, e.g. "vanilla", "vanilla-xformer".
        """
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

        self.use_rel_pos = use_rel_pos

    def forward(self, x):
        B, H, W, _ = x.shape
        # qkv with shape (3, B, nHead, H * W, C)
        qkv = (
            self.qkv(x).reshape(B, H * W, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        )
        # q, k, v with shape (B * nHead, H * W, C)
        q, k, v = qkv.reshape(3, B * self.num_heads, H * W, -1).unbind(0)

        q = q.view(B, self.num_heads, H * W, -1)
        k = k.view(B, self.num_heads, H * W, -1)
        v = v.view(B, self.num_heads, H * W, -1)

        x = F.scaled_dot_product_attention(q, k, v)

        x = (
            x.view(B, self.num_heads, H, W, -1)
            .permute(0, 2, 3, 1, 4)
            .reshape(B, H, W, -1)
        )
        x = self.proj(x)

        return x


class Block(nn.Module):
    """Transformer blocks with support of window attention"""

    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=True,
        drop_path=0.0,
        norm_layer=nn.LayerNorm,
        act_layer=nn.GELU,
        use_rel_pos=False,
        rel_pos_zero_init=True,
        window_size=0,
        input_size=None,
        dropout=0.0,
        init_values=None,
    ):
        """
        Args:
            dim (int): Number of input channels.
            num_heads (int): Number of attention heads in each ViT block.
            mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
            qkv_bias (bool): If True, add a learnable bias to query, key, value.
            drop_path (float): Stochastic depth rate.
            norm_layer (nn.Module): Normalization layer.
            act_layer (nn.Module): Activation layer.
            use_rel_pos (bool): If True, add relative positional embeddings to the attention map.
            rel_pos_zero_init (bool): If True, zero initialize relative positional parameters.
            window_size (int): Window size for window attention blocks. If it equals 0, then not
                use window attention.
            input_size (int or None): Input resolution for calculating the relative positional
                parameter size.
            dropout (float): Dropout rate.
        /
        参数:
            dim (int): 输入通道数。
            num_heads (int): 每个 ViT 块中的注意力头数。
            mlp_ratio (float): MLP 隐藏层维度与输入维度的比值。
            qkv_bias (bool): 如果为 True，则为查询、键和值添加可学习偏置。
            drop_path (float): 随机深度率，表示丢弃的概率。
            norm_layer (nn.Module): 使用的归一化层。
            act_layer (nn.Module): 激活函数层。
            use_rel_pos (bool): 如果为 True，则在注意力图中添加相对位置编码。
            rel_pos_zero_init (bool): 如果为 True，则将相对位置编码参数初始化为零。
            window_size (int): 窗口注意力块的窗口大小。为 0 时，不使用窗口注意力。
            input_size (int 或 None): 输入分辨率，用于计算相对位置编码的大小。
            dropout (float): dropout 比例。
        """
        super().__init__()
        # 第一层归一化
        self.norm1 = norm_layer(dim)
        # 注意力层
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            use_rel_pos=use_rel_pos,
            rel_pos_zero_init=rel_pos_zero_init,
            input_size=input_size if window_size == 0 else (window_size, window_size),
        )
        # 如果使用 LayerScale，则为注意力输出添加层级缩放
        self.ls1 = (
            LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        )
        # 随机深度（DropPath）层。随机丢弃某些路径（即随机跳过某些网络层的计算）
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        # 第二层归一化
        self.norm2 = norm_layer(dim)
        # MLP 层
        self.mlp = MLP(
            dim,
            int(dim * mlp_ratio),
            dim,
            num_layers=2,
            activation=act_layer,
        )
        # MLP 输出加权（如果使用 LayerScale）
        self.ls2 = (
            LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        )
        # dropout 层
        self.dropout = nn.Dropout(dropout)
        # 窗口大小
        self.window_size = window_size

    def forward(self, x):
        shortcut = x  # 保存输入，用于残差连接
        # 归一化输入
        x = self.norm1(x)

        # 如果使用窗口注意力，将输入拆分成多个窗口 / Window partition
        if self.window_size > 0:
            H, W = x.shape[1], x.shape[2]
            x, pad_hw = window_partition(x, self.window_size)

        x = self.ls1(self.attn(x))  # 计算注意力

        # 如果使用窗口注意力，恢复窗口拆分后的输入 / Reverse window partition
        if self.window_size > 0:
            x = window_unpartition(x, self.window_size, pad_hw, (H, W))

        # 残差连接和 DropPath
        x = shortcut + self.dropout(self.drop_path(x))
        # MLP 和第二个残差连接
        x = x + self.dropout(self.drop_path(self.ls2(self.mlp(self.norm2(x)))))

        return x


class ViT(nn.Module):
    """
    This module implements Vision Transformer (ViT) backbone in :paper:`vitdet`.
    "Exploring Plain Vision Transformer Backbones for Object Detection",
    https://arxiv.org/abs/2203.16527
    """

    def __init__(
        self,
        img_size=1024,
        patch_size=16,
        in_chans=3,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        qkv_bias=True,
        drop_path_rate=0.0,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        act_layer=nn.GELU,
        use_abs_pos=True,
        use_rel_pos=False,
        rel_pos_zero_init=True,
        window_size=14,
        window_block_indexes=(0, 1, 3, 4, 6, 7, 9, 10),
        use_act_checkpoint=False,
        pretrain_img_size=224,
        pretrain_use_cls_token=True,
        dropout=0.0,
        weights_path=None,
        return_interm_layers=False,
        init_values=None,
    ):
        """
        Args:
            img_size (int): Input image size. Only relevant for rel pos.
            patch_size (int): Patch size.
            in_chans (int): Number of input image channels.
            embed_dim (int): Patch embedding dimension.
            depth (int): Depth of ViT.
            num_heads (int): Number of attention heads in each ViT block.
            mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
            qkv_bias (bool): If True, add a learnable bias to query, key, value.
            drop_path_rate (float): Stochastic depth rate.
            norm_layer (nn.Module): Normalization layer.
            act_layer (nn.Module): Activation layer.
            use_abs_pos (bool): If True, use absolute positional embeddings.
            use_rel_pos (bool): If True, add relative positional embeddings to the attention map.
            rel_pos_zero_init (bool): If True, zero initialize relative positional parameters.
            window_size (int): Window size for window attention blocks.
            window_block_indexes (list): Indexes for blocks using window attention.
            residual_block_indexes (list): Indexes for blocks using conv propagation.
            use_act_checkpoint (bool): If True, use activation checkpointing.
            pretrain_img_size (int): input image size for pretraining models.
            pretrain_use_cls_token (bool): If True, pretrainig models use class token.
            dropout (float): Dropout rate. Applied in residual blocks of attn, mlp and inside the mlp.
            path (str or None): Path to the pretrained weights.
            return_interm_layers (bool): Whether to return intermediate layers (all global attention blocks).
            freezing (BackboneFreezingType): Type of freezing.
        """
        super().__init__()
        self.pretrain_use_cls_token = pretrain_use_cls_token

        # 图像块嵌入（Patch embedding）
        self.patch_embed = PatchEmbed(
            kernel_size=(patch_size, patch_size),
            stride=(patch_size, patch_size),
            padding=(0, 0),
            in_chans=in_chans,
            embed_dim=embed_dim,
        )

        # 如果使用绝对位置嵌入
        if use_abs_pos:
            # 初始化预训练图像大小的绝对位置编码 / Initialize absolute positional embedding with pretrain image size.
            num_patches = (pretrain_img_size // patch_size) * (
                pretrain_img_size // patch_size
            )
            num_positions = (num_patches + 1) if pretrain_use_cls_token else num_patches
            # 初始化位置嵌入
            self.pos_embed = nn.Parameter(torch.zeros(1, num_positions, embed_dim))
        else:
            self.pos_embed = None

        # 随机深度衰减规则 / stochastic depth decay rule
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]

        self.blocks = nn.ModuleList()  # Transformer block 的列表

        # 在前向传播时，这些层的输出会被保留下来，供后续使用或返回中间层的特征。
        # 通过将全局注意力与局部窗口注意力结合使用，ViT能够在计算效率和表现力之间找到一个平衡。
        self.full_attn_ids = []  # 全局注意力的 block 索引，列表记录了哪些 Transformer block 采用了全局注意力机制
        cur_stage = 1
        for i in range(depth):
            # 创建每个 ViT block
            block = Block(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop_path=dpr[i],
                norm_layer=norm_layer,
                act_layer=act_layer,
                use_rel_pos=use_rel_pos,
                rel_pos_zero_init=rel_pos_zero_init,
                window_size=window_size if i in window_block_indexes else 0,
                input_size=(img_size // patch_size, img_size // patch_size),
                dropout=dropout,
                init_values=init_values,
            )
            # 如果该 block 不是窗口注意力 block，则将其加入全局注意力 block 列表
            if i not in window_block_indexes:
                self.full_attn_ids.append(i)
                cur_stage += 1

            # 将 block 加入网络中
            self.blocks.append(block)

        # 是否返回中间层
        self.return_interm_layers = return_interm_layers
        self.channel_list = (
            [embed_dim] * len(self.full_attn_ids)
            if return_interm_layers
            else [embed_dim]
        )

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        # 图像块嵌入（Patch embedding）
        x = self.patch_embed(x)
        # 如果使用位置嵌入
        if self.pos_embed is not None:
            x = x + get_abs_pos(
                self.pos_embed, self.pretrain_use_cls_token, (x.shape[1], x.shape[2])
            )

        outputs = []  # 存储输出的中间层
        for i, blk in enumerate(self.blocks):
            # 每个 Transformer block 的计算
            x = blk(x)
            # 如果是最后一个全局注意力 block 或需要返回中间层
            if (i == self.full_attn_ids[-1]) or (
                self.return_interm_layers and i in self.full_attn_ids
            ):
                # 将输出从 (B, C, H, W) 转为 (B, H, W, C)
                feats = x.permute(0, 3, 1, 2)
                outputs.append(feats)

        return outputs

    # 获取层的 ID，根据给定的层名
    def get_layer_id(self, layer_name):
        # https://github.com/microsoft/unilm/blob/master/beit/optim_factory.py#L33
        num_layers = self.get_num_layers()

        if layer_name.find("rel_pos") != -1:
            return num_layers + 1
        elif layer_name.find("pos_embed") != -1:
            return 0
        elif layer_name.find("patch_embed") != -1:
            return 0
        elif layer_name.find("blocks") != -1:
            return int(layer_name.split("blocks")[1].split(".")[1]) + 1
        else:
            return num_layers + 1

    # 获取 ViT 中 block 的数量
    def get_num_layers(self) -> int:
        return len(self.blocks)
