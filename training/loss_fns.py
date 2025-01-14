# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from collections import defaultdict
from typing import Dict, List

import torch
import torch.distributed
import torch.nn as nn
import torch.nn.functional as F

from training.trainer import CORE_LOSS_KEY

from training.utils.distributed import get_world_size, is_dist_avail_and_initialized


def dice_loss(inputs, targets, num_objects, loss_on_multimask=False):
    """
    Compute the DICE loss, similar to generalized IOU for masks
    计算DICE损失，类似于广义的IOU用于掩膜
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
                任意形状的浮点张量，表示每个样本的预测结果。
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
                 与inputs形状相同的浮点张量，存储每个元素的二进制分类标签
                 (0表示负类，1表示正类)。
        num_objects: Number of objects in the batch
                批次中物体的数量。
        loss_on_multimask: True if multimask prediction is enabled
                如果启用了多掩膜预测，则为True。
    Returns:
        Dice loss tensor
        DICE损失张量
    """
    inputs = inputs.sigmoid()  # 对输入进行sigmoid激活，得到概率
    if loss_on_multimask:
        # inputs and targets are [N, M, H, W] where M corresponds to multiple predicted masks
        # 如果启用了多掩膜预测，inputs和targets是[N, M, H, W]，其中M表示多个预测掩膜
        assert inputs.dim() == 4 and targets.dim() == 4
        # flatten spatial dimension while keeping multimask channel dimension
        # 将空间维度展平，同时保持多掩膜通道维度
        inputs = inputs.flatten(2)
        targets = targets.flatten(2)
        numerator = 2 * (inputs * targets).sum(-1)
    else:
        inputs = inputs.flatten(1)  # 将输入展平成一维
        numerator = 2 * (inputs * targets).sum(1)  # 计算分子
    denominator = inputs.sum(-1) + targets.sum(-1)  # 计算分母
    loss = 1 - (numerator + 1) / (denominator + 1)  # 计算DICE损失
    if loss_on_multimask:
        return loss / num_objects  # 如果启用了多掩膜预测，返回平均损失
    return loss.sum() / num_objects  # 返回所有物体的平均损失


def sigmoid_focal_loss(
    inputs,
    targets,
    num_objects,
    alpha: float = 0.25,
    gamma: float = 2,
    loss_on_multimask=False,
):
    """
    Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.
    RetinaNet中用于密集检测的焦点损失：https://arxiv.org/abs/1708.02002
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
                任意形状的浮点张量，表示每个样本的预测结果。
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
                与inputs形状相同的浮点张量，存储每个元素的二进制分类标签
                 (0表示负类，1表示正类)。
        num_objects: Number of objects in the batch
                批次中物体的数量。
        alpha: (optional) Weighting factor in range (0,1) to balance
                positive vs negative examples. Default = -1 (no weighting).
                (可选) 用于平衡正负样本的权重因子，范围在(0,1)之间，默认值为0.25。
        gamma: Exponent of the modulating factor (1 - p_t) to
               balance easy vs hard examples.
               调制因子的指数，用于平衡容易与困难样本的损失，默认值为2。
        loss_on_multimask: True if multimask prediction is enabled
                如果启用了多掩膜预测，则为True。
    Returns:
        focal loss tensor
        焦点损失张量
    """
    prob = inputs.sigmoid()  # 对输入进行sigmoid激活，得到概率
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")  # 计算二元交叉熵损失
    p_t = prob * targets + (1 - prob) * (1 - targets)  # 计算p_t（目标类概率）
    loss = ce_loss * ((1 - p_t) ** gamma)  # 计算焦点损失

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)  # 计算加权因子
        loss = alpha_t * loss  # 乘以加权因子

    if loss_on_multimask:
        # loss is [N, M, H, W] where M corresponds to multiple predicted masks
        # 如果启用了多掩膜预测，loss的形状为[N, M, H, W]，其中M表示多个预测掩膜
        assert loss.dim() == 4
        return loss.flatten(2).mean(-1) / num_objects  # 在空间维度上求平均并除以物体数量 / average over spatial dims
    return loss.mean(1).sum() / num_objects  # 返回所有物体的平均损失


def iou_loss(
    inputs, targets, pred_ious, num_objects, loss_on_multimask=False, use_l1_loss=False
):
    """
    计算IoU损失
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
                任意形状的浮点张量，表示每个样本的预测结果。
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
                与inputs形状相同的浮点张量，存储每个元素的二进制分类标签 (0表示负类，1表示正类)。
        pred_ious: A float tensor containing the predicted IoUs scores per mask
                一个浮点张量，包含每个掩膜的预测IoU分数
        num_objects: Number of objects in the batch
                批次中物体的数量。
        loss_on_multimask: True if multimask prediction is enabled
                如果启用了多掩膜预测，则为True。
        use_l1_loss: Whether to use L1 loss is used instead of MSE loss
                是否使用L1损失，默认为False，表示使用MSE损失。
    Returns:
        IoU loss tensor
        IoU损失张量
    """
    assert inputs.dim() == 4 and targets.dim() == 4  # 确保输入和目标都是四维张量
    pred_mask = inputs.flatten(2) > 0  # 预测掩膜（阈值为0）
    gt_mask = targets.flatten(2) > 0  # 真实掩膜（阈值为0）
    area_i = torch.sum(pred_mask & gt_mask, dim=-1).float()  # 计算交集区域
    area_u = torch.sum(pred_mask | gt_mask, dim=-1).float()  # 计算并集区域
    actual_ious = area_i / torch.clamp(area_u, min=1.0)  # 计算IoU

    if use_l1_loss:
        loss = F.l1_loss(pred_ious, actual_ious, reduction="none")  # 使用L1损失
    else:
        loss = F.mse_loss(pred_ious, actual_ious, reduction="none")  # 使用MSE损失
    if loss_on_multimask:
        return loss / num_objects  # 如果启用了多掩膜预测，返回平均损失
    return loss.sum() / num_objects  # 返回所有物体的平均损失


class MultiStepMultiMasksAndIous(nn.Module):
    def __init__(
        self,
        weight_dict,
        focal_alpha=0.25,
        focal_gamma=2,
        supervise_all_iou=False,
        iou_use_l1_loss=False,
        pred_obj_scores=False,
        focal_gamma_obj_score=0.0,
        focal_alpha_obj_score=-1,
    ):
        """
        This class computes the multi-step multi-mask and IoU losses.
        该类用于计算多步多掩码（multi-mask）和 IoU 损失。

        Args:
            weight_dict: dict containing weights for focal, dice, iou losses
                一个字典，包含 focal、dice 和 iou 损失的权重。
            focal_alpha: alpha for sigmoid focal loss
                用于 sigmoid focal loss 的 alpha 参数。
            focal_gamma: gamma for sigmoid focal loss
                用于 sigmoid focal loss 的 gamma 参数
            supervise_all_iou: if True, back-prop iou losses for all predicted masks
                如果为 True，计算所有预测掩码的 IoU 损失。
            iou_use_l1_loss: use L1 loss instead of MSE loss for iou
                如果为 True，使用 L1 损失代替 MSE 损失来计算 IoU 损失。
            pred_obj_scores: if True, compute loss for object scores
                如果为 True，计算物体分数的损失。
            focal_gamma_obj_score: gamma for sigmoid focal loss on object scores
                用于物体分数的 sigmoid focal loss 的 gamma 参数。
            focal_alpha_obj_score: alpha for sigmoid focal loss on object scores
                用于物体分数的 sigmoid focal loss 的 alpha 参数。
        """

        super().__init__()
        self.weight_dict = weight_dict
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma
        assert "loss_mask" in self.weight_dict
        assert "loss_dice" in self.weight_dict
        assert "loss_iou" in self.weight_dict
        if "loss_class" not in self.weight_dict:
            self.weight_dict["loss_class"] = 0.0

        self.focal_alpha_obj_score = focal_alpha_obj_score
        self.focal_gamma_obj_score = focal_gamma_obj_score
        self.supervise_all_iou = supervise_all_iou
        self.iou_use_l1_loss = iou_use_l1_loss
        self.pred_obj_scores = pred_obj_scores

    def forward(self, outs_batch: List[Dict], targets_batch: torch.Tensor):
        # 确保 outs_batch 和 targets_batch 的长度相同
        assert len(outs_batch) == len(targets_batch)

        # 计算每个 batch 中物体的数量，使用 targets_batch 的第一个维度（即目标的数量）
        num_objects = torch.tensor(
            (targets_batch.shape[1]), device=targets_batch.device, dtype=torch.float
        )  # 每个 batch 中物体的数量 / Number of objects is fixed within a batch

        # 如果分布式训练可用且已初始化，进行跨设备的 all_reduce 操作，确保每个进程的 num_objects 一致
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_objects)
        # 将 num_objects 限制在每个设备上的最小值为 1，防止出现除零错误
        num_objects = torch.clamp(num_objects / get_world_size(), min=1).item()

        # 初始化损失字典，用于存储每个 batch 中的各项损失
        losses = defaultdict(int)

        # 遍历 outs_batch 和 targets_batch 中的每一对输出和目标
        for outs, targets in zip(outs_batch, targets_batch):
            # 对每一对 outs 和 targets 计算损失
            cur_losses = self._forward(outs, targets, num_objects)
            # 将当前计算的损失加入总损失字典中
            for k, v in cur_losses.items():
                losses[k] += v

        # 返回计算出的所有损失
        return losses

    def _forward(self, outputs: Dict, targets: torch.Tensor, num_objects):
        """
        Compute the losses related to the masks: the focal loss and the dice loss.
        and also the MAE or MSE loss between predicted IoUs and actual IoUs.
        计算与掩码相关的损失：包括 focal 损失和 dice 损失，
        以及预测的 IoU 与实际 IoU 之间的 MAE 或 MSE 损失。

        Here "multistep_pred_multimasks_high_res" is a list of multimasks (tensors
        of shape [N, M, H, W], where M could be 1 or larger, corresponding to
        one or multiple predicted masks from a click.
        这里的 "multistep_pred_multimasks_high_res" 是一个多掩码的列表
        （张量形状为 [N, M, H, W]，其中 M 可能为 1 或更大，表示来自一次点击的一个或多个预测掩码）。

        We back-propagate focal, dice losses only on the prediction channel
        with the lowest focal+dice loss between predicted mask and ground-truth.
        If `supervise_all_iou` is True, we backpropagate ious losses for all predicted masks.
        我们只在预测掩码与真实掩码之间的 focal 和 dice 损失最小的预测通道上反向传播 focal 和 dice 损失。
        如果 supervise_all_iou 为 True，我们将对所有预测的掩码反向传播 IoU 损失。
        """

        target_masks = targets.unsqueeze(1).float()
        # 将目标张量添加一个维度，并转换为 float 类型，形状变为 [N, 1, H, W]，用于后续计算。
        assert target_masks.dim() == 4  # [N, 1, H, W]

        # 从输出中提取多步预测的高分辨率掩膜、预测的IoU以及对象得分 logits
        src_masks_list = outputs["multistep_pred_multimasks_high_res"]
        ious_list = outputs["multistep_pred_ious"]
        object_score_logits_list = outputs["multistep_object_score_logits"]

        # 确保多步掩膜、IoU 和对象得分 logits 列表的长度一致
        assert len(src_masks_list) == len(ious_list)
        assert len(object_score_logits_list) == len(ious_list)

        # 在每个预测步骤中累积损失 / accumulate the loss over prediction steps
        losses = {"loss_mask": 0, "loss_dice": 0, "loss_iou": 0, "loss_class": 0}
        for src_masks, ious, object_score_logits in zip(
            src_masks_list, ious_list, object_score_logits_list
        ):
            # 使用 _update_losses 方法更新损失
            self._update_losses(
                losses, src_masks, target_masks, ious, num_objects, object_score_logits
            )
        # 使用 reduce_loss 方法计算总损失，并将其存入字典中
        losses[CORE_LOSS_KEY] = self.reduce_loss(losses)
        return losses

    def _update_losses(
        self, losses, src_masks, target_masks, ious, num_objects, object_score_logits
    ):
        # 扩展目标掩码，确保与预测的掩码形状一致
        target_masks = target_masks.expand_as(src_masks)
        # 计算每个预测步骤中的 focal、dice 和 iou 损失 / get focal, dice and iou loss on all output masks in a prediction step
        loss_multimask = sigmoid_focal_loss(
            src_masks,
            target_masks,
            num_objects,
            alpha=self.focal_alpha,
            gamma=self.focal_gamma,
            loss_on_multimask=True,
        )
        # 计算 dice 损失
        loss_multidice = dice_loss(
            src_masks, target_masks, num_objects, loss_on_multimask=True
        )

        # 如果不计算目标分类损失
        if not self.pred_obj_scores:
            # 初始化分类损失为 0
            loss_class = torch.tensor(
                0.0, dtype=loss_multimask.dtype, device=loss_multimask.device
            )
            # 假设目标物体存在，生成全 1 的目标对象标记
            target_obj = torch.ones(
                loss_multimask.shape[0],
                1,
                dtype=loss_multimask.dtype,
                device=loss_multimask.device,
            )
        else:
            # 计算目标物体是否存在
            target_obj = torch.any((target_masks[:, 0] > 0).flatten(1), dim=-1)[
                ..., None
            ].float()
            # 计算目标物体的分类损失
            loss_class = sigmoid_focal_loss(
                object_score_logits,
                target_obj,
                num_objects,
                alpha=self.focal_alpha_obj_score,
                gamma=self.focal_gamma_obj_score,
            )

        # 计算 IoU 损失
        loss_multiiou = iou_loss(
            src_masks,
            target_masks,
            ious,
            num_objects,
            loss_on_multimask=True,
            use_l1_loss=self.iou_use_l1_loss,
        )

        # 确保各个损失的维度为2
        assert loss_multimask.dim() == 2
        assert loss_multidice.dim() == 2
        assert loss_multiiou.dim() == 2

        # 如果有多个预测掩码
        if loss_multimask.size(1) > 1:
            # take the mask indices with the smallest focal + dice loss for back propagation
            # 选择具有最小 focal + dice 损失的掩码进行反向传播
            loss_combo = (
                loss_multimask * self.weight_dict["loss_mask"]
                + loss_multidice * self.weight_dict["loss_dice"]
            )
            best_loss_inds = torch.argmin(loss_combo, dim=-1)
            batch_inds = torch.arange(loss_combo.size(0), device=loss_combo.device)
            loss_mask = loss_multimask[batch_inds, best_loss_inds].unsqueeze(1)
            loss_dice = loss_multidice[batch_inds, best_loss_inds].unsqueeze(1)
            # calculate the iou prediction and slot losses only in the index
            # with the minimum loss for each mask (to be consistent w/ SAM)
            # 只在最小损失的掩码索引上计算 IoU 预测和槽损失（与 SAM 保持一致）
            if self.supervise_all_iou:
                loss_iou = loss_multiiou.mean(dim=-1).unsqueeze(1)
            else:
                loss_iou = loss_multiiou[batch_inds, best_loss_inds].unsqueeze(1)
        else:
            # 如果只有一个掩码，直接使用相应的损失
            loss_mask = loss_multimask
            loss_dice = loss_multidice
            loss_iou = loss_multiiou

        # backprop focal, dice and iou loss only if obj present
        # 仅在目标存在的情况下反向传播 focal、dice 和 IoU 损失
        loss_mask = loss_mask * target_obj
        loss_dice = loss_dice * target_obj
        loss_iou = loss_iou * target_obj

        # sum over batch dimension (note that the losses are already divided by num_objects)
        # 在 batch 维度上求和（注意损失已根据物体数量进行了归一化）
        losses["loss_mask"] += loss_mask.sum()
        losses["loss_dice"] += loss_dice.sum()
        losses["loss_iou"] += loss_iou.sum()
        losses["loss_class"] += loss_class

    def reduce_loss(self, losses):
        reduced_loss = 0.0
        # 根据权重字典中的权重计算最终的总损失
        for loss_key, weight in self.weight_dict.items():
            if loss_key not in losses:
                raise ValueError(f"{type(self)} doesn't compute {loss_key}")
            if weight != 0:
                reduced_loss += losses[loss_key] * weight

        return reduced_loss
