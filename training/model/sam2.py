# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import logging

import numpy as np
import torch
import torch.distributed
from sam2.modeling.sam2_base import SAM2Base
from sam2.modeling.sam2_utils import (
    get_1d_sine_pe,
    get_next_point,
    sample_box_points,
    select_closest_cond_frames,
)

from sam2.utils.misc import concat_points

from training.utils.data_utils import BatchedVideoDatapoint


class SAM2Train(SAM2Base):
    def __init__(
        self,
        image_encoder,  # 图像编码器
        memory_attention=None,  # 记忆注意力模块
        memory_encoder=None,  # 记忆编码器
        prob_to_use_pt_input_for_train=0.0,  # 训练时使用点输入的概率
        prob_to_use_pt_input_for_eval=0.0,  # 评估时使用点输入的概率
        prob_to_use_box_input_for_train=0.0,  # 训练时使用框输入的概率
        prob_to_use_box_input_for_eval=0.0,  # 评估时使用框输入的概率
        # if it is greater than 1, we interactive point sampling in the 1st frame and other randomly selected frames
        # 如果该值大于1，则在第1帧和其他随机选择的帧中进行交互式点采样
        num_frames_to_correct_for_train=1,  # 训练时纠正的帧数 default: 仅在第一帧上进行迭代采样 / only iteratively sample on first frame
        num_frames_to_correct_for_eval=1,  # 评估时纠正的帧数 default: 仅在第一帧上进行迭代采样 / only iteratively sample on first frame
        rand_frames_to_correct_for_train=False,  # 训练时是否随机选择纠正帧
        rand_frames_to_correct_for_eval=False,  # 评估时是否随机选择纠正帧
        # how many frames to use as initial conditioning frames (for both point input and mask input; the first frame is always used as an initial conditioning frame)
        # - if `rand_init_cond_frames` below is True, we randomly sample 1~num_init_cond_frames initial conditioning frames
        # - otherwise we sample a fixed number of num_init_cond_frames initial conditioning frames
        # note: for point input, we sample correction points on all such initial conditioning frames, and we require that `num_frames_to_correct` >= `num_init_cond_frames`;
        # these are initial conditioning frames because as we track the video, more conditioning frames might be added
        # when a frame receives correction clicks under point input if `add_all_frames_to_correct_as_cond=True`
        # /
        # 使用多少帧作为初始条件帧（对于点输入和掩码输入；第一帧始终作为初始条件帧）
        # - 如果下面的 `rand_init_cond_frames` 为 True，我们会随机选择 1 到 `num_init_cond_frames` 帧作为初始条件帧
        # - 否则，我们会选择固定数量的 `num_init_cond_frames` 帧作为初始条件帧
        # 注意：对于点输入，我们会在所有这些初始条件帧上采样修正点，并且我们要求 `num_frames_to_correct` 必须大于等于 `num_init_cond_frames`；
        # 这些帧被称为初始条件帧，因为在追踪视频的过程中，如果某帧在点输入下接收到修正点击，更多的条件帧可能会被添加进来
        # 如果 `add_all_frames_to_correct_as_cond=True`，所有接收到修正点击的帧都将作为条件帧添加。
        num_init_cond_frames_for_train=1,  # 训练时的初始条件帧数 default: 仅使用第一帧作为初始条件帧 / only use the first frame as initial conditioning frame
        num_init_cond_frames_for_eval=1,  # 评估时的初始条件帧数 default: 仅使用第一帧作为初始条件帧 / only use the first frame as initial conditioning frame
        rand_init_cond_frames_for_train=True,  # 训练时是否随机选择初始条件帧 default: 随机选择 1 到 num_init_cond_frames_for_train 个初始条件帧（与之前的 TA 数据加载器保持一致） / random 1~num_init_cond_frames_for_train cond frames (to be constent w/ previous TA data loader)
        rand_init_cond_frames_for_eval=False,  # 评估时是否随机选择初始条件帧
        # if `add_all_frames_to_correct_as_cond` is True, we also append to the conditioning frame list any frame that receives a later correction click
        # if `add_all_frames_to_correct_as_cond` is False, we conditioning frame list to only use those initial conditioning frames
        # /
        # 如果 `add_all_frames_to_correct_as_cond` 为 True，我们还会将任何收到后续修正点击的帧添加到条件帧列表中
        # 如果 `add_all_frames_to_correct_as_cond` 为 False，我们的条件帧列表仅使用那些初始条件帧
        add_all_frames_to_correct_as_cond=False,  # 是否将所有被纠正的帧添加到条件帧中
        # how many additional correction points to sample (on each frame selected to be corrected)
        # note that the first frame receives an initial input click (in addition to any correction clicks)
        # /
        # 每个被选中进行修正的帧要采样多少个额外的修正点。 注意，第一帧除了收到初始输入点击外，还会收到任何修正点击
        num_correction_pt_per_frame=7,  # 每帧纠正点的数量
        # method for point sampling during evaluation
        # "uniform" (sample uniformly from error region) or "center" (use the point with the largest distance to error region boundary)
        # default to "center" to be consistent with evaluation in the SAM paper
        # /
        # 在评估期间用于点采样的方法 "uniform"（从错误区域均匀采样）或 "center"（使用与错误区域边界最远的点）
        # 默认为 "center"，以与SAM论文中的评估一致
        pt_sampling_for_eval="center",  # 评估时的点采样方法
        # During training, we optionally allow sampling the correction points from GT regions
        # instead of the prediction error regions with a small probability. This might allow the
        # model to overfit less to the error regions in training datasets
        # /
        # 在训练期间，我们可选地允许以小概率从GT区域采样修正点，而不是从预测误差区域中采样。这样可以减少模型在训练数据集中过拟合误差区域的情况
        prob_to_sample_from_gt_for_train=0.0,  # 训练时从GT区域采样的概率
        use_act_ckpt_iterative_pt_sampling=False,  # 是否使用活动检查点进行迭代点采样
        # whether to forward image features per frame (as it's being tracked) during evaluation, instead of forwarding image features
        # of all frames at once. This avoids backbone OOM errors on very long videos in evaluation, but could be slightly slower.
        # /
        # 在评估过程中，是否在每一帧被跟踪时单独前向图像特征，
        # 而不是一次性地前向所有帧的图像特征。这样可以避免在评估非常长的视频时出现骨干网络（backbone）内存溢出（OOM）错误，但可能会稍微变慢。
        forward_backbone_per_frame_for_eval=False,  # 是否逐帧计算图像特征
        freeze_image_encoder=False,  # 是否冻结图像编码器
        **kwargs,
    ):
        super().__init__(image_encoder, memory_attention, memory_encoder, **kwargs)
        self.use_act_ckpt_iterative_pt_sampling = use_act_ckpt_iterative_pt_sampling
        self.forward_backbone_per_frame_for_eval = forward_backbone_per_frame_for_eval

        # 设置点采样和条件帧 / Point sampler and conditioning frames
        self.prob_to_use_pt_input_for_train = prob_to_use_pt_input_for_train
        self.prob_to_use_box_input_for_train = prob_to_use_box_input_for_train
        self.prob_to_use_pt_input_for_eval = prob_to_use_pt_input_for_eval
        self.prob_to_use_box_input_for_eval = prob_to_use_box_input_for_eval
        if prob_to_use_pt_input_for_train > 0 or prob_to_use_pt_input_for_eval > 0:
            logging.info(
                f"Training with points (sampled from masks) as inputs with p={prob_to_use_pt_input_for_train}"
            )
            assert num_frames_to_correct_for_train >= num_init_cond_frames_for_train
            assert num_frames_to_correct_for_eval >= num_init_cond_frames_for_eval

        self.num_frames_to_correct_for_train = num_frames_to_correct_for_train
        self.num_frames_to_correct_for_eval = num_frames_to_correct_for_eval
        self.rand_frames_to_correct_for_train = rand_frames_to_correct_for_train
        self.rand_frames_to_correct_for_eval = rand_frames_to_correct_for_eval
        # 初始多条件帧 / Initial multi-conditioning frames
        self.num_init_cond_frames_for_train = num_init_cond_frames_for_train
        self.num_init_cond_frames_for_eval = num_init_cond_frames_for_eval
        self.rand_init_cond_frames_for_train = rand_init_cond_frames_for_train
        self.rand_init_cond_frames_for_eval = rand_init_cond_frames_for_eval
        self.add_all_frames_to_correct_as_cond = add_all_frames_to_correct_as_cond
        self.num_correction_pt_per_frame = num_correction_pt_per_frame
        self.pt_sampling_for_eval = pt_sampling_for_eval
        self.prob_to_sample_from_gt_for_train = prob_to_sample_from_gt_for_train
        # A random number generator with a fixed initial seed across GPUs
        # 使用一个固定初始种子的随机数生成器，确保在多个GPU上使用相同的随机数种子
        self.rng = np.random.default_rng(seed=42)

        if freeze_image_encoder:
            for p in self.image_encoder.parameters():
                p.requires_grad = False  # 冻结图像编码器的参数

    def forward(self, input: BatchedVideoDatapoint):
        if self.training or not self.forward_backbone_per_frame_for_eval:
            # precompute image features on all frames before tracking
            # 如果是训练模式或评估时不逐帧计算图像特征，预先计算所有帧的图像特征
            backbone_out = self.forward_image(input.flat_img_batch)
        else:
            # defer image feature computation on a frame until it's being tracked
            # 如果是评估模式且逐帧计算图像特征，则推迟计算
            backbone_out = {"backbone_fpn": None, "vision_pos_enc": None}
        backbone_out = self.prepare_prompt_inputs(backbone_out, input)  # 准备输入的提示
        previous_stages_out = self.forward_tracking(backbone_out, input)  # 向前传播到跟踪阶段

        return previous_stages_out

    def _prepare_backbone_features_per_frame(self, img_batch, img_ids):
        """
        Compute the image backbone features on the fly for the given img_ids.
        在给定的img_ids上计算图像的骨干特征。
        """
        # Only forward backbone on unique image ids to avoid repetitive computation
        # (if `img_ids` has only one element, it's already unique so we skip this step).
        # 只对唯一的图像ID计算骨干特征，避免重复计算
        if img_ids.numel() > 1:
            unique_img_ids, inv_ids = torch.unique(img_ids, return_inverse=True)
        else:
            unique_img_ids, inv_ids = img_ids, None

        # 对这些唯一的图像ID计算图像特征 / Compute the image features on those unique image ids
        image = img_batch[unique_img_ids]
        backbone_out = self.forward_image(image)
        (
            _,
            vision_feats,
            vision_pos_embeds,
            feat_sizes,
        ) = self._prepare_backbone_features(backbone_out)
        # Inverse-map image features for `unique_img_ids` to the final image features
        # for the original input `img_ids`.
        # 将图像特征从唯一图像ID映射回原始输入的img_ids
        if inv_ids is not None:
            image = image[inv_ids]
            vision_feats = [x[:, inv_ids] for x in vision_feats]
            vision_pos_embeds = [x[:, inv_ids] for x in vision_pos_embeds]

        return image, vision_feats, vision_pos_embeds, feat_sizes

    def prepare_prompt_inputs(self, backbone_out, input, start_frame_idx=0):
        """
        Prepare input mask, point or box prompts. Optionally, we allow tracking from
        a custom `start_frame_idx` to the end of the video (for evaluation purposes).
        /
        准备输入掩膜、点或框提示。可以选择从自定义的 `start_frame_idx` 跟踪视频的其余部分（用于评估）。
        """
        # Load the ground-truth masks on all frames (so that we can later
        # sample correction points from them)
        # 加载所有帧的地面真值掩膜（以便稍后从中采样修正点）

        # gt_masks_per_frame = {
        #     stage_id: targets.segments.unsqueeze(1)  # [B, 1, H_im, W_im]
        #     for stage_id, targets in enumerate(input.find_targets)
        # }
        gt_masks_per_frame = {
            stage_id: masks.unsqueeze(1)  # [B, 1, H_im, W_im]
            for stage_id, masks in enumerate(input.masks)
        }
        # gt_masks_per_frame = input.masks.unsqueeze(2) # [T,B,1,H_im,W_im] keep everything in tensor form
        backbone_out["gt_masks_per_frame"] = gt_masks_per_frame
        num_frames = input.num_frames
        backbone_out["num_frames"] = num_frames

        # Randomly decide whether to use point inputs or mask inputs
        # 随机决定是否使用点输入或掩膜输入
        if self.training:
            prob_to_use_pt_input = self.prob_to_use_pt_input_for_train
            prob_to_use_box_input = self.prob_to_use_box_input_for_train
            num_frames_to_correct = self.num_frames_to_correct_for_train
            rand_frames_to_correct = self.rand_frames_to_correct_for_train
            num_init_cond_frames = self.num_init_cond_frames_for_train
            rand_init_cond_frames = self.rand_init_cond_frames_for_train
        else:
            prob_to_use_pt_input = self.prob_to_use_pt_input_for_eval
            prob_to_use_box_input = self.prob_to_use_box_input_for_eval
            num_frames_to_correct = self.num_frames_to_correct_for_eval
            rand_frames_to_correct = self.rand_frames_to_correct_for_eval
            num_init_cond_frames = self.num_init_cond_frames_for_eval
            rand_init_cond_frames = self.rand_init_cond_frames_for_eval
        if num_frames == 1:
            # here we handle a special case for mixing video + SAM on image training,
            # where we force using point input for the SAM task on static images
            # 在混合视频 + SAM 训练时处理一个特殊情况，
            # 强制在静态图像的 SAM 任务中使用点输入
            prob_to_use_pt_input = 1.0
            num_frames_to_correct = 1
            num_init_cond_frames = 1
        assert num_init_cond_frames >= 1
        # (here `self.rng.random()` returns value in range 0.0 <= X < 1.0)
        # (这里 `self.rng.random()` 返回一个范围为 0.0 <= X < 1.0 的值)
        use_pt_input = self.rng.random() < prob_to_use_pt_input
        if rand_init_cond_frames and num_init_cond_frames > 1:
            # randomly select 1 to `num_init_cond_frames` frames as initial conditioning frames
            # 随机选择 1 到 `num_init_cond_frames` 帧作为初始条件帧
            num_init_cond_frames = self.rng.integers(
                1, num_init_cond_frames, endpoint=True
            )
        if (
            use_pt_input
            and rand_frames_to_correct
            and num_frames_to_correct > num_init_cond_frames
        ):
            # randomly select `num_init_cond_frames` to `num_frames_to_correct` frames to sample
            # correction clicks (only for the case of point input)
            # /
            # 随机选择 `num_init_cond_frames` 到 `num_frames_to_correct` 帧来采样修正点击（仅在使用点输入时）
            num_frames_to_correct = self.rng.integers(
                num_init_cond_frames, num_frames_to_correct, endpoint=True
            )
        backbone_out["use_pt_input"] = use_pt_input

        # 采样初始条件帧 / Sample initial conditioning frames
        if num_init_cond_frames == 1:
            init_cond_frames = [start_frame_idx]  # 起始帧 / starting frame
        else:
            # starting frame + randomly selected remaining frames (without replacement)
            # 起始帧 + 随机选择剩余的帧（不重复选择）
            init_cond_frames = [start_frame_idx] + self.rng.choice(
                range(start_frame_idx + 1, num_frames),
                num_init_cond_frames - 1,
                replace=False,
            ).tolist()
        backbone_out["init_cond_frames"] = init_cond_frames
        backbone_out["frames_not_in_init_cond"] = [
            t for t in range(start_frame_idx, num_frames) if t not in init_cond_frames
        ]
        # 在初始条件帧上准备掩膜或点输入 / Prepare mask or point inputs on initial conditioning frames
        backbone_out["mask_inputs_per_frame"] = {}  # {frame_idx: <input_masks>}
        backbone_out["point_inputs_per_frame"] = {}  # {frame_idx: <input_points>}
        for t in init_cond_frames:
            if not use_pt_input:
                backbone_out["mask_inputs_per_frame"][t] = gt_masks_per_frame[t]
            else:
                # During training # P(box) = prob_to_use_pt_input * prob_to_use_box_input
                # 在训练时 # P(box) = prob_to_use_pt_input * prob_to_use_box_input
                use_box_input = self.rng.random() < prob_to_use_box_input
                if use_box_input:
                    points, labels = sample_box_points(
                        gt_masks_per_frame[t],
                    )
                else:
                    # (here we only sample **one initial point** on initial conditioning frames from the
                    # ground-truth mask; we may sample more correction points on the fly)
                    # （这里我们仅在初始条件帧上从地面真值掩膜中采样 **一个初始点**；我们可能会在运行时采样更多修正点）
                    points, labels = get_next_point(
                        gt_masks=gt_masks_per_frame[t],
                        pred_masks=None,
                        method=(
                            "uniform" if self.training else self.pt_sampling_for_eval
                        ),
                    )

                point_inputs = {"point_coords": points, "point_labels": labels}
                backbone_out["point_inputs_per_frame"][t] = point_inputs

        # Sample frames where we will add correction clicks on the fly
        # based on the error between prediction and ground-truth masks
        # 采样在运行时我们将要添加修正点击的帧
        # 基于预测与地面真值掩膜之间的误差
        if not use_pt_input:
            # 在使用掩膜输入时不会采样修正点 / no correction points will be sampled when using mask inputs
            frames_to_add_correction_pt = []
        elif num_frames_to_correct == num_init_cond_frames:
            frames_to_add_correction_pt = init_cond_frames
        else:
            assert num_frames_to_correct > num_init_cond_frames
            # 初始条件帧 + 随机选择剩余的帧（不重复选择） / initial cond frame + randomly selected remaining frames (without replacement)
            extra_num = num_frames_to_correct - num_init_cond_frames
            frames_to_add_correction_pt = (
                init_cond_frames
                + self.rng.choice(
                    backbone_out["frames_not_in_init_cond"], extra_num, replace=False
                ).tolist()
            )
        backbone_out["frames_to_add_correction_pt"] = frames_to_add_correction_pt

        return backbone_out

    def forward_tracking(
        self, backbone_out, input: BatchedVideoDatapoint, return_dict=False
    ):
        """
        Forward video tracking on each frame (and sample correction clicks).
        执行视频跟踪，每一帧都进行前向传播（并采样修正点击）
        """
        # 检查图像特征是否已经计算
        img_feats_already_computed = backbone_out["backbone_fpn"] is not None
        if img_feats_already_computed:
            # Prepare the backbone features
            # - vision_feats and vision_pos_embeds are in (HW)BC format
            # 如果图像特征已经计算，则准备骨干特征
            # - vision_feats 和 vision_pos_embeds 的格式为 (HW)BC
            (
                _,
                vision_feats,
                vision_pos_embeds,
                feat_sizes,
            ) = self._prepare_backbone_features(backbone_out)

        # 开始阶段循环 / Starting the stage loop
        num_frames = backbone_out["num_frames"]
        init_cond_frames = backbone_out["init_cond_frames"]
        frames_to_add_correction_pt = backbone_out["frames_to_add_correction_pt"]
        # first process all the initial conditioning frames to encode them as memory,
        # and then conditioning on them to track the remaining frames
        # 首先处理所有的初始条件帧，将它们编码为记忆，
        # 然后以这些记忆为条件，跟踪剩余的帧
        processing_order = init_cond_frames + backbone_out["frames_not_in_init_cond"]
        output_dict = {
            "cond_frame_outputs": {},  # dict containing {frame_idx: <out>}
            "non_cond_frame_outputs": {},  # dict containing {frame_idx: <out>}
        }
        for stage_id in processing_order:
            # # 获取当前帧的图像特征 / Get the image features for the current frames
            # img_ids = input.find_inputs[stage_id].img_ids
            img_ids = input.flat_obj_to_img_idx[stage_id]
            if img_feats_already_computed:
                # Retrieve image features according to img_ids (if they are already computed).
                # 如果图像特征已经计算，则根据 img_ids 获取图像特征
                current_vision_feats = [x[:, img_ids] for x in vision_feats]
                current_vision_pos_embeds = [x[:, img_ids] for x in vision_pos_embeds]
            else:
                # Otherwise, compute the image features on the fly for the given img_ids
                # (this might be used for evaluation on long videos to avoid backbone OOM).
                # 否则，动态计算给定 img_ids 的图像特征
                # （这对于评估长视频时避免骨干网络内存溢出非常有用）
                (
                    _,
                    current_vision_feats,
                    current_vision_pos_embeds,
                    feat_sizes,
                ) = self._prepare_backbone_features_per_frame(
                    input.flat_img_batch, img_ids
                )

            # Get output masks based on this frame's prompts and previous memory
            # 基于当前帧的提示和之前的记忆，获取输出的掩码
            current_out = self.track_step(
                frame_idx=stage_id,
                is_init_cond_frame=stage_id in init_cond_frames,
                current_vision_feats=current_vision_feats,
                current_vision_pos_embeds=current_vision_pos_embeds,
                feat_sizes=feat_sizes,
                point_inputs=backbone_out["point_inputs_per_frame"].get(stage_id, None),
                mask_inputs=backbone_out["mask_inputs_per_frame"].get(stage_id, None),
                gt_masks=backbone_out["gt_masks_per_frame"].get(stage_id, None),
                frames_to_add_correction_pt=frames_to_add_correction_pt,
                output_dict=output_dict,
                num_frames=num_frames,
            )
            # Append the output, depending on whether it's a conditioning frame
            # 根据是否是条件帧，决定是否将输出添加到条件帧输出中
            add_output_as_cond_frame = stage_id in init_cond_frames or (
                self.add_all_frames_to_correct_as_cond
                and stage_id in frames_to_add_correction_pt
            )
            if add_output_as_cond_frame:
                output_dict["cond_frame_outputs"][stage_id] = current_out
            else:
                output_dict["non_cond_frame_outputs"][stage_id] = current_out

        if return_dict:
            return output_dict
        # turn `output_dict` into a list for loss function
        # 将 `output_dict` 转换为列表，以供损失函数使用
        all_frame_outputs = {}
        all_frame_outputs.update(output_dict["cond_frame_outputs"])
        all_frame_outputs.update(output_dict["non_cond_frame_outputs"])
        all_frame_outputs = [all_frame_outputs[t] for t in range(num_frames)]
        # Make DDP happy with activation checkpointing by removing unused keys
        # 使 DDP 在激活检查点时不报错，去除未使用的键
        all_frame_outputs = [
            {k: v for k, v in d.items() if k != "obj_ptr"} for d in all_frame_outputs
        ]

        return all_frame_outputs

    def track_step(
        self,
        frame_idx,
        is_init_cond_frame,
        current_vision_feats,
        current_vision_pos_embeds,
        feat_sizes,
        point_inputs,
        mask_inputs,
        output_dict,
        num_frames,
        track_in_reverse=False,  # 是否反向时间顺序进行跟踪（用于演示） / tracking in reverse time order (for demo usage)
        run_mem_encoder=True,  # 是否对预测的掩码运行记忆编码器 / Whether to run the memory encoder on the predicted masks.
        prev_sam_mask_logits=None,  # 上一帧预测的 SAM 掩码 logits / The previously predicted SAM mask logits.
        frames_to_add_correction_pt=None,  # 需要添加修正点击的帧
        gt_masks=None,  # 当前帧的 ground truth 掩码
    ):
        if frames_to_add_correction_pt is None:
            frames_to_add_correction_pt = []
        # 调用 `_track_step` 函数获取当前帧的输出，以及相关的 SAM 输出和特征
        current_out, sam_outputs, high_res_features, pix_feat = self._track_step(
            frame_idx,
            is_init_cond_frame,
            current_vision_feats,
            current_vision_pos_embeds,
            feat_sizes,
            point_inputs,
            mask_inputs,
            output_dict,
            num_frames,
            track_in_reverse,
            prev_sam_mask_logits,
        )

        # 解包 SAM 输出，包括多分辨率的掩码和其他相关信息
        (
            low_res_multimasks,
            high_res_multimasks,
            ious,
            low_res_masks,
            high_res_masks,
            obj_ptr,
            object_score_logits,
        ) = sam_outputs

        # 将低分辨率和高分辨率的预测掩码保存到当前输出中
        current_out["multistep_pred_masks"] = low_res_masks
        current_out["multistep_pred_masks_high_res"] = high_res_masks
        current_out["multistep_pred_multimasks"] = [low_res_multimasks]
        current_out["multistep_pred_multimasks_high_res"] = [high_res_multimasks]
        current_out["multistep_pred_ious"] = [ious]
        current_out["multistep_point_inputs"] = [point_inputs]
        current_out["multistep_object_score_logits"] = [object_score_logits]

        # Optionally, sample correction points iteratively to correct the mask
        # 如果当前帧在需要添加修正点击的帧列表中，进行修正点击的迭代采样
        if frame_idx in frames_to_add_correction_pt:
            # 调用修正点击采样函数进行迭代修正
            point_inputs, final_sam_outputs = self._iter_correct_pt_sampling(
                is_init_cond_frame,
                point_inputs,
                gt_masks,
                high_res_features,
                pix_feat,
                low_res_multimasks,
                high_res_multimasks,
                ious,
                low_res_masks,
                high_res_masks,
                object_score_logits,
                current_out,
            )
            # 解包修正后的 SAM 输出，得到最终的掩码和其他信息
            (
                _,
                _,
                _,
                low_res_masks,
                high_res_masks,
                obj_ptr,
                object_score_logits,
            ) = final_sam_outputs

        # Use the final prediction (after all correction steps for output and eval)
        # 使用最终的预测掩码（经过所有修正步骤后的输出）
        current_out["pred_masks"] = low_res_masks
        current_out["pred_masks_high_res"] = high_res_masks
        current_out["obj_ptr"] = obj_ptr

        # Finally run the memory encoder on the predicted mask to encode
        # it into a new memory feature (that can be used in future frames)
        # 最后，使用记忆编码器对预测的掩码进行编码，生成新的记忆特征
        self._encode_memory_in_output(
            current_vision_feats,
            feat_sizes,
            point_inputs,
            run_mem_encoder,
            high_res_masks,
            object_score_logits,
            current_out,
        )
        return current_out

    def _iter_correct_pt_sampling(
        self,
        is_init_cond_frame,  # 是否是初始条件帧
        point_inputs,  # 当前输入的点（点位置和标签）
        gt_masks,  # 真实掩码
        high_res_features,  # 高分辨率特征
        pix_feat_with_mem,  # 包含内存特征的像素特征
        low_res_multimasks,  # 低分辨率多掩码
        high_res_multimasks,  # 高分辨率多掩码
        ious,  # 交并比
        low_res_masks,  # 低分辨率掩码
        high_res_masks,  # 高分辨率掩码
        object_score_logits,  # 对象得分的逻辑回归输出
        current_out,  # 当前输出字典
    ):

        assert gt_masks is not None  # 确保真实掩码存在
        all_pred_masks = [low_res_masks]
        all_pred_high_res_masks = [high_res_masks]
        all_pred_multimasks = [low_res_multimasks]
        all_pred_high_res_multimasks = [high_res_multimasks]
        all_pred_ious = [ious]
        all_point_inputs = [point_inputs]
        all_object_score_logits = [object_score_logits]

        for _ in range(self.num_correction_pt_per_frame):
            # sample a new point from the error between prediction and ground-truth
            # (with a small probability, directly sample from GT masks instead of errors)
            # 从预测与真实值之间的误差中采样新点（以小概率直接从真实掩码采样）
            if self.training and self.prob_to_sample_from_gt_for_train > 0:
                sample_from_gt = (
                    self.rng.random() < self.prob_to_sample_from_gt_for_train
                )
            else:
                sample_from_gt = False

            # if `pred_for_new_pt` is None, only GT masks will be used for point sampling
            # 如果 `pred_for_new_pt` 为 None，则仅使用真实掩码进行点采样
            pred_for_new_pt = None if sample_from_gt else (high_res_masks > 0)
            new_points, new_labels = get_next_point(
                gt_masks=gt_masks,  # 真实掩码
                pred_masks=pred_for_new_pt,  # 预测掩码
                method="uniform" if self.training else self.pt_sampling_for_eval,  # 根据训练或评估选择点采样方法
            )
            point_inputs = concat_points(point_inputs, new_points, new_labels)  # 合并新的点输入
            # Feed the mask logits of the previous SAM outputs in the next SAM decoder step.
            # For tracking, this means that when the user adds a correction click, we also feed
            # the tracking output mask logits along with the click as input to the SAM decoder.
            # /
            # 将先前 SAM 输出的掩码 logits 作为下一步 SAM 解码器的输入。
            # 对于追踪任务，这意味着当用户添加纠正点击时，我们不仅将追踪输出的掩码 logits
            # 作为输入，还将点击信息一起传递给 SAM 解码器。
            mask_inputs = low_res_masks
            multimask_output = self._use_multimask(is_init_cond_frame, point_inputs)
            # 在启用激活检查点时，执行 SAM 解码器前向计算
            if self.use_act_ckpt_iterative_pt_sampling and not multimask_output:
                sam_outputs = torch.utils.checkpoint.checkpoint(
                    self._forward_sam_heads,
                    backbone_features=pix_feat_with_mem,
                    point_inputs=point_inputs,
                    mask_inputs=mask_inputs,
                    high_res_features=high_res_features,
                    multimask_output=multimask_output,
                    use_reentrant=False,
                )
            else:
                sam_outputs = self._forward_sam_heads(
                    backbone_features=pix_feat_with_mem,  # 像素特征（包含内存）
                    point_inputs=point_inputs,  # 输入点
                    mask_inputs=mask_inputs,  # 输入掩码
                    high_res_features=high_res_features,  # 高分辨率特征
                    multimask_output=multimask_output,  # 是否输出多掩码
                )
            (
                low_res_multimasks,
                high_res_multimasks,
                ious,
                low_res_masks,
                high_res_masks,
                _,
                object_score_logits,
            ) = sam_outputs  # 更新输出结果

            # 将每次迭代的结果保存到对应列表中
            all_pred_masks.append(low_res_masks)
            all_pred_high_res_masks.append(high_res_masks)
            all_pred_multimasks.append(low_res_multimasks)
            all_pred_high_res_multimasks.append(high_res_multimasks)
            all_pred_ious.append(ious)
            all_point_inputs.append(point_inputs)
            all_object_score_logits.append(object_score_logits)

        # Concatenate the masks along channel (to compute losses on all of them,
        # using `MultiStepIteractiveMasks`)
        # 将掩码按通道拼接（用于后续多步交互掩码的损失计算）
        current_out["multistep_pred_masks"] = torch.cat(all_pred_masks, dim=1)
        current_out["multistep_pred_masks_high_res"] = torch.cat(
            all_pred_high_res_masks, dim=1
        )
        current_out["multistep_pred_multimasks"] = all_pred_multimasks
        current_out["multistep_pred_multimasks_high_res"] = all_pred_high_res_multimasks
        current_out["multistep_pred_ious"] = all_pred_ious
        current_out["multistep_point_inputs"] = all_point_inputs
        current_out["multistep_object_score_logits"] = all_object_score_logits

        return point_inputs, sam_outputs  # 返回最终的点输入和 SAM 输出
