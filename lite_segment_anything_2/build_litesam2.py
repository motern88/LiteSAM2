# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os

import lite_segment_anything_2

import torch
from hydra import compose
from hydra.utils import instantiate
from omegaconf import OmegaConf

if os.path.isdir(
    os.path.join(lite_segment_anything_2.__path__[0], "lite_segment_anything_2")
):
    raise RuntimeError(
        "You're likely running Python from the parent directory of the LiteSAM2 repository "
    )

# Working on putting efficient track anything models on Facebook Hugging Face Hub.
# This is just for demonstration.
# Please download efficient track anything models from https://huggingface.co/yunyangx/efficient-track-anything.
# and use build_litesam2/build_litesam2_video_predictor for loading them.
HF_MODEL_ID_TO_FILENAMES = {
    "ATA-space/litesam2": (
        "configs/litesam2/litesam2_512x512.yaml",
        "litesam2_512x512.pt",
    ),
}


def build_litesam2(
    config_file,
    ckpt_path=None,
    device="cuda",
    mode="eval",
    hydra_overrides_extra=[],
    apply_postprocessing=True,
    **kwargs,
):

    if apply_postprocessing:
        hydra_overrides_extra = hydra_overrides_extra.copy()
        hydra_overrides_extra += [
            # dynamically fall back to multi-mask if the single mask is not stable
            "++model.sam_mask_decoder_extra_args.dynamic_multimask_via_stability=true",
            "++model.sam_mask_decoder_extra_args.dynamic_multimask_stability_delta=0.05",
            "++model.sam_mask_decoder_extra_args.dynamic_multimask_stability_thresh=0.98",
        ]
    # Read config and init model
    cfg = compose(config_name=config_file, overrides=hydra_overrides_extra)
    OmegaConf.resolve(cfg)
    model = instantiate(cfg.model, _recursive_=True)
    _load_checkpoint(model, ckpt_path)
    model = model.to(device)
    if mode == "eval":
        model.eval()
    return model


def build_litesam2_video_predictor(
    config_file,
    ckpt_path=None,
    device="cuda",
    mode="eval",
    hydra_overrides_extra=[],
    apply_postprocessing=True,
    vos_optimized=False,
    **kwargs,
):
    # 检查GPU是否支持torch编译
    if not torch.cuda.is_available() or torch.cuda.get_device_properties(0).major < 8:
        print("Disable torch compile due to unsupported GPU.")
        hydra_overrides_extra = ["++model.compile_image_encoder=False"]
        vos_optimized = False

    hydra_overrides = [
        "++model._target_=lite_segment_anything_2.litesam2_video_predictor.LiteSAM2VideoPredictor",
    ]
    if vos_optimized:
        hydra_overrides = [
            "++model._target_=lite_segment_anything_2.litesam2_video_predictor.LiteSAM2VideoPredictorVOS",
            "++model.compile_image_encoder=True",  # Let litesam2_base handle this / 让 LiteSAM2 基础模块处理编译
        ]

    if apply_postprocessing:
        hydra_overrides_extra = hydra_overrides_extra.copy()
        hydra_overrides_extra += [
            # dynamically fall back to multi-mask if the single mask is not stable
            # /
            # 1. 在单掩膜不稳定时动态启用多掩膜
            "++model.sam_mask_decoder_extra_args.dynamic_multimask_via_stability=true",
            "++model.sam_mask_decoder_extra_args.dynamic_multimask_stability_delta=0.05",
            "++model.sam_mask_decoder_extra_args.dynamic_multimask_stability_thresh=0.98",
            # the sigmoid mask logits on interacted frames with clicks in the memory encoder so that the encoded masks are exactly as what users see from clicking
            # /
            # 2. 使用内存编码器时二值化掩膜 (使点击结果与可视化一致)
            "++model.binarize_mask_from_pts_for_mem_enc=true",
            # fill small holes in the low-res masks up to `fill_hole_area` (before resizing them to the original video resolution)
            # /
            # 3. 在低分辨率掩膜中填充小孔 (放大前填补面积小于 8 像素的孔)
            "++model.fill_hole_area=8",
        ]
    hydra_overrides.extend(hydra_overrides_extra)

    # Read config and init model
    cfg = compose(config_name=config_file, overrides=hydra_overrides)
    OmegaConf.resolve(cfg)
    model = instantiate(cfg.model, _recursive_=True)
    if ckpt_path is not None:
        _load_checkpoint(model, ckpt_path)
    model = model.to(device)
    if mode == "eval":
        model.eval()
    return model


def _hf_download(model_id):
    from huggingface_hub import hf_hub_download

    config_name, checkpoint_name = HF_MODEL_ID_TO_FILENAMES[model_id]
    ckpt_path = hf_hub_download(repo_id=model_id, filename=checkpoint_name)
    return config_name, ckpt_path


def build_litesam2_hf(model_id, **kwargs):
    config_name, ckpt_path = _hf_download(model_id)
    return build_litesam2(config_file=config_name, ckpt_path=ckpt_path, **kwargs)


def build_litesam2_video_predictor_hf(model_id, **kwargs):
    config_name, ckpt_path = _hf_download(model_id)
    return build_litesam2_video_predictor(
        config_file=config_name, ckpt_path=ckpt_path, **kwargs
    )


def _load_checkpoint(model, ckpt_path):
    if ckpt_path is not None:
        sd = torch.load(ckpt_path, map_location="cpu", weights_only=True)["model"]
        missing_keys, unexpected_keys = model.load_state_dict(sd)
        if missing_keys:
            logging.error(missing_keys)
            raise RuntimeError()
        if unexpected_keys:
            logging.error(unexpected_keys)
            raise RuntimeError()
        logging.info("Loaded checkpoint sucessfully")
