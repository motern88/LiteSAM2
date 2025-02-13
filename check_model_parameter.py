import torch
from efficient_track_anything.build_efficienttam import (
    build_efficienttam_video_predictor,
)

checkpoint = "./checkpoints/efficienttam_ti_512x512.pt"
model_cfg = "configs/efficienttam/efficienttam_ti_512x512.yaml"

model = build_efficienttam_video_predictor(model_cfg, checkpoint)

print("模型结构：\n", model)

# 定义各外层模块
modules = {
    'image_encoder': model.image_encoder,
    'mask_downsample': model.mask_downsample,
    'memory_attention': model.memory_attention,
    'memory_encoder': model.memory_encoder,
    'sam_prompt_encoder': model.sam_prompt_encoder,
    'sam_mask_decoder': model.sam_mask_decoder,
    'obj_ptr_proj': model.obj_ptr_proj,
    'obj_ptr_tpos_proj': model.obj_ptr_tpos_proj,
}

# 计算各模块参数并打印
total_params = 0
for name, module in modules.items():
    params = sum(p.numel() for p in module.parameters())
    print(f"{name}: {params}")
    total_params += params

# 定义参数量格式化函数
def format_params(num_params):
    if num_params >= 1e6:  # 百万
        return f"{num_params / 1e6:.2f}M"
    elif num_params >= 1e3:  # 千
        return f"{num_params / 1e3:.2f}K"
    else:
        return f"{num_params}"
    
# # 验证总和
# model_total = sum(p.numel() for p in model.parameters())
# print(f"\n各模块参数量总和: {total_params}")
# print(f"模型总参数量: {model_total}")
# print(f"是否一致: {total_params == model_total}")

# 计算各模块参数并打印
total_params = 0
for name, module in modules.items():
    params = sum(p.numel() for p in module.parameters())
    formatted_params = format_params(params)
    print(f"{name}: {formatted_params}")
    total_params += params

# 验证总和
model_total = sum(p.numel() for p in model.parameters())
formatted_total = format_params(total_params)
formatted_model_total = format_params(model_total)
print(f"\n各模块参数量总和: {formatted_total}")
print(f"模型总参数量: {formatted_model_total}")
print(f"是否一致: {total_params == model_total}")

