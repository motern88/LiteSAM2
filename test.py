import hydra
from hydra.core.global_hydra import GlobalHydra
from omegaconf import OmegaConf

import torch

GlobalHydra.instance().clear()

@hydra.main(config_path="efficient_track_anything/configs/efficienttam",
            config_name="efficienttam_ti_512x512.yaml",
            version_base="1.2")
def test_image_encoder(cfg):
    # 根据配置文件实例化模型
    image_encoder = hydra.utils.instantiate(cfg.model.image_encoder)
    print("############## image_encoder ##############")
    print(image_encoder)
    print("###########################################")

    # 准备输入数据，假设输入的图像是一个批次（batch_size, channels, height, width）的四维张量
    sample = torch.randn(1, 3, 512, 512)  # 假设图像是 512x512 的 RGB 图像

    # 调用 forward 方法
    output = image_encoder(sample)  # 或者 image_encoder.forward(sample)

    print("############## output ##############")
    print("vision_features:", output['vision_features'].shape)

    print("vision_pos_enc:")
    for i in range(len(output['vision_pos_enc'])):
        print(output['vision_pos_enc'][i].shape)

    print("backbone_fpn:")
    for i in range(len(output['backbone_fpn'])):
        print(output['backbone_fpn'][i].shape)

def test_loss_fn()

    # 实例化学生模型
    teacher_model = hydra.utils.instantiate(teacher_model_conf, _convert_="all")
    teacher_model.eval()
    student_model = hydra.utils.instantiate(student_model_conf, _convert_="all")

    # 前向传播
    teacher_outputs, teacher_backbone_outputs = teacher_model(batch)  # 教师模型的输出
    student_outputs, student_backbone_outputs = student_model(batch)  # 学生模型的输出

    targets = batch.masks  # 获取目标数据（mask）
    batch_size = len(batch.img_batch)

    key = batch.dict_key  # 数据集的键 key for dataset



if __name__ == "__main__":
    # test_image_encoder()