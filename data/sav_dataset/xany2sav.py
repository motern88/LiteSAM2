'''
转换脚本，用于将X-AnyLabeling标注输出格式转换为SA-V数据集的格式

SA-V格式:
{
	"video_id": "sav_000001",
    "video_duration": 20.125,
    "video_frame_count": 483.0,
    "video_height": 848.0,
    "video_width": 480.0,
    "video_resolution": 407040.0,
    "video_environment": "Indoor",
    "video_split": "train",
    "masklet":
    [
        [
            {"size": [848, 480], "counts": <此处为RLE掩码格式>},
			{"size": [848, 480], "counts": <此处为RLE掩码格式>},
			...
        ],
        ...
        [
            {"size": [848, 480], "counts": <此处为RLE掩码格式>},
			{"size": [848, 480], "counts": <此处为RLE掩码格式>},
			...
        ]
	],
    "masklet_id": [0, 1, 2, 3, 4],
	"masklet_size_rel":
		[0.0035249812, 0.0946159778, 0.011285757, 0.0091357729, 0.0090703819],
	"masklet_size_abs":
		[1434.8083333333, 38512.4876033058, 4593.7545454545, 3718.625, 3692.0082644628],
	"masklet_size_bucket": ["medium", "large", "medium", "medium", "medium"],
	"masklet_visibility_changes": [2, 0, 10, 0, 0],
	"masklet_first_appeared_frame": [0.0, 0.0, 0.0, 113.0, 0.0],
	"masklet_frame_count": [121, 121, 121, 121, 121],
	"masklet_edited_frame_count": [41, 11, 22, 4, 115],
	"masklet_type": ["manual", "manual", "manual", "manual", "manual"],
	"masklet_stability_score": [null, null, null, null, null],
	"masklet_num": 5
}

读取X-AnyLabeling标注输出文件夹:
├──X-AnyLabeling_output
|   ├──video001.mp4
|   ├──video001_annotation
|   |   ├──00001.json  # video001视频第00001帧的标注
|   |   ├──00002.json  # video001视频第00002帧的标注
|   |   ├──00003.json
|   |   ...
|   ├──video002.mp4
|   ├──video002_annotation
|   |   ├──00001.json  # video002视频第00001帧的标注
|   |   ├──00002.json
|   |   ├──00003.json
|   |   ...
...

生成SA-V数据集的格式：
├──sav_train
|   ├──vos_table
|   |   ├──sav_000001.mp4
|   |   ├──sav_000001_manual.json
|   |   ├──sav_000002.mp4
|   |   ├──sav_000002_manual.json
|   |   ├──sav_000003.mp4
|   |   ├──sav_000003_manual.json
...

'''
from contextlib import nullcontext

import cv2
import os
import json
import shutil
import numpy as np
from pathlib import Path
from pycocotools import mask as mask_utils

def read_XAny_annotation(annotation_folder):
    '''
    读取XAny格式视频中的所有的帧标注文件
    args:
        annotation_folder: 存放视频的所有帧标注文件的路径
    return:
        video_annotation:
            所有帧的标注信息
            list[dict{"frame_idx": int, "annotations": list[dict{"label": str, "points": list[list[float]]}]}]
    '''
    video_annotation = []

    # 确保按数字顺序读取标注文件
    annotation_files = sorted(Path(annotation_folder).glob("*.json"), key=lambda x: int(x.stem))

    # 遍历X-AnyLabeling_output/videoXXX_annotation下的每一帧的标注文件
    for annotation_file in annotation_files:
        with open(annotation_file, 'r') as f:
            frame_annotation = json.load(f)

            # 提取当前帧标注的label和points
            frame_data = []
            for shape in frame_annotation.get("shapes", []):
                shape_data = {
                    "label": shape.get("label", ""),
                    "points": shape.get("points", [])
                }
                frame_data.append(shape_data)

            # 将当前帧的标注加入到整体标注中
            video_annotation.append({
                "frame_idx": int(annotation_file.stem) - 1,  # 例如文件名为 00001.json ,则填入帧索引 0
                "annotations": frame_data
            })

    return video_annotation

def encode_RLE(mask):
    '''
    将二值掩码转换为RLE格式
    args:
        mask: 二值掩码 (numpy array)
    return:
        RLE编码后的掩码
    '''
    return mask_utils.encode(np.asfortranarray(mask))[0]

def build_SAV_annotation(XAny_video_id, XAny_video_annotation):
    '''
    将获取到的X-AnyLabeling标注信息转换为SA-V数据集的格式
    args:
        XAny_video_id: XAny视频id (video001)
        XAny_video_annotation: 视频标注信息 (list[dict{"frame_idx": int, "annotations": list[dict{"label": str, "points": list[list[float]]}]}])
    return:
        sav_000001_manual.json中的所有内容 (dict{})

    需要构造的内容:
        video_id，video_duration
        video_frame_count，video_height，video_width，video_resolution
        video_environment，video_split

        masklet
        masklet_id
        masklet_size_rel
        masklet_size_abs
        masklet_size_bucket
        masklet_frame_count
        masklet_edited_frame_count

        masklet_visibility_changes，masklet_first_appeared_frame 留白，待手工标注


    '''

    # 生成SA-V的视频ID
    video_id = f"sav_{int(XAny_video_id.split('video')[1]):06d}"
    # 获取视频路径
    video_source = Path(input_dir, f"{XAny_video_id}.mp4")

    # 打开视频文件并获取相关信息
    cap = cv2.VideoCapture(str(video_source))
    if not cap.isOpened():
        raise ValueError(f"无法打开视频文件 {video_source}")

    # 从视频源获取时长(秒),总帧数,高度,宽度和分辨率
    video_duration = np.float64(cap.get(cv2.CAP_PROP_POS_MSEC) / 1000)  # 转换为秒 float64
    video_frame_count = np.float64(cap.get(cv2.CAP_PROP_FRAME_COUNT))  # 总帧数 float64
    video_height = np.float64(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # 视频高度 float64
    video_width = np.float64(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  # 视频宽度 float64
    video_resolution = np.float64(video_height * video_width)

    # 释放视频资源
    cap.release()

    # 固定设置
    video_environment = "Indoor"
    video_split = "train"
    masklet_type = "manual"


    #### 构造 masklet和 masklet_id, masklet_size 的rel和abs ####

    # 创建masklet结构
    masklet = [[] for _ in range(video_frame_count)]  # 每帧的对象掩码列表

    # 创建一个字典来映射每个label到唯一的掩码ID
    label_to_mask_id = {}  # dict{"label": int} 类别ID映射字典 # TODO：需要将其存入最终标注中，以便一些字段人工修正查看
    next_mask_id = 0  # 从0开始分配掩码ID

    # 创建一个字典来保存每个mask_id的总面积
    mask_area_sum = {}  # dict{mask_id: list[第一帧mask_area,第二帧mask_area...]} 掩码ID的总面积

    # 从XAny_video_annotation中获取所有"label"字段，为每种label字段自动分配唯一掩码ID
    for frame_data in XAny_video_annotation:
        frame_idx = frame_data["frame_idx"]

        temp_frame_masklet = []  # 用于临时存储当前帧的多个mask的列表

        # 遍历每帧中的标注数据
        for shape in frame_data["annotations"]:
            label = shape["label"]
            points = np.array(shape["points"], dtype=np.float64)  # 确保 points 为 float64 类型

            # 如果label不在label_to_mask_id映射表中，则为每个label分配唯一的掩码ID
            if label not in label_to_mask_id:
                label_to_mask_id[label] = next_mask_id
                next_mask_id += 1

            # 获取掩码ID
            mask_id = label_to_mask_id[label]

            # 创建一个空白掩码图像
            mask = np.zeros((video_height, video_width), dtype=np.uint8)

            # 填充掩码图像（根据多边形的点）
            points = points.astype(int)
            cv2.fillPoly(mask, [points], 1)

            # 计算该掩码的面积（即掩码中值为1的像素数量）
            mask_area = np.sum(mask)  # 计算掩码的面积，面积为掩码图像中像素值为1的数量
            # 将该掩码的面积加入到对应mask_id的面积总和中
            mask_area_sum[mask_id].append(mask_area) # mask_area_sum = dict{mask_id: list[mask_area, mask_area...]}

            # 对掩码进行RLE编码
            rle = encode_RLE(mask)

            # 构造当前mask的RLE的字典
            temp_mask = {
                "mask_id": mask_id,
                "size": [video_height, video_width],
                "counts": rle["counts"]
            }
            temp_frame_masklet.append(temp_mask) # 将当前mask添加到当前帧的masklet临时列表中

        # 将temp_frame_masklet中的temp_masklet按照id顺序添加到masklet列表中
        masklet[frame_idx] = sorted(temp_frame_masklet, key=lambda x: x["mask_id"])  # list[dict{"mask_id": int, "size": list[int], "counts": str}]
        # 移除masklet[frame_idx]中的mask_id，只保留size和counts  # list[dict{"size": list[int], "counts": str}]
        for mask_dict in masklet[frame_idx]:
            # 删除字典中mask_id的键值对
            mask_dict.pop("mask_id")

    # 构造掩码的ID
    masklet_id = list(label_to_mask_id.values())  # [0,1,2,3...N]

    # List[float] 在对象所有可见帧中，平均 mask 面积（像素）
    masklet_size_abs = []
    for mask_id in masklet_id:
        # 计算每个mask_id的平均(归一化后的)面积,并添加到masklet_size_rel中
        masklet_size_abs.append(
            sum(mask_area_sum[mask_id]) / len(mask_area_sum[mask_id])
        )

    # List[float] 在对象所有可见帧中，平均 mask 面积（按分辨率归一化）
    masklet_size_rel = []
    for abs in masklet_size_abs:
        # 计算每个mask_id的平均(归一化后的)面积,并添加到masklet_size_rel中
        masklet_size_rel.append( abs / video_resolution )  # 按分辨率归一化

    # List[str] masklet_size_bucket 的分类
    masklet_size_bucket = []
    for abs in masklet_size_abs:
        if abs < 32**2:
            masklet_size_bucket.append("small")
        elif abs < 96**2:
            masklet_size_bucket.append("medium")
        else:
            masklet_size_bucket.append("large")

    # List[int] 每个对象的帧数
    masklet_frame_count = []
    for mask_id in masklet_id:
        masklet_frame_count.append(len(mask_area_sum[mask_id]))


    #### 从videoXXX_manual.json手工编辑的文件中获取 ####

    video_manual_path = Path(input_dir, f"{XAny_video_id}_manual.json")

    masklet_visibility_changes = []  # List[int] 每个对象的可见性变化次数
    masklet_edited_frame_count = []  # List[int] 每个对象手工编辑的帧数
    masklet_first_appeared_frame = []  # List[int] 每个对象第一次出现的帧索引

    with open(video_manual_path, 'r') as f:
        '''
        "masklet_visibility_changes": List[Dict{"label": str, "value": int}] 每个对象的可见性变化次数
        "masklet_edited_frame_count": List[Dict{"label": str, "value": float}] 每个对象手工编辑的帧数
        "masklet_first_appeared_frame": List[Dict{"label": str, "value": int}] 每个对象第一次出现的帧索引
        '''
        video_manual = json.load(f)

        # 根据label-ID映射字典label_to_mask_id = dict{"label": int}，将ID索引对应的label信息依次提取
        for mask_id in masklet_id:
            # 获取索引对应的label的str
            label = list(label_to_mask_id.keys())[list(label_to_mask_id.values()).index(mask_id)]

            # 获取由label对应的value的值
            masklet_visibility_changes.append(
                video_manual.get("masklet_visibility_changes", {}).get(label, 0)
            )
            masklet_edited_frame_count.append(
                video_manual.get("masklet_edited_frame_count", {}).get(label, 0)
            )
            masklet_first_appeared_frame.append(
                video_manual.get("masklet_first_appeared_frame", {}).get(label, 0)
            )

    # Optional[List[List[float]]]  每个 mask 的稳定性分数。仅自动注释。固定设置长度为masklet_num的”null“填充list
    masklet_stability_score = [None] * len(masklet_id)  # TODO:这里应当填充null

    return {
        "video_id": video_id,
        "video_duration": video_duration,
        "video_frame_count": video_frame_count,
        "video_height": video_height,
        "video_width": video_width,
        "video_resolution": video_resolution,
        "video_environment": video_environment,
        "video_split": video_split,
        "masklet": masklet,
        "masklet_id": masklet_id,
        "masklet_size_rel": masklet_size_rel,
        "masklet_size_abs": masklet_size_abs,
        "masklet_size_bucket": masklet_size_bucket,
        "masklet_visibility_changes": masklet_visibility_changes,  # 由读取额外json获取
        "masklet_first_appeared_frame": masklet_first_appeared_frame,  # 由读取额外json获取
        "masklet_frame_count": masklet_frame_count,
        "masklet_edited_frame_count": masklet_edited_frame_count,  # 由读取额外json获取
        "masklet_type": masklet_type,
        "masklet_stability_score": masklet_stability_score,
        "masklet_num": len(masklet_id),
    }

def convert_XAny_to_SAV(input_dir, output_dir):
    '''
    将X-AnyLabeling标注输出格式转换为SA-V数据集的格式
    args:
        input_dir: X-AnyLabeling标注输出文件夹路径
        output_dir: SA-V数据集输出文件夹路径
    '''

    # 在sav_train下创建输出文件夹vos_table
    Path(output_dir, 'vos_table').mkdir(parents=True, exist_ok=True)

    # 遍历X-AnyLabeling_output文件夹下的每一个_annotation后缀的文件夹
    for video_folder in Path(input_dir).glob("*_annotation"):  # 视频文件夹以 "_annotation" 结尾

        # 获取video_id和视频标注
        XAny_video_id = video_folder.name.split('_')[0]  # 从video001_annotation中提取video001
        XAny_video_annotation = read_XAny_annotation(video_folder)

        # 构建SAV的video_annotation (sav_XXXXXX_manual.json)
        SAV_video_annotation = build_SAV_annotation(XAny_video_id, XAny_video_annotation)

        # 生成SAV_video_id, 例如从video001到sav_000001
        SAV_video_id = f"sav_{int(XAny_video_id.split('video')[1]):06d}"

        # 将XAny_video_id.mp4复制为sav_train/vos_table/SAV_video_id.mp4
        shutil.copy(
            Path(input_dir, f"{XAny_video_id}.mp4"),
            Path(output_dir, 'vos_table', f"{SAV_video_id}.mp4")
        )

        # 将SAV_video_annotation写入sav_train/vos_table/sav_XXXXXX_manual.json
        with open(Path(output_dir, 'vos_table', f"{SAV_video_id}_manual.json"), 'w') as f:
            json.dump(SAV_video_annotation, f, indent=4)

    print("X-AnyLabeling标注输出格式转换为SA-V数据集的格式完成！")


if __name__ == "__main__":
    # 设置路径
    input_dir = './X-AnyLabeling_output'  # 输入文件夹路径
    output_dir = './sav_train'  # 输出文件夹路径

    # 创建输出文件夹
    Path(output_dir, 'vos_table').mkdir(parents=True, exist_ok=True)

    # 开始转换
    convert_XAny_to_SAV(input_dir, output_dir)











































