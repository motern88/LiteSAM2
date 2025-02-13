'''
依赖库:
matplotlib
opencv-python
numpy
pycocoevalcap
pandas
'''


import json
import pandas as pd
from utils.sav_utils import SAVDataset

# 初始化类，指定路径
sav_dataset = SAVDataset(sav_dir="sav_train/vos_table/")  # "example/"
frames, manual_annot, auto_annot = sav_dataset.get_frames_and_annotations("sav_000002")

# 显示 SA-V 在第 annotated_frame_id 帧上的注释
sav_dataset.visualize_annotation(
    frames, manual_annot, auto_annot,
    annotated_frame_id=0,  # 显示视频第n帧的可视化
    show_auto=False,  # 是否显示自动标注
    show_manual=True,  # 是否显示手动标注
)

'''
统计列表包含以下列
video_id	video_duration	video_frame_count	video_height	video_width	video_resolution	video_environment	
video_split	masklet	masklet_id	masklet_size_rel	masklet_size_abs	masklet_size_bucket	masklet_visibility_changes	
masklet_first_appeared_frame	masklet_frame_count	masklet_edited_frame_count	masklet_type	masklet_stability_score	masklet_num
'''
# print("手动标注数据统计：\n",pd.DataFrame([manual_annot]))

# print("自动标注数据统计：\n",pd.DataFrame([auto_annot]))








