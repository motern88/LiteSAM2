# Segment Anything Video (SA-V) Dataset

## Overview

[Segment Anything Video (SA-V)](https://ai.meta.com/datasets/segment-anything-video/), consists of 51K diverse videos and 643K high-quality spatio-temporal segmentation masks (i.e., masklets). The dataset is released under the CC by 4.0 license. Browse the dataset [here](https://sam2.metademolab.com/dataset).

![SA-V dataset](../assets/sa_v_dataset.jpg?raw=true)

## Getting Started

### Download the dataset

Visit [here](https://ai.meta.com/datasets/segment-anything-video-downloads/) to download SA-V including the training, val and test sets.

### Dataset Stats

|            | Num Videos | Num Masklets                              |
| ---------- | ---------- | ----------------------------------------- |
| SA-V train | 50,583     | 642,036 (auto 451,720 and manual 190,316) |
| SA-V val   | 155        | 293                                       |
| SA-V test  | 150        | 278                                       |

### Notebooks

To load and visualize the SA-V training set annotations, refer to the example [sav_visualization_example.ipynb](./sav_visualization_example.ipynb) notebook.

要加载和可视化 SA-V 训练集的注释信息，请参考示例 notebook sav_visualization_example.ipynb。

### SA-V train

For SA-V training set we release the mp4 videos and store the masklet annotations per video as json files . Automatic masklets and manual masklets are stored separately as two json files: `{video_id}_auto.json` and `{video_id}_manual.json`. They can be loaded as dictionaries in python in the format below.

对于 SA-V 训练集，我们提供 MP4 格式的视频，并将每个视频的 masklet 注释 存储为 JSON 文件。
自动 masklets 和 手动 masklets 分别存储在两个独立的 JSON 文件中，命名格式如下：
{video_id}_auto.json 和 {video_id}_manual.json。 

这些 JSON 文件可以在 Python 中加载为字典，格式如下所示。

```
{
    "video_id"                        : str; video id / 视频 ID
    "video_duration"                  : float64; the duration in seconds of this video / 该视频的时长（秒）
    "video_frame_count"               : float64; the number of frames in the video / 该视频的总帧数
    "video_height"                    : float64; the height of the video / 该视频的高度
    "video_width"                     : float64; the width of the video / 该视频的宽度
    "video_resolution"                : float64; video_height $\times$ video_width / 视频分辨率
    "video_environment"               : List[str]; "Indoor" or "Outdoor" / 室内或室外
    "video_split"                     : str; "train" for training set / 训练集
    "masklet"                         : List[List[Dict]]; masklet annotations in list of list of RLEs. / masklet 注释，列表中的列表是视频中的帧，内部列表是视频中的对象。
                                        The outer list is over frames in the video and the inner list / 外部列表是视频中的帧，
                                        is over objects in the video. / 内部列表是视频中的对象。
    "masklet_id"                      : List[int]; the masklet ids
    "masklet_size_rel"                : List[float]; the average mask area normalized by resolution 
                                        across all the frames where the object is visible / 在对象可见的所有帧中 平均 mask 面积，按分辨率归一化
    "masklet_size_abs"                : List[float]; the average mask area (in pixels)
                                        across all the frames where the object is visible / 在对象可见的所有帧中 平均 mask 面积（像素）
    "masklet_size_bucket"             : List[str]; "small": $1$ <= masklet_size_abs < $32^2$,
                                        "medium": $32^2$ <= masklet_size_abs < $96^2$,
                                        and "large": masklet_size_abs > $96^2$
    "masklet_visibility_changes"      : List[int]; the number of times where the visibility changes
                                        after the first appearance (e.g., invisible -> visible
                                        or visible -> invisible) / 在第一次出现后，可见性发生变化的次数（例如，不可见 -> 可见 或 可见 -> 不可见）
    "masklet_first_appeared_frame"    : List[int]; the index of the frame where the object appears
                                        the first time in the video. Always 0 for auto masklets. / 对象第一次出现的帧索引。自动 masklets 始终为 0。
    "masklet_frame_count"             : List[int]; the number of frames being annotated. Note that
                                        videos are annotated at 6 fps (annotated every 4 frames)
                                        while the videos are at 24 fps. / 正在注释的帧数。请注意，视频以 6 fps 注释（每 4 帧注释一次），
    "masklet_edited_frame_count"      : List[int]; the number of frames being edited by human annotators.
                                        Always 0 for auto masklets. / 人类标注者编辑的帧数。自动 masklets 始终为 0。
    "masklet_type"                    : List[str]; "auto" or "manual" / 自动或手动
    "masklet_stability_score"         : Optional[List[List[float]]]; per-mask stability scores. Auto annotation only. / 每个 mask 的稳定性分数。仅自动注释。
    "masklet_num"                     : int; the number of manual/auto masklets in the video / 视频中的手动/自动 masklets 数量

}
```

Note that in SA-V train, there are in total 50,583 videos where all of them have manual annotations. Among the 50,583 videos there are 48,436 videos that also have automatic annotations.

请注意，在 SA-V 训练集（train）中，共有 50,583 个视频，这些视频全部包含人工标注（manual annotations）。在这 50,583 个视频中，有 48,436 个视频同时包含自动标注（automatic annotations）。

### SA-V val and test

For SA-V val and test sets, we release the extracted frames as jpeg files, and the masks as png files with the following directory structure:

对于 SA-V 的验证集（val）和测试集（test），我们将提取的帧以 JPEG 文件格式发布，并将掩码以 PNG 文件格式发布，目录结构如下：

```
sav_val(sav_test)
├── sav_val.txt (sav_test.txt): a list of video ids in the split
├── JPEGImages_24fps # videos are extracted at 24 fps
│   ├── {video_id}
│   │     ├── 00000.jpg        # video frame
│   │     ├── 00001.jpg        # video frame
│   │     ├── 00002.jpg        # video frame
│   │     ├── 00003.jpg        # video frame
│   │     └── ...
│   ├── {video_id}
│   ├── {video_id}
│   └── ...
└── Annotations_6fps # videos are annotated at 6 fps
    ├── {video_id}
    │     ├── 000               # obj 000
    │     │    ├── 00000.png    # mask for object 000 in 00000.jpg
    │     │    ├── 00004.png    # mask for object 000 in 00004.jpg
    │     │    ├── 00008.png    # mask for object 000 in 00008.jpg
    │     │    ├── 00012.png    # mask for object 000 in 00012.jpg
    │     │    └── ...
    │     ├── 001               # obj 001
    │     ├── 002               # obj 002
    │     └── ...
    ├── {video_id}
    ├── {video_id}
    └── ...
```

All masklets in val and test sets are manually annotated in every frame by annotators. For each annotated object in a video, we store the annotated masks in a single png. This is because the annotated objects may overlap, e.g., it is possible in our SA-V dataset for there to be a mask for the whole person as well as a separate mask for their hands.

验证集（val）和测试集（test）中的所有 masklet 都由标注人员逐帧手动标注。对于视频中的每个被标注对象，我们将所有标注的掩码存储在单个 PNG 文件中。这是因为标注对象可能会相互重叠，例如，在 SA-V 数据集中，可能同时存在一个人的整体掩码以及他们手部的单独掩码。

## SA-V Val and Test Evaluation

We provide an evaluator to compute the common J and F metrics on SA-V val and test sets. To run the evaluation, we need to first install a few dependencies as follows:

我们提供了一个评估工具，用于计算 SA-V 验证集和测试集上的常见 J 和 F 指标（Jaccard 相似度和边界 F 度量）。要运行评估，首先需要安装一些依赖项，如下所示：

```
pip install -r requirements.txt
```

Then we can evaluate the predictions as follows:

然后可以使用以下命令进行预测评估：

```
python sav_evaluator.py --gt_root {GT_ROOT} --pred_root {PRED_ROOT}
```

or run

或者运行：

```
python sav_evaluator.py --help
```

to print a complete help message.

The evaluator expects the `GT_ROOT` to be one of the following folder structures, and `GT_ROOT` and `PRED_ROOT` to have the same structure.

- Same as SA-V val and test directory structure

要打印完整的帮助信息，请运行相应的命令。

评估工具（evaluator）要求 GT_ROOT 具有以下目录结构之一，并且 GT_ROOT 和 PRED_ROOT 需要保持相同的结构。

- 与 SA-V 验证集和测试集目录结构相同

```
{GT_ROOT}  # gt root folder
├── {video_id}
│     ├── 000               # all masks associated with obj 000
│     │    ├── 00000.png    # mask for object 000 in frame 00000 (binary mask)
│     │    └── ...
│     ├── 001               # all masks associated with obj 001
│     ├── 002               # all masks associated with obj 002
│     └── ...
├── {video_id}
├── {video_id}
└── ...
```

In the paper for the experiments on SA-V val and test, we run inference on the 24 fps videos, and evaluate on the subset of frames where we have ground truth annotations (first and last annotated frames dropped). The evaluator will ignore the masks in frames where we don't have ground truth annotations.

- Same as [DAVIS](https://github.com/davisvideochallenge/davis2017-evaluation) directory structure

在 SA-V 论文中的实验中，我们在 24 fps 的视频上运行推理，并在包含 人工标注 的帧子集上进行评估（首尾两帧的标注数据被丢弃）。评估工具（evaluator）会自动忽略那些没有人工标注的帧中的掩码

- 目录结构与 DAVIS 数据集相同。

```
{GT_ROOT}  # gt root folder
├── {video_id}
│     ├── 00000.png        # annotations in frame 00000 (may contain multiple objects)
│     └── ...
├── {video_id}
├── {video_id}
└── ...
```

## License

The evaluation code is licensed under the [BSD 3 license](./LICENSE). Please refer to the paper for more details on the models. The videos and annotations in SA-V Dataset are released under CC BY 4.0.

Third-party code: the evaluation software is heavily adapted from [`VOS-Benchmark`](https://github.com/hkchengrex/vos-benchmark) and [`DAVIS`](https://github.com/davisvideochallenge/davis2017-evaluation) (with their licenses in [`LICENSE_DAVIS`](./LICENSE_DAVIS) and [`LICENSE_VOS_BENCHMARK`](./LICENSE_VOS_BENCHMARK)).
