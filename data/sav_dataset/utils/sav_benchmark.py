# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the sav_dataset directory of this source tree.

# adapted from https://github.com/hkchengrex/vos-benchmark
# and  https://github.com/davisvideochallenge/davis2017-evaluation
# with their licenses found in the LICENSE_VOS_BENCHMARK and LICENSE_DAVIS files
# in the sav_dataset directory.
import math
import os
import time
from collections import defaultdict
from multiprocessing import Pool
from os import path
from typing import Dict, List, Tuple

import cv2
import numpy as np
import tqdm
from PIL import Image
from skimage.morphology import disk


class VideoEvaluator:
    def __init__(self, gt_root, pred_root, skip_first_and_last=True) -> None:
        """
        gt_root: path to the folder storing the gt masks
        pred_root: path to the folder storing the predicted masks
        skip_first_and_last: whether we should skip the evaluation of the first and the last frame.
                             True for SA-V val and test, same as in DAVIS semi-supervised evaluation.
        /
        gt_root: 存储 GT 掩膜的文件夹路径
        pred_root: 存储预测掩膜的文件夹路径
        skip_first_and_last: 是否跳过第一帧和最后一帧的评估。
                             对于 SA-V val 和 test 需要设置为 True，类似于 DAVIS 半监督评估的设置
        """
        self.gt_root = gt_root
        self.pred_root = pred_root
        self.skip_first_and_last = skip_first_and_last

    def __call__(self, vid_name: str) -> Tuple[str, Dict[str, float], Dict[str, float]]:
        """
        vid_name: name of the video to evaluate / 需要评估的视频名称
        """

        # scan the folder to find subfolders for evaluation and
        # check if the folder structure is SA-V
        # /
        # 扫描文件夹结构，检查视频文件夹是否符合 SA-V 格式
        to_evaluate, is_sav_format = self.scan_vid_folder(vid_name)

        # evaluate each (gt_path, pred_path) pair
        # /
        # 对每个 (gt_path, pred_path) 配对进行评估
        eval_results = []
        for all_frames, obj_id, gt_path, pred_path in to_evaluate:
            if self.skip_first_and_last:
                # skip the first and the last frames
                # /
                # 跳过第一帧和最后一帧
                all_frames = all_frames[1:-1]

            evaluator = Evaluator(name=vid_name, obj_id=obj_id)
            for frame in all_frames:
                # 获取每帧的 GT 和预测掩膜
                gt_array, pred_array = self.get_gt_and_pred(
                    gt_path, pred_path, frame, is_sav_format
                )
                evaluator.feed_frame(mask=pred_array, gt=gt_array)

            # 计算 IOU 和边界精度
            iou, boundary_f = evaluator.conclude()
            eval_results.append((obj_id, iou, boundary_f))

        if is_sav_format:
            # 如果是 SA-V 格式，合并所有物体的评估结果
            iou_output, boundary_f_output = self.consolidate(eval_results)
        else:
            # 否则，仅有一个评估对象
            assert len(eval_results) == 1
            iou_output = eval_results[0][1]
            boundary_f_output = eval_results[0][2]

        return vid_name, iou_output, boundary_f_output

    def get_gt_and_pred(
        self,
        gt_path: str,
        pred_path: str,
        f_name: str,
        is_sav_format: bool,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get the ground-truth and predicted masks for a single frame.
        /
        获取单帧的 GT 和预测掩膜
        """
        gt_mask_path = path.join(gt_path, f_name)  # GT 掩膜路径
        pred_mask_path = path.join(pred_path, f_name)  # 预测掩膜路径
        assert os.path.exists(pred_mask_path), f"{pred_mask_path} not found"

        # 读取掩膜文件
        gt_array = np.array(Image.open(gt_mask_path))
        pred_array = np.array(Image.open(pred_mask_path))

        # 检查 GT 和预测掩膜的尺寸是否匹配
        assert (
            gt_array.shape[-2:] == pred_array.shape[-2:]
        ), f"shape mismatch: {gt_mask_path}, {pred_mask_path}"

        # 如果是 SA-V 格式，将掩膜值大于0的部分视为前景
        if is_sav_format:
            assert len(np.unique(gt_array)) <= 2, (
                f"found more than 1 object in {gt_mask_path} "
                "SA-V format assumes one object mask per png file."
            )
            assert len(np.unique(pred_array)) <= 2, (
                f"found more than 1 object in {pred_mask_path} "
                "SA-V format assumes one object mask per png file."
            )
            gt_array = gt_array > 0  # 将 GT 掩膜转换为二值掩膜
            pred_array = pred_array > 0  # 将预测掩膜转换为二值掩膜

        return gt_array, pred_array

    def scan_vid_folder(self, vid_name) -> Tuple[List, bool]:
        """
        Scan the folder structure of the video and return a list of folders for evaluate.
        /
        扫描视频文件夹结构，返回待评估的文件夹列表
        """

        vid_gt_path = path.join(self.gt_root, vid_name)  # 获取视频 GT 路径
        vid_pred_path = path.join(self.pred_root, vid_name)  # 获取视频预测路径
        all_files_and_dirs = sorted(os.listdir(vid_gt_path))  # 获取所有文件和目录，排序
        to_evaluate = []  # 存储待评估的结果
        if all(name.endswith(".png") for name in all_files_and_dirs):
            # All files are png files, dataset structure similar to DAVIS
            # /
            # 如果所有文件都是 png 文件，类似于 DAVIS 数据集的结构
            is_sav_format = False
            frames = all_files_and_dirs
            obj_dir = None
            to_evaluate.append((frames, obj_dir, vid_gt_path, vid_pred_path))
        else:
            # SA-V dataset structure, going one layer down into each subdirectory
            # /
            # 如果是 SA-V 数据集结构，遍历每个子目录
            is_sav_format = True
            for obj_dir in all_files_and_dirs:
                obj_gt_path = path.join(vid_gt_path, obj_dir)  # 获取目标的 GT 路径
                obj_pred_path = path.join(vid_pred_path, obj_dir)  # 获取目标的预测路径
                frames = sorted(os.listdir(obj_gt_path))  # 获取目标目录下的所有帧文件
                to_evaluate.append((frames, obj_dir, obj_gt_path, obj_pred_path))
        return to_evaluate, is_sav_format

    def consolidate(
        self, eval_results
    ) -> Tuple[str, Dict[str, float], Dict[str, float]]:
        """
        Consolidate the results of all the objects from the video into one dictionary.
        /
        合并视频中所有物体的评估结果到一个字典
        """
        iou_output = {}  # 存储每个物体的 IOU 结果
        boundary_f_output = {}  # 存储每个物体的边界精度结果
        for obj_id, iou, boundary_f in eval_results:
            assert len(iou) == 1  # IOU 应该只有一个值
            key = list(iou.keys())[0]  # 获取 IOU 的键
            iou_output[obj_id] = iou[key]  # 存储 IOU 结果
            boundary_f_output[obj_id] = boundary_f[key]  # 存储边界精度结果
        return iou_output, boundary_f_output


#################################################################################################################
# Functions below are from https://github.com/hkchengrex/vos-benchmark with minor modifications
# _seg2bmap from https://github.com/hkchengrex/vos-benchmark/blob/main/vos_benchmark/utils.py
# get_iou and Evaluator from https://github.com/hkchengrex/vos-benchmark/blob/main/vos_benchmark/evaluator.py
# benchmark from https://github.com/hkchengrex/vos-benchmark/blob/main/vos_benchmark/benchmark.py with slight mod
# /
# 以下函数来自于 https://github.com/hkchengrex/vos-benchmark，做了少量修改
# _seg2bmap 来自 https://github.com/hkchengrex/vos-benchmark/blob/main/vos_benchmark/utils.py
# get_iou 和 Evaluator 来自 https://github.com/hkchengrex/vos-benchmark/blob/main/vos_benchmark/evaluator.py
# benchmark 来自 https://github.com/hkchengrex/vos-benchmark/blob/main/vos_benchmark/benchmark.py，做了轻微修改
#################################################################################################################


def _seg2bmap(seg, width=None, height=None):
    """
    From a segmentation, compute a binary boundary map with 1 pixel wide
    boundaries.  The boundary pixels are offset by 1/2 pixel towards the
    origin from the actual segment boundary.
    /
    从分割结果中计算出一个二值边界图，边界宽度为1像素。
    边界像素从实际分割边界向原点偏移1/2个像素。

    Arguments:
        seg     : Segments labeled from 1..k. / 标注的分割结果，标签从1到k。
        width	  :	Width of desired bmap  <= seg.shape[1] / 目标边界图宽度，<= seg.shape[1]
        height  :	Height of desired bmap <= seg.shape[0] / 目标边界图高度，<= seg.shape[0]
    Returns:
        bmap (ndarray):	Binary boundary map. / 二值边界图。

     David Martin <dmartin@eecs.berkeley.edu>
     January 2003
    """

    seg = seg.astype(bool)
    seg[seg > 0] = 1  # 非零值标记为1

    assert np.atleast_3d(seg).shape[2] == 1  # 确保输入是二维分割图像

    width = seg.shape[1] if width is None else width  # 如果未指定宽度，则使用分割图的宽度
    height = seg.shape[0] if height is None else height  # 如果未指定高度，则使用分割图的高度

    h, w = seg.shape[:2]  # 获取分割图像的高度和宽度

    ar1 = float(width) / float(height)  # 目标图像的宽高比
    ar2 = float(w) / float(h)  # 原始图像的宽高比

    # 确保宽高比差异小于0.01且目标尺寸不超过原始图像尺寸
    assert not (
        width > w | height > h | abs(ar1 - ar2) > 0.01
    ), "Cannot convert %dx%d seg to %dx%d bmap." % (w, h, width, height)

    # 创建边界图
    e = np.zeros_like(seg)
    s = np.zeros_like(seg)
    se = np.zeros_like(seg)

    # 计算边界：与右、下、右下像素比较
    e[:, :-1] = seg[:, 1:]
    s[:-1, :] = seg[1:, :]
    se[:-1, :-1] = seg[1:, 1:]

    b = seg ^ e | seg ^ s | seg ^ se  # 计算边界

    # 对最后一行和最后一列进行特殊处理
    b[-1, :] = seg[-1, :] ^ e[-1, :]
    b[:, -1] = seg[:, -1] ^ s[:, -1]
    b[-1, -1] = 0  # 最后的像素设为0

    # 如果目标尺寸与原图相同，则直接返回边界图
    if w == width and h == height:
        bmap = b
    else:
        # 如果目标尺寸不同，调整图像尺寸
        bmap = np.zeros((height, width))
        for x in range(w):
            for y in range(h):
                if b[y, x]:
                    # 将边界图缩放至目标尺寸
                    j = 1 + math.floor((y - 1) + height / h)
                    i = 1 + math.floor((x - 1) + width / h)
                    bmap[j, i] = 1

    return bmap


def get_iou(intersection, pixel_sum):
    # handle edge cases without resorting to epsilon
    # /
    # 处理边界情况，不使用epsilon
    if intersection == pixel_sum:
        # both mask and gt have zero pixels in them
        # /
        # 如果mask和gt都有零像素
        assert intersection == 0
        return 1  # 完全重合时返回1

    return intersection / (pixel_sum - intersection)  # 计算IoU


class Evaluator:
    def __init__(self, boundary=0.008, name=None, obj_id=None):
        # boundary: used in computing boundary F-score / 用于计算边界F值
        self.boundary = boundary
        self.name = name
        self.obj_id = obj_id
        self.objects_in_gt = set()
        self.objects_in_masks = set()

        self.object_iou = defaultdict(list)
        self.boundary_f = defaultdict(list)

    def feed_frame(self, mask: np.ndarray, gt: np.ndarray):
        """
        Compute and accumulate metrics for a single frame (mask/gt pair)
        /
        计算并累计一帧的指标（mask/gt对）
        """

        # get all objects in the ground-truth / 获取ground-truth中的所有对象
        gt_objects = np.unique(gt)
        gt_objects = gt_objects[gt_objects != 0].tolist()

        # get all objects in the predicted mask / 获取预测mask中的所有对象
        mask_objects = np.unique(mask)
        mask_objects = mask_objects[mask_objects != 0].tolist()

        self.objects_in_gt.update(set(gt_objects))
        self.objects_in_masks.update(set(mask_objects))

        all_objects = self.objects_in_gt.union(self.objects_in_masks)

        # boundary disk for boundary F-score. It is the same for all objects.
        # /
        # 计算边界F值所需的边界磁盘，所有对象使用相同的边界磁盘
        bound_pix = np.ceil(self.boundary * np.linalg.norm(mask.shape))
        boundary_disk = disk(bound_pix)  # 创建边界磁盘

        for obj_idx in all_objects:
            obj_mask = mask == obj_idx  # 获取当前对象的mask
            obj_gt = gt == obj_idx  # 获取当前对象的gt

            # object iou / 计算对象的IoU
            self.object_iou[obj_idx].append(
                get_iou((obj_mask * obj_gt).sum(), obj_mask.sum() + obj_gt.sum())
            )
            """
            # boundary f-score / 计算边界F值
            This part is copied from davis2017-evaluation
            """
            mask_boundary = _seg2bmap(obj_mask)
            gt_boundary = _seg2bmap(obj_gt)
            mask_dilated = cv2.dilate(mask_boundary.astype(np.uint8), boundary_disk)
            gt_dilated = cv2.dilate(gt_boundary.astype(np.uint8), boundary_disk)

            # Get the intersection / 计算交集
            gt_match = gt_boundary * mask_dilated
            fg_match = mask_boundary * gt_dilated

            # Area of the intersection / 计算前景和gt区域的面积
            n_fg = np.sum(mask_boundary)
            n_gt = np.sum(gt_boundary)

            # Compute precision and recall / 计算精度和召回率
            if n_fg == 0 and n_gt > 0:
                precision = 1
                recall = 0
            elif n_fg > 0 and n_gt == 0:
                precision = 0
                recall = 1
            elif n_fg == 0 and n_gt == 0:
                precision = 1
                recall = 1
            else:
                precision = np.sum(fg_match) / float(n_fg)
                recall = np.sum(gt_match) / float(n_gt)

            # Compute F measure / 计算F值
            if precision + recall == 0:
                F = 0
            else:
                F = 2 * precision * recall / (precision + recall)
            self.boundary_f[obj_idx].append(F)

    def conclude(self):
        # 计算所有对象的平均IoU和边界F值
        all_iou = {}
        all_boundary_f = {}

        for object_id in self.objects_in_gt:
            all_iou[object_id] = np.mean(self.object_iou[object_id]) * 100
            all_boundary_f[object_id] = np.mean(self.boundary_f[object_id]) * 100

        return all_iou, all_boundary_f


def benchmark(
    gt_roots,
    mask_roots,
    strict=True,
    num_processes=None,
    *,
    verbose=True,
    skip_first_and_last=True,
):
    """
    gt_roots: a list of paths to datasets, i.e., [path_to_DatasetA, path_to_DatasetB, ...]
    mask_roots: same as above, but the .png are masks predicted by the model
    strict: when True, all videos in the dataset must have corresponding predictions.
            Setting it to False is useful in cases where the ground-truth contains both train/val
                sets, but the model only predicts the val subset.
            Either way, if a video is predicted (i.e., the corresponding folder exists),
                then it must at least contain all the masks in the ground truth annotations.
                Masks that are in the prediction but not in the ground-truth
                (i.e., sparse annotations) are ignored.
    skip_first_and_last: whether we should skip the first and the last frame in evaluation.
                            This is used by DAVIS 2017 in their semi-supervised evaluation.
                            It should be disabled for unsupervised evaluation.
    /
    gt_roots: 数据集的路径列表，例如：[path_to_DatasetA, path_to_DatasetB, ...]
    mask_roots: 与gt_roots相同，但是 .png 文件是模型预测的掩码
    strict: 当为True时，数据集中的所有视频必须有对应的预测。
            设置为False时，适用于ground-truth包含训练集/验证集，但模型仅预测验证集的情况。
            无论哪种情况，如果视频被预测（即对应的文件夹存在），则至少必须包含ground-truth注释中的所有掩码。
            在预测中有但在ground-truth中没有的掩码（即稀疏标注）将被忽略。
    skip_first_and_last: 是否跳过评估的第一帧和最后一帧。
                         这是DAVIS 2017半监督评估中的标准做法。
                         对于无监督评估应该禁用此选项。
    """

    assert len(gt_roots) == len(mask_roots)  # 确保gt_roots和mask_roots长度一致
    single_dataset = len(gt_roots) == 1  # 判断是否只有一个数据集

    if verbose:
        if skip_first_and_last:
            print(
                "We are *SKIPPING* the evaluation of the first and the last frame (standard for semi-supervised video object segmentation)."
                "/"
                "我们将 *SKIPPING* 评估第一帧和最后一帧（这是半监督视频物体分割中的标准做法）。"
            )
        else:
            print(
                "We are *NOT SKIPPING* the evaluation of the first and the last frame (*NOT STANDARD* for semi-supervised video object segmentation)."
                "我们将 *NOT SKIPPING* 评估第一帧和最后一帧（ *NOT STANDARD* 的半监督视频物体分割做法）。"
            )

    pool = Pool(num_processes)
    start = time.time()
    to_wait = []
    for gt_root, mask_root in zip(gt_roots, mask_roots):
        # Validate folders / 验证文件夹
        validated = True
        gt_videos = os.listdir(gt_root)  # 获取ground-truth视频文件夹中的文件
        mask_videos = os.listdir(mask_root)  # 获取mask文件夹中的文件

        # if the user passed the root directory instead of Annotations
        # /
        # 如果用户传入的是根目录而不是Annotations文件夹
        if len(gt_videos) != len(mask_videos):
            if "Annotations" in gt_videos:
                if ".png" not in os.listdir(path.join(gt_root, "Annotations"))[0]:
                    gt_root = path.join(gt_root, "Annotations")
                    gt_videos = os.listdir(gt_root)

        # remove non-folder items / 删除非文件夹项
        gt_videos = list(filter(lambda x: path.isdir(path.join(gt_root, x)), gt_videos))
        mask_videos = list(
            filter(lambda x: path.isdir(path.join(mask_root, x)), mask_videos)
        )

        if not strict:
            # 在非严格模式下，只考虑gt和mask文件夹中都有的视频
            videos = sorted(list(set(gt_videos) & set(mask_videos)))
        else:
            # 在严格模式下，检查ground-truth和mask中不匹配的视频
            gt_extras = set(gt_videos) - set(mask_videos)
            mask_extras = set(mask_videos) - set(gt_videos)

            if len(gt_extras) > 0:
                print(
                    f"Videos that are in {gt_root} but not in {mask_root}: {gt_extras}"
                )
                validated = False
            if len(mask_extras) > 0:
                print(
                    f"Videos that are in {mask_root} but not in {gt_root}: {mask_extras}"
                )
                validated = False
            if not validated:
                print("Validation failed. Exiting.")
                exit(1)

            # 严格模式下，只评估gt中的视频
            videos = sorted(gt_videos)

        if verbose:
            print(
                f"In dataset {gt_root}, we are evaluating on {len(videos)} videos: {videos}"
            )

        if single_dataset:
            if verbose:
                results = tqdm.tqdm(
                    pool.imap(
                        VideoEvaluator(
                            gt_root, mask_root, skip_first_and_last=skip_first_and_last
                        ),
                        videos,
                    ),
                    total=len(videos),
                )
            else:
                results = pool.map(
                    VideoEvaluator(
                        gt_root, mask_root, skip_first_and_last=skip_first_and_last
                    ),
                    videos,
                )
        else:
            to_wait.append(
                pool.map_async(
                    VideoEvaluator(
                        gt_root, mask_root, skip_first_and_last=skip_first_and_last
                    ),
                    videos,
                )
            )

    pool.close()  # 关闭进程池

    all_global_jf, all_global_j, all_global_f = [], [], []  # 存储评估结果
    all_object_metrics = []  # 存储每个物体的评估结果
    for i, mask_root in enumerate(mask_roots):
        if not single_dataset:
            results = to_wait[i].get()

        all_iou = []
        all_boundary_f = []
        object_metrics = {}
        for name, iou, boundary_f in results:
            all_iou.extend(list(iou.values()))
            all_boundary_f.extend(list(boundary_f.values()))
            object_metrics[name] = (iou, boundary_f)

        global_j = np.array(all_iou).mean()  # 平均IoU
        global_f = np.array(all_boundary_f).mean()  # 平均边界F值
        global_jf = (global_j + global_f) / 2  # 综合J&F得分

        time_taken = time.time() - start  # 计算耗时
        """
        Build string for reporting results / 构建报告结果的字符串
        """
        # find max length for padding / 查找最大长度以进行填充
        ml = max(*[len(n) for n in object_metrics.keys()], len("Global score"))
        # build header / 构建标题
        out_string = f'{"sequence":<{ml}},{"obj":>3}, {"J&F":>4}, {"J":>4}, {"F":>4}\n'
        out_string += f'{"Global score":<{ml}},{"":>3}, {global_jf:.1f}, {global_j:.1f}, {global_f:.1f}\n'
        # append one line for each object / 为每个对象添加一行
        for name, (iou, boundary_f) in object_metrics.items():
            for object_idx in iou.keys():
                j, f = iou[object_idx], boundary_f[object_idx]
                jf = (j + f) / 2
                out_string += (
                    f"{name:<{ml}},{object_idx:03}, {jf:>4.1f}, {j:>4.1f}, {f:>4.1f}\n"
                )

        # print to console / 打印到控制台
        if verbose:
            print(out_string.replace(",", " "), end="")
            print("\nSummary:")
            print(
                f"Global score: J&F: {global_jf:.1f} J: {global_j:.1f} F: {global_f:.1f}"
            )
            print(f"Time taken: {time_taken:.2f}s")

        # print to file / 打印到文件
        result_path = path.join(mask_root, "results.csv")
        print(f"Saving the results to {result_path}")
        with open(result_path, "w") as f:
            f.write(out_string)

        all_global_jf.append(global_jf)
        all_global_j.append(global_j)
        all_global_f.append(global_f)
        all_object_metrics.append(object_metrics)

    return all_global_jf, all_global_j, all_global_f, all_object_metrics
