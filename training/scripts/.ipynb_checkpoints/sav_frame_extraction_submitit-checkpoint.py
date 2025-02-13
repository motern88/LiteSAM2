# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
import argparse
import os
from pathlib import Path

import cv2

import numpy as np
import submitit
import tqdm


def get_args_parser():
    """获取命令行参数解析器"""
    parser = argparse.ArgumentParser(
        description="[SA-V Preprocessing] 提取JPEG帧 / [SA-V Preprocessing] Extracting JPEG frames",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ------------
    # DATA
    # ------------
    data_parser = parser.add_argument_group(
        title="SA-V dataset data root",
        description="要加载的数据以及如何处理它 / What data to load and how to process it.",
    )
    data_parser.add_argument(
        "--sav-vid-dir",
        type=str,
        required=True,
        help=("视频文件的存放路径 / Where to find the SAV videos"),
    )
    data_parser.add_argument(
        "--sav-frame-sample-rate",
        type=int,
        default=4,
        help="帧采样率，表示每隔多少帧取一帧 / Rate at which to sub-sample frames",
    )

    # ------------
    # LAUNCH
    # ------------
    launch_parser = parser.add_argument_group(
        title="Cluster launch settings",
        description="作业数量和重试设置 / Number of jobs and retry settings.",
    )
    launch_parser.add_argument(
        "--n-jobs",
        type=int,
        required=True,
        help="将运行分割为多个作业的数量 / Shard the run over this many jobs.",
    )
    launch_parser.add_argument(
        "--timeout", type=int, required=True, help="SLURM 超时参数，单位为分钟 / SLURM timeout parameter in minutes."
    )
    launch_parser.add_argument(
        "--partition", type=str, required=True, help="提交作业的分区 / Partition to launch on."
    )
    launch_parser.add_argument(
        "--account", type=str, required=True, help="SLURM的账户名 / Partition to launch on."
    )
    launch_parser.add_argument("--qos", type=str, required=True, help="QOS.")

    # ------------
    # OUTPUT
    # ------------
    output_parser = parser.add_argument_group(
        title="Setting for results output", description="指定结果保存的位置及方式 / Where and how to save results."
    )
    output_parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help=("提取的JPEG帧保存的目录路径 / Where to dump the extracted jpeg frames"),
    )
    output_parser.add_argument(
        "--slurm-output-root-dir",
        type=str,
        required=True,
        help=("保存SLURM输出日志的目录路径 / Where to save slurm outputs"),
    )
    return parser

# 解码视频文件，读取视频帧
def decode_video(video_path: str):
    assert os.path.exists(video_path)
    video = cv2.VideoCapture(video_path)
    video_frames = []
    while video.isOpened():
        ret, frame = video.read()
        if ret:
            video_frames.append(frame)
        else:
            break
    return video_frames

# 根据采样率从视频中提取帧
def extract_frames(video_path, sample_rate):
    frames = decode_video(video_path)
    return frames[::sample_rate]


# 使用Submitit提交任务并保存帧
def submitit_launch(video_paths, sample_rate, save_root):
    for path in tqdm.tqdm(video_paths):
        frames = extract_frames(path, sample_rate)
        output_folder = os.path.join(save_root, Path(path).stem)
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        for fid, frame in enumerate(frames):
            frame_path = os.path.join(output_folder, f"{fid*sample_rate:05d}.jpg")
            cv2.imwrite(frame_path, frame)
    print(f"Saved output to {save_root}")


if __name__ == "__main__":
    # 获取命令行参数
    parser = get_args_parser()
    args = parser.parse_args()

    # 获取视频目录、保存目录和帧采样率
    sav_vid_dir = args.sav_vid_dir
    save_root = args.output_dir
    sample_rate = args.sav_frame_sample_rate

    # 列出所有SA-V视频文件 / List all SA-V videos
    mp4_files = sorted([str(p) for p in Path(sav_vid_dir).glob("*/*.mp4")])
    mp4_files = np.array(mp4_files)
    # 将视频文件分块，依据作业数量进行分割
    chunked_mp4_files = [x.tolist() for x in np.array_split(mp4_files, args.n_jobs)]

    print(f"Processing videos in: {sav_vid_dir}")
    print(f"Processing {len(mp4_files)} files")
    print(f"Beginning processing in {args.n_jobs} processes")

    # Submitit参数设置 / Submitit params
    jobs_dir = os.path.join(args.slurm_output_root_dir, "%j")  # 保存SLURM输出日志的目录
    cpus_per_task = 4  # 每个任务分配的CPU核心数
    executor = submitit.AutoExecutor(folder=jobs_dir)  # 提交作业的执行器
    executor.update_parameters(
        timeout_min=args.timeout,  # 设置作业的超时时间（分钟）
        gpus_per_node=0,  # 每个节点分配的GPU数量（这里不使用GPU）
        tasks_per_node=1,  # 每个节点分配的任务数
        slurm_array_parallelism=args.n_jobs,  # SLURM数组并行度，作业数
        cpus_per_task=cpus_per_task,  # 每个任务的CPU核心数
        slurm_partition=args.partition,  # SLURM分区设置
        slurm_account=args.account,  # SLURM账户
        slurm_qos=args.qos,  # SLURM QOS
    )
    executor.update_parameters(slurm_srun_args=["-vv", "--cpu-bind", "none"])  # 设置SLURM srun参数

    # 提交作业 / Launch
    jobs = []
    with executor.batch():  # 批量提交作业
        for _, mp4_chunk in tqdm.tqdm(enumerate(chunked_mp4_files)):
            # 提交每个视频文件块的处理任务
            job = executor.submit(
                submitit_launch,  # 提交的任务函数
                video_paths=mp4_chunk,  # 当前块的视频文件路径
                sample_rate=sample_rate,  # 帧采样率
                save_root=save_root,  # 保存帧的根目录
            )
            jobs.append(job)  # 将提交的作业添加到作业列表中

    # 打印每个作业的SLURM JobID
    for j in jobs:
        print(f"Slurm JobID: {j.job_id}")
    print(f"Saving outputs to {save_root}")
    print(f"Slurm outputs at {args.slurm_output_root_dir}")
