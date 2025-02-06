# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os
import random
import sys
import traceback
from argparse import ArgumentParser

import submitit
import torch

from hydra import compose, initialize_config_module
from hydra.utils import instantiate

from iopath.common.file_io import g_pathmgr
from omegaconf import OmegaConf

from training.utils.train_utils import makedir, register_omegaconf_resolvers

os.environ["HYDRA_FULL_ERROR"] = "1"


def single_proc_run(local_rank, main_port, cfg, world_size):
    """单GPU进程运行 / Single GPU process"""
    os.environ["MASTER_ADDR"] = "localhost"  # 设置主节点地址
    os.environ["MASTER_PORT"] = str(main_port)  # 设置主节点端口
    os.environ["RANK"] = str(local_rank)  # 设置当前进程的全局排名
    os.environ["LOCAL_RANK"] = str(local_rank)  # 设置当前进程在本地节点的排名
    os.environ["WORLD_SIZE"] = str(world_size)  # 设置进程总数
    try:
        register_omegaconf_resolvers()  # 注册OmegaConf解析器
    except Exception as e:
        logging.info(e)

    # 实例化并运行训练器
    trainer = instantiate(cfg.trainer, _recursive_=False)
    trainer.run()


def single_node_runner(cfg, main_port: int):
    assert cfg.launcher.num_nodes == 1  # 确保只有一个节点
    num_proc = cfg.launcher.gpus_per_node  # 获取每个节点的GPU数量
    torch.multiprocessing.set_start_method(
        "spawn"
    )  # CUDA运行时不支持`fork`，所以使用`spawn`方法启动子进程 / CUDA runtime does not support `fork`
    if num_proc == 1:
        # 如果只有一个GPU，直接调用single_proc以便于设置断点 /directly call single_proc so we can easily set breakpoints
        # 因为mp.spawn不允许设置断点 / mp.spawn does not let us set breakpoints
        single_proc_run(local_rank=0, main_port=main_port, cfg=cfg, world_size=num_proc)
    else:
        # 启动多个进程进行并行训练
        mp_runner = torch.multiprocessing.start_processes
        args = (main_port, cfg, num_proc)
        # Note: using "fork" below, "spawn" causes time and error regressions. Using
        # spawn changes the default multiprocessing context to spawn, which doesn't
        # interact well with the dataloaders (likely due to the use of OpenCV).
        # 注意：使用"fork"，因为"spawn"会导致时间和错误回归。使用spawn会将默认的多进程上下文更改为spawn，
        # 这与数据加载器不兼容（可能是由于OpenCV的使用）
        mp_runner(single_proc_run, args=args, nprocs=num_proc, start_method="spawn")


# 格式化异常信息
def format_exception(e: Exception, limit=20):
    traceback_str = "".join(traceback.format_tb(e.__traceback__, limit=limit))
    return f"{type(e).__name__}: {e}\nTraceback:\n{traceback_str}"


class SubmititRunner(submitit.helpers.Checkpointable):
    """
    A callable which is passed to submitit to launch the jobs.
    一个可以传递给submitit来启动作业的可调用类
    """

    def __init__(self, port, cfg):
        self.cfg = cfg
        self.port = port
        self.has_setup = False

    def run_trainer(self):
        job_env = submitit.JobEnvironment()  # 获取作业环境
        # Need to add this again so the hydra.job.set_env PYTHONPATH
        # is also set when launching jobs.
        # 需要再次添加这个，以便在启动作业时设置hydra.job.set_env PYTHONPATH
        add_pythonpath_to_sys_path()  # 将Python路径添加到系统路径中
        os.environ["MASTER_ADDR"] = job_env.hostnames[0]
        os.environ["MASTER_PORT"] = str(self.port)
        os.environ["RANK"] = str(job_env.global_rank)
        os.environ["LOCAL_RANK"] = str(job_env.local_rank)
        os.environ["WORLD_SIZE"] = str(job_env.num_tasks)

        # 注册OmegaConf解析器
        register_omegaconf_resolvers()
        # 解析配置
        cfg_resolved = OmegaConf.to_container(self.cfg, resolve=False)
        cfg_resolved = OmegaConf.create(cfg_resolved)

        # 实例化并运行训练器
        trainer = instantiate(cfg_resolved.trainer, _recursive_=False)
        trainer.run()

    def __call__(self):
        job_env = submitit.JobEnvironment()  # 获取作业环境
        self.setup_job_info(job_env.job_id, job_env.global_rank)
        try:
            self.run_trainer()
        except Exception as e:
            # 记录异常信息，然后重新抛出 / Log the exception. Then raise it again (as what SubmititRunner currently does).
            message = format_exception(e)
            logging.error(message)
            raise e

    def setup_job_info(self, job_id, rank):
        """Set up slurm job info"""
        self.job_info = {
            "job_id": job_id,
            "rank": rank,
            "cluster": self.cfg.get("cluster", None),  # 集群信息
            "experiment_log_dir": self.cfg.launcher.experiment_log_dir,  # 实验日志目录
        }

        self.has_setup = True  # 设置完成

# 将PYTHONPATH添加到系统路径
def add_pythonpath_to_sys_path():
    if "PYTHONPATH" not in os.environ or not os.environ["PYTHONPATH"]:
        return
    sys.path = os.environ["PYTHONPATH"].split(":") + sys.path


def main(args) -> None:
    # 读取并解析配置文件
    cfg = compose(config_name=args.config)
    # 如果没有指定实验日志目录，则使用默认目录
    if cfg.launcher.experiment_log_dir is None:
        cfg.launcher.experiment_log_dir = os.path.join(
            os.getcwd(), "ETAM_logs", args.config
        )
    print("###################### Train App Config ####################")
    print(OmegaConf.to_yaml(cfg))
    print("############################################################")

    # 将PYTHONPATH添加到系统路径
    add_pythonpath_to_sys_path()
    # 创建实验日志目录
    makedir(cfg.launcher.experiment_log_dir)
    # 将配置保存到日志目录中的config.yaml文件
    with g_pathmgr.open(
        os.path.join(cfg.launcher.experiment_log_dir, "config.yaml"), "w"
    ) as f:
        f.write(OmegaConf.to_yaml(cfg))
    # 解析配置文件并保存解析后的配置
    cfg_resolved = OmegaConf.to_container(cfg, resolve=False)
    cfg_resolved = OmegaConf.create(cfg_resolved)

    with g_pathmgr.open(
        os.path.join(cfg.launcher.experiment_log_dir, "config_resolved.yaml"), "w"
    ) as f:
        f.write(OmegaConf.to_yaml(cfg_resolved, resolve=True))

    # 获取submitit配置，如果没有则抛出异常
    submitit_conf = cfg.get("submitit", None)
    assert submitit_conf is not None, "Missing submitit config"

    submitit_dir = cfg.launcher.experiment_log_dir
    submitit_dir = os.path.join(submitit_dir, "submitit_logs")
    # 优先使用命令行传入的参数 / Priotrize cmd line args
    cfg.launcher.gpus_per_node = (
        args.num_gpus if args.num_gpus is not None else cfg.launcher.gpus_per_node
    )
    cfg.launcher.num_nodes = (
        args.num_nodes if args.num_nodes is not None else cfg.launcher.num_nodes
    )
    submitit_conf.use_cluster = (
        args.use_cluster if args.use_cluster is not None else submitit_conf.use_cluster
    )
    if submitit_conf.use_cluster:
        # 如果使用集群，初始化submitit执行器
        executor = submitit.AutoExecutor(folder=submitit_dir)
        submitit_conf.partition = (
            args.partition
            if args.partition is not None
            else submitit_conf.get("partition", None)
        )
        submitit_conf.account = (
            args.account
            if args.account is not None
            else submitit_conf.get("account", None)
        )
        submitit_conf.qos = (
            args.qos if args.qos is not None else submitit_conf.get("qos", None)
        )
        job_kwargs = {
            "timeout_min": 60 * submitit_conf.timeout_hour,  # 设置作业超时时间
            "name": (
                submitit_conf.name if hasattr(submitit_conf, "name") else args.config
            ),
            "slurm_partition": submitit_conf.partition,  # 设置SLURM分区
            "gpus_per_node": cfg.launcher.gpus_per_node,  # 每个节点的GPU数量
            "tasks_per_node": cfg.launcher.gpus_per_node,  # 每个GPU分配一个任务 / one task per GPU
            "cpus_per_task": submitit_conf.cpus_per_task,  # 每个任务分配的CPU数
            "nodes": cfg.launcher.num_nodes,  # 节点数量
            "slurm_additional_parameters": {
                "exclude": " ".join(submitit_conf.get("exclude_nodes", [])),  # 排除节点
            },
        }
        # 设置包含节点的配置
        if "include_nodes" in submitit_conf:
            assert (
                len(submitit_conf["include_nodes"]) >= cfg.launcher.num_nodes
            ), "Not enough nodes"
            job_kwargs["slurm_additional_parameters"]["nodelist"] = " ".join(
                submitit_conf["include_nodes"]
            )
        # 设置其他SLURM参数
        if submitit_conf.account is not None:
            job_kwargs["slurm_additional_parameters"]["account"] = submitit_conf.account
        if submitit_conf.qos is not None:
            job_kwargs["slurm_additional_parameters"]["qos"] = submitit_conf.qos

        # 设置内存配置
        if submitit_conf.get("mem_gb", None) is not None:
            job_kwargs["mem_gb"] = submitit_conf.mem_gb
        elif submitit_conf.get("mem", None) is not None:
            job_kwargs["slurm_mem"] = submitit_conf.mem

        # 设置约束条件
        if submitit_conf.get("constraints", None) is not None:
            job_kwargs["slurm_constraint"] = submitit_conf.constraints

        # 设置作业备注
        if submitit_conf.get("comment", None) is not None:
            job_kwargs["slurm_comment"] = submitit_conf.comment

        # Supports only cpu-bind option within srun_args. New options can be added here
        # 支持设置srun_args（目前只支持cpu-bind选项）
        if submitit_conf.get("srun_args", None) is not None:
            job_kwargs["slurm_srun_args"] = []
            if submitit_conf.srun_args.get("cpu_bind", None) is not None:
                job_kwargs["slurm_srun_args"].extend(
                    ["--cpu-bind", submitit_conf.srun_args.cpu_bind]
                )

        print("###################### SLURM Config ####################")
        print(job_kwargs)
        print("##########################################")
        executor.update_parameters(**job_kwargs)

        # 随机生成主节点端口
        main_port = random.randint(
            submitit_conf.port_range[0], submitit_conf.port_range[1]
        )
        # 创建SubmititRunner并提交作业
        runner = SubmititRunner(main_port, cfg)
        job = executor.submit(runner)
        print(f"Submitit Job ID: {job.job_id}")
        runner.setup_job_info(job.job_id, rank=0)
    else:
        # 如果不使用集群，单节点运行
        cfg.launcher.num_nodes = 1
        main_port = random.randint(
            submitit_conf.port_range[0], submitit_conf.port_range[1]
        )
        single_node_runner(cfg, main_port)


if __name__ == "__main__":

    initialize_config_module("sam2", version_base="1.2")
    parser = ArgumentParser()
    parser.add_argument(
        "-c",
        "--config",
        required=True,
        type=str,
        help="path to config file (e.g. configs/sam2.1_training/sam2.1_hiera_b+_MOSE_finetune.yaml)",
    )
    # 是否使用集群，0表示本地运行，1表示集群运行
    parser.add_argument(
        "--use-cluster",
        type=int,
        default=None,
        help="whether to launch on a cluster, 0: run locally, 1: run on a cluster",
    )
    # SLURM的分区名称
    parser.add_argument("--partition", type=str, default=None, help="SLURM partition")
    # SLURM的账户名称
    parser.add_argument("--account", type=str, default=None, help="SLURM account")
    # SLURM的QOS（服务质量）
    parser.add_argument("--qos", type=str, default=None, help="SLURM qos")
    # 每个节点的GPU数量
    parser.add_argument(
        "--num-gpus", type=int, default=None, help="number of GPUS per node"
    )
    parser.add_argument("--num-nodes", type=int, default=None, help="Number of nodes")
    args = parser.parse_args()
    # 将use_cluster的值转换为布尔类型（0 -> False, 1 -> True）
    args.use_cluster = bool(args.use_cluster) if args.use_cluster is not None else None
    register_omegaconf_resolvers()  # 注册OmegaConf解析器
    main(args)
