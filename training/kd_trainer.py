import gc
import json
import logging
import math
import os
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional

import numpy as np

import torch
import torch.distributed as dist
import torch.nn as nn
from hydra.utils import instantiate
from iopath.common.file_io import g_pathmgr

from training.optimizer import construct_optimizer

from training.utils.checkpoint_utils import (
    assert_skipped_parameters_are_frozen,
    exclude_params_matching_unix_pattern,
    load_state_dict_into_model,
    with_check_parameter_frozen,
)
from training.utils.data_utils import BatchedVideoDatapoint
from training.utils.distributed import all_reduce_max, barrier, get_rank

from training.utils.logger import Logger, setup_logging

from training.utils.train_utils import (
    AverageMeter,
    collect_dict_keys,
    DurationMeter,
    get_amp_type,
    get_machine_local_and_dist_rank,
    get_resume_checkpoint,
    human_readable_time,
    is_dist_avail_and_initialized,
    log_env_variables,
    makedir,
    MemMeter,
    Phase,
    ProgressMeter,
    set_seeds,
    setup_distributed_backend,
)


CORE_LOSS_KEY = "core_loss"


@dataclass
class OptimAMPConf:
    enabled: bool = False  # 是否启用自动混合精度（AMP）
    amp_dtype: str = "float16"  # 使用的精度类型

# 优化器配置类
@dataclass
class OptimConf:
    optimizer: torch.optim.Optimizer = None  # 优化器
    options: Optional[Dict[str, Any]] = None  # 优化器的配置
    param_group_modifiers: Optional[List] = None  # 参数组的修改器，用于对不同参数组使用不同的优化策略
    amp: Optional[Dict[str, Any]] = None  # 自动混合精度配置（如果启用）
    gradient_clip: Any = None  # 梯度裁剪的配置
    gradient_logger: Any = None  # 梯度日志记录

    def __post_init__(self):
        # amp 配置的处理：如果 `amp` 不是 OptimAMPConf 实例，则将其转换为该类型的实例
        if not isinstance(self.amp, OptimAMPConf):
            if self.amp is None:
                self.amp = {}  # 如果未提供 amp 配置，则默认设置为空字典
            assert isinstance(self.amp, Mapping)  # 确保 amp 配置是一个映射类型（即字典）
            # 使用提供的配置字典来初始化 OptimAMPConf 实例
            self.amp = OptimAMPConf(**self.amp)

# 分布式配置类
@dataclass
class DistributedConf:
    backend: Optional[str] = None  # 从加速器类型推断 inferred from accelerator type, (e.g. "nccl" for cuda)
    comms_dtype: Optional[str] = None  # 通信数据类型，用于指定分布式训练中数据传输的精度类型，如 "float32" 或 "float16" 等。
    find_unused_parameters: bool = False  # 是否查找未使用的模型参数，若为 True，分布式训练时将检查并标记模型中未使用的参数，防止在反向传播时出错。
    timeout_mins: int = 30  # 分布式训练的超时限制（单位：分钟）

@dataclass
class CudaConf:
    cudnn_deterministic: bool = False  # 是否启用 cuDNN 的确定性算法。若启用，则每次运行相同的操作时，结果应该一致。
    cudnn_benchmark: bool = True  # 是否启用 cuDNN 的自动优化。可以提高卷积操作的性能，适用于固定输入尺寸的情况。
    allow_tf32: bool = False  # 是否允许 TensorFloat-32（TF32）精度。TF32 在一些硬件上可提升性能。
    # if not None, `matmul_allow_tf32` key will override `allow_tf32` for matmul
    # 如果不是 None，'matmul_allow_tf32' Key 将覆盖 allow_tf32 对于矩阵乘法的设置。
    matmul_allow_tf32: Optional[bool] = None
    # if not None, `cudnn_allow_tf32` key will override `allow_tf32` for cudnn
    # 如果不是 None，cudnn_allow_tf32 key将覆盖 allow_tf32 对于 cudnn 的设置。
    cudnn_allow_tf32: Optional[bool] = None

@dataclass
class CheckpointConf:
    save_dir: str  # 保存检查点的目录
    save_freq: int  # 保存频率，单位为迭代次数（iterations）
    save_list: List[int] = field(default_factory=list)  # 需要保存的迭代次数列表，默认空列表
    model_weight_initializer: Any = None  # 模型权重初始化器（可选）
    save_best_meters: List[str] = None  # 用于保存最佳指标的列表，如损失、准确率等
    skip_saving_parameters: List[str] = field(default_factory=list)  # 跳过保存的参数列表
    initialize_after_preemption: Optional[bool] = None  # 是否在训练中断后重新初始化模型（如果为 None，则根据其他参数决定）
    # if not None, training will be resumed from this checkpoint
    # /
    # 如果不是 None，训练将从此检查点恢复
    resume_from: Optional[str] = None

    def infer_missing(self):
        # 如果未指定 `initialize_after_preemption`，则根据 `skip_saving_parameters` 来推断
        if self.initialize_after_preemption is None:
            with_skip_saving = len(self.skip_saving_parameters) > 0
            self.initialize_after_preemption = with_skip_saving
        return self

@dataclass
class LoggingConf:
    log_dir: str  # 日志文件目录
    log_freq: int  # 日历频率 单位为迭代次数（iterations）
    tensorboard_writer: Any
    log_level_primary: str = "INFO"  # 主日志记录的级别（默认是 INFO）
    log_level_secondary: str = "ERROR"  # 次日志记录的级别（默认是 ERROR）
    log_scalar_frequency: int = 100  # 记录标量信息的频率，单位为迭代次数
    log_visual_frequency: int = 100  # 记录可视化信息的频率，单位为迭代次数
    scalar_keys_to_log: Optional[Dict[str, Any]] = None  # 需要记录的标量的字典，键是标量名，值是对应的值
    log_batch_stats: bool = False  # 是否记录批次统计信息（如每批次的平均损失等）

# 知识蒸馏训练类
class KnowledgeDistillationTrainer:
    '''
    KD Trainer supporting the DDP training strategies.
    支持 DDP 训练的 KD Trainer
    '''

    EPSILON = 1e-8  # 小的常数，防止数值不稳定

    # -----------------------------------------------------------------------------------------
    # 下：各组件初始化与构建类

    def __init__(
        self,
        # 这些参数的顺序随时可能改变，因此它们是仅限关键字参数
        *,  # the order of these args can change at any time, so they are keyword-only
        data: Dict[str, Any],
        teacher_model: Dict[str, Any],  # 教师模型配置
        student_model: Dict[str, Any],  # 学生模型配置
        logging: Dict[str, Any],
        checkpoint: Dict[str, Any],
        max_epochs: int,
        mode: str = "train",  # "train", "val", "train_only"
        accelerator: str = "cuda",
        seed_value: int = 123,
        val_epoch_freq: int = 1,  # 验证频率（每几轮验证一次）
        distributed: Dict[str, bool] = None,
        cuda: Dict[str, bool] = None,
        env_variables: Optional[Dict[str, Any]] = None,
        optim: Optional[Dict[str, Any]] = None,
        optim_overrides: Optional[List[Dict[str, Any]]] = None,  # 优化器覆盖配置
        meters: Optional[Dict[str, Any]] = None,  # 训练过程中需要的计量器
        loss: Optional[Dict[str, Any]] = None,
    ):
        self._setup_env_variables(env_variables)  # 设置环境变量
        self._setup_timers()  # 初始化定时器
        # 数据、模型配置文件
        self.data_conf = data
        self.teacher_model_conf = teacher_model
        self.student_model_conf = student_model
        # 日志、检查点配置文件（只针对学生模型）
        self.logging_conf = LoggingConf(**logging)
        self.checkpoint_conf = CheckpointConf(**checkpoint).infer_missing()  # 检查点配置
        # 训练配置
        self.max_epochs = max_epochs
        self.mode = mode
        self.val_epoch_freq = val_epoch_freq  # 验证频率
        self.optim_conf = OptimConf(**optim) if optim is not None else None
        self.meters_conf = meters  # 计量器配置
        self.loss_conf = loss
        distributed = DistributedConf(**distributed or {})
        cuda = CudaConf(**cuda or {})
        self.where = 0.0

        self._infer_distributed_backend_if_none(distributed, accelerator)  # 推断分布式后端

        self._setup_device(accelerator)  # 设置设备（GPU 或 CPU）

        self._setup_torch_dist_and_backend(cuda, distributed)  # 设置分布式训练环境

        # 设置日志文件夹并初始化日志
        makedir(self.logging_conf.log_dir)
        setup_logging(
            __name__,
            output_dir=self.logging_conf.log_dir,
            rank=self.rank,
            log_level_primary=self.logging_conf.log_level_primary,
            log_level_secondary=self.logging_conf.log_level_secondary,
        )

        set_seeds(seed_value, self.max_epochs, self.distributed_rank)
        log_env_variables()  # 输出环境变量

        # 确保分布式环境已初始化
        assert (
            is_dist_avail_and_initialized()
        ), "Torch分布式需要在调用Trainer之前初始化 / Torch distributed needs to be initialized before calling the trainer."

        self._setup_components()  # 除了优化器，其他都在这里设置 / Except Optimizer everything is setup here.
        self._move_to_device()
        self._construct_optimizers()  # 构建优化器
        self._setup_dataloaders()  # 设置数据加载器

        # 初始化经过时间计量器
        self.time_elapsed_meter = DurationMeter("Time Elapsed", self.device, ":.2f")

        # 如果有恢复检查点的需求，处理恢复逻辑
        if self.checkpoint_conf.resume_from is not None:
            # 确保 'resume_from' 指定的检查点文件存在
            assert os.path.exists(
                self.checkpoint_conf.resume_from
            ), f"The 'resume_from' checkpoint {self.checkpoint_conf.resume_from} does not exist!"

            # 定义目标路径（检查点存放的路径）
            dst = os.path.join(self.checkpoint_conf.save_dir, "checkpoint.pt")

            # 如果当前进程是分布式训练中的主进程（rank == 0），并且目标路径不存在
            if self.distributed_rank == 0 and not os.path.exists(dst):
                # Copy the "resume_from" checkpoint to the checkpoint folder
                # if there is not a checkpoint to resume from already there
                # /
                # 如果检查点文件夹中没有可恢复的检查点，则将“resume_from”检查点复制到检查点文件夹。
                makedir(self.checkpoint_conf.save_dir)
                g_pathmgr.copy(self.checkpoint_conf.resume_from, dst)

            # 同步其他进程，确保在恢复检查点后，所有进程都在同一阶段
            barrier()

        self.load_checkpoint()
        self._setup_ddp_distributed_training(distributed, accelerator)  # 设置 DDP 分布式训练
        barrier()  # 等待所有进程完成

    # 其他参数初始化设置
    def _setup_components(self):
        # Get the keys for all the val datasets, if any
        # 获取所有验证数据集的键（如果有的话）
        val_phase = Phase.VAL  # 设置验证阶段的标识符
        val_keys = None

        # 如果验证阶段的数据配置存在，则获取验证数据集的所有键
        if self.data_conf.get(val_phase, None) is not None:
            val_keys = collect_dict_keys(self.data_conf[val_phase])
        # Additional checks on the sanity of the config for val datasets
        # 对验证数据集的配置进行额外的合理性检查
        self._check_val_key_match(val_keys, phase=val_phase)

        logging.info("Setting up components: Model, loss, optim, meters etc.")
        self.epoch = 0
        self.steps = {Phase.TRAIN: 0, Phase.VAL: 0}  # 初始化训练和验证阶段的步数

        # 初始化日志记录器
        self.logger = Logger(self.logging_conf)

        # 实例化教师模型和学生
        self.teacher_model = instantiate(self.teacher_model_conf, _convert_="all")
        self.teacher_model.eval()  # 设置教师模型为评估模式
        self.student_model = instantiate(self.student_model_conf, _convert_="all")

        logging.info(f"////////Teacher Model////////")
        print_model_summary(self.teacher_model)  # 打印学生模型结构
        logging.info(f"////////Student Model////////")
        print_model_summary(self.student_model)  # 打印学生模型结构

        # 初始化损失函数  TODO:这里应该是初始化total_loss还是分别初始化hard_loss和soft_loss？
        self.hard_loss = None
        self.soft_loss = None
        if self.loss_conf:
            # 知识蒸馏硬损失和软损失
            self.hard_loss = {
                key: el  # wrap_base_loss(el)
                for (key, el) in instantiate(self.loss_conf, _convert_="hard_loss").items()
            }
            self.hard_loss = nn.ModuleDict(self.hard_loss)  # 用 nn.ModuleDict 包装以便在 PyTorch 中当作模块管理

            self.soft_loss = {
                key: el  # wrap_base_loss(el)
                for (key, el) in instantiate(self.loss_conf, _convert_="soft_loss").items()
            }
            self.soft_loss = nn.ModuleDict(self.soft_loss)  # 用 nn.ModuleDict 包装以便在 PyTorch 中当作模块管理

        # 初始化度量器和最佳度量值字典
        self.meters = {}
        self.best_meter_values = {}
        if self.meters_conf:
            self.meters = instantiate(self.meters_conf, _convert_="all")

        # 初始化混合精度缩放器，用于 AMP（自动混合精度）训练
        self.scaler = torch.amp.GradScaler(
            self.device,
            enabled=self.optim_conf.amp.enabled if self.optim_conf else False,
        )

        # 如果配置了梯度裁剪，则实例化梯度裁剪器
        self.gradient_clipper = (
            instantiate(self.optim_conf.gradient_clip) if self.optim_conf else None
        )
        # 如果配置了梯度日志记录器，则实例化
        self.gradient_logger = (
            instantiate(self.optim_conf.gradient_logger) if self.optim_conf else None
        )

        logging.info("完成设置组件：模型、损失、优化器、计量器等。"
                     " / "
                     "Finished setting up components: Model, loss, optim, meters etc.")

    # 实例化数据集
    def _setup_dataloaders(self):
        self.train_dataset = None
        self.val_dataset = None

        # 如果模式是训练或验证
        if self.mode in ["train", "val"]:
            # 使用配置中 Phase.VAL 阶段的配置来实例化验证数据集
            self.val_dataset = instantiate(self.data_conf.get(Phase.VAL, None))

        # 如果模式是训练或仅训练
        if self.mode in ["train", "train_only"]:
            # 使用配置中 train 阶段的配置来实例化训练数据集
            self.train_dataset = instantiate(self.data_conf.train)

    # 构建优化器
    def _construct_optimizers(self):
        self.optim = construct_optimizer(
            self.student_model,  # 只对学生模型构建优化器
            self.optim_conf.optimizer,
            self.optim_conf.options,
            self.optim_conf.param_group_modifiers,
        )

    # 设置环境变量
    def _setup_env_variables(self, env_variables_conf) -> None:
        if env_variables_conf is not None:
            for variable_name, value in env_variables_conf.items():
                os.environ[variable_name] = value

    # 上：各组件初始化与构建类
    # -----------------------------------------------------------------------------------------
    # 下：日志记录类

    # 记录loss中的核心loss到日志
    def _log_loss_detailed_and_return_core_loss(self, loss, loss_str, step):
        # 从损失字典中提取核心损失（核心任务的主要损失值）
        core_loss = loss.pop(CORE_LOSS_KEY)
        # 如果当前步数是记录间隔的倍数，则记录损失值
        if step % self.logging_conf.log_scalar_frequency == 0:
            for k in loss:
                log_str = os.path.join(loss_str, k)
                self.logger.log(log_str, loss[k], step)
        return core_loss

    # 记录数据加载时间
    def _log_sync_data_times(self, phase, data_times):
        # 同步所有进程中的数据加载时间 / Synchronize data loading times across all processes
        data_times = all_reduce_max(torch.tensor(data_times)).tolist()
        steps = range(self.steps[phase] - len(data_times), self.steps[phase])
        for step, data_time in zip(steps, data_times):
            if step % self.logging_conf.log_scalar_frequency == 0:
                self.logger.log(
                    os.path.join("Step_Stats", phase, "Data Time Synced"),
                    data_time,
                    step,
                )

    # 同步并记录计量器的值，同时根据最佳指标保存检查点
    def _log_meters_and_save_best_ckpts(self, phases: List[str]):
        logging.info("Synchronizing meters")
        out_dict = {}
        checkpoint_save_keys = []

        for key, meter in self._get_meters(phases).items():
            # 计算同步后的计量器值。
            meter_output = meter.compute_synced()
            is_better_check = getattr(meter, "is_better", None)

            for meter_subkey, meter_value in meter_output.items():
                # 记录当前的计量器值。
                out_dict[os.path.join("Meters_train", key, meter_subkey)] = meter_value

                if is_better_check is None:  # 如果该计量器支持“更好”的检查，则更新最佳值。
                    continue

                tracked_meter_key = os.path.join(key, meter_subkey)
                if tracked_meter_key not in self.best_meter_values or is_better_check(
                        meter_value,
                        self.best_meter_values[tracked_meter_key],
                ):
                    # 更新最佳计量器值
                    self.best_meter_values[tracked_meter_key] = meter_value

                    # 如果需要保存该指标的最佳检查点，则记录对应的键。
                    if (
                            self.checkpoint_conf.save_best_meters is not None
                            and key in self.checkpoint_conf.save_best_meters
                    ):
                        checkpoint_save_keys.append(tracked_meter_key.replace("/", "_"))

        # 如果有需要保存的检查点，则执行保存操作。
        if len(checkpoint_save_keys) > 0:
            self.save_checkpoint(self.epoch + 1, checkpoint_save_keys)

        return out_dict

    # 计算从当前阶段（训练或验证）到训练结束所需的剩余时间
    def _log_timers(self, phase):
        time_remaining = 0
        # 计算剩余的训练轮数
        epochs_remaining = self.max_epochs - self.epoch - 1
        # 计算剩余的验证轮数
        val_epochs_remaining = sum(
            n % self.val_epoch_freq == 0 for n in range(self.epoch, self.max_epochs)
        )

        # Adding the guaranteed val run at the end if val_epoch_freq doesn't coincide with the end epoch.
        # /
        # 如果 val_epoch_freq 与结束轮次不一致，则在最后添加保证的验证运行。
        if (self.max_epochs - 1) % self.val_epoch_freq != 0:
            val_epochs_remaining += 1

        # Remove the current val run from estimate
        # /
        # 从 estimate 中删除当前 val run
        if phase == Phase.VAL:
            val_epochs_remaining -= 1

        # 估算剩余时间，包括训练和验证所需时间。
        time_remaining += (
            epochs_remaining * self.est_epoch_time[Phase.TRAIN]
            + val_epochs_remaining * self.est_epoch_time[Phase.VAL]
        )

        # 记录当前的已用时间。
        self.logger.log(
            os.path.join("Step_Stats", phase, self.time_elapsed_meter.name),
            self.time_elapsed_meter.val,
            self.steps[phase],
        )

        logging.info(f"Estimated time remaining: {human_readable_time(time_remaining)}")

    # 重置指定阶段的所有计量器。
    def _reset_meters(self, phases: str) -> None:
        for meter in self._get_meters(phases).values():
            meter.reset()

    # 上：日志记录类
    # -----------------------------------------------------------------------------------------
    # 下：训练与验证

    # 判断是否是验证epoch
    def is_intermediate_val_epoch(self, epoch):
        # 如果当前 epoch 能被 val_epoch_freq 整除，并且当前 epoch 小于 max_epochs-1，
        # 则认为是一个中间验证的 epoch
        return epoch % self.val_epoch_freq == 0 and epoch < self.max_epochs - 1

    # 计算总损失  # TODO:实现更好的总损失方式
    def _combined_loss(self, hard_loss, soft_loss):
        return hard_loss + soft_loss

    # 获取trainer状态
    def _get_trainer_state(self, phase):
        return {
            "Trainer/where": self.where,
            "Trainer/epoch": self.epoch,
            f"Trainer/steps_{phase}": self.steps[phase],
        }

    # step内部方法
    def _step(
        self,
        batch: BatchedVideoDatapoint,
        teacher_model: nn.Module,  # 教师模型
        student_model: nn.Module,  # 学生模型
        phase: str,  # 当前阶段（如训练或验证）
    ):
        # 模型的前向传播
        teacher_outputs, teacher_backbone_outputs = teacher_model(batch)  # 教师模型的输出
        student_outputs, student_backbone_outputs = student_model(batch)  # 学生模型的输出
        targets = batch.masks  # 获取目标数据（mask）
        batch_size = len(batch.img_batch)

        key = batch.dict_key  # 数据集的键 key for dataset

        # 计算KD hard_loss
        hard_loss = self.hard_loss[key](student_outputs, targets)  # TODO：hardloss取student_outputs最后输出结果
        # 计算KD soft_loss
        soft_loss = self.soft_loss[key](teacher_backbone_outputs, student_backbone_outputs) # TODO:实现对中间结果计算soft_loss


        loss_str = f"Losses/{phase}_{key}_loss"
        loss_log_str = os.path.join("Step_Losses", loss_str)

        # loss contains multiple sub-components we wish to log
        # 损失包含多个子组件，我们希望记录这些子组件
        step_losses = {}

        # 分别记录硬损失和软损失
        if isinstance(hard_loss, dict):
            step_losses.update(
                {f"Losses/{phase}_{key}_hard_{k}": v for k, v in hard_loss.items()}
            )
            # 记录详细的损失信息并返回核心损失
            hard_loss = self._log_loss_detailed_and_return_core_loss(
                hard_loss, loss_log_str, self.steps[phase]
            )
        if isinstance(soft_loss, dict):
            step_losses.update(
                {f"Losses/{phase}_{key}_soft_{k}": v for k, v in soft_loss.items()}
            )
            # 记录详细的损失信息并返回核心损失
            soft_loss = self._log_loss_detailed_and_return_core_loss(
                soft_loss, loss_log_str, self.steps[phase]
            )

        # 计算总损失
        total_loss = self._combined_loss(hard_loss,soft_loss)  # TODO:实现计算总损失_combined_loss

        # 每隔一定步数记录一次损失
        if self.steps[phase] % self.logging_conf.log_scalar_frequency == 0:
            self.logger.log(
                loss_log_str,
                total_loss,  # 记录合并后的总损失
                self.steps[phase],
            )

        # 更新步骤数
        self.steps[phase] += 1
        # 生成返回的元组，包含总损失、批量大小和详细损失
        ret_tuple = {loss_str: total_loss}, batch_size, step_losses

        # 更新指标（meters）
        if phase in self.meters and key in self.meters[phase]:
            meters_dict = self.meters[phase][key]
            if meters_dict is not None:
                for _, meter in meters_dict.items():
                    meter.update(
                        find_stages=student_outputs,  # 更新学生模型输出
                        find_metadatas=batch.metadata,
                    )

        return ret_tuple

    # 根据当前阶段（训练或验证）运行步骤
    def run(self):
        assert self.mode in ["train", "train_only", "val"]

        if self.mode == "train":
            # 如果当前 epoch 大于 0（表示不是从头开始训练）
            if self.epoch > 0:
                logging.info(f"Resuming training from epoch: {self.epoch}")
                # 从检查点恢复 / resuming from a checkpoint
                if self.is_intermediate_val_epoch(self.epoch - 1):
                    logging.info("Running previous val epoch")
                    # 如果当前 epoch 之前有验证步骤（即上一轮是验证阶段）
                    self.epoch -= 1  # 将 epoch 减 1，表示回退到上一轮进行验证
                    self.run_val()  # 运行验证阶段（用于评估模型性能）
                    self.epoch += 1  # 恢复回当前 epoch（恢复正常训练）

            # 运行训练阶段
            self.run_train()
            # 训练后运行验证阶段（每一轮训练后进行一次验证）
            self.run_val()

        # 如果当前模式是 "val"（验证模式）
        elif self.mode == "val":
            self.run_val()

        # 如果当前模式是 "train_only"（仅训练模式）
        elif self.mode == "train_only":
            self.run_train()

    # 运行训练
    def run_train(self):
        # 循环训练，直到达到最大 epoch 数
        while self.epoch < self.max_epochs:
            # 获取当前 epoch 对应的训练数据加载器
            dataloader = self.train_dataset.get_loader(epoch=int(self.epoch))
            barrier()  # 同步各个进程的状态

            outs = self.train_epoch(dataloader)
            self.logger.log_dict(outs, self.epoch)  # 仅在 rank 0 上记录 / Logged only on rank 0

            # 将训练日志记录到文本文件 / log train to text file.
            if self.distributed_rank == 0:
                with g_pathmgr.open(
                    os.path.join(self.logging_conf.log_dir, "train_stats.json"),
                    "a",
                ) as f:
                    f.write(json.dumps(outs) + "\n")

            # 在验证之前保存检查点 / Save checkpoint before validating
            self.save_checkpoint(self.epoch + 1)

            del dataloader  # 清除数据加载器，释放内存
            gc.collect()  # 强制进行垃圾回收，回收不再使用的内存

            # Run val, not running on last epoch since will run after the loop anyway
            # 运行验证，而不是在最后一轮运行，因为无论如何验证将在循环后运行。
            if self.is_intermediate_val_epoch(self.epoch):
                self.run_val()

            if self.distributed_rank == 0:
                # 更新最佳指标（根据训练状态）
                self.best_meter_values.update(self._get_trainer_state("train"))
                # 将最佳指标记录到文件
                with g_pathmgr.open(
                    os.path.join(self.logging_conf.log_dir, "best_stats.json"),
                    "a",
                ) as f:
                    f.write(json.dumps(self.best_meter_values) + "\n")

            self.epoch += 1

        # epoch was incremented in the loop but the val step runs out of the loop
        # 在循环中，epoch 已被递增，但验证步骤是在循环外运行的，因此将其恢复到正确的值
        self.epoch -= 1

    # 训练一个 epoch
    def train_epoch(self, train_loader):
        # 初始化统计计数器 / Init stat meters
        batch_time_meter = AverageMeter("Batch Time", self.device, ":.2f")
        data_time_meter = AverageMeter("Data Time", self.device, ":.2f")
        mem_meter = MemMeter("Mem (GB)", self.device, ":.2f")
        data_times = []
        phase = Phase.TRAIN  # 当前阶段为训练阶段 / Current phase is training

        # 计算每个 epoch 的迭代次数
        iters_per_epoch = len(train_loader)

        # 初始化损失统计计数器  TODO:这里是只定义hard_loss还是应该也一起定义soft_loss？
        loss_names = []
        for batch_key in self.hard_loss.keys():
            loss_names.append(f"Losses/{phase}_{batch_key}_hard_loss")
        for batch_key in self.soft_loss.keys():
            loss_names.append(f"Losses/{phase}_{batch_key}_soft_loss")

        loss_mts = OrderedDict(
            [(name, AverageMeter(name, self.device, ":.2e")) for name in loss_names]
        )  # 为hard_loss中每个损失创建一个 AverageMeter 对象，用于计算和记录损失的平均值

        extra_loss_mts = {}  # 额外损失统计计数器（用于记录soft_loss与hard_loss的详细损失）

        # 初始化进度条，显示训练过程的各类统计信息
        progress = ProgressMeter(
            iters_per_epoch,
            [
                batch_time_meter,
                data_time_meter,
                mem_meter,
                self.time_elapsed_meter,
                *loss_mts.values(),
            ],
            self._get_meters([phase]),
            prefix="Train Epoch: [{}]".format(self.epoch),
        )

        # 模型训练循环 / Model training loop
        self.student_model.train()
        end = time.time()

        # 训练循环，遍历每个 batch
        for data_iter, batch in enumerate(train_loader):
            # 测量数据加载时间
            data_time_meter.update(time.time() - end)
            data_times.append(data_time_meter.val)  # 记录数据加载时间
            batch = batch.to(
                self.device, non_blocking=True
            )  # 在 TensorClass 中移动张量 / move tensors in a tensorclass

            try:
                # 执行当前 batch 的训练步骤，包括计算损失、反向传播等
                self._run_step(batch, phase, loss_mts, extra_loss_mts)

                # 计算梯度并执行优化步骤 / compute gradient and do optim step
                exact_epoch = self.epoch + float(data_iter) / iters_per_epoch  # 计算当前的 epoch（包含部分迭代的时间）
                self.where = float(exact_epoch) / self.max_epochs  # 计算当前的训练进度（从 0 到 1）
                assert self.where <= 1 + self.EPSILON  # 确保训练进度小于等于 1

                # 更新调度器（若未到最后） / Update schedulers (if not at the end)
                if self.where < 1.0:
                    self.optim.step_schedulers(
                        self.where, step=int(exact_epoch * iters_per_epoch)
                    )
                else:
                    logging.warning(
                        f"Skipping scheduler update since the training is at the end, i.e, {self.where} of [0,1]."
                    )

                # 日志调度器 / Log schedulers
                if data_iter % self.logging_conf.log_scalar_frequency == 0:
                    for j, param_group in enumerate(self.optim.optimizer.param_groups):
                        for option in self.optim.schedulers[j]:
                            optim_prefix = (
                                "" + f"{j}_"
                                if len(self.optim.optimizer.param_groups) > 1
                                else ""
                            )
                            self.logger.log(
                                os.path.join("Optim", f"{optim_prefix}", option),
                                param_group[option],
                                self.steps[phase],
                            )

                # 裁剪梯度并检测发散梯度 / Clipping gradients and detecting diverging gradients
                if self.gradient_clipper is not None:
                    self.scaler.unscale_(self.optim.optimizer)
                    self.gradient_clipper(model=self.student_model)

                if self.gradient_logger is not None:
                    self.gradient_logger(
                        self.student_model, rank=self.distributed_rank, where=self.where
                    )

                # Optimizer step: the scaler will make sure gradients are not
                # applied if the gradients are infinite
                # /
                # 优化器步骤：scaler 会确保在梯度为无限大时，梯度不会被应用
                self.scaler.step(self.optim.optimizer)
                self.scaler.update()  # 更新 scaler，准备下一次的梯度更新

                # 测量经过的时间 / measure elapsed time
                batch_time_meter.update(time.time() - end)
                end = time.time()

                # 更新时间计数器 / Update time meters
                self.time_elapsed_meter.update(
                    time.time() - self.start_time + self.ckpt_time_elapsed
                )

                # 更新内存使用计数器
                mem_meter.update(reset_peak_usage=True)

                # 每隔一定步数，显示进度条
                if data_iter % self.logging_conf.log_freq == 0:
                    progress.display(data_iter)

                # 每隔一定步数，记录进度
                if data_iter % self.logging_conf.log_scalar_frequency == 0:
                    # 日志进度计数器 /  Log progress meters.
                    for progress_meter in progress.meters:
                        self.logger.log(
                            os.path.join("Step_Stats", phase, progress_meter.name),
                            progress_meter.val,
                            self.steps[phase],
                        )

            # 捕捉损失中的 NaN 或 Inf 错误 / Catching NaN/Inf errors in the loss
            except FloatingPointError as e:
                raise e

        # 记录每个 epoch 的预计时间 / Log estimated time for each epoch
        self.est_epoch_time[Phase.TRAIN] = batch_time_meter.avg * iters_per_epoch
        self._log_timers(Phase.TRAIN)
        self._log_sync_data_times(Phase.TRAIN, data_times)

        # 记录损失和保存检查点 / Log losses and save checkpoints
        out_dict = self._log_meters_and_save_best_ckpts([Phase.TRAIN])

        # 将损失值记录到输出字典 / Log loss values to the output dictionary
        for k, v in loss_mts.items():
            out_dict[k] = v.avg
        for k, v in extra_loss_mts.items():
            out_dict[k] = v.avg

        # 更新输出字典，包含训练状态
        out_dict.update(self._get_trainer_state(phase))
        logging.info(f"Losses and meters: {out_dict}")

        # 重置统计计数器 / Reset meters
        self._reset_meters([phase])
        return out_dict

    # 训练步骤内部方法
    def _run_step(
        self,
        batch: BatchedVideoDatapoint,
        phase: str,
        loss_mts: Dict[str, AverageMeter],  # 总损失
        extra_loss_mts: Dict[str, AverageMeter],  # 知识蒸馏软硬损失的详细损失
        raise_on_error: bool = True,
    ):
        """
        Run the forward / backward
        运行前向传播/反向传播
        """

        # it's important to set grads to None, especially with Adam since 0
        # grads will also update a model even if the step doesn't produce gradients
        # /
        # 特别是对于 Adam，设置梯度为 None 是很重要的，因为即使步骤没有产生梯度，梯度也会更新模型
        self.optim.zero_grad(set_to_none=True)

        with torch.cuda.amp.autocast(
            enabled=self.optim_conf.amp.enabled,  # 是否启用 AMP（自动混合精度）。
            dtype=get_amp_type(self.optim_conf.amp.amp_dtype),  # AMP 的精度类型。
        ):
            # 执行前向传播步骤，计算损失。
            loss_dict, batch_size, extra_losses = self._step(
                batch,
                self.teacher_model,
                self.student_model,
                phase,
            )  # 这里loss_dict为total_loss, extra_losses为hard_loss与soft_loss的详细损失

        # 确保只有一个损失值。
        assert len(loss_dict) == 1
        loss_key, loss = loss_dict.popitem()

        # 检查损失值是否为有限值（即不是 NaN 或 Inf）。
        if not math.isfinite(loss.item()):
            error_msg = f"Loss is {loss.item()}, attempting to stop training"
            logging.error(error_msg)
            if raise_on_error:
                raise FloatingPointError(error_msg)
            else:
                return

        # 反向传播计算梯度。
        self.scaler.scale(loss).backward()

        # 更新损失的计量器
        loss_mts[loss_key].update(loss.item(), batch_size)
        for extra_loss_key, extra_loss in extra_losses.items():
            if extra_loss_key not in extra_loss_mts:
                # 如果是额外损失，初始化对应的计量器。
                extra_loss_mts[extra_loss_key] = AverageMeter(
                    extra_loss_key, self.device, ":.2e"
                )
            extra_loss_mts[extra_loss_key].update(extra_loss.item(), batch_size)

    # 运行验证
    def run_val(self):
        if not self.val_dataset:
            return

        dataloader = self.val_dataset.get_loader(epoch=int(self.epoch))
        outs = self.val_epoch(dataloader, phase=Phase.VAL)  # 运行一个验证 epoch

        # 删除 dataloader 并进行垃圾回收
        del dataloader
        gc.collect()

        self.logger.log_dict(outs, self.epoch)  # 仅在 rank 0 上记录 / Logged only on rank 0

        if self.distributed_rank == 0:  # 确保只在主进程执行
            with g_pathmgr.open(
                os.path.join(self.logging_conf.log_dir, "val_stats.json"),
                "a",
            ) as f:
                f.write(json.dumps(outs) + "\n")

    # 验证一个 epoch
    def val_epoch(self, val_loader, phase):
        # 定义时间和内存统计工具
        batch_time = AverageMeter("Batch Time", self.device, ":.2f")
        data_time = AverageMeter("Data Time", self.device, ":.2f")
        mem = MemMeter("Mem (GB)", self.device, ":.2f")

        iters_per_epoch = len(val_loader)

        # 当前阶段和模型
        curr_phases = [phase]
        curr_models = [self.student_model]

        # 定义损失名称  TODO:这里是只定义hard_loss还是应该也一起定义soft_loss？
        loss_names = []
        for p in curr_phases:
            for key in self.hard_loss.keys():
                loss_names.append(f"Losses/{p}_{key}_hard_loss")
            for key in self.soft_loss.keys():
                loss_names.append(f"Losses/{p}_{key}_soft_loss")

        # 初始化损失计数器
        loss_mts = OrderedDict(
            [(name, AverageMeter(name, self.device, ":.2e")) for name in loss_names]
        )
        extra_loss_mts = {}

        # 设置模型为评估模式
        for model in curr_models:
            model.eval()
            # 如果模型支持验证阶段的回调，则调用
            if hasattr(self.unwrap_ddp_if_wrapped(model), "on_validation_epoch_start"):
                self.unwrap_ddp_if_wrapped(model).on_validation_epoch_start()

        # 创建进度条显示工具
        progress = ProgressMeter(
            iters_per_epoch,
            [batch_time, data_time, mem, self.time_elapsed_meter, *loss_mts.values()],
            self._get_meters(curr_phases),
            prefix="Val Epoch: [{}]".format(self.epoch),
        )

        end = time.time()

        for data_iter, batch in enumerate(val_loader):

            # 测量数据加载时间 / measure data loading time
            data_time.update(time.time() - end)

            batch = batch.to(self.device, non_blocking=True)

            # 计算输出 / compute output
            with torch.no_grad():
                with torch.cuda.amp.autocast(
                        enabled=(self.optim_conf.amp.enabled if self.optim_conf else False),
                        dtype=(
                                get_amp_type(self.optim_conf.amp.amp_dtype)
                                if self.optim_conf
                                else None
                        ),
                ):
                    for phase, model in zip(curr_phases, curr_models):
                        # 调用 `_step` 函数计算损失和额外信息
                        loss_dict, batch_size, extra_losses = self._step(
                            batch,
                            model,
                            phase,
                        )

                        assert len(loss_dict) == 1  # 验证损失字典格式
                        loss_key, loss = loss_dict.popitem()

                        # 更新损失计数器
                        loss_mts[loss_key].update(loss.item(), batch_size)

                        # 更新额外损失计数器
                        for k, v in extra_losses.items():
                            if k not in extra_loss_mts:
                                extra_loss_mts[k] = AverageMeter(k, self.device, ":.2e")
                            extra_loss_mts[k].update(v.item(), batch_size)

            # 测量经过的时间 / measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            # 更新累计运行时间
            self.time_elapsed_meter.update(
                time.time() - self.start_time + self.ckpt_time_elapsed
            )

            # 如果可用，记录 GPU 内存使用情况
            if torch.cuda.is_available():
                mem.update(reset_peak_usage=True)

            # 每隔指定步数显示进度
            if data_iter % self.logging_conf.log_freq == 0:
                progress.display(data_iter)

            if data_iter % self.logging_conf.log_scalar_frequency == 0:
                # 记录进度计数器 / Log progress meters.
                for progress_meter in progress.meters:
                    self.logger.log(
                        os.path.join("Step_Stats", phase, progress_meter.name),
                        progress_meter.val,
                        self.steps[Phase.VAL],
                    )
            # 每隔 10 个步骤进行同步
            if data_iter % 10 == 0:
                dist.barrier()

        self.est_epoch_time[phase] = batch_time.avg * iters_per_epoch  # 记录每个 epoch 的平均时间
        self._log_timers(phase)  # 记录计时器

        # 如果模型支持验证结束阶段的回调，则调用
        for model in curr_models:
            if hasattr(self.unwrap_ddp_if_wrapped(model), "on_validation_epoch_end"):
                self.unwrap_ddp_if_wrapped(model).on_validation_epoch_end()

        # 记录指标和保存最佳模型检查点
        out_dict = self._log_meters_and_save_best_ckpts(curr_phases)

        # 将损失的平均值加入输出字典
        for k, v in loss_mts.items():
            out_dict[k] = v.avg
        for k, v in extra_loss_mts.items():
            out_dict[k] = v.avg

        # 更新阶段状态
        for phase in curr_phases:
            out_dict.update(self._get_trainer_state(phase))

        self._reset_meters(curr_phases)  # 重置计数器
        logging.info(f"Meters: {out_dict}")
        return out_dict

    # 上：训练与验证
    # -----------------------------------------------------------------------------------------
    # 下：ckpt加载与恢复

    # 检查传入的 model 是否被 DistributedDataParallel (DDP) 包装过
    def unwrap_ddp_if_wrapped(self, model):
        if isinstance(model, torch.nn.parallel.DistributedDataParallel):
            return model.module  # DDP包装后的模型会多一个module层级
        return model

    # 保存检查点
    def save_checkpoint(self, epoch, checkpoint_names=None):

        checkpoint_folder = self.checkpoint_conf.save_dir  # 获取检查点文件夹的路径
        makedir(checkpoint_folder)  # 创建检查点保存文件夹（如果不存在的话）

        if checkpoint_names is None:
            checkpoint_names = ["checkpoint"]  # 如果未指定检查点名称，则使用默认名称 "checkpoint"

            # 如果设置了保存频率并且当前 epoch 满足保存条件，或者当前 epoch 在保存列表中，则添加带有当前 epoch 的名称
            if (
                    self.checkpoint_conf.save_freq > 0
                    and (int(epoch) % self.checkpoint_conf.save_freq == 0)
            ) or int(epoch) in self.checkpoint_conf.save_list:
                checkpoint_names.append(f"checkpoint_{int(epoch)}")

        # 为每个检查点名称生成完整的文件路径
        checkpoint_paths = []
        for ckpt_name in checkpoint_names:
            checkpoint_paths.append(os.path.join(checkpoint_folder, f"{ckpt_name}.pt"))

        # 获取模型的 state_dict（模型的权重参数）
        state_dict = self.unwrap_ddp_if_wrapped(self.student_model).state_dict()
        state_dict = exclude_params_matching_unix_pattern(
            patterns=self.checkpoint_conf.skip_saving_parameters, state_dict=state_dict
        )

        # 创建保存的检查点字典，包含模型参数、优化器状态、当前 epoch、损失函数状态、训练步数等信息
        checkpoint = {
            "student_model": state_dict,  # ckpt只应存student_model,不需要存固定权重的teacher_model
            "optimizer": self.optim.optimizer.state_dict(),
            "epoch": epoch,
            "hard_loss": self.hard_loss.state_dict(),
            "soft_loss": self.soft_loss.state_dict(),
            "steps": self.steps,
            "time_elapsed": self.time_elapsed_meter.val,
            "best_meter_values": self.best_meter_values,
        }

        # 如果启用了混合精度训练（AMP），将 scaler 的状态添加到检查点字典中
        if self.optim_conf.amp.enabled:
            checkpoint["scaler"] = self.scaler.state_dict()

        # DDP checkpoints are only saved on rank 0 (all workers are identical)
        # /
        # DDP 检查点只会在 rank 0 上保存（所有工作进程是相同的）
        if self.distributed_rank != 0:
            return

        # 遍历所有的检查点路径，保存检查点
        for checkpoint_path in checkpoint_paths:
            self._save_checkpoint(checkpoint, checkpoint_path)

    # 保存检查点的内部函数
    def _save_checkpoint(self, checkpoint, checkpoint_path):
        """
        Save a checkpoint while guarding against the job being killed in the middle
        of checkpoint saving (which corrupts the checkpoint file and ruins the
        entire training since usually only the last checkpoint is kept per run).

        We first save the new checkpoint to a temp file (with a '.tmp' suffix), and
        and move it to overwrite the old checkpoint_path.

        在保存检查点时，防止作业在保存过程中被终止
        （这会损坏检查点文件并破坏整个训练，因为通常每次运行只保留最后一个检查点）。
        我们首先将新的检查点保存到一个临时文件（带有 .tmp 后缀），然后将其移动并覆盖旧的检查点路径。
        """
        checkpoint_path_tmp = f"{checkpoint_path}.tmp"
        with g_pathmgr.open(checkpoint_path_tmp, "wb") as f:
            torch.save(checkpoint, f)

        # after torch.save is completed, replace the old checkpoint with the new one
        # /
        # 在 torch.save 完成后，替换旧的检查点为新的检查点。
        if g_pathmgr.exists(checkpoint_path):
            # remove the old checkpoint_path file first (otherwise g_pathmgr.mv fails)
            # /
            # 首先移除旧的 checkpoint_path 文件（否则 g_pathmgr.mv 会失败）。
            g_pathmgr.rm(checkpoint_path)
        success = g_pathmgr.mv(checkpoint_path_tmp, checkpoint_path)
        assert success

    # 保存检查点
    def load_checkpoint(self):
        # 获取恢复的检查点路径
        ckpt_path = get_resume_checkpoint(self.checkpoint_conf.save_dir)

        # 如果没有找到恢复的检查点（ckpt_path 为 None）
        if ckpt_path is None:
            self._init_model_state()    # 初始化模型状态（如果没有恢复的检查点）
        else:
            # 如果配置要求在预期外中断后初始化模型（initialize_after_preemption）
            if self.checkpoint_conf.initialize_after_preemption:
                self._call_model_initializer()  # 调用模型初始化器（重新初始化模型）

            # 加载恢复的检查点
            self._load_resuming_checkpoint(ckpt_path)

    # 初始化模型状态
    def _init_model_state(self):
        # Checking that parameters that won't be saved are indeed frozen
        # We do this check here before even saving the model to catch errors
        # are early as possible and not at the end of the first epoch
        # /
        # 检查那些不会被保存的参数是否确实被冻结。
        # 我们在这里进行检查，在保存模型之前，以尽早捕捉错误，而不是等到第一轮训练结束时再发现问题。
        assert_skipped_parameters_are_frozen(
            patterns=self.checkpoint_conf.skip_saving_parameters,
            model=self.student_model,
        )

        # Checking that parameters that won't be saved are initialized from
        # within the model definition, unless `initialize_after_preemption`
        # is explicitly set to `True`. If not, this is a bug, and after
        # preemption, the `skip_saving_parameters` will have random values
        # /
        # 检查那些不会被保存的参数是否在模型定义中初始化，除非显式将 initialize_after_preemption 设置为 True。
        # 如果没有这样做，这是一个 bug，且在中断后，skip_saving_parameters 将具有随机值。
        allow_init_skip_parameters = self.checkpoint_conf.initialize_after_preemption
        with with_check_parameter_frozen(
            patterns=self.checkpoint_conf.skip_saving_parameters,
            model=self.student_model,
            disabled=allow_init_skip_parameters,
        ):
            self._call_model_initializer()

    # 调用模型初始化器
    def _call_model_initializer(self):
        # 实例化模型权重初始化器
        model_weight_initializer = instantiate(
            self.checkpoint_conf.model_weight_initializer
        )
        # 如果初始化器不为 None，则加载预训练的模型权重
        if model_weight_initializer is not None:
            logging.info(
                f"Loading pretrained checkpoint from {self.checkpoint_conf.model_weight_initializer}"
            )
            self.student_model = model_weight_initializer(model=self.student_model)

    # 加载恢复的检查点
    def _load_resuming_checkpoint(self, ckpt_path: str):
        logging.info(f"Resuming training from {ckpt_path}")

        # 打开并加载检查点文件
        with g_pathmgr.open(ckpt_path, "rb") as f:
            checkpoint = torch.load(f, map_location="cpu", weights_only=True)

        # 加载学生模型的状态字典（模型的参数）
        load_state_dict_into_model(
            model=self.student_model,
            state_dict=checkpoint["student_model"],  # 从ckpt中获取
            ignore_missing_keys=self.checkpoint_conf.skip_saving_parameters,
        )
        # 加载固定权重拿到教师模型  TODO:不改变teacher_model权重的话这里还需要load_state_dict_into_model吗？
        load_state_dict_into_model(
            model=self.teacher_model,
            state_dict=self.teacher_model.state_dict(),
        )

        # 加载优化器、损失函数和其他训练状态
        self.optim.optimizer.load_state_dict(checkpoint["optimizer"])
        self.hard_loss.load_state_dict(checkpoint["hard_loss"], strict=True)
        self.soft_loss.load_state_dict(checkpoint["soft_loss"], strict=True)
        self.epoch = checkpoint["epoch"]
        self.steps = checkpoint["steps"]
        self.ckpt_time_elapsed = checkpoint.get("time_elapsed")

        # 如果启用了混合精度训练（AMP），恢复 scaler 状态
        if self.optim_conf.amp.enabled and "scaler" in checkpoint:
            self.scaler.load_state_dict(checkpoint["scaler"])

        # 恢复最佳指标的值（如果有的话）
        self.best_meter_values = checkpoint.get("best_meter_values", {})

        # 如果检查点中包含训练数据集的状态且训练数据集存在，恢复训练数据集的状态
        if "train_dataset" in checkpoint and self.train_dataset is not None:
            self.train_dataset.load_checkpoint_state(checkpoint["train_dataset"])

    # 上：ckpt加载与恢复
    # ------------------------------------------------------------------------------------------
    # 下：分布式与device工具类

    # 设置 DDP 分布式训练
    def _setup_ddp_distributed_training(self, distributed_conf, accelerator):
        # 确保模型是 nn.Module 类型
        assert isinstance(self.student_model, torch.nn.Module)

        # 使用 DistributedDataParallel 包装模型进行分布式训练
        self.student_model = nn.parallel.DistributedDataParallel(
            self.student_model,
            device_ids=[self.local_rank] if accelerator == "cuda" else [],
            find_unused_parameters=distributed_conf.find_unused_parameters,  # 是否查找未使用的参数
        )

        # 如果配置中指定了通信数据类型
        if distributed_conf.comms_dtype is not None:  # noqa
            from torch.distributed.algorithms import ddp_comm_hooks

            # 根据配置的数据类型，获取相应的自动混合精度类型
            amp_type = get_amp_type(distributed_conf.comms_dtype)

            # 如果是 bfloat16 精度，使用对应的通信钩子
            if amp_type == torch.bfloat16:
                hook = ddp_comm_hooks.default_hooks.bf16_compress_hook  # bfloat16 梯度压缩钩子
                logging.info("Enabling bfloat16 grad communication")
            else:
                hook = ddp_comm_hooks.default_hooks.fp16_compress_hook  # fp16 梯度压缩钩子
                logging.info("Enabling fp16 grad communication")

            process_group = None  # 此处未指定通信组
            # 注册通信钩子，以优化梯度通信
            self.student_model.register_comm_hook(process_group, hook)

    # 推断分布式后端，如果accelerator是cuda则为nccl，否则为gloo
    def _infer_distributed_backend_if_none(self, distributed_conf, accelerator):
        if distributed_conf.backend is None:
            distributed_conf.backend = "nccl" if accelerator == "cuda" else "gloo"

    # 设置device
    def _setup_device(self, accelerator):
        # 获取本地设备的 rank 和分布式设备的 rank
        self.local_rank, self.distributed_rank = get_machine_local_and_dist_rank()
        if accelerator == "cuda":
            # 如果选择了 "cuda" 作为加速器（即使用 GPU）
            self.device = torch.device("cuda", self.local_rank)  # 设置当前设备为本地 GPU
            torch.cuda.set_device(self.local_rank)  # 设置当前进程使用的 GPU 为本地 GPU
        elif accelerator == "cpu":
            self.device = torch.device("cpu")
        else:
            raise ValueError(f"Unsupported accelerator: {accelerator}")

    # 设置torch分布式和后端
    def _setup_torch_dist_and_backend(self, cuda_conf, distributed_conf) -> None:
        # 检查是否有可用的 CUDA 设备（即是否存在 GPU）
        if torch.cuda.is_available():
            # 设置 cuDNN 的确定性算法模式
            torch.backends.cudnn.deterministic = cuda_conf.cudnn_deterministic
            # 设置 cuDNN 的自动优化模式
            torch.backends.cudnn.benchmark = cuda_conf.cudnn_benchmark
            # 设置是否允许矩阵乘法使用 TF32 精度
            torch.backends.cuda.matmul.allow_tf32 = (
                cuda_conf.matmul_allow_tf32
                if cuda_conf.matmul_allow_tf32 is not None
                else cuda_conf.allow_tf32
            )
            # 设置是否允许 cudnn 使用 TF32
            torch.backends.cudnn.allow_tf32 = (
                cuda_conf.cudnn_allow_tf32
                if cuda_conf.cudnn_allow_tf32 is not None
                else cuda_conf.allow_tf32
            )
        # 设置分布式训练的后端，并返回当前进程的 rank
        self.rank = setup_distributed_backend(
            distributed_conf.backend, distributed_conf.timeout_mins
        )

    # 上：分布式后端与device工具类
    # ------------------------------------------------------------------------------------------
    # 下：其它trainer工具类

    # 检查验证数据集中的键是否符合预期
    def _check_val_key_match(self, val_keys, phase):

        if val_keys is not None:
            # Check if there are any duplicates
            # 检查是否有任何重复项
            assert len(val_keys) == len(
                set(val_keys)
            ), f"验证集中有重复的keys / Duplicate keys in val datasets, keys: {val_keys}"

            # Check that the keys match the meter keys
            # 检查键是否与计数器的键匹配
            if self.meters_conf is not None and phase in self.meters_conf:  # 如果配置了度量器并且验证阶段存在度量器配置
                assert set(val_keys) == set(self.meters_conf[phase].keys()), (
                    f"Keys in val datasets do not match the keys in meters."
                    f"\nMissing in meters: {set(val_keys) - set(self.meters_conf[phase].keys())}"
                    f"\nMissing in val datasets: {set(self.meters_conf[phase].keys()) - set(val_keys)}"
                )

            if self.loss_conf is not None:  # 如果损失函数配置存在
                loss_keys = set(self.loss_conf.keys()) - set(["all"])  # 获取损失函数的键，去掉 "all" 键
                assert all([k in loss_keys for k in val_keys]), (
                    f"Keys in val datasets do not match the keys in losses."
                    f"\nMissing in losses: {set(val_keys) - loss_keys}"
                    f"\nMissing in val datasets: {loss_keys - set(val_keys)}"
                )

    # 将模型转移到设备上
    def _move_to_device(self):
        logging.info(
            f"将教师模型与学生模型移动到设备 {self.device} 和本地 rank {self.local_rank}."
            f" / "
            f"Moving components to device {self.device} and local rank {self.local_rank}."
        )

        self.teacher_model.to(self.device)
        self.student_model.to(self.device)

        logging.info(
            f"完成将教师模型与学生模型移动到设备 {self.device} 和本地 rank {self.local_rank}. "
            f" / "
            f"Done moving components to device {self.device} and local rank {self.local_rank}."
        )

    # 初始化经过时间和预计时间（ETA）的计数器
    def _setup_timers(self):
        """
        Initializes counters for elapsed time and eta.
        初始化经过时间和预计时间（ETA）的计数器。
        """
        self.start_time = time.time()  # 记录开始时间
        self.ckpt_time_elapsed = 0  # 初始化检查点时间
        self.est_epoch_time = dict.fromkeys([Phase.TRAIN, Phase.VAL], 0)  # 初始化训练和验证阶段的预计时间

    # 获取模型训练过程中的所有指标
    def _get_meters(self, phase_filters=None):
        if self.meters is None:
            return {}

        meters = {}
        # 遍历所有阶段（phase）和对应的指标（phase_meters）
        for phase, phase_meters in self.meters.items():
            # 如果 phase_filters 不为 None 且当前 phase 不在 phase_filters 中，则跳过当前 phase
            if phase_filters is not None and phase not in phase_filters:
                continue

            # 遍历每个 phase 下的各个指标（key 和 key_meters）
            for key, key_meters in phase_meters.items():
                # 如果 key_meters 为 None，则跳过
                if key_meters is None:
                    continue

                # 遍历每个 key_meters 下的每个具体的计量器（name 和 meter）
                for name, meter in key_meters.items():
                    # 将指标信息添加到 meters 字典中，键名格式为 "phase_key/name"
                    meters[f"{phase}_{key}/{name}"] = meter
        return meters

    # 上：其他trainer工具类
# -----------------------------------------------------------------------------------------------
# 下：打印模型总览与统计

PARAMETER_NUM_UNITS = [" ", "K", "M", "B", "T"]

def print_model_summary(model: torch.nn.Module, log_dir: str = ""):
    """
    Prints the model and the number of parameters in the model.
    # Multiple packages provide this info in a nice table format
    # However, they need us to provide an `input` (as they also write down the output sizes)
    # Our models are complex, and a single input is restrictive.
    # https://github.com/sksq96/pytorch-summary
    # https://github.com/nmhkahn/torchsummaryX

    打印模型及模型中的参数数量。
    # 多个库提供了以表格格式显示这些信息
    # 但是，它们需要我们提供一个`input`（因为它们还需要写出输出尺寸）
    # 我们的模型很复杂，单一的输入限制了这种方式。
    # 参考：https://github.com/sksq96/pytorch-summary
    # 参考：https://github.com/nmhkahn/torchsummaryX
    """
    if get_rank() != 0:
        return
    param_kwargs = {}
    trainable_parameters = sum(
        p.numel() for p in model.parameters(**param_kwargs) if p.requires_grad
    )
    total_parameters = sum(p.numel() for p in model.parameters(**param_kwargs))
    non_trainable_parameters = total_parameters - trainable_parameters
    logging.info("==" * 10)
    logging.info(f"Summary for model {type(model)}")
    logging.info(f"Model is {model}")
    logging.info(f"\tTotal parameters {get_human_readable_count(total_parameters)}")
    logging.info(
        f"\tTrainable parameters {get_human_readable_count(trainable_parameters)}"
    )
    logging.info(
        f"\tNon-Trainable parameters {get_human_readable_count(non_trainable_parameters)}"
    )
    logging.info("==" * 10)

    if log_dir:
        output_fpath = os.path.join(log_dir, "model.txt")
        with g_pathmgr.open(output_fpath, "w") as f:
            print(model, file=f)

def get_human_readable_count(number: int) -> str:
    """
    获取人类可读的计数

    Abbreviates an integer number with K, M, B, T for thousands, millions,
    billions and trillions, respectively.
    将整数简写为 K、M、B、T，分别表示千、百万、十亿和万亿。
    Examples:
        >>> get_human_readable_count(123)
        '123  '
        >>> get_human_readable_count(1234)  # (one thousand)
        '1.2 K'
        >>> get_human_readable_count(2e6)   # (two million)
        '2.0 M'
        >>> get_human_readable_count(3e9)   # (three billion)
        '3.0 B'
        >>> get_human_readable_count(4e14)  # (four hundred trillion)
        '400 T'
        >>> get_human_readable_count(5e15)  # (more than trillion)
        '5,000 T'
    Args:
        number: a positive integer number
        number：一个正整数
    Return:
        A string formatted according to the pattern described above.
        按照上述模式格式化的字符串
    """
    assert number >= 0
    labels = PARAMETER_NUM_UNITS
    num_digits = int(np.floor(np.log10(number)) + 1 if number > 0 else 1)
    num_groups = int(np.ceil(num_digits / 3))
    num_groups = min(num_groups, len(labels))  # 不要将数字简写到万亿以上 don't abbreviate beyond trillions
    shift = -3 * (num_groups - 1)
    number = number * (10**shift)
    index = num_groups - 1
    if index < 1 or number >= 100:
        return f"{int(number):,d} {labels[index]}"
    else:
        return f"{number:,.1f} {labels[index]}"

# 上：打印模型总览与统计
# -----------------------------------------------------------------------------------------------