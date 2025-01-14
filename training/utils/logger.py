# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# Code borrowed from TLC - https://www.internalfb.com/code/fbsource/fbcode/pytorch/tlc/torchtlc/loggers/tensorboard.py
import atexit
import functools
import logging
import sys
import uuid
from typing import Any, Dict, Optional, Union

from hydra.utils import instantiate

from iopath.common.file_io import g_pathmgr
from numpy import ndarray
from torch import Tensor
from torch.utils.tensorboard import SummaryWriter

from training.utils.train_utils import get_machine_local_and_dist_rank, makedir

Scalar = Union[Tensor, ndarray, int, float]

# 创建一个 TensorBoard 日志记录器。
def make_tensorboard_logger(log_dir: str, **writer_kwargs: Any):
    makedir(log_dir)  # 确保日志目录存在
    summary_writer_method = SummaryWriter
    return TensorBoardLogger(
        path=log_dir, summary_writer_method=summary_writer_method, **writer_kwargs
    )


class TensorBoardWriterWrapper:
    """
    A wrapper around a SummaryWriter object.
    TensorBoard SummaryWriter 的包装类。
    """

    def __init__(
        self,
        path: str,
        *args: Any,
        filename_suffix: str = None,
        summary_writer_method: Any = SummaryWriter,
        **kwargs: Any,
    ) -> None:
        """Create a new TensorBoard logger.
        On construction, the logger creates a new events file that logs
        will be written to.  If the environment variable `RANK` is defined,
        logger will only log if RANK = 0.
        /
        在构造时，记录器会创建一个新的事件文件，日志会写入该文件。如果环境变量 `RANK` 被定义，日志仅在 `RANK = 0` 时记录。

        NOTE: If using the logger with distributed training:
        - This logger can call collective operations
        - Logs will be written on rank 0 only
        - Logger must be constructed synchronously *after* initializing distributed process group.
        /
        注意：如果在分布式训练中使用该记录器：
        - 该记录器可以执行集合操作
        - 日志仅会写入 rank 为 0 的进程
        - 记录器必须在初始化分布式进程组后同步构造

        Args:
            path (str): path to write logs to
            *args, **kwargs: Extra arguments to pass to SummaryWriter
        /
        参数：
            path (str): 日志保存的路径
            *args, **kwargs: 传递给 SummaryWriter 的额外参数
        """
        self._writer: Optional[SummaryWriter] = None
        _, self._rank = get_machine_local_and_dist_rank()  # 获取当前进程的 rank
        self._path: str = path

        # 如果是 rank 0，则创建 SummaryWriter
        if self._rank == 0:
            logging.info(
                f"TensorBoard SummaryWriter instantiated. Files will be stored in: {path}"
            )
            self._writer = summary_writer_method(
                log_dir=path,
                *args,
                filename_suffix=filename_suffix or str(uuid.uuid4()),  # 默认使用 UUID 作为文件名后缀
                **kwargs,
            )
        else:
            logging.debug(
                f"Not logging meters on this host because env RANK: {self._rank} != 0"
            )

        # 注册退出时自动关闭日志文件
        atexit.register(self.close)

    @property
    def writer(self) -> Optional[SummaryWriter]:
        return self._writer

    @property
    def path(self) -> str:
        return self._path

    def flush(self) -> None:
        """刷新日志，写入磁盘 / Writes pending logs to disk."""

        if not self._writer:
            return

        self._writer.flush()

    def close(self) -> None:
        """
        Close writer, flushing pending logs to disk.
        Logs cannot be written after `close` is called.
        关闭记录器并刷新所有待处理的日志到磁盘。关闭后无法再写入日志。
        """

        if not self._writer:
            return

        self._writer.close()
        self._writer = None


class TensorBoardLogger(TensorBoardWriterWrapper):
    """
    A simple logger for TensorBoard.
    一个简单的 TensorBoard 日志记录器，用于记录标量数据。
    """

    def log_dict(self, payload: Dict[str, Scalar], step: int) -> None:
        """
        Add multiple scalar values to TensorBoard.
        向 TensorBoard 添加多个标量值。

        Args:
            payload (dict): dictionary of tag name and scalar value
            step (int, Optional): step value to record
        /
        参数：
            payload (dict): 标签名与标量值的字典
            step (int, optional): 记录的步骤值
        """
        if not self._writer:
            return
        for k, v in payload.items():
            self.log(k, v, step)

    def log(self, name: str, data: Scalar, step: int) -> None:
        """
        Add scalar data to TensorBoard.
        向 TensorBoard 添加一个标量数据。

        Args:
            name (string): tag name used to group scalars
            data (float/int/Tensor): scalar data to log
            step (int, optional): step value to record
        /
        参数：
            name (str): 标签名，用于组织标量数据
            data (float/int/Tensor): 要记录的标量数据
            step (int, optional): 记录的步骤值
        """
        if not self._writer:
            return
        self._writer.add_scalar(name, data, global_step=step, new_style=True)

    def log_hparams(
        self, hparams: Dict[str, Scalar], meters: Dict[str, Scalar]
    ) -> None:
        """
        Add hyperparameter data to TensorBoard.
        向 TensorBoard 添加超参数数据。

        Args:
            hparams (dict): dictionary of hyperparameter names and corresponding values
            meters (dict): dictionary of name of meter and corersponding values
        /
        参数：
            hparams (dict): 超参数名称与对应值的字典
            meters (dict): 记录的度量值字典
        """
        if not self._writer:
            return
        self._writer.add_hparams(hparams, meters)


class Logger:
    """
    A logger class that can interface with multiple loggers. It now supports tensorboard only for simplicity, but you can extend it with your own logger.
    一个通用的日志记录器类，可以与多个日志记录器接口对接。当前仅支持 TensorBoard，但可以根据需要扩展其他日志记录器。
    """

    def __init__(self, logging_conf):
        # allow turning off TensorBoard with "should_log: false" in config
        # 根据配置决定是否启用 TensorBoard
        tb_config = logging_conf.tensorboard_writer
        tb_should_log = tb_config and tb_config.pop("should_log", True)
        self.tb_logger = instantiate(tb_config) if tb_should_log else None

    def log_dict(self, payload: Dict[str, Scalar], step: int) -> None:
        if self.tb_logger:
            self.tb_logger.log_dict(payload, step)

    def log(self, name: str, data: Scalar, step: int) -> None:
        if self.tb_logger:
            self.tb_logger.log(name, data, step)

    def log_hparams(
        self, hparams: Dict[str, Scalar], meters: Dict[str, Scalar]
    ) -> None:
        if self.tb_logger:
            self.tb_logger.log_hparams(hparams, meters)


# cache the opened file object, so that different calls to `setup_logger`
# with the same file name can safely write to the same file.
# 缓存打开的文件对象，以便相同文件名的多个调用可以安全地写入相同文件。
@functools.lru_cache(maxsize=None)
def _cached_log_stream(filename):
    # we tune the buffering value so that the logs are updated frequently.
    # 我们调整缓冲区大小，以便日志能够更频繁地更新。
    log_buffer_kb = 10 * 1024  # 10KB
    io = g_pathmgr.open(filename, mode="a", buffering=log_buffer_kb)
    # 注册退出时自动关闭文件
    atexit.register(io.close)
    return io


def setup_logging(
    name,
    output_dir=None,
    rank=0,
    log_level_primary="INFO",
    log_level_secondary="ERROR",
):
    """
    Setup various logging streams: stdout and file handlers.
    For file handlers, we only setup for the master gpu.
    /
    设置各种日志流：标准输出（stdout）和文件处理器。
    对于文件处理器，我们只为主 GPU 设置日志。
    """
    # get the filename if we want to log to the file as well
    # 如果需要将日志输出到文件，获取文件名
    log_filename = None
    if output_dir:
        makedir(output_dir)
        if rank == 0:
            log_filename = f"{output_dir}/log.txt"

    # 获取日志记录器
    logger = logging.getLogger(name)
    logger.setLevel(log_level_primary)

    # 创建日志格式化器 / create formatter
    FORMAT = "%(levelname)s %(asctime)s %(filename)s:%(lineno)4d: %(message)s"
    formatter = logging.Formatter(FORMAT)

    # 清除任何现有的处理器 / Cleanup any existing handlers
    for h in logger.handlers:
        logger.removeHandler(h)
    logger.root.handlers = []

    # 设置控制台处理器 / setup the console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    if rank == 0:
        console_handler.setLevel(log_level_primary)
    else:
        console_handler.setLevel(log_level_secondary)

    # 如果用户希望，我们也将日志输出到文件 / we log to file as well if user wants
    if log_filename and rank == 0:
        file_handler = logging.StreamHandler(_cached_log_stream(log_filename))
        file_handler.setLevel(log_level_primary)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    logging.root = logger


def shutdown_logging():
    """
    After training is done, we ensure to shut down all the logger streams.
    训练完成后，确保关闭所有日志流。
    """
    logging.info("Shutting down loggers...")
    handlers = logging.root.handlers
    for handler in handlers:
        handler.close()
