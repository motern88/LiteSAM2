# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import datetime
import functools
import io
import logging
import os
import random
import tempfile
import time
from typing import Any, Callable, List, Tuple

import torch
import torch.autograd as autograd
import torch.distributed as dist


# 默认使用 GPU 0 / Default to GPU 0
_cuda_device_index: int = 0

# Setting _cuda_device_index to -1 internally implies that we should use CPU
# 如果内部将 _cuda_device_index 设置为 -1，则表示应使用 CPU
_CPU_DEVICE_INDEX = -1
_PRIMARY_RANK = 0


@functools.lru_cache()
def _get_global_gloo_group():
    """
    Return a process group based on gloo backend, containing all the ranks
    The result is cached.

    返回一个基于 gloo 后端的进程组，包含所有 rank。结果会被缓存。
    """

    if dist.get_backend() == "nccl":
        # Increase timeout from 1800 sec to 43200 sec (12 hr) to avoid some processes
        # being much slower than others causing a timeout (which can happen in relation
        # or LVIS class mAP evaluation).
        # 将超时时间从 1800 秒增加到 43200 秒（12 小时），以避免某些进程比其他进程慢得多
        # 导致超时（这可能发生在评估如 LVIS 类 mAP 时）
        timeout = 43200
        return dist.new_group(
            backend="gloo",
            timeout=datetime.timedelta(seconds=timeout),
        )

    return dist.group.WORLD


def is_main_process():
    """
    Return true if the current process is the main one
    返回当前进程是否为主进程
    """
    return get_rank() == 0


def all_gather_via_filesys(data, filesys_save_dir=None, gather_to_rank_0_only=False):
    """
    Run all_gather on arbitrary picklable data (not necessarily tensors), similar to
    `all_gather` above, but using filesystem instead of collective ops.

    使用文件系统进行 all_gather 操作，用于任意可序列化的数据（不一定是张量）。
    类似于上面的 `all_gather`，但通过文件系统而非集体通信操作实现。

    If gather_to_rank_0_only is True, only rank 0 will load the gathered object list
    (and other ranks will have an empty list).

    如果 `gather_to_rank_0_only` 为 True，仅 rank 0 会加载收集的数据列表（其他 rank 的列表将为空）。
    """
    world_size = get_world_size()
    if world_size == 1:
        return [data]

    print("gathering via files")
    cpu_group = _get_global_gloo_group()

    # if unspecified, we will save to the current python file dir
    # 如果未指定保存目录，则尝试使用当前 Python 文件所在目录
    if filesys_save_dir is not None:
        save_dir = filesys_save_dir
    elif "EXP_DIR" in os.environ:
        save_dir = os.environ["EXP_DIR"]
    else:
        # try the same directory where the code is stored
        # 尝试使用代码所在的目录作为保存目录
        save_dir = filesys_save_dir or os.path.dirname(__file__)
    save_dir = os.path.join(save_dir, "all_gather_via_filesys")
    if is_main_process():
        os.makedirs(save_dir, exist_ok=True)

    # use a timestamp and salt to distinguish different all_gather
    # 使用时间戳和随机数区分不同的 all_gather 操作
    timestamp = int(time.time()) if is_main_process() else 0
    salt = random.randint(0, 2**31 - 1) if is_main_process() else 0
    # broadcast the timestamp and salt across ranks
    # (all-reduce will do the broadcasting since only rank 0 is non-zero)
    # 通过广播将时间戳和随机数传递给所有 rank
    timestamp_and_salt = torch.tensor([timestamp, salt], dtype=torch.long)
    dist.all_reduce(timestamp_and_salt, group=cpu_group)
    timestamp, salt = timestamp_and_salt.tolist()

    # save the data to a file on the disk
    # 将数据保存到磁盘上的文件
    rank_save = get_rank()
    save_data_filename = f"data_to_gather_{timestamp}_{salt}_{rank_save}.pkl"
    save_data_path = os.path.join(save_dir, save_data_filename)
    assert not os.path.exists(save_data_path), f"{save_data_path} already exists"
    torch.save(data, save_data_path)
    dist.barrier(group=cpu_group)

    # read the data from the files
    # 从文件中读取数据
    data_list = []
    if rank_save == 0 or not gather_to_rank_0_only:
        for rank_load in range(world_size):
            load_data_filename = f"data_to_gather_{timestamp}_{salt}_{rank_load}.pkl"
            load_data_path = os.path.join(save_dir, load_data_filename)
            assert os.path.exists(load_data_path), f"cannot read {save_data_path}"
            data_list.append(torch.load(load_data_path))
    dist.barrier(group=cpu_group)

    # delete the saved file
    # 删除保存的文件
    os.remove(save_data_path)
    return data_list


def all_gather(data, force_cpu=False, force_filesys=False, filesys_save_dir=None):
    """
    Run all_gather on arbitrary picklable data (not necessarily tensors)

    使用 all_gather 操作收集任意可序列化的数据（不一定是张量）。

    Args:
        data: any picklable object
            任意可序列化的对象
    Returns:
        list[data]: list of data gathered from each rank
            来自每个 rank 收集到的数据列表
    """

    world_size = get_world_size()
    if world_size == 1:
        return [data]

    if os.getenv("MDETR_FILESYS_REDUCE_RANK_0_ONLY") == "1":
        return all_gather_via_filesys(
            data, filesys_save_dir, gather_to_rank_0_only=True
        )

    if os.getenv("MDETR_FILESYS_REDUCE") == "1" or force_filesys:
        return all_gather_via_filesys(data, filesys_save_dir)

    cpu_group = None
    if os.getenv("MDETR_CPU_REDUCE") == "1" or force_cpu:
        cpu_group = _get_global_gloo_group()

    buffer = io.BytesIO()
    torch.save(data, buffer)
    data_view = buffer.getbuffer()
    device = "cuda" if cpu_group is None else "cpu"
    tensor = torch.ByteTensor(data_view).to(device)

    # 获取每个 rank 的 Tensor 大小 / obtain Tensor size of each rank
    local_size = torch.tensor([tensor.numel()], device=device, dtype=torch.long)
    size_list = [
        torch.tensor([0], device=device, dtype=torch.long) for _ in range(world_size)
    ]
    if cpu_group is None:
        dist.all_gather(size_list, local_size)
    else:
        print("gathering on cpu")
        dist.all_gather(size_list, local_size, group=cpu_group)
    size_list = [int(size.item()) for size in size_list]
    max_size = max(size_list)
    assert isinstance(local_size.item(), int)
    local_size = int(local_size.item())

    # receiving Tensor from all ranks
    # we pad the tensor because torch all_gather does not support
    # gathering tensors of different shapes
    # 从所有 rank 接收 Tensor
    # 因为 torch 的 all_gather 不支持不同形状的张量，需对张量进行填充
    tensor_list = []
    for _ in size_list:
        tensor_list.append(torch.empty((max_size,), dtype=torch.uint8, device=device))
    if local_size != max_size:
        padding = torch.empty(
            size=(max_size - local_size,), dtype=torch.uint8, device=device
        )
        tensor = torch.cat((tensor, padding), dim=0)
    if cpu_group is None:
        dist.all_gather(tensor_list, tensor)
    else:
        dist.all_gather(tensor_list, tensor, group=cpu_group)

    data_list = []
    for size, tensor in zip(size_list, tensor_list):
        tensor = torch.split(tensor, [size, max_size - size], dim=0)[0]
        buffer = io.BytesIO(tensor.cpu().numpy())
        obj = torch.load(buffer)
        data_list.append(obj)

    return data_list


def convert_to_distributed_tensor(tensor: torch.Tensor) -> Tuple[torch.Tensor, str]:
    """
    For some backends, such as NCCL, communication only works if the
    tensor is on the GPU. This helper function converts to the correct
    device and returns the tensor + original device.

    对于某些后端（如 NCCL），通信仅在张量位于 GPU 上时可用。
    此辅助函数将张量转换到正确的设备并返回转换后的张量以及原始设备信息。
    """
    orig_device = "cpu" if not tensor.is_cuda else "gpu"  # 记录原始设备类型
    if (
        torch.distributed.is_available()
        and torch.distributed.get_backend() == torch.distributed.Backend.NCCL
        and not tensor.is_cuda
    ):
        tensor = tensor.cuda()  # 如果 NCCL 后端可用且张量不在 GPU 上，则将其转移到 GPU
    return (tensor, orig_device)


def convert_to_normal_tensor(tensor: torch.Tensor, orig_device: str) -> torch.Tensor:
    """
    For some backends, such as NCCL, communication only works if the
    tensor is on the GPU. This converts the tensor back to original device.

    对于某些后端（如 NCCL），通信仅在张量位于 GPU 上时可用。
    此函数将张量转换回原始设备。
    """
    if tensor.is_cuda and orig_device == "cpu":
        tensor = tensor.cpu()  # 如果原始设备是 CPU，且张量当前在 GPU 上，则转移回 CPU
    return tensor


# 检查是否正在运行分布式训练。
def is_distributed_training_run() -> bool:
    return (
        torch.distributed.is_available()
        and torch.distributed.is_initialized()
        and (torch.distributed.get_world_size() > 1)
    )

# 检查当前进程是否为主进程（rank 0）
def is_primary() -> bool:
    """
    Returns True if this is rank 0 of a distributed training job OR if it is
    a single trainer job. Otherwise False.

    如果是分布式训练中的 rank 0 或单进程训练，则返回 True；否则返回 False。
    """
    return get_rank() == _PRIMARY_RANK


def all_reduce_mean(tensor: torch.Tensor) -> torch.Tensor:
    """
    Wrapper over torch.distributed.all_reduce for performing mean reduction
    of tensor over all processes.

    使用 torch.distributed.all_reduce 对张量执行均值（mean）归约。
    """
    return all_reduce_op(
        tensor,
        torch.distributed.ReduceOp.SUM,
        lambda t: t / torch.distributed.get_world_size(),
    )


def all_reduce_sum(tensor: torch.Tensor) -> torch.Tensor:
    """
    Wrapper over torch.distributed.all_reduce for performing sum
    reduction of tensor over all processes in both distributed /
    non-distributed scenarios.

    使用 torch.distributed.all_reduce 对张量执行求和（sum）归约。
    """
    return all_reduce_op(tensor, torch.distributed.ReduceOp.SUM)


def all_reduce_min(tensor: torch.Tensor) -> torch.Tensor:
    """
    Wrapper over torch.distributed.all_reduce for performing min
    reduction of tensor over all processes in both distributed /
    non-distributed scenarios.

    使用 torch.distributed.all_reduce 对张量执行最小值（min）归约。
    """
    return all_reduce_op(tensor, torch.distributed.ReduceOp.MIN)


def all_reduce_max(tensor: torch.Tensor) -> torch.Tensor:
    """
    Wrapper over torch.distributed.all_reduce for performing min
    reduction of tensor over all processes in both distributed /
    non-distributed scenarios.

    使用 torch.distributed.all_reduce 对张量执行最大值（max）归约。
    """
    return all_reduce_op(tensor, torch.distributed.ReduceOp.MAX)


def all_reduce_op(
    tensor: torch.Tensor,
    op: torch.distributed.ReduceOp,
    after_op_func: Callable[[torch.Tensor], torch.Tensor] = None,
) -> torch.Tensor:
    """
    Wrapper over torch.distributed.all_reduce for performing
    reduction of tensor over all processes in both distributed /
    non-distributed scenarios.

    使用 torch.distributed.all_reduce 对张量执行归约操作，支持分布式和非分布式场景。
    """
    if is_distributed_training_run():
        tensor, orig_device = convert_to_distributed_tensor(tensor)
        torch.distributed.all_reduce(tensor, op)  # 执行归约操作
        if after_op_func is not None:
            tensor = after_op_func(tensor)  # 如果有后处理函数，应用它
        tensor = convert_to_normal_tensor(tensor, orig_device)  # 将张量转换回原始设备
    return tensor


def gather_tensors_from_all(tensor: torch.Tensor) -> List[torch.Tensor]:
    """
    Wrapper over torch.distributed.all_gather for performing
    'gather' of 'tensor' over all processes in both distributed /
    non-distributed scenarios.

    使用 torch.distributed.all_gather 对张量执行 'gather' 操作，
    将张量从所有进程收集到当前进程，支持分布式和非分布式场景。
    """
    if tensor.ndim == 0:
        # 0 dim tensors cannot be gathered. so unsqueeze
        # 0维张量不能直接聚合，因此需要 unsqueeze 扩展维度
        tensor = tensor.unsqueeze(0)

    if is_distributed_training_run():
        tensor, orig_device = convert_to_distributed_tensor(tensor)
        gathered_tensors = [
            torch.zeros_like(tensor) for _ in range(torch.distributed.get_world_size())
        ]
        torch.distributed.all_gather(gathered_tensors, tensor)  # 聚合所有进程的张量
        gathered_tensors = [
            convert_to_normal_tensor(_tensor, orig_device)
            for _tensor in gathered_tensors
        ]
    else:
        gathered_tensors = [tensor]  # 如果不是分布式训练，直接返回输入张量

    return gathered_tensors


# 使用 `gather_tensors_from_all` 函数对张量进行收集，并将所有收集到的张量拼接成一个大的张量。
def gather_from_all(tensor: torch.Tensor) -> torch.Tensor:
    gathered_tensors = gather_tensors_from_all(tensor)
    gathered_tensor = torch.cat(gathered_tensors, 0)  # 将收集到的张量拼接在一起
    return gathered_tensor


def broadcast(tensor: torch.Tensor, src: int = 0) -> torch.Tensor:
    """
    Wrapper over torch.distributed.broadcast for broadcasting a tensor from the source
    to all processes in both distributed / non-distributed scenarios.

    使用 torch.distributed.broadcast 将一个张量从源进程广播到所有进程。
    """
    if is_distributed_training_run():
        tensor, orig_device = convert_to_distributed_tensor(tensor)
        torch.distributed.broadcast(tensor, src)  # 从源进程广播张量
        tensor = convert_to_normal_tensor(tensor, orig_device)  # 将张量转换回原始设备
    return tensor


def barrier() -> None:
    """
    Wrapper over torch.distributed.barrier, returns without waiting
    if the distributed process group is not initialized instead of throwing error.

    使用 torch.distributed.barrier 在所有进程之间同步，如果分布式进程组未初始化，则不进行等待。
    """
    if not torch.distributed.is_available() or not torch.distributed.is_initialized():
        return  # 如果没有分布式训练环境，则不执行 barrier
    torch.distributed.barrier()  # 等待所有进程同步


def get_world_size() -> int:
    """
    Simple wrapper for correctly getting worldsize in both distributed
    / non-distributed settings

    获取分布式训练中的世界大小（即进程数）。如果没有分布式环境，则返回 1。
    """
    return (
        torch.distributed.get_world_size()
        if torch.distributed.is_available() and torch.distributed.is_initialized()
        else 1
    )


def get_rank() -> int:
    """
    Simple wrapper for correctly getting rank in both distributed
    / non-distributed settings

    获取当前进程的 rank（即进程编号）。如果没有分布式环境，则返回 0。
    """
    return (
        torch.distributed.get_rank()
        if torch.distributed.is_available() and torch.distributed.is_initialized()
        else 0
    )


# 获取主进程的 rank（通常是 rank 0）。
def get_primary_rank() -> int:
    return _PRIMARY_RANK

# 设置当前使用的 CUDA 设备索引。
def set_cuda_device_index(idx: int) -> None:
    global _cuda_device_index
    _cuda_device_index = idx
    torch.cuda.set_device(_cuda_device_index)

# 设置当前使用的设备为 CPU。
def set_cpu_device() -> None:
    global _cuda_device_index
    _cuda_device_index = _CPU_DEVICE_INDEX

# 获取当前使用的 CUDA 设备索引。
def get_cuda_device_index() -> int:
    return _cuda_device_index

# 初始化分布式数据并行模型（DistributedDataParallel）。如果使用 CPU 设备，则返回非 GPU 模式的 DDP。
def init_distributed_data_parallel_model(
    model: torch.nn.Module,
    broadcast_buffers: bool = False,
    find_unused_parameters: bool = True,
    bucket_cap_mb: int = 25,
) -> torch.nn.parallel.DistributedDataParallel:
    global _cuda_device_index

    if _cuda_device_index == _CPU_DEVICE_INDEX:
        # 如果使用的是 CPU 设备，则不指定设备 / CPU-only model, don't specify device
        return torch.nn.parallel.DistributedDataParallel(
            model,
            broadcast_buffers=broadcast_buffers,
            find_unused_parameters=find_unused_parameters,
            bucket_cap_mb=bucket_cap_mb,
        )
    else:
        # GPU model
        return torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[_cuda_device_index],
            output_device=_cuda_device_index,
            broadcast_buffers=broadcast_buffers,
            find_unused_parameters=find_unused_parameters,
            bucket_cap_mb=bucket_cap_mb,
        )


def broadcast_object(obj: Any, src: int = _PRIMARY_RANK, use_disk: bool = True) -> Any:
    """
    Broadcast an object from a source to all workers.
    从源进程广播一个对象到所有工作进程。该对象必须是可序列化的。


    Args:
        obj: Object to broadcast, must be serializable
            要广播的对象，必须是可序列化的。
        src: Source rank for broadcast (default is primary)
            广播的源进程 rank（默认是主进程）。
        use_disk: If enabled, removes redundant CPU memory copies by writing to disk
            如果启用，则通过写入磁盘来减少冗余的 CPU 内存副本（默认 True）。
    """
    # Either broadcast from primary to the fleet (default),
    # or use the src setting as the original rank
    # 如果当前进程是源进程，则发送数据
    if get_rank() == src:
        # Emit data
        buffer = io.BytesIO()
        torch.save(obj, buffer)
        data_view = buffer.getbuffer()  # 将对象序列化并存储在内存缓冲区中
        length_tensor = torch.LongTensor([len(data_view)])
        length_tensor = broadcast(length_tensor, src=src)  # 广播数据长度
        data_tensor = torch.ByteTensor(data_view)
        data_tensor = broadcast(data_tensor, src=src)  # 广播数据
    else:
        # 否则，接收数据 / Fetch from the source
        length_tensor = torch.LongTensor([0])
        length_tensor = broadcast(length_tensor, src=src)  # 广播数据长度
        data_tensor = torch.empty([length_tensor.item()], dtype=torch.uint8)
        data_tensor = broadcast(data_tensor, src=src)  # 广播数据
        if use_disk:
            # 如果启用了 use_disk，使用临时文件进行磁盘存储
            with tempfile.TemporaryFile("r+b") as f:
                f.write(data_tensor.numpy())
                # remove reference to the data tensor and hope that Python garbage
                # collects it
                # 删除数据张量以减少内存占用
                del data_tensor
                f.seek(0)
                obj = torch.load(f)  # 从磁盘加载对象
        else:
            # 否则直接从内存缓冲区加载
            buffer = io.BytesIO(data_tensor.numpy())
            obj = torch.load(buffer)  # 从缓冲区加载对象
    return obj


def all_gather_tensor(tensor: torch.Tensor, world_size=None):
    if world_size is None:
        world_size = get_world_size()
    # make contiguous because NCCL won't gather the tensor otherwise
    # 确保 tensor 是连续的，否则 NCCL 可能无法进行收集
    assert tensor.is_contiguous(), f"{tensor.shape} is not contiguous!"
    tensor, orig_device = convert_to_distributed_tensor(tensor)
    tensor_all = [torch.ones_like(tensor) for _ in range(world_size)]
    # 使用 `all_gather` 收集来自各个进程的 tensor，性能优化（同步操作）
    dist.all_gather(tensor_all, tensor, async_op=False)  # performance opt
    tensor_all = [
        convert_to_normal_tensor(tensor, orig_device) for tensor in tensor_all
    ]
    return tensor_all


def all_gather_batch(tensors: List[torch.Tensor]):
    """
    Performs all_gather operation on the provided tensors.
    对提供的张量执行 all_gather 操作，将各个进程中的张量合并到一起。
    """
    # 获取总的进程数 / Queue the gathered tensors
    world_size = get_world_size()
    # There is no need for reduction in the single-proc case
    # 如果是单进程模式，不需要归约，直接返回原始张量列表
    if world_size == 1:
        return tensors
    tensor_list = []
    output_tensor = []
    # 对每个张量执行 all_gather 操作
    for tensor in tensors:
        tensor_all = all_gather_tensor(tensor, world_size)
        tensor_list.append(tensor_all)

    # 合并所有进程的张量
    for tensor_all in tensor_list:
        output_tensor.append(torch.cat(tensor_all, dim=0))
    return output_tensor


class GatherLayer(autograd.Function):
    """
    Gather tensors from all workers with support for backward propagation:
    This implementation does not cut the gradients as torch.distributed.all_gather does.

    从所有工作节点收集张量并支持反向传播：
    该实现不会像 `torch.distributed.all_gather` 那样切断梯度计算图。
    """

    # 前向传播：收集来自所有进程的张量。
    @staticmethod
    def forward(ctx, x):
        output = [torch.zeros_like(x) for _ in range(dist.get_world_size())]
        dist.all_gather(output, x)
        return tuple(output)

    # 反向传播：收集来自所有进程的梯度，并执行 all_reduce 操作。
    @staticmethod
    def backward(ctx, *grads):
        all_gradients = torch.stack(grads)
        dist.all_reduce(all_gradients)
        return all_gradients[dist.get_rank()]


def all_gather_batch_with_grad(tensors):
    """
    Performs all_gather operation on the provided tensors.
    Graph remains connected for backward grad computation.

    对提供的张量执行 all_gather 操作，同时保留梯度计算图，支持反向传播。
    """
    # 获取总的进程数 / Queue the gathered tensors
    world_size = get_world_size()
    # There is no need for reduction in the single-proc case
    # 如果是单进程模式，则直接返回原始张量列表
    if world_size == 1:
        return tensors
    tensor_list = []
    output_tensor = []

    # 对每个张量执行 GatherLayer 操作，保持梯度图
    for tensor in tensors:
        tensor_all = GatherLayer.apply(tensor)
        tensor_list.append(tensor_all)

    # 合并所有进程的张量
    for tensor_all in tensor_list:
        output_tensor.append(torch.cat(tensor_all, dim=0))
    return output_tensor


# 如果模型被 DistributedDataParallel 包装过，返回原始的模型。
def unwrap_ddp_if_wrapped(model):
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        return model.module
    return model


def create_new_process_group(group_size):
    """
    Creates process groups of a gives `group_size` and returns
    process group that current GPU participates in.
    创建指定大小的进程组，并返回当前 GPU 所在的进程组。

    `group_size` must divide the total number of GPUs (world_size).
    每个进程组的 GPU 数量，必须能整除总的 GPU 数量。

    Modified from
    https://github.com/NVIDIA/apex/blob/4e1ae43f7f7ac69113ef426dd15f37123f0a2ed3/apex/parallel/__init__.py#L60

    Args:
        group_size (int): number of GPU's to collaborate for sync bn
            每个进程组的 GPU 数量
    """

    assert group_size > 0

    world_size = torch.distributed.get_world_size()
    if world_size <= 8:
        if group_size > world_size:
            logging.warning(
                f"Requested group size [{group_size}] > world size [{world_size}]. "
                "Assuming local debug run and capping it to world size."
            )
            group_size = world_size
    assert world_size >= group_size
    assert world_size % group_size == 0

    group = None
    # 为每个进程组创建新组，并返回当前进程所在的组
    for group_num in range(world_size // group_size):
        group_ids = range(group_num * group_size, (group_num + 1) * group_size)
        cur_group = torch.distributed.new_group(ranks=group_ids)
        if torch.distributed.get_rank() // group_size == group_num:
            group = cur_group
            # can not drop out and return here, every process must go through creation of all subgroups
            # 不能在这里提前返回，所有进程都必须创建完所有子组

    assert group is not None
    return group

# 检查分布式训练是否可用且已初始化。
def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True
