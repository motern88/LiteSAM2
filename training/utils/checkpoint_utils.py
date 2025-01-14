# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import contextlib
import fnmatch
import logging
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Mapping,
    Optional,
    Sequence,
    Set,
    Tuple,
    Union,
)

import numpy as np
import torch
import torch.nn as nn
from iopath.common.file_io import g_pathmgr
from torch.jit._script import RecursiveScriptModule


def unix_pattern_to_parameter_names(
    constraints: List[str], all_parameter_names: Sequence[str]
) -> Union[None, Set[str]]:
    """
    Go through the list of parameter names and select those that match
    any of the provided constraints
    遍历给定的参数名列表，选择那些符合提供的约束条件的参数名
    """
    parameter_names = []
    for param_name in constraints:
        # 使用 fnmatch 过滤出与约束条件匹配的参数名
        matching_parameters = set(fnmatch.filter(all_parameter_names, param_name))
        assert (
            len(matching_parameters) > 0
        ), f"param_names {param_name} don't match any param in the given names."
        parameter_names.append(matching_parameters)
    # 返回所有匹配的参数名的并集
    return set.union(*parameter_names)


def filter_params_matching_unix_pattern(
    patterns: List[str], state_dict: Dict[str, torch.Tensor]
) -> Dict[str, torch.Tensor]:
    """
    Remove from the state dictionary the parameters matching the provided unix patterns
    从状态字典中移除与提供的 unix 模式匹配的参数

    Args:
        patterns: the list of unix patterns to exclude
            需要排除的 unix 模式列表
        state_dict: the dictionary to filter
            需要过滤的状态字典

    Returns:
        A new state dictionary
        过滤后的新的状态字典
    """
    if len(patterns) == 0:
        return {}

    # 获取所有的键名
    all_keys = list(state_dict.keys())
    # 获取匹配 unix 模式的参数名
    included_keys = unix_pattern_to_parameter_names(patterns, all_keys)
    # 返回过滤后的状态字典，仅保留匹配的参数
    return {k: state_dict[k] for k in included_keys}


def exclude_params_matching_unix_pattern(
    patterns: List[str], state_dict: Dict[str, torch.Tensor]
) -> Dict[str, torch.Tensor]:
    """
    Remove from the state dictionary the parameters matching the provided unix patterns
    从状态字典中移除与提供的 unix 模式匹配的参数

    Args:
        patterns: the list of unix patterns to exclude
            需要排除的 unix 模式列表
        state_dict: the dictionary to filter
            需要过滤的状态字典

    Returns:
        A new state dictionary
        过滤后的新的状态字典
    """
    if len(patterns) == 0:
        return state_dict

    # 获取所有的键名
    all_keys = list(state_dict.keys())
    # 获取要排除的键名
    excluded_keys = unix_pattern_to_parameter_names(patterns, all_keys)
    # 返回移除匹配键名后的字典
    return {k: v for k, v in state_dict.items() if k not in excluded_keys}

# 获取状态字典的简要总结，包括所有键名及其对应的张量和求和结果
def _get_state_dict_summary(state_dict: Dict[str, torch.Tensor]):
    keys = []
    trace = []
    for k, v in state_dict.items():
        keys.append(k)
        trace.append(v.sum().item())
    # 按照键名对 trace 进行排序
    trace = np.array(trace)[np.argsort(keys)]
    return trace


def assert_skipped_parameters_are_frozen(model: nn.Module, patterns: List[str]):
    """
    Verifies that all the parameters matching the provided patterns
    are frozen - this acts as a safeguard when ignoring parameter
    when saving checkpoints - if the parameters are in fact trainable
    验证所有匹配给定模式的参数是否被冻结
    这作为一个保护措施，防止在保存检查点时忽略的参数仍然是可训练的
    """
    if not patterns:
        return

    # 获取过滤后的状态字典，其中只包含匹配的参数
    frozen_state_dict = filter_params_matching_unix_pattern(
        patterns=patterns, state_dict=model.state_dict()
    )
    # 获取那些被排除但仍然可训练的参数
    non_frozen_keys = {
        n
        for n, p in model.named_parameters()
        if n in frozen_state_dict and p.requires_grad
    }
    # 如果有未冻结的参数，抛出异常
    if non_frozen_keys:
        raise ValueError(
            f"Parameters excluded with `skip_saving_parameters` should be frozen: {non_frozen_keys}"
        )


@contextlib.contextmanager
def with_check_parameter_frozen(
    model: nn.Module, patterns: List[str], disabled: bool = True
):
    """
    Context manager that inspects a model surrounding a piece of code
    and verifies if the model has been updated by this piece of code
    上下文管理器，用于检查在代码块执行过程中模型的更新情况，
    并验证模型中是否有匹配给定模式的参数被更新。

    The function will raise an exception if the model has been updated
    on at least one of the parameter that matches one of the pattern
    如果模型在至少一个匹配的参数上发生了更新，函数将抛出异常。

    Args:
        model: the model that might have been updated
            可能被更新的模型
        patterns: for the parameters we want to observe
            需要观察的参数模式列表
        disabled: if True, the context manager does nothing
            是否禁用检查，默认为 True
    """
    if not patterns or disabled:
        yield
        return

    # 获取符合给定模式的参数，记录更新前的状态
    frozen_state_dict = filter_params_matching_unix_pattern(
        patterns=patterns, state_dict=model.state_dict()
    )
    summary_before = _get_state_dict_summary(frozen_state_dict)

    yield  # 执行代码块

    # 执行完代码块后，再次获取符合给定模式的参数，记录更新后的状态
    frozen_state_dict = filter_params_matching_unix_pattern(
        patterns=patterns, state_dict=model.state_dict()
    )
    summary_after = _get_state_dict_summary(frozen_state_dict)

    # 如果更新前后状态不同，抛出异常
    if not np.allclose(summary_before, summary_after, atol=1e-6):
        raise ValueError(
            f"""
            The `model_weight_initializer` has initialized parameters frozen with `skip_saving_parameters`.
            You can resolve this error by either initializing those parameters from within the model definition
            or using the flag `trainer.checkpoint.initialize_after_preemption` to True.
        """
        )


class CkptExcludeKernel:
    """
    Removes the keys from the given model state_dict that match the key_pattern.
    从给定模型的 state_dict 中移除匹配指定模式的键值。

    Args:
        key_pattern: Patterns used to select the keys in the state_dict
            that are eligible for this kernel.
            用于选择符合模式的 state_dict 键
    """

    def __init__(self, key_pattern: List[str]):
        self.key_pattern = key_pattern

    def __call__(self, state_dict: Dict):
        """
        Args:
            state_dict: A dictionary representing the given checkpoint's state dict.
                表示给定检查点状态字典的字典
        """
        if len(self.key_pattern) == 0:
            return state_dict
        # 获取符合模式的键
        exclude_keys = unix_pattern_to_parameter_names(
            self.key_pattern, state_dict.keys()
        )
        # 返回移除符合模式键后的字典
        return {k: v for k, v in state_dict.items() if k not in exclude_keys}


def load_checkpoint(
    path_list: List[str],
    pick_recursive_keys: Optional[List[str]] = None,
    map_location: str = "cpu",
) -> Any:
    """
    Loads a checkpoint from the specified path.
    从指定路径加载检查点。

    Args:
        path_list: A list of paths which contain the checkpoint. Each element
            is tried (in order) until a file that exists is found. That file is then
            used to read the checkpoint.
            包含检查点路径的列表。每个元素都会按顺序尝试，直到找到一个存在的文件，
            然后使用该文件读取检查点。
        pick_recursive_keys: Picks sub dicts from the loaded checkpoint if not None.
            For pick_recursive_keys = ["a", "b"], will return checkpoint_dict["a"]["b"]
            如果不为 None，选择加载的检查点中的子字典。
            比如 pick_recursive_keys = ["a", "b"]，将返回 checkpoint_dict["a"]["b"]
        map_location (str): a function, torch.device, string or a dict specifying how to
            remap storage locations
            一个函数、torch.device、字符串或字典，用于指定如何重新映射存储位置


    Returns: Model with the matchin pre-trained weights loaded.
            加载了预训练权重的模型。
    """
    path_exists = False
    # 遍历路径列表，查找第一个存在的路径
    for path in path_list:
        if g_pathmgr.exists(path):
            path_exists = True
            break

    if not path_exists:
        raise ValueError(f"No path exists in {path_list}")

    # 打开文件并加载检查点
    with g_pathmgr.open(path, "rb") as f:
        checkpoint = torch.load(f, map_location=map_location)
    # 如果指定了 pick_recursive_keys，从检查点中提取子字典
    logging.info(f"Loaded checkpoint from {path}")
    if pick_recursive_keys is not None:
        for key in pick_recursive_keys:
            checkpoint = checkpoint[key]
    return checkpoint


def get_state_dict(checkpoint, ckpt_state_dict_keys):
    if isinstance(checkpoint, RecursiveScriptModule):
        # 如果是一个 TorchScript JIT 模型，直接返回它的 state_dict / This is a torchscript JIT model
        return checkpoint.state_dict()
    pre_train_dict = checkpoint
    # 遍历键路径
    for i, key in enumerate(ckpt_state_dict_keys):
        # 如果是字典且没有找到该键，或是序列且索引越界，抛出异常
        if (isinstance(pre_train_dict, Mapping) and key not in pre_train_dict) or (
            isinstance(pre_train_dict, Sequence) and key >= len(pre_train_dict)
        ):
            # 生成错误信息时的键路径字符串
            key_str = (
                '["' + '"]["'.join(list(map(ckpt_state_dict_keys[:i], str))) + '"]'
            )
            raise KeyError(
                f"'{key}' not found in checkpoint{key_str} "
                f"with keys: {pre_train_dict.keys()}"
            )
        # 如果找到键，更新 pre_train_dict
        pre_train_dict = pre_train_dict[key]
    return pre_train_dict


def load_checkpoint_and_apply_kernels(
    checkpoint_path: str,
    checkpoint_kernels: List[Callable] = None,
    ckpt_state_dict_keys: Tuple[str] = ("state_dict",),
    map_location: str = "cpu",
) -> nn.Module:
    """
    Performs checkpoint loading with a variety of pre-processing kernel applied in
    sequence.
    加载检查点并应用一系列预处理核函数。

    Args:
        checkpoint_path (str): Path to the checkpoint.
            检查点的路径。
        checkpoint_kernels List(Callable): A list of checkpoint processing kernels
            to apply in the specified order. Supported kernels include `CkptIncludeKernel`,
            `CkptExcludeKernel`, etc. These kernels are applied in the
            given order.
            要应用的检查点处理核函数的列表，这些核函数按给定的顺序应用。
            支持的核函数包括 `CkptIncludeKernel`、`CkptExcludeKernel` 等。
        ckpt_state_dict_keys (str): Keys containing the model state dict.
            包含模型状态字典的键路径。
        map_location (str): a function, torch.device, string or a dict specifying how to
            remap storage locations
            用于指定如何重新映射存储位置的函数、设备、字符串或字典。

    Returns: Model with the matchin pre-trained weights loaded.
        加载了匹配的预训练权重的模型。
    """
    # 检查路径是否存在
    assert g_pathmgr.exists(checkpoint_path), "Checkpoint '{}' not found".format(
        checkpoint_path
    )

    # Load the checkpoint on CPU to avoid GPU mem spike.
    # 在 CPU 上加载检查点，避免 GPU 内存突增。
    with g_pathmgr.open(checkpoint_path, "rb") as f:
        checkpoint = torch.load(f, map_location=map_location)
    # 获取检查点的状态字典
    pre_train_dict = get_state_dict(checkpoint, ckpt_state_dict_keys)

    # Not logging into info etc since it's a huge log
    # 不进行日志记录，以免产生过大的日志
    logging.debug(
        "Loaded Checkpoint State Dict pre-kernel application: %s"
        % str(", ".join(list(pre_train_dict.keys())))
    )
    # 应用核函数 / Apply kernels
    if checkpoint_kernels is not None:
        for f in checkpoint_kernels:
            pre_train_dict = f(state_dict=pre_train_dict)

    logging.debug(
        "Loaded Checkpoint State Dict Post-kernel application %s"
        % str(", ".join(list(pre_train_dict.keys())))
    )

    return pre_train_dict

# 检查加载模型状态字典时的错误，确保没有遗漏或不符合的键。
def check_load_state_dict_errors(
    missing_keys,
    unexpected_keys,
    strict: bool,
    ignore_missing_keys: List[str] = None,
    ignore_unexpected_keys: List[str] = None,
):
    # 忽略指定的缺失键
    if ignore_missing_keys is not None and len(ignore_missing_keys) > 0:
        ignored_keys = unix_pattern_to_parameter_names(
            ignore_missing_keys, missing_keys
        )
        missing_keys = [key for key in missing_keys if key not in ignored_keys]

    # 忽略指定的意外键
    if ignore_unexpected_keys is not None and len(ignore_unexpected_keys) > 0:
        ignored_unexpected_keys = unix_pattern_to_parameter_names(
            ignore_unexpected_keys, unexpected_keys
        )
        unexpected_keys = [
            key for key in unexpected_keys if key not in ignored_unexpected_keys
        ]

    # 错误信息构建
    err = "State key mismatch."
    if unexpected_keys:
        err += f" Unexpected keys: {unexpected_keys}."
    if missing_keys:
        err += f" Missing keys: {missing_keys}."

    # 如果有缺失或意外键，进行警告并根据 strict 参数决定是否抛出异常
    if unexpected_keys or missing_keys:
        logging.warning(err)
        if unexpected_keys or strict:
            raise KeyError(err)


def load_state_dict_into_model(
    state_dict: Dict,
    model: nn.Module,
    strict: bool = True,
    ignore_missing_keys: List[str] = None,
    ignore_unexpected_keys: List[str] = None,
    checkpoint_kernels: List[Callable] = None,
):
    """
    Loads a state dict into the given model.
    将状态字典加载到指定的模型中。

    Args:
        state_dict: A dictionary containing the model's
            state dict, or a subset if strict is False
            包含模型状态字典的字典，或者如果 strict 为 False，则为其子集。
        model: Model to load the checkpoint weights into
            要加载权重的模型。
        strict: raise if the state_dict has missing state keys
            是否严格要求状态字典中的键匹配。
        ignore_missing_keys: unix pattern of keys to ignore
            要忽略的缺失键的 unix 模式。
    """
    # 应用检查点核函数 / Apply kernels
    if checkpoint_kernels is not None:
        for f in checkpoint_kernels:
            state_dict = f(state_dict=state_dict)
    # 加载状态字典到模型
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)

    # 检查加载时的错误
    check_load_state_dict_errors(
        missing_keys,
        unexpected_keys,
        strict=strict,
        ignore_missing_keys=ignore_missing_keys,
        ignore_unexpected_keys=ignore_unexpected_keys,
    )
    return model
