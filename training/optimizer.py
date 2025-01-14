# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import fnmatch
import inspect
import itertools
import logging
import types
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Mapping,
    Optional,
    Set,
    Tuple,
    Type,
    Union,
)

import hydra

import torch
import torch.nn as nn
from omegaconf import DictConfig
from torch import Tensor


class Optimizer:
    def __init__(self, optimizer, schedulers=None) -> None:
        # 初始化优化器和调度器
        self.optimizer = optimizer
        self.schedulers = schedulers
        self._validate_optimizer_schedulers()  # 验证优化器和调度器的有效性
        self.step_schedulers(0.0, 0)  # 初始化调度器

    def _validate_optimizer_schedulers(self):
        # 验证优化器中的参数和调度器之间的匹配关系
        if self.schedulers is None:
            return
        for _, set_of_schedulers in enumerate(self.schedulers):
            for option, _ in set_of_schedulers.items():
                # 如果调度器选项不在优化器的默认选项中，抛出异常
                assert option in self.optimizer.defaults, (
                    "Optimizer option "
                    f"{option} not found in {self.optimizer}. Valid options are "
                    f"{self.optimizer.defaults.keys()}"
                )

    def step_schedulers(self, where: float, step: int) -> None:
        # 更新调度器的值
        if self.schedulers is None:
            return
        for i, param_group in enumerate(self.optimizer.param_groups):
            for option, scheduler in self.schedulers[i].items():
                # 如果调度器的__call__方法接收step参数，则调用调度器并传入step
                if "step" in inspect.signature(scheduler.__call__).parameters:
                    new_value = scheduler(step=step, where=where)
                # 如果调度器是一个包含scheduler的对象，并且该scheduler的__call__方法接收step参数
                # 处理ValueScaler包装的调度器
                elif (
                    hasattr(scheduler, "scheduler")
                    and "step"
                    in inspect.signature(scheduler.scheduler.__call__).parameters
                ):
                    # To handle ValueScaler wrappers
                    new_value = scheduler(step=step, where=where)
                else:
                    new_value = scheduler(where)
                # 更新参数组中的调度器选项
                param_group[option] = new_value

    def step(self, where, step, closure=None):
        # 更新调度器的值并执行优化器的step
        self.step_schedulers(where, step)
        return self.optimizer.step(closure)

    def zero_grad(self, *args, **kwargs):
        # 清空梯度
        return self.optimizer.zero_grad(*args, **kwargs)


def set_default_parameters(
    scheduler_cfgs: List[DictConfig], all_parameter_names: Set[str]
) -> None:
    """
    Set up the "default" scheduler with the right parameters.
    设置“默认”调度器并指定适当的参数

    Args:
        scheduler_cgfs: A list of scheduler configs, where each scheduler also
            specifies which parameters it applies to, based on the names of parameters
            or the class of the modules. At most one scheduler is allowed to skip this
            specification, which is used as a "default" specification for any remaining
            parameters.
            一组调度器配置，每个调度器都指定其适用的参数，
            基于参数名称或模块类。至多允许一个调度器跳过此规格，它将作为
            默认规格应用于其他剩余的参数。
        all_parameter_names: Names of all the parameters to consider.
            所有参数的名称
    """
    constraints = [
        scheduler_cfg.parameter_names
        for scheduler_cfg in scheduler_cfgs
        if scheduler_cfg.parameter_names is not None
    ]
    if len(constraints) == 0:
        # 如果没有任何参数名称约束，则默认参数是所有参数
        default_params = set(all_parameter_names)
    else:
        # 否则，默认参数是所有参数减去约束的参数
        default_params = all_parameter_names - set.union(*constraints)
    default_count = 0
    for scheduler_cfg in scheduler_cfgs:
        if scheduler_cfg.parameter_names is None:
            # 如果某个调度器没有指定参数名称，则将其指定为默认参数
            scheduler_cfg.parameter_names = default_params
            default_count += 1
    # 确保至多只有一个调度器不指定参数名称作为默认值
    assert default_count <= 1, "Only one scheduler per option can be default"
    if default_count == 0:
        # No default scheduler specified, add a default, but without any scheduler
        # for that option
        # 如果没有指定默认调度器，则添加一个默认调度器，但不为该选项指定任何调度器
        scheduler_cfgs.append({"parameter_names": default_params})


def name_constraints_to_parameters(
    param_constraints: List[Set[str]], named_parameters: Dict[str, Tensor]
) -> List[torch.nn.Parameter]:
    """
    Return parameters which match the intersection of parameter constraints.
    返回与参数约束交集匹配的参数。

    Note that this returns the parameters themselves, not their names.
    注意，这返回的是参数本身，而不是它们的名称。

    Args:
        param_constraints: A list, with each element being a set of allowed parameters.
            一个列表，其中每个元素是允许的参数集合。
        named_parameters: Mapping from a parameter name to the parameter itself.
            从参数名称到参数本身的映射。

    Returns:
        A list containing the parameters which overlap with _each_ constraint set from
        param_constraints.
        一个列表，包含与param_constraints中每个约束集重叠的参数。
    """
    # 计算所有约束集合的交集，即所有约束条件下都包含的参数名称
    matching_names = set.intersection(*param_constraints)
    # 根据参数名称匹配参数，并返回这些参数
    return [value for name, value in named_parameters.items() if name in matching_names]


def map_scheduler_cfgs_to_param_groups(
    all_scheduler_cfgs: Iterable[List[Dict]],
    named_parameters: Dict[str, Tensor],
) -> Tuple[List[Dict[Any, Any]], List[Dict[str, List[torch.nn.Parameter]]]]:
    """
    Produce parameter groups corresponding to all the scheduler configs.
    根据所有调度器配置生成对应的参数组。

    Takes all the scheduler configs, each of which applies to a specific optimizer
    option (like "lr" or "weight_decay") and has a set of parameter names which it
    applies to, and produces a final set of param groups where each param group
    covers all the options which apply to a particular set of parameters.
    该方法接受所有调度器配置，每个配置都应用于特定的优化器选项（例如“lr”或“weight_decay”），
    并指定其适用的参数名称。它生成最终的参数组，其中每个参数组包含应用于一组特定参数的所有选项。

    Args:
        all_scheduler_cfgs: All the scheduler configs covering every option.
            所有调度器配置，覆盖每个选项。
        named_parameters: Mapping from a parameter name to the parameter itself.
            从参数名称到参数本身的映射。

    Returns:
        Tuple of lists of schedulers and param_groups, where schedulers[i]
        applies to param_groups[i].
        返回两个列表的元组：一个是调度器列表，另一个是参数组列表，调度器列表的第i个元素
        对应于参数组列表的第i个元素。
    """
    # 获取所有调度器配置的笛卡尔积，表示每种调度器组合
    scheduler_cfgs_per_param_group = itertools.product(*all_scheduler_cfgs)
    schedulers = []
    param_groups = []
    for scheduler_cfgs in scheduler_cfgs_per_param_group:
        # 提取每个调度器配置中的参数名称
        param_constraints = [
            scheduler_cfg["parameter_names"] for scheduler_cfg in scheduler_cfgs
        ]
        # 获取符合这些约束条件的参数
        matching_parameters = name_constraints_to_parameters(
            param_constraints, named_parameters
        )

        if len(matching_parameters) == 0:  # 如果没有匹配的参数，跳过该组合 / If no overlap of parameters, skip
            continue
        # 为当前参数组创建调度器字典
        schedulers_for_group = {
            scheduler_cfg["option"]: scheduler_cfg["scheduler"]
            for scheduler_cfg in scheduler_cfgs
            if "option" in scheduler_cfg # 如果调度器配置中有option项
        }
        # 将调度器和对应的参数组添加到列表中
        schedulers.append(schedulers_for_group)
        param_groups.append({"params": matching_parameters})
    return schedulers, param_groups


def validate_param_group_params(param_groups: List[Dict], model: nn.Module):
    """
    Check that the param groups are non-overlapping and cover all the parameters.
    检查参数组是否不重叠并且涵盖了所有模型参数。

    Args:
        param_groups: List of all param groups
            所有参数组的列表
        model: Model to validate against. The check ensures that all the model
            parameters are part of param_groups
            用于验证的模型。检查确保所有模型参数都包含在param_groups中
    """
    for pg in param_groups:
        # 确保每个参数组中的参数不重复 / no param should be repeated within a group
        assert len(pg["params"]) == len(set(pg["params"]))

    # 获取所有参数组的参数集合
    parameters = [set(param_group["params"]) for param_group in param_groups]
    # 获取模型中的所有参数
    model_parameters = {parameter for _, parameter in model.named_parameters()}

    # 确保每对参数组之间没有重叠
    for p1, p2 in itertools.permutations(parameters, 2):
        assert p1.isdisjoint(p2), "Scheduler generated param_groups should be disjoint"
    # 确保所有模型参数都包含在参数组中
    assert set.union(*parameters) == model_parameters, (
        "Scheduler generated param_groups must include all parameters of the model."
        f" Found {len(set.union(*parameters))} params whereas model has"
        f" {len(model_parameters)} params"
    )


def unix_module_cls_pattern_to_parameter_names(
    filter_module_cls_names: List[str],
    module_cls_to_param_names: Dict[Type, str],
) -> Union[None, Set[str]]:
    """
    Returns param names which pass the filters specified in filter_module_cls_names.
    返回通过filter_module_cls_names指定的过滤器的参数名称

    Args:
        filter_module_cls_names: A list of filter strings containing class names, like
            ["torch.nn.LayerNorm", "torch.nn.BatchNorm2d"]
            一个包含类名的过滤器字符串列表，类似 ["torch.nn.LayerNorm", "torch.nn.BatchNorm2d"]
        module_cls_to_param_names: Mapping from module classes to the parameter names
            they contain. See `get_module_cls_to_param_names`.
            一个将模块类映射到它们包含的参数名称的字典。参考 `get_module_cls_to_param_names`
    """
    # 如果没有指定过滤器，则返回一个空的集合
    if filter_module_cls_names is None:
        return set()
    allowed_parameter_names = []  # 用于存储符合条件的参数名称列表
    for module_cls_name in filter_module_cls_names:
        module_cls = hydra.utils.get_class(module_cls_name)  # 获取模块类
        # 如果该模块类没有对应的参数名称，则抛出错误
        if module_cls not in module_cls_to_param_names:
            raise AssertionError(
                f"module_cls_name {module_cls_name} does not "
                "match any classes in the model"
            )
        # 获取该模块类对应的参数名称
        matching_parameters = module_cls_to_param_names[module_cls]
        # 如果该模块类没有任何参数，则抛出错误
        assert (
            len(matching_parameters) > 0
        ), f"module_cls_name {module_cls_name} does not contain any parameters in the model"
        logging.info(
            f"Matches for module_cls_name [{module_cls_name}]: {matching_parameters} "
        )
        # 将符合条件的参数名称添加到列表中
        allowed_parameter_names.append(matching_parameters)
    # 返回所有符合条件的参数名称的集合
    return set.union(*allowed_parameter_names)


def unix_param_pattern_to_parameter_names(
    filter_param_names: Optional[List[str]],
    parameter_names: Dict[str, torch.Tensor],
) -> Union[None, Set[str]]:
    """
    Returns param names which pass the filters specified in filter_param_names.
    返回通过filter_param_names指定的过滤器的参数名称

    Args:
        filter_param_names: A list of unix-style filter strings with optional
            wildcards, like ["block.2.*", "block.2.linear.weight"]
            一个包含 Unix 风格过滤器字符串（可选通配符）的列表，像 ["block.2.*", "block.2.linear.weight"]
        module_cls_to_param_names: Mapping from module classes to the parameter names
            they contain. See `get_module_cls_to_param_names`.
            一个字典，映射参数名称到对应的tensor。
    """
    # 如果没有指定过滤器，则返回一个空集合
    if filter_param_names is None:
        return set()
    allowed_parameter_names = []  # 用于存储符合条件的参数名称列表
    for param_name in filter_param_names:
        # 使用 fnmatch 根据通配符过滤参数名称
        matching_parameters = set(fnmatch.filter(parameter_names, param_name))
        # 如果没有匹配的参数，抛出错误
        assert (
            len(matching_parameters) >= 1
        ), f"param_name {param_name} does not match any parameters in the model"
        # 记录符合条件的参数名称
        logging.info(f"Matches for param_name [{param_name}]: {matching_parameters}")
        # 将符合条件的参数名称添加到列表中
        allowed_parameter_names.append(matching_parameters)
    # 返回所有符合条件的参数名称的集合
    return set.union(*allowed_parameter_names)


def _unix_pattern_to_parameter_names(
    scheduler_cfg: DictConfig,
    parameter_names: Set[str],
    module_cls_to_param_names: Dict[Type, str],
) -> Union[None, Set[str]]:
    """
    Returns param names which pass the filters specified in scheduler_cfg.
    返回通过scheduler_cfg指定的过滤器的参数名称

    Args:
        scheduler_cfg: The config for the scheduler
            调度器配置
        parameter_names: The set of all parameter names which will be filtered
            需要过滤的所有参数名称集合
    """
    # 如果配置中没有提供 "param_names" 或 "module_cls_names"，则返回 None
    if "param_names" not in scheduler_cfg and "module_cls_names" not in scheduler_cfg:
        return None
    # 根据 "param_names" 过滤参数名称，并与基于模块类名称的过滤结果合并
    return unix_param_pattern_to_parameter_names(
        scheduler_cfg.get("param_names"), parameter_names
    ).union(
        unix_module_cls_pattern_to_parameter_names(
            scheduler_cfg.get("module_cls_names"), module_cls_to_param_names
        )
    )


def get_module_cls_to_param_names(
    model: nn.Module, param_allowlist: Set[str] = None
) -> Dict[Type, str]:
    """
    Produce a mapping from all the modules classes to the names of parames they own.
    生成模块类到其拥有的参数名称的映射

    Only counts a parameter as part of the immediate parent module, i.e. recursive
    parents do not count.
    只计算每个模块直接拥有的参数，不递归父模块

    Args:
        model: Model to iterate over
            要遍历的模型
        param_allowlist: If specified, only these param names will be processed
            如果指定，只有这些参数名称会被处理
    """

    module_cls_to_params = {}
    for module_name, module in model.named_modules():
        module_cls = type(module)
        module_cls_to_params.setdefault(module_cls, set())  # 如果模块类不存在，则初始化为空集合
        for param_name, _ in module.named_parameters(recurse=False):  # 遍历模块的参数，不递归子模块
            full_param_name = get_full_parameter_name(module_name, param_name)  # 获取参数的完整名称
            if param_allowlist is None or full_param_name in param_allowlist:  # 如果没有允许列表或参数在列表中
                module_cls_to_params[module_cls].add(full_param_name)  # 将参数名称添加到模块类的集合中
    return module_cls_to_params


def construct_optimizer(
    model: torch.nn.Module,
    optimizer_conf: Any,
    options_conf: Mapping[str, List] = None,
    param_group_modifiers_conf: List[Callable] = None,
    param_allowlist: Optional[Set[str]] = None,
    validate_param_groups=True,
) -> Optimizer:
    """
    Constructs a stochastic gradient descent or ADAM (or ADAMw) optimizer
    with momentum. i.e, constructs a torch.optim.Optimizer with zero-weight decay
    Batchnorm and/or no-update 1-D parameters support, based on the config.
    构建带动量的随机梯度下降(SGD)或ADAM（或ADAMw）优化器。
    即根据配置构建一个torch.optim.Optimizer，支持零权重衰减，支持BatchNorm和/或不更新1-D参数。

    Supports wrapping the optimizer with Layer-wise Adaptive Rate Scaling
    (LARS): https://arxiv.org/abs/1708.03888
    支持使用层自适应学习率缩放（LARS）包装优化器： https://arxiv.org/abs/1708.03888

    Args:
        model: model to perform stochastic gradient descent
            optimization or ADAM optimization.
            要进行优化的模型
        optimizer_conf: Hydra config consisting a partial torch optimizer like SGD or
            ADAM, still missing the params argument which this function provides to
            produce the final optimizer
            Hydra 配置，包含一个部分的torch优化器（如SGD或ADAM），
            此函数将为其提供缺少的params参数以生成最终优化器
        param_group_modifiers_conf: Optional user specified functions which can modify
            the final scheduler configs before the optimizer's param groups are built
            可选的用户指定函数，用于在构建优化器的参数组之前修改最终的调度器配置
        param_allowlist: The parameters to optimize. Parameters which are not part of
            this allowlist will be skipped.
            需要优化的参数。未包含在此允许列表中的参数将被跳过
        validate_param_groups: If enabled, valides that the produced param_groups don't
            overlap and cover all the model parameters.
            如果启用，则验证生成的参数组不重叠并覆盖所有模型参数
    """
    # 如果没有指定允许的参数列表，则默认为模型中所有参数
    if param_allowlist is None:
        param_allowlist = {name for name, _ in model.named_parameters()}

    # 获取模型中所有参数的名称和值，且名称在允许列表中
    named_parameters = {
        name: param
        for name, param in model.named_parameters()
        if name in param_allowlist
    }

    # 如果没有指定调度器配置，则直接创建优化器
    if not options_conf:
        optimizer = hydra.utils.instantiate(optimizer_conf, named_parameters.values())
        return Optimizer(optimizer)
    # 获取所有参数名称集合
    all_parameter_names = {
        name for name, _ in model.named_parameters() if name in param_allowlist
    }
    # 获取模块类到所有参数名称的映射
    module_cls_to_all_param_names = get_module_cls_to_param_names(
        model, param_allowlist
    )
    # 创建调度器配置
    scheduler_cfgs_per_option = hydra.utils.instantiate(options_conf)
    all_scheduler_cfgs = []
    # 为每个调度器选项创建调度器配置
    for option, scheduler_cfgs in scheduler_cfgs_per_option.items():
        for config in scheduler_cfgs:
            config.option = option
            # 根据参数名称和模块类名称过滤参数
            config.parameter_names = _unix_pattern_to_parameter_names(
                config, all_parameter_names, module_cls_to_all_param_names
            )

        # 设置调度器的默认参数
        set_default_parameters(scheduler_cfgs, all_parameter_names)
        all_scheduler_cfgs.append(scheduler_cfgs)

    # 如果有用户自定义的参数组修改函数，则执行修改
    if param_group_modifiers_conf:
        for custom_param_modifier in param_group_modifiers_conf:
            custom_param_modifier = hydra.utils.instantiate(custom_param_modifier)
            all_scheduler_cfgs = custom_param_modifier(
                scheduler_cfgs=all_scheduler_cfgs, model=model
            )
    # 映射调度器配置到参数组
    schedulers, param_groups = map_scheduler_cfgs_to_param_groups(
        all_scheduler_cfgs, named_parameters
    )
    # 如果需要验证参数组，确保没有重叠并且覆盖所有模型参数
    if validate_param_groups:
        validate_param_group_params(param_groups, model)
    # 实例化优化器并返回
    optimizer = hydra.utils.instantiate(optimizer_conf, param_groups)
    return Optimizer(optimizer, schedulers)


def get_full_parameter_name(module_name, param_name):
    if module_name == "":
        return param_name
    return f"{module_name}.{param_name}"


class GradientClipper:
    """
    Gradient clipping utils that works for DDP
    用于分布式数据并行（DDP）的梯度裁剪工具类
    """

    def __init__(self, max_norm: float = 1.0, norm_type: int = 2):
        # 确保 max_norm 是数值类型或 None
        assert isinstance(max_norm, (int, float)) or max_norm is None
        self.max_norm = max_norm if max_norm is None else float(max_norm)  # 设置最大范数
        self.norm_type = norm_type  # 设置范数类型

    def __call__(self, model: nn.Module):
        # 对模型的参数应用梯度裁剪。
        if self.max_norm is None:
            return  # 不执行任何操作 / no-op

        # 使用 PyTorch 的 `clip_grad_norm_` 函数进行梯度裁剪
        nn.utils.clip_grad_norm_(
            model.parameters(), max_norm=self.max_norm, norm_type=self.norm_type
        )


class ValueScaler:
    # 值缩放器，用于调整调度器输出的值。
    def __init__(self, scheduler, mult_val: float):
        self.scheduler = scheduler  # 保存调度器
        self.mult_val = mult_val  # 保存缩放因子

    def __call__(self, *args, **kwargs):
        val = self.scheduler(*args, **kwargs)  # 获取调度器的输出
        return val * self.mult_val  # 返回缩放后的值


def rgetattr(obj, rattrs: str = None):
    """
    Like getattr(), but supports dotted notation for nested objects.
    rattrs is a str of form 'attr1.attr2', returns obj.attr1.attr2
    类似于 `getattr()`，但支持嵌套对象的点符号访问。
    `rattrs` 是一个形如 'attr1.attr2' 的字符串，返回 obj.attr1.attr2

    """
    if rattrs is None:
        return obj  # 如果没有提供属性字符串，直接返回对象本身
    attrs = rattrs.split(".")  # 按 '.' 分割属性字符串
    for attr in attrs:
        obj = getattr(obj, attr)  # 逐级获取属性
    return obj  # 返回最终属性值


def layer_decay_param_modifier(
    scheduler_cfgs: List[List[Dict]],
    model,
    layer_decay_value: float,
    layer_decay_min: Optional[float] = None,
    apply_to: Optional[str] = None,
    overrides: List[Dict] = (),
) -> List[List[Dict]]:
    """
    层衰减参数修改器，根据模型的层级调整学习率或其他参数的衰减。

    Args
    - scheduler_cfgs: a list of omegaconf.ListConfigs.
        Each element in the list is a omegaconfg.DictConfig with the following structure
        {
            "scheduler": <some fvcore scheduler>
            "option": <value> possible options are "lr", "weight_decay" etc.
            "parameter_names": Set of str indicating param names that this scheduler applies to
        }
        一个列表，其中每个元素是一个 omegaconf.ListConfig。每个列表中的字典包含：
        {
            "scheduler": 调度器对象
            "option": 要调整的参数（如学习率 "lr" 或权重衰减 "weight_decay"）
            "parameter_names": 该调度器适用的参数名称集合
        }
    - model: a model that implements a method `get_layer_id` that maps layer_name to an integer and
            and a method get_num_layers.
            Alternatively, use apply_to argument to select a specific component of the model.
            一个实现了 `get_layer_id` 和 `get_num_layers` 方法的模型，或者通过 `apply_to` 参数指定的模型组件。
    - layer_decay_value: float
        层衰减的初始值，用于根据层数逐步衰减。
    - layer_decay_min: min val for layer decay
        可选的，最小层衰减值，如果提供，层衰减值将不小于该值。
    - apply_to: optional arg to select which component of the model to apply the the layer decay modifier to
        可选的，指定应用层衰减的模型组件的名称（默认为 None）。
    - overrides: to manually override lr for specific patterns. Is a list of dicts. Each dict, has keys "pattern", "value".
        一个列表，手动覆盖某些模式的学习率设置。每个字典有两个键：

    Returns
    - final_scheduler_cfgs: same structure as the input, elements can be modified
        修改后的调度器配置，具有与输入相同的结构。
    """
    # 获取应用层衰减的模型组件
    model = rgetattr(model, apply_to)
    num_layers = model.get_num_layers() + 1  # 获取模型的层数
    # 根据层数计算每一层的衰减因子
    layer_decays = [
        layer_decay_value ** (num_layers - i) for i in range(num_layers + 1)
    ]
    # 如果提供了最小衰减值，确保所有层的衰减值不小于该最小值
    if layer_decay_min is not None:
        layer_decays = [max(val, layer_decay_min) for val in layer_decays]
    final_scheduler_cfgs = []

    # scheduler_cfgs is a list of lists
    # 遍历每组调度器配置
    for scheduler_cfg_group in scheduler_cfgs:
        curr_cfg_group = []
        # scheduler_cfg_group is a list of dictionaries
        # 遍历配置组中的每个调度器配置
        for scheduler_cfg in scheduler_cfg_group:
            # 如果配置的选项不是学习率（lr），直接添加到结果中
            if scheduler_cfg["option"] != "lr":
                curr_cfg_group.append(scheduler_cfg)
                continue
            # Need sorted so that the list of parameter names is deterministic and consistent
            # across re-runs of this job. Else it was causing issues with loading the optimizer
            # state during a job restart (D38591759)
            # 需要排序，以确保参数名称的列表在重新运行作业时是确定性且一致的。
            # 否则，在作业重新启动时加载优化器状态时会出现问题 (D38591759)
            parameter_names = sorted(scheduler_cfg["parameter_names"])

            # 用于每层只需要一个配置 / Only want one cfg group per layer
            layer_cfg_groups = {}
            # 遍历每个参数名
            for param_name in parameter_names:
                layer_id = num_layers  # 默认层ID为最大层数
                this_scale = layer_decays[layer_id]  # 获取对应层的衰减因子
                # 如果参数名称以 apply_to 开头，表示属于某一特定组件，获取该层的 ID
                if param_name.startswith(apply_to):
                    layer_id = model.get_layer_id(param_name)  # 获取层ID
                    this_scale = layer_decays[layer_id]  # 获取对应的衰减因子
                    # 如果有手动覆盖设置，则应用覆盖 / Overrides
                    for override in overrides:
                        if fnmatch.fnmatchcase(param_name, override["pattern"]):
                            this_scale = float(override["value"])  # 覆盖衰减因子
                            layer_id = override["pattern"]  # 记录覆盖的模式
                            break

                # 将每层的配置按层ID分组
                if layer_id not in layer_cfg_groups:
                    curr_param = {
                        "option": scheduler_cfg["option"],  # 选项（如学习率、权重衰减等）
                        "scheduler": ValueScaler(
                            scheduler_cfg["scheduler"], this_scale
                        ),  # 应用缩放因子的调度器
                        "parameter_names": {param_name},  # 当前层的参数名集合
                    }
                else:
                    curr_param = layer_cfg_groups[layer_id]
                    curr_param["parameter_names"].add(param_name)  # 将参数名添加到对应的层配置中
                layer_cfg_groups[layer_id] = curr_param  # 更新层配置

            # 将每层的配置添加到当前配置组中
            for layer_cfg in layer_cfg_groups.values():
                curr_cfg_group.append(layer_cfg)
        # 将当前配置组添加到最终配置中
        final_scheduler_cfgs.append(curr_cfg_group)
    return final_scheduler_cfgs  # 返回修改后的调度器配置
