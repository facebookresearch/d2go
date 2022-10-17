# g!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import itertools
import logging
from typing import Any, Dict, Iterable, List, Optional, Type, Union

import torch
from d2go.config import CfgNode

# FIXME: optimizer should not depend on quantization (or vice versa)
from d2go.quantization.learnable_qat import iterate_module_named_parameters
from detectron2.solver.build import (
    maybe_add_gradient_clipping as d2_maybe_add_gradient_clipping,
    reduce_param_groups,
)
from detectron2.utils.registry import Registry


D2GO_OPTIM_MAPPER_REGISTRY = Registry("D2GO_OPTIM_MAPPER")

logger = logging.getLogger(__name__)


OptimizerModelsType = Union[torch.nn.Module, torch.nn.parallel.DistributedDataParallel]
OptimizerParams = Union[Iterable[torch.Tensor], Iterable[Dict[str, Any]]]


NORM_MODULE_TYPES = (
    torch.nn.BatchNorm1d,
    torch.nn.BatchNorm2d,
    torch.nn.BatchNorm3d,
    torch.nn.SyncBatchNorm,
    torch.nn.GroupNorm,
    torch.nn.InstanceNorm1d,
    torch.nn.InstanceNorm2d,
    torch.nn.InstanceNorm3d,
    torch.nn.LayerNorm,
    torch.nn.LocalResponseNorm,
)


def get_optimizer_param_groups_from_model(
    cfg: CfgNode, model: OptimizerModelsType, required: bool = False
) -> List[Dict[str, Any]]:
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        model = model.module
    if not hasattr(model, "get_optimizer_param_groups"):
        if required:
            raise Exception(
                "Expected model to implement get_optimizer_param_groups "
                "to specify parameters for the optimizer(-s)"
            )
        else:
            return []

    return model.get_optimizer_param_groups(cfg)


def get_lr_and_weight_decay_optimizer_param_groups(
    cfg: CfgNode, model: OptimizerModelsType, param_groups: List[Dict[str, Any]] = None
) -> List[Dict[str, Any]]:
    # parameter groups for lr
    params = get_optimizer_param_groups_lr(
        model,
        base_lr=cfg.SOLVER.BASE_LR,
        bias_lr_factor=cfg.SOLVER.BIAS_LR_FACTOR,
        lr_multipliers_overwrite=_merge_dict(cfg.SOLVER.LR_MULTIPLIER_OVERWRITE),
        param_groups=param_groups,
    )

    # parameter groups for normalization, bias, and embedding
    params += get_optimizer_param_groups_weight_decay(
        model,
        weight_decay=cfg.SOLVER.WEIGHT_DECAY,
        weight_decay_norm=cfg.SOLVER.WEIGHT_DECAY_NORM,
        weight_decay_bias=cfg.SOLVER.WEIGHT_DECAY_BIAS,
        weight_decay_embed=cfg.SOLVER.WEIGHT_DECAY_EMBED,
        weight_decay_overwrite=_merge_dict(cfg.SOLVER.WEIGHT_DECAY_OVERWRITE),
        param_groups=param_groups,
    )
    return params


def get_optimizer_param_groups(
    model: OptimizerModelsType,
    cfg: CfgNode,
) -> List[Dict[str, Any]]:
    """
    Get override optimizer parameter groups
       * Get all default parameters
       # Get parameter groups for normalization and bias
       # Get parameter groups from model if the model implements `get_optimizer_param_groups()`
    Parameters appear later will override parameters appear earlier
    """
    params = get_optimizer_param_groups_default(model)
    params += get_lr_and_weight_decay_optimizer_param_groups(cfg, model)
    params += get_optimizer_param_groups_from_model(cfg, model)

    return reduce_param_groups(params)


def get_optimizer_param_groups_default(
    model: OptimizerModelsType,
) -> List[Dict[str, Any]]:
    ret = [
        {
            "params": list(
                filter(
                    lambda x: x.requires_grad,
                    model.parameters(),
                )
            )
        }
    ]
    return ret


def get_optimizer_param_groups_lr(
    model: OptimizerModelsType,
    base_lr: float,
    bias_lr_factor: float = 1.0,
    lr_multipliers_overwrite: Optional[Dict[str, float]] = None,
    param_groups: List[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """
    Allow setting up lr for modules
    base_lr: lr for all modules
    bias_lr_factor: scale factor for lr for bias term
    lr_multipliers_overwrite (dict: str-> float):
        Applying different lr multiplier to a set of parameters whose names
        containing certain keys. For example, if lr_multipliers_overwrite={'backbone': 0.1},
        the LR for the parameters whose names containing 'backbone' will be scaled to 0.1x.
        Set lr_multipliers_overwrite=None if no multipliers required.
    """
    params_set = set()
    if param_groups is not None:
        for pg in param_groups:
            params_set.update(pg["params"])
    params: List[Dict[str, Any]] = []
    for (
        module_name,
        _module,
        module_param_name,
        value,
    ) in iterate_module_named_parameters(model):
        if params_set and value not in params_set:
            continue
        cur_lr = base_lr
        if module_param_name == "bias":
            cur_lr = base_lr * bias_lr_factor
        if lr_multipliers_overwrite is not None:
            for kname, mult in lr_multipliers_overwrite.items():
                if kname in module_name:
                    # apply multiplier for the params containing kname, e.g. backbone
                    cur_lr = cur_lr * mult

        params += [
            {
                "params": [value],
                "lr": cur_lr,
            }
        ]

    return params


def get_optimizer_param_groups_weight_decay(
    model: OptimizerModelsType,
    weight_decay: Optional[float],
    weight_decay_norm: Optional[float] = None,
    weight_decay_bias: Optional[float] = None,
    weight_decay_embed: Optional[float] = None,
    weight_decay_overwrite: Optional[Dict[str, float]] = None,
    param_groups: List[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """
    Allow setting up weight decay for normalization, embedding and bias
    """
    params_set = set()
    if param_groups is not None:
        for pg in param_groups:
            params_set.update(pg["params"])

    weight_decay_norm = weight_decay if weight_decay_norm is None else weight_decay_norm
    weight_decay_bias = weight_decay if weight_decay_bias is None else weight_decay_bias
    weight_decay_embed = (
        weight_decay if weight_decay_embed is None else weight_decay_embed
    )

    params: List[Dict[str, Any]] = []

    for (
        _module_name,
        module,
        module_param_name,
        value,
    ) in iterate_module_named_parameters(model):
        if params_set and value not in params_set:
            continue

        cur_wd = weight_decay
        if isinstance(module, NORM_MODULE_TYPES):
            cur_wd = weight_decay_norm
        elif isinstance(module, torch.nn.Embedding):
            cur_wd = weight_decay_embed
        elif module_param_name == "bias":
            cur_wd = weight_decay_bias
        if weight_decay_overwrite is not None:
            for kname, wd in weight_decay_overwrite.items():
                if kname in module_param_name:
                    cur_wd = wd

        if cur_wd is not None:
            params += [
                {
                    "params": [value],
                    "weight_decay": cur_wd,
                }
            ]

    return params


def get_optimizer_param_groups_override(
    model: OptimizerModelsType,
    overrides: Optional[Dict[str, Dict[str, float]]] = None,
) -> List[Dict[str, Any]]:
    """
    Allow setting up overrides for parameter groups
    overrides (dict: str -> (dict: str -> float)):
        if not `None`, provides values for optimizer hyperparameters
        (LR, weight decay) for module parameters with a given name; e.g.
        {"embedding": {"lr": 0.01, "weight_decay": 0.1}} will set the LR and
        weight decay values for all module parameters named `embedding` (default: None)
    """

    params: List[Dict[str, Any]] = []

    if overrides is None:
        return params

    for (
        _module_name,
        _module,
        module_param_name,
        value,
    ) in iterate_module_named_parameters(model):
        schedule_params = {}
        if module_param_name in overrides:
            schedule_params.update(overrides[module_param_name])
            params += [{"params": [value], **schedule_params}]

    return params


def maybe_add_gradient_clipping(
    cfg: CfgNode, optim: Type[torch.optim.Optimizer]
) -> Type[torch.optim.Optimizer]:
    # detectron2 doesn't have full model gradient clipping now
    clip_norm_val = cfg.SOLVER.CLIP_GRADIENTS.CLIP_VALUE
    enable = (
        cfg.SOLVER.CLIP_GRADIENTS.ENABLED
        and cfg.SOLVER.CLIP_GRADIENTS.CLIP_TYPE == "full_model"
        and clip_norm_val > 0.0
    )

    class FullModelGradientClippingOptimizer(optim):
        def step(self, closure=None):
            all_params = itertools.chain(*[x["params"] for x in self.param_groups])
            torch.nn.utils.clip_grad_norm_(all_params, clip_norm_val)
            super().step(closure=closure)

    if enable:
        return FullModelGradientClippingOptimizer
    return d2_maybe_add_gradient_clipping(cfg, optim)


def _merge_dict(in_dict):
    ret_dict = {}
    assert all(isinstance(x, dict) for x in in_dict)
    for dic in in_dict:
        ret_dict.update(dic)
    return ret_dict


@D2GO_OPTIM_MAPPER_REGISTRY.register()
def sgd(
    cfg: CfgNode, model: torch.nn.Module, params: OptimizerParams
) -> torch.optim.Optimizer:
    """
    Build an optimizer from config.
    """
    return maybe_add_gradient_clipping(cfg, torch.optim.SGD)(
        params,
        cfg.SOLVER.BASE_LR,
        momentum=cfg.SOLVER.MOMENTUM,
        nesterov=cfg.SOLVER.NESTEROV,
    )


@D2GO_OPTIM_MAPPER_REGISTRY.register()
def adam(
    cfg: CfgNode, model: torch.nn.Module, params: OptimizerParams
) -> torch.optim.Optimizer:
    """
    Build an optimizer from config.
    """
    return maybe_add_gradient_clipping(cfg, torch.optim.Adam)(
        params, cfg.SOLVER.BASE_LR, betas=cfg.SOLVER.BETAS
    )


@D2GO_OPTIM_MAPPER_REGISTRY.register()
def adamw(
    cfg: CfgNode, model: torch.nn.Module, params: OptimizerParams
) -> torch.optim.Optimizer:
    """
    Build an optimizer from config.
    """
    return maybe_add_gradient_clipping(cfg, torch.optim.AdamW)(
        params, cfg.SOLVER.BASE_LR, betas=cfg.SOLVER.BETAS
    )


@D2GO_OPTIM_MAPPER_REGISTRY.register()
def sgd_mt(
    cfg: CfgNode, model: torch.nn.Module, params: OptimizerParams
) -> torch.optim.Optimizer:
    """
    Build a multi_tensor SGD optimizer that works significantly faster.
    This version is expected to be the default implementation for SGD
    optimizer by end of H1'21. To benefit from the speedup, the number
    of parameter groups needs to be reduced using `reduce_param_groups`.
    """
    return maybe_add_gradient_clipping(cfg, torch.optim._multi_tensor.SGD)(
        params,
        cfg.SOLVER.BASE_LR,
        momentum=cfg.SOLVER.MOMENTUM,
        nesterov=cfg.SOLVER.NESTEROV,
    )


@D2GO_OPTIM_MAPPER_REGISTRY.register()
def adamw_mt(
    cfg: CfgNode, model: torch.nn.Module, params: OptimizerParams
) -> torch.optim.Optimizer:
    """
    Build a multi_tensor adamw optimizer that works significantly faster.
    This version is expected to be the default implementation for adamw
    optimizer by end of H1'21. To benefit from the speedup, the number
    of parameter groups needs to be reduced using `reduce_param_groups`.
    """
    return maybe_add_gradient_clipping(cfg, torch.optim._multi_tensor.AdamW)(
        params, cfg.SOLVER.BASE_LR
    )


def get_param_groups_for_multiple_optimizers(model, cfg, optimizer_keys):
    # In case of multiple optimizers we expect user to provide param_groups
    # for each of the optimizers
    params = {
        k.lower(): param_group
        for k, param_group in get_optimizer_param_groups_from_model(
            cfg, model, required=True
        ).items()
    }
    for key in optimizer_keys:
        if key not in params:
            raise Exception(
                f"Missing parameters for optimizer: '{key.upper()}'. Make sure "
                "model.get_optimizer_param_groups returns Dict containint entries"
                " for all optimizers."
            )
    for key in optimizer_keys:
        solver_cfg = cfg.SOLVERS[key.upper()]
        params[key] += get_lr_and_weight_decay_optimizer_param_groups(
            solver_cfg, model, param_groups=params[key]
        )
        params[key] = reduce_param_groups(params[key])

    return params


def build_multiple_optimizers(
    cfg: CfgNode, model: OptimizerModelsType
) -> Dict[str, torch.optim.Optimizer]:
    """Assume cfg has "SOLVERS" section and will instantiate one more optimizer
    from that section of the config.
    """
    # In multiple optimizers case we assume that model has to implement get_optimizer_param_groups and
    # return Dict[str, OptimizerParams] where each keys match optimizer keys under
    # cfg.SOLVERS part of the config.

    optimizer_keys = [key.lower() for key in list(cfg.SOLVERS.keys())]
    params: Dict[str, OptimizerParams] = {}

    if len(optimizer_keys) == 1:
        # In case there is single solver then default set of param_groups is used.
        params[optimizer_keys[0]] = get_optimizer_param_groups(model, cfg)
    else:
        params = get_param_groups_for_multiple_optimizers(model, cfg, optimizer_keys)

    result: Dict[str, torch.optim.Optimizer] = {}
    for optimizer_key, opt_cfg in cfg.SOLVERS.items():
        name = opt_cfg.SOLVER.OPTIMIZER
        optimizer_key = optimizer_key.lower()
        if optimizer_key in result:
            raise ValueError(f"Optimizers with duplicate key: '{optimizer_key}'")
        result[optimizer_key] = D2GO_OPTIM_MAPPER_REGISTRY.get(name.lower())(
            opt_cfg, model, params=params[optimizer_key]
        )
    return result


def _param_group_str(group):
    # ret = {x: y if x != "params" else len(y) for x, y in group.items()}
    ret = {x: y if x != "params" else len(y) for x, y in group.items()}
    ret = sorted(ret.items())
    ret = [f"{x[0]}: {x[1]}" for x in ret]
    ret = "{" + ", ".join(ret) + "}"
    return ret


def _param_groups_str(groups):
    ret = ""
    for idx, group in enumerate(groups):
        ret += f"Param group {idx}: {_param_group_str(group)}\n"
    return ret


def build_optimizer_mapper(
    cfg: CfgNode, model
) -> Union[torch.optim.Optimizer, Dict[str, torch.optim.Optimizer]]:
    """Instantiates one or more optimizer depending on the config.

    By default optimizer is specified under SOLVER.
    If needed it's possible to specify more optimizers under SOLVERS section.

    TODO: add an example config.

    Returns:
        Either single optimizer object (if only one is specified) or dictionary
        of optimizers.
    """
    optimizers = {}

    if ("SOLVER" in cfg and "SOLVERS" in cfg) or (
        "SOLVER" not in cfg and "SOLVERS" not in cfg
    ):
        raise ValueError(
            f"Config must contain 'SOLVER' or 'SOLVERS' section. Config: {cfg}"
        )

    # Single solver use case
    if "SOLVER" in cfg:
        name = cfg.SOLVER.OPTIMIZER
        opt = D2GO_OPTIM_MAPPER_REGISTRY.get(name.lower())(
            cfg, model, params=get_optimizer_param_groups(model, cfg)
        )
        logger.info(f"parameter groups:\n{_param_groups_str(opt.param_groups)}")
        return opt

    # Multiple solvers use case
    optimizers.update(build_multiple_optimizers(cfg, model))

    for optimizer_key, optimizer in optimizers.items():
        logger.info(
            f"optimizer ({optimizer_key}) parameter groups:\n{_param_groups_str(optimizer.param_groups)}"
        )

    if len(optimizers) == 1:
        return list(optimizers.values())[0]

    return optimizers
