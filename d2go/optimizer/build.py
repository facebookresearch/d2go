#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import itertools
from typing import Any, Dict, List, Optional, Set

import torch
from detectron2.solver.build import (
    maybe_add_gradient_clipping as d2_maybe_add_gradient_clipping,
)
from detectron2.utils.registry import Registry

D2GO_OPTIM_MAPPER_REGISTRY = Registry("D2GO_OPTIM_MAPPER")


def reduce_param_groups(param_groups: List[Dict[str, Any]]):
    # The number of parameter groups needs to be as small as possible in order
    # to efficiently use the PyTorch multi-tensor optimizer. Therefore instead
    # of using a parameter_group per single parameter, we group all the params
    # with the same lr and weight_decay in a single group. This approach speeds
    # up optimizer step significantly.

    dict_new_groups: Dict[str, Dict[str, Any]] = {}

    for param_group in param_groups:
        # value is a list of parameters from the previous group
        value = param_group["params"]

        # lr and weight_decay are floating point values
        lr = param_group["lr"]
        weight_decay = param_group["weight_decay"]

        # Create the new groups using combinations of lr and weight_decay
        group_key = (lr, weight_decay)
        if group_key not in dict_new_groups:
            dict_new_groups[group_key] = {
                "params": value,
                "lr": lr,
                "weight_decay": weight_decay,
            }
        else:
            # Add elements from an existing group to the new larger group
            dict_new_groups[group_key]["params"].extend(value)

    return list(dict_new_groups.values())


def get_default_optimizer_params(
    model: torch.nn.Module,
    base_lr,
    weight_decay,
    weight_decay_norm,
    weight_decay_embed,
    bias_lr_factor=1.0,
    weight_decay_bias=None,
    use_param_group_reduction=False,
    overrides: Optional[Dict[str, Dict[str, float]]] = None,
    lr_multipliers_overwrite: Optional[Dict[str, float]] = None,
):
    """
    Get default param list for optimizer
    Args:
        overrides (dict: str -> (dict: str -> float)):
            if not `None`, provides values for optimizer hyperparameters
            (LR, weight decay) for module parameters with a given name; e.g.
            {"embedding": {"lr": 0.01, "weight_decay": 0.1}} will set the LR and
            weight decay values for all module parameters named `embedding` (default: None)
        lr_multipliers_overwrite (dict: str-> float):
            Applying different lr multiplier to a set of parameters whose names
            containing certain keys. For example, if lr_multipliers_overwrite={'backbone': 0.1},
            the LR for the parameters whose names containing 'backbone' will be scaled to 0.1x.
            Set lr_multipliers_overwrite={} if no multipliers required.
        use_param_group_reduction:
            if set to `False` we will have a parameter group for each parameter which makes
            the optimizer very slow. This option should be used when using checkpoints of models
            that were created using a parameter group for each param.
    """
    if weight_decay_bias is None:
        weight_decay_bias = weight_decay
    norm_module_types = (
        torch.nn.BatchNorm1d,
        torch.nn.BatchNorm2d,
        torch.nn.BatchNorm3d,
        torch.nn.SyncBatchNorm,
        # NaiveSyncBatchNorm inherits from BatchNorm2d
        torch.nn.GroupNorm,
        torch.nn.InstanceNorm1d,
        torch.nn.InstanceNorm2d,
        torch.nn.InstanceNorm3d,
        torch.nn.LayerNorm,
        torch.nn.LocalResponseNorm,
    )
    params: List[Dict[str, Any]] = []
    memo: Set[torch.nn.parameter.Parameter] = set()
    for module_name, module in model.named_modules():
        for module_param_name, value in module.named_parameters(recurse=False):
            if not value.requires_grad:
                continue
            # Avoid duplicating parameters
            if value in memo:
                continue
            memo.add(value)

            schedule_params = {
                "lr": base_lr,
                "weight_decay": weight_decay,
            }
            if isinstance(module, norm_module_types):
                schedule_params["weight_decay"] = weight_decay_norm
            elif module_param_name == "bias":
                # NOTE: unlike Detectron v1, we now default BIAS_LR_FACTOR to 1.0
                # and WEIGHT_DECAY_BIAS to WEIGHT_DECAY so that bias optimizer
                # hyperparameters are by default exactly the same as for regular
                # weights.
                schedule_params["lr"] = base_lr * bias_lr_factor
                schedule_params["weight_decay"] = weight_decay_bias
            if isinstance(module, torch.nn.Embedding):
                schedule_params["weight_decay"] = weight_decay_embed
            if overrides is not None and module_param_name in overrides:
                schedule_params.update(overrides[module_param_name])
            if lr_multipliers_overwrite is not None:
                for kname, mult in lr_multipliers_overwrite.items():
                    if kname in module_name:
                        # apply multiplier for the params containing kname, e.g. backbone
                        schedule_params["lr"] = schedule_params["lr"] * mult
            params += [
                {
                    "params": [value],
                    "lr": schedule_params["lr"],
                    "weight_decay": schedule_params["weight_decay"],
                }
            ]

    if use_param_group_reduction:
        # Reduce number of param groups to speed-up optimizer step
        return reduce_param_groups(params)

    return params


def maybe_add_gradient_clipping(cfg, optim):  # optim: the optimizer class
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
def sgd(cfg, model: torch.nn.Module) -> torch.optim.Optimizer:
    """
    Build an optimizer from config.
    """
    params = get_default_optimizer_params(
        model,
        base_lr=cfg.SOLVER.BASE_LR,
        weight_decay=cfg.SOLVER.WEIGHT_DECAY,
        weight_decay_norm=cfg.SOLVER.WEIGHT_DECAY_NORM,
        weight_decay_embed=cfg.SOLVER.WEIGHT_DECAY_EMBED,
        bias_lr_factor=cfg.SOLVER.BIAS_LR_FACTOR,
        weight_decay_bias=cfg.SOLVER.WEIGHT_DECAY_BIAS,
        lr_multipliers_overwrite=_merge_dict(cfg.SOLVER.LR_MULTIPLIER_OVERWRITE),
    )
    return maybe_add_gradient_clipping(cfg, torch.optim.SGD)(
        params,
        cfg.SOLVER.BASE_LR,
        momentum=cfg.SOLVER.MOMENTUM,
        nesterov=cfg.SOLVER.NESTEROV,
    )


@D2GO_OPTIM_MAPPER_REGISTRY.register()
def adamw(cfg, model: torch.nn.Module) -> torch.optim.Optimizer:
    """
    Build an optimizer from config.
    """
    params = get_default_optimizer_params(
        model,
        base_lr=cfg.SOLVER.BASE_LR,
        weight_decay=cfg.SOLVER.WEIGHT_DECAY,
        weight_decay_norm=cfg.SOLVER.WEIGHT_DECAY_NORM,
        weight_decay_embed=cfg.SOLVER.WEIGHT_DECAY_EMBED,
        bias_lr_factor=cfg.SOLVER.BIAS_LR_FACTOR,
        weight_decay_bias=cfg.SOLVER.WEIGHT_DECAY_BIAS,
        lr_multipliers_overwrite=_merge_dict(cfg.SOLVER.LR_MULTIPLIER_OVERWRITE),
    )
    return maybe_add_gradient_clipping(cfg, torch.optim.AdamW)(
        params, cfg.SOLVER.BASE_LR
    )


@D2GO_OPTIM_MAPPER_REGISTRY.register()
def sgd_mt(cfg, model: torch.nn.Module) -> torch.optim.Optimizer:
    """
    Build a multi_tensor SGD optimizer that works significantly faster.
    This version is expected to be the default implementation for SGD
    optimizer by end of H1'21. To benefit from the speedup, the number
    of parameter groups needs to be reduced using `reduce_param_groups`.
    """
    params = get_default_optimizer_params(
        model,
        base_lr=cfg.SOLVER.BASE_LR,
        weight_decay=cfg.SOLVER.WEIGHT_DECAY,
        weight_decay_norm=cfg.SOLVER.WEIGHT_DECAY_NORM,
        weight_decay_embed=cfg.SOLVER.WEIGHT_DECAY_EMBED,
        bias_lr_factor=cfg.SOLVER.BIAS_LR_FACTOR,
        use_param_group_reduction=True,
        weight_decay_bias=cfg.SOLVER.WEIGHT_DECAY_BIAS,
        lr_multipliers_overwrite=_merge_dict(cfg.SOLVER.LR_MULTIPLIER_OVERWRITE),
    )
    return maybe_add_gradient_clipping(cfg, torch.optim._multi_tensor.SGD)(
        params,
        cfg.SOLVER.BASE_LR,
        momentum=cfg.SOLVER.MOMENTUM,
        nesterov=cfg.SOLVER.NESTEROV,
    )


@D2GO_OPTIM_MAPPER_REGISTRY.register()
def adamw_mt(cfg, model: torch.nn.Module) -> torch.optim.Optimizer:
    """
    Build a multi_tensor adamw optimizer that works significantly faster.
    This version is expected to be the default implementation for adamw
    optimizer by end of H1'21. To benefit from the speedup, the number
    of parameter groups needs to be reduced using `reduce_param_groups`.
    """
    params = get_default_optimizer_params(
        model,
        base_lr=cfg.SOLVER.BASE_LR,
        weight_decay=cfg.SOLVER.WEIGHT_DECAY,
        weight_decay_norm=cfg.SOLVER.WEIGHT_DECAY_NORM,
        weight_decay_embed=cfg.SOLVER.WEIGHT_DECAY_EMBED,
        bias_lr_factor=cfg.SOLVER.BIAS_LR_FACTOR,
        use_param_group_reduction=True,
        weight_decay_bias=cfg.SOLVER.WEIGHT_DECAY_BIAS,
        lr_multipliers_overwrite=_merge_dict(cfg.SOLVER.LR_MULTIPLIER_OVERWRITE),
    )
    return maybe_add_gradient_clipping(cfg, torch.optim._multi_tensor.AdamW)(
        params, cfg.SOLVER.BASE_LR
    )


def build_optimizer_mapper(cfg, model):
    name = cfg.SOLVER.OPTIMIZER
    return D2GO_OPTIM_MAPPER_REGISTRY.get(name.lower())(cfg, model)
