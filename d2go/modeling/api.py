#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import logging
from dataclasses import dataclass
from typing import List

import torch
import torch.nn as nn

from d2go.config import CfgNode
from d2go.modeling import modeling_hook as mh
from d2go.registry.builtin import META_ARCH_REGISTRY
from d2go.utils.misc import _log_api_usage
from detectron2.modeling import META_ARCH_REGISTRY as D2_META_ARCH_REGISTRY

logger = logging.getLogger(__name__)


@dataclass
class D2GoModelBuildResult:
    """Class to store the output of build_d2go_model.
    It stores the model, a key-value mapping of modeling hooks and can be further
    extended with other fields, e.g. state_dict.
    """

    # Stores model with applied modeling hooks.
    # If modeling hooks (e.g. EMA) are not enabled in config
    # the modeling hook will be no-op (e.g. return original model)
    model: nn.Module
    modeling_hooks: List[mh.ModelingHook]


def build_meta_arch(cfg):
    """
    Build the whole model architecture, defined by ``cfg.MODEL.META_ARCHITECTURE``.
    Note that it does not load any weights from ``cfg``.
    """

    # initialize the meta-arch and cast to the device
    meta_arch = cfg.MODEL.META_ARCHITECTURE
    # NOTE: during transition we also check if meta_arch is registered as D2 MetaArch
    # TODO: remove this check after Sep 2022.
    if meta_arch not in META_ARCH_REGISTRY and meta_arch in D2_META_ARCH_REGISTRY:
        raise KeyError(
            f"Can't find '{meta_arch}' in D2Go's META_ARCH_REGISTRY, although it is in"
            f" D2's META_ARCH_REGISTRY, now D2Go uses its own registry, please register"
            f" it in D2Go's META_ARCH_REGISTRY."
        )
    model = META_ARCH_REGISTRY.get(meta_arch)(cfg)
    model.to(torch.device(cfg.MODEL.DEVICE))

    _log_api_usage("modeling.meta_arch." + meta_arch)
    return model


def build_d2go_model(
    cfg: CfgNode,
) -> D2GoModelBuildResult:
    # NOTE distributed initialization path (using FSDP) for large models
    if (
        hasattr(cfg.MODEL, "MODELING_HOOKS")
        and "FSDPModelingHook" in cfg.MODEL.MODELING_HOOKS
        and hasattr(cfg, "FSDP")
        and hasattr(cfg.FSDP, "DISTRIBUTED_INIT")
        and cfg.FSDP.DISTRIBUTED_INIT
    ):
        logger.info("Using distributed initialization path.")
        import torch.distributed as dist

        if dist.is_initialized():
            from d2go.trainer.fsdp import CpuOverrideMode
            from torch._subclasses import FakeTensorMode

            # NOTE (global) rank 0 will build the whole model on cpu
            # other ranks will build the model on fake tensors
            if dist.get_rank() == 0:
                with CpuOverrideMode():
                    model = build_meta_arch(cfg)
            else:
                with FakeTensorMode(allow_non_fake_inputs=True):
                    model = build_meta_arch(cfg)
        else:
            raise RuntimeError(
                "torch.distributed is not initialized. cannot process with distributed init."
            )
    else:
        model = build_meta_arch(cfg)
    modeling_hooks: List[mh.ModelingHook] = []
    # apply modeling hooks
    # some custom projects bypass d2go's default config so may not have the
    # MODELING_HOOKS key
    if hasattr(cfg.MODEL, "MODELING_HOOKS"):
        hook_names = cfg.MODEL.MODELING_HOOKS
        model, modeling_hooks = mh.build_and_apply_modeling_hooks(
            model, cfg, hook_names
        )
    return D2GoModelBuildResult(model=model, modeling_hooks=modeling_hooks)
