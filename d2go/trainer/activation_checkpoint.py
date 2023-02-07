#!/usr/bin/env python3
# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
import logging
from functools import partial

import torch.nn as nn
from d2go.config import CfgNode as CN
from d2go.modeling import modeling_hook as mh
from d2go.registry.builtin import MODELING_HOOK_REGISTRY
from d2go.trainer.fsdp import get_module_class_from_name
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    apply_activation_checkpointing,
    checkpoint_wrapper,
    CheckpointImpl,
)


logger = logging.getLogger(__name__)


def add_activation_checkpoint_configs(_C: CN):
    _C.ACTIVATION_CHECKPOINT = CN()
    # A list of layer cls names to wrap, case sensitive
    _C.ACTIVATION_CHECKPOINT.WRAP_LAYER_CLS = []


@MODELING_HOOK_REGISTRY.register()
class ActivationCheckpointModelingHook(mh.ModelingHook):
    """Modeling hook that wraps model in activation checkpoint based on config"""

    def apply(self, model: nn.Module) -> nn.Module:
        logger.info("Activation Checkpointing is used")
        non_reentrant_wrapper = partial(
            checkpoint_wrapper,
            checkpoint_impl=CheckpointImpl.NO_REENTRANT,
        )
        layer_cls = []
        for name in self.cfg.ACTIVATION_CHECKPOINT.WRAP_LAYER_CLS:
            closure = get_module_class_from_name(model, name)
            layer_cls.append(closure)

        check_fn = lambda submodule: isinstance(submodule, tuple(layer_cls))
        apply_activation_checkpointing(
            model, checkpoint_wrapper_fn=non_reentrant_wrapper, check_fn=check_fn
        )
        return model

    def unapply(self, model: nn.Module) -> nn.Module:
        raise NotImplementedError(
            "ActivationCheckpointModelingHook.unapply() not implemented: can't unwrap an activation checkpoint module"
        )
