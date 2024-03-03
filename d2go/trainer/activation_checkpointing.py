#!/usr/bin/env python3
# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
import logging
from functools import partial

import torch.nn as nn
from d2go.config import CfgNode as CN
from d2go.modeling import modeling_hook as mh
from d2go.registry.builtin import MODELING_HOOK_REGISTRY
from d2go.trainer.helper import D2GO_WRAP_POLICY_REGISTRY
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    apply_activation_checkpointing,
    checkpoint_wrapper,
    CheckpointImpl,
)


logger = logging.getLogger(__name__)


def add_activation_checkpoint_configs(_C: CN):
    _C.ACTIVATION_CHECKPOINT = CN()
    _C.ACTIVATION_CHECKPOINT.REENTRANT = False
    # Find autowrap policy at D2GO_WRAP_POLICY_REGISTRY, or use '' to disable autowrap
    _C.ACTIVATION_CHECKPOINT.AUTO_WRAP_POLICY = "always_wrap_policy"
    # A list of layer cls names to wrap, case sensitive
    _C.ACTIVATION_CHECKPOINT.AUTO_WRAP_LAYER_CLS = []


@MODELING_HOOK_REGISTRY.register()
class ActivationCheckpointModelingHook(mh.ModelingHook):
    """Modeling hook that wraps model in activation checkpoint based on config"""

    def apply(self, model: nn.Module) -> nn.Module:
        logger.info("Activation Checkpointing is used")
        wrapper_fn = partial(
            checkpoint_wrapper,
            checkpoint_impl=(
                CheckpointImpl.NO_REENTRANT
                if not self.cfg.ACTIVATION_CHECKPOINT.REENTRANT
                else CheckpointImpl.REENTRANT
            ),
        )
        policy_name = self.cfg.ACTIVATION_CHECKPOINT.AUTO_WRAP_POLICY
        assert (
            policy_name != "size_based_auto_wrap_policy"
        ), "ActivationCheckpointing should always be wrapped at module boundary"
        policy_kwargs = {
            "layer_names": self.cfg.ACTIVATION_CHECKPOINT.AUTO_WRAP_LAYER_CLS,
        }
        auto_wrap_policy = (
            D2GO_WRAP_POLICY_REGISTRY.get(policy_name)(model, **policy_kwargs)
            if policy_name != ""
            else lambda _: True
        )

        apply_activation_checkpointing(
            model, checkpoint_wrapper_fn=wrapper_fn, auto_wrap_policy=auto_wrap_policy
        )
        return model

    def unapply(self, model: nn.Module) -> nn.Module:
        raise NotImplementedError(
            "ActivationCheckpointModelingHook.unapply() not implemented: can't unwrap an activation checkpoint module"
        )
