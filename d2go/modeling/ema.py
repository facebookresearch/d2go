#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.

import logging

from d2go.config import CfgNode
from d2go.modeling import modeling_hook as mh
from d2go.registry.builtin import MODELING_HOOK_REGISTRY
from d2go.utils.ema_state import EMAState
from torch import nn

logger = logging.getLogger(__name__)


@MODELING_HOOK_REGISTRY.register("EMA")
class EMAModelingHook(mh.ModelingHook):
    """Modeling hook to attach"""

    def __init__(self, cfg: CfgNode):
        super().__init__(cfg)

        self.cfg = cfg

        # "EMA" will be always specified in MODEL.MODELING_HOOKS in config
        self.ema_enabled = cfg.MODEL_EMA.ENABLED

        if not self.ema_enabled:
            return

        self.ema_state = EMAState(
            decay=cfg.MODEL_EMA.DECAY, device=cfg.MODEL_EMA.DEVICE or cfg.MODEL.DEVICE
        )

    def apply(self, model: nn.Module) -> nn.Module:
        """If EMA is not enabled, this is no-op, returning original model."""
        if not self.ema_enabled:
            return model
        model.ema_state = self.ema_state

        return model

    def unapply(self, model: nn.Module) -> nn.Module:
        if not self.ema_enabled:
            return model
        del model.ema_state
        return model
