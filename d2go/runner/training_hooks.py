#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import logging
from typing import List

from d2go.config import CfgNode

from detectron2.engine.train_loop import HookBase
from detectron2.utils.registry import Registry


logger = logging.getLogger(__name__)

# List of functions to add hooks for trainer, all functions in the registry will
# be called to add hooks
#   func(hooks: List[HookBase]) -> None
TRAINER_HOOKS_REGISTRY = Registry("TRAINER_HOOKS_REGISTRY")


def update_hooks_from_registry(hooks: List[HookBase], cfg: CfgNode):
    for name, hook_func in TRAINER_HOOKS_REGISTRY:
        logger.info(f"Update trainer hooks from {name}...")
        hook_func(hooks, cfg)
