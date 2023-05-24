#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import logging
from typing import List

from d2go.config import CfgNode

from d2go.utils.gpu_memory_profiler import log_memory_snapshot, record_memory_history

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


class D2GoGpuMemorySnapshot(HookBase):
    """
    A profiler that logs GPU memory snapshot during training.
    There are three places that logging could happen:
    1. start of training
        d2go records memory snapshots before model instantiation and logs snapshots after `log_n_steps` iterations.
        This is to capture the typical memory peak at model instantiation and the first few iterations
    2. during training
        d2go records memory snapshots at `log_during_train_at` iteration and logs snapshots after `log_n_steps` iterations.
        This is to capture the stabilized memory utilization during training.
    3. OOM
        Right before OOM, the GPU memory snapshot will be logged to help diagnose OOM issues.
    """

    def __init__(
        self,
        output_dir,
        log_n_steps: int = 3,
        log_during_train_at: int = 550,
        trace_max_entries: int = 1000000,
    ) -> None:
        self.output_dir = output_dir
        self.step = 0
        self.log_n_steps = log_n_steps
        self.log_during_train_at = log_during_train_at
        self.trace_max_entries = trace_max_entries

    def before_step(self):
        if self.trainer.iter == self.log_during_train_at:
            record_memory_history(self.trace_max_entries)

    def after_step(self):
        if self.step == self.log_n_steps - 1:
            log_memory_snapshot(self.output_dir, file_prefix=f"iter{self.trainer.iter}")

        if self.trainer.iter == self.log_during_train_at + self.log_n_steps - 1:
            log_memory_snapshot(self.output_dir, file_prefix=f"iter{self.trainer.iter}")

        self.step += 1
