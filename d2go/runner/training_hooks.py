#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import logging
from typing import List

from aiplatform.monitoring.unitrace.memory_snapshot import (
    export_memory_snapshot,
    start_record_memory_history,
    stop_record_memory_history,
)

from d2go.config import CfgNode

from detectron2.engine.train_loop import HookBase
from detectron2.utils.registry import Registry
from mobile_cv.torch.utils_pytorch import comm


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
        log_n_steps: int = 3,
        log_during_train_at: int = 550,
        manifold_bucket: str = "d2go_traces",
        root_manifold_path: str = "tree/memory_snapshot",
    ) -> None:
        self.log_n_steps = log_n_steps
        self.log_during_train_at = log_during_train_at
        self.manifold_bucket = manifold_bucket
        self.root_manifold_path = root_manifold_path
        logger.warning(
            "WARNING: Memory snapshot profiler is enabled. This may cause ranks to die and training jobs to get stuck. Please use with caution."
        )

    def before_step(self):
        if self.trainer.iter == self.log_during_train_at:
            logger.info(
                f"[itrn-{self.trainer.iter}] Starting memory snapshot recording"
            )
            start_record_memory_history()

    def after_step(self):
        if self.trainer.iter == self.log_during_train_at + self.log_n_steps - 1:
            export_memory_snapshot(
                worker_name=f"rank-{comm.get_rank()}",
                bucket=self.manifold_bucket,
                root_manifold_path=self.root_manifold_path,
            )
            logger.info(
                f"[itrn-{self.trainer.iter}] Stopping memory snapshot recording"
            )
            stop_record_memory_history()
