#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved


import os
import random

import torch
import torch.nn as nn
from d2go.quantization.modeling import QATCheckpointer
from d2go.registry.builtin import CONFIG_UPDATER_REGISTRY
from d2go.runner.default_runner import BaseRunner
from d2go.runner.defaults import (
    add_base_runner_default_cfg,
    add_tensorboard_default_configs,
)
from detectron2.utils.file_io import PathManager


@CONFIG_UPDATER_REGISTRY.register("DebugRunner")
def add_debug_runner_default_cfg(cfg):
    assert len(cfg) == 0, "start from scratch, but previous cfg is non-empty!"
    _C = add_base_runner_default_cfg(cfg)

    # _C.TENSORBOARD...
    add_tensorboard_default_configs(_C)

    # target metric
    _C.TEST.TARGET_METRIC = "dataset0:dummy0:metric1"
    return _C


class DebugRunner(BaseRunner):
    get_default_cfg = None

    def build_model(self, cfg, eval_only=False):
        return nn.Sequential()

    def do_test(self, cfg, model, train_iter=None):
        return {
            "dataset0": {
                "dummy0": {"metric0": random.random(), "metric1": random.random()}
            }
        }

    def do_train(self, cfg, model, resume):
        # save a dummy checkpoint file

        save_file = os.path.join(cfg.OUTPUT_DIR, "model_123.pth")
        with PathManager.open(save_file, "wb") as f:
            torch.save({"model": model.state_dict()}, f)

        save_file = os.path.join(cfg.OUTPUT_DIR, "model_12345.pth")
        with PathManager.open(save_file, "wb") as f:
            torch.save({"model": model.state_dict()}, f)

        save_file = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
        with PathManager.open(save_file, "wb") as f:
            torch.save({"model": model.state_dict()}, f)

    def build_checkpointer(self, cfg, model, save_dir, **kwargs):
        checkpointer = QATCheckpointer(model, save_dir=save_dir, **kwargs)
        return checkpointer

    @staticmethod
    def final_model_name():
        return "model_final"
