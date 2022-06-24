#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved


import os
import random

import torch
import torch.nn as nn
from d2go.quantization.modeling import QATCheckpointer
from d2go.runner.default_runner import BaseRunner
from d2go.utils.visualization import add_tensorboard_default_configs
from detectron2.utils.file_io import PathManager


class DebugRunner(BaseRunner):
    @classmethod
    def get_default_cfg(cls):
        _C = super().get_default_cfg()

        # _C.TENSORBOARD...
        add_tensorboard_default_configs(_C)

        # target metric
        _C.TEST.TARGET_METRIC = "dataset0:dummy0:metric1"
        return _C

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
