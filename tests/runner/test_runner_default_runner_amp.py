#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import os
import tempfile
import unittest

import d2go.runner.default_runner as default_runner
import torch
from d2go.registry.builtin import META_ARCH_REGISTRY
from d2go.utils.testing.data_loader_helper import create_local_dataset
from detectron2.structures import Boxes, Instances


@META_ARCH_REGISTRY.register()
class MetaArchForTestSingleValueAMP(torch.nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.scale_weight = torch.nn.Parameter(torch.Tensor([1.0]))
        self.counter = 0

    @property
    def device(self):
        return self.scale_weight.device

    def forward(self, inputs):
        if not self.training:
            return self.inference(inputs)

        ret = {"loss": self.scale_weight.norm() * 10.0}
        if self.counter not in [2, 6]:
            ret["loss"] = ret["loss"] / 0.0
        print(f"Iter {self.counter}: scale_weight={self.scale_weight}")
        print(f"Iter {self.counter}: loss={ret}")
        self.counter += 1
        return ret

    def inference(self, inputs):
        instance = Instances((10, 10))
        instance.pred_boxes = Boxes(
            torch.tensor([[2.5, 2.5, 7.5, 7.5]], device=self.device) * self.scale_weight
        )
        instance.scores = torch.tensor([0.9])
        instance.pred_classes = torch.tensor([1], dtype=torch.int32)
        ret = [{"instances": instance}]
        return ret


def _get_cfg(runner, output_dir, dataset_name):
    cfg = runner.get_default_cfg()
    cfg.MODEL.DEVICE = "cuda"
    cfg.MODEL.META_ARCHITECTURE = "MetaArchForTestSingleValueAMP"

    cfg.DATASETS.TRAIN = (dataset_name,)
    cfg.DATASETS.TEST = (dataset_name,)

    cfg.INPUT.MIN_SIZE_TRAIN = (10,)
    cfg.INPUT.MIN_SIZE_TEST = (10,)

    cfg.SOLVER.MAX_ITER = 20
    cfg.SOLVER.STEPS = []
    cfg.SOLVER.WARMUP_ITERS = 1
    cfg.SOLVER.CHECKPOINT_PERIOD = 100000
    cfg.SOLVER.IMS_PER_BATCH = 2
    cfg.WRITER_PERIOD = 1

    cfg.OUTPUT_DIR = output_dir

    return cfg


class TestDefaultRunnerAMP(unittest.TestCase):
    def test_d2go_runner_train_amp(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            ds_name = create_local_dataset(tmp_dir, 5, 10, 10)
            runner = default_runner.Detectron2GoRunner()
            cfg = _get_cfg(runner, tmp_dir, ds_name)
            cfg.SOLVER.AMP.ENABLED = True

            model = runner.build_model(cfg)
            runner.do_train(cfg, model, resume=True)
            final_model_path = os.path.join(tmp_dir, "model_final.pth")
            self.assertTrue(os.path.isfile(final_model_path))
            default_runner._close_all_tbx_writers()
