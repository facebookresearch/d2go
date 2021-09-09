#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved


import os
import tempfile
import unittest

import d2go.projects.detr.runner as oss_runner
import d2go.runner.default_runner as default_runner
from d2go.utils.testing.data_loader_helper import create_local_dataset

# RUN:
# buck test mobile-vision/d2go/projects_oss/detr:test_detr_runner


def _get_cfg(runner, output_dir, dataset_name):
    cfg = runner.get_default_cfg()
    cfg.MODEL.DEVICE = "cpu"
    cfg.MODEL.META_ARCHITECTURE = "Detr"

    cfg.DATASETS.TRAIN = (dataset_name,)
    cfg.DATASETS.TEST = (dataset_name,)

    cfg.INPUT.MIN_SIZE_TRAIN = (10,)
    cfg.INPUT.MIN_SIZE_TEST = (10,)

    cfg.SOLVER.MAX_ITER = 5
    cfg.SOLVER.STEPS = []
    cfg.SOLVER.WARMUP_ITERS = 1
    cfg.SOLVER.CHECKPOINT_PERIOD = 1
    cfg.SOLVER.IMS_PER_BATCH = 2

    cfg.OUTPUT_DIR = output_dir

    return cfg


class TestOssRunner(unittest.TestCase):
    def test_build_model(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            ds_name = create_local_dataset(tmp_dir, 5, 10, 10)
            runner = oss_runner.DETRRunner()
            cfg = _get_cfg(runner, tmp_dir, ds_name)
            model = runner.build_model(cfg)
            dl = runner.build_detection_train_loader(cfg)
            batch = next(iter(dl))
            output = model(batch)
            self.assertIsInstance(output, dict)

            model.eval()
            output = model(batch)
            self.assertIsInstance(output, list)
            default_runner._close_all_tbx_writers()

    def test_runner_train(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            ds_name = create_local_dataset(tmp_dir, 5, 10, 10, num_classes=1000)
            runner = oss_runner.DETRRunner()
            cfg = _get_cfg(runner, tmp_dir, ds_name)
            model = runner.build_model(cfg)

            runner.do_train(cfg, model, True)
            final_model_path = os.path.join(tmp_dir, "model_final.pth")
            self.assertTrue(os.path.isfile(final_model_path))
            default_runner._close_all_tbx_writers()
