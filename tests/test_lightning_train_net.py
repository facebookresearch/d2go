#!/usr/bin/env python3

import os
import tempfile
import unittest

import d2go.runner.default_runner as default_runner
import torch
from d2go.config import CfgNode
from d2go.tools.lightning_train_net import main

from . import meta_arch_helper as mah


class TestLightningTrainNet(unittest.TestCase):
    def _get_cfg(self, tmp_dir) -> CfgNode:
        runner = default_runner.Detectron2GoRunner()
        cfg = mah.create_detection_cfg(runner, tmp_dir)
        cfg.TEST.EVAL_PERIOD = cfg.SOLVER.MAX_ITER
        return cfg

    def test_train_net_main(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            cfg = self._get_cfg(tmp_dir)
            # set distributed backend to none to avoid spawning child process,
            # which doesn't inherit the temporary dataset
            main(cfg, accelerator=None)

    def test_qat_config(self):
        """ Sanity test to validate training occurs when using QAT. """
        with tempfile.TemporaryDirectory() as tmp_dir:
            cfg = self._get_cfg(tmp_dir)
            cfg.SOLVER.MAX_ITER = 5
            cfg.SOLVER.STEPS = []
            cfg.SOLVER.CHECKPOINT_PERIOD = 1
            cfg.QUANTIZATION.QAT.ENABLED = True
            cfg.QUANTIZATION.QAT.START_ITER = 0
            cfg.QUANTIZATION.QAT.FREEZE_BN_ITER = 3
            cfg.QUANTIZATION.QAT.FREEZE_BN_ITER = 3
            cfg.QUANTIZATION.QAT.ENABLE_OBSERVER_ITER = 1
            cfg.QUANTIZATION.MODULES = [
                # "model.conv", # raises "RuntimeError: dimensions of scale and zero-point are not consistent with input tensor"
                # "model.bn", # Not a traceable layer.
                "model.relu",
                "model.avgpool",
            ]

            main(cfg, accelerator=None)

            ckpts = [file for file in os.listdir(tmp_dir) if file.endswith(".ckpt")]
            self.assertCountEqual(
                [
                    "step=0.ckpt",
                    "last.ckpt",
                    "step=1.ckpt",
                    "step=2.ckpt",
                    "step=3.ckpt",
                    "step=4.ckpt",
                ],
                ckpts,
            )

    def test_checkpointing(self):
        """ Test Model Checkpointing """
        with tempfile.TemporaryDirectory() as tmp_dir:
            cfg = self._get_cfg(tmp_dir)
            cfg.SOLVER.MAX_ITER = 3
            cfg.SOLVER.STEPS = []
            cfg.SOLVER.CHECKPOINT_PERIOD = 1

            main(cfg, accelerator=None)
            ckpts = [file for file in os.listdir(tmp_dir) if file.endswith(".ckpt")]
            self.assertCountEqual(
                ["step=0.ckpt", "last.ckpt", "step=1.ckpt", "step=2.ckpt"], ckpts
            )

            with tempfile.TemporaryDirectory() as tmp_dir2:
                cfg2 = cfg.clone()
                cfg2.defrost()
                cfg2.OUTPUT_DIR = tmp_dir2
                cfg2.SOLVER.MAX_ITER = 1
                # load the second from the last checkpoint
                cfg2.MODEL.WEIGHTS = os.path.join(tmp_dir, "step=1.ckpt")

                main(cfg2, accelerator=None)
                ckpts = [
                    file for file in os.listdir(tmp_dir2) if file.endswith(".ckpt")
                ]
                self.assertCountEqual(["step=0.ckpt", "last.ckpt"], ckpts)

                # the last checkpoints should be the same
                ckpt = torch.load(os.path.join(tmp_dir, "last.ckpt"))
                ckpt2 = torch.load(os.path.join(tmp_dir2, "last.ckpt"))
                for k, v in ckpt["state_dict"].items():
                    self.assertTrue(torch.allclose(v, ckpt2["state_dict"][k]))
