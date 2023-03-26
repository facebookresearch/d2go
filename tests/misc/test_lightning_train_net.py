#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import os
import unittest

import numpy as np
from d2go.config import CfgNode
from d2go.config.utils import flatten_config_dict
from d2go.runner.lightning_task import GeneralizedRCNNTask
from d2go.tools.lightning_train_net import FINAL_MODEL_CKPT, main
from d2go.utils.testing import meta_arch_helper as mah
from d2go.utils.testing.helper import enable_ddp_env, tempdir


class TestLightningTrainNet(unittest.TestCase):
    def setUp(self):
        # set distributed backend to none to avoid spawning child process,
        # which doesn't inherit the temporary dataset
        patcher = unittest.mock.patch(
            "d2go.tools.lightning_train_net._get_accelerator", return_value=None
        )
        self.addCleanup(patcher.stop)
        patcher.start()

    def _get_cfg(self, tmp_dir) -> CfgNode:
        return mah.create_detection_cfg(GeneralizedRCNNTask, tmp_dir)

    @tempdir
    @enable_ddp_env()
    def test_train_net_main(self, root_dir):
        """tests the main training entry point."""
        cfg = self._get_cfg(root_dir)
        # set distributed backend to none to avoid spawning child process,
        # which doesn't inherit the temporary dataset
        main(cfg, root_dir, GeneralizedRCNNTask)

    @tempdir
    @enable_ddp_env()
    def test_checkpointing(self, tmp_dir):
        """tests saving and loading from checkpoint."""
        cfg = self._get_cfg(tmp_dir)

        out = main(cfg, tmp_dir, GeneralizedRCNNTask)
        ckpts = [f for f in os.listdir(tmp_dir) if f.endswith(".ckpt")]
        expected_ckpts = ("last.ckpt", FINAL_MODEL_CKPT)
        for ckpt in expected_ckpts:
            self.assertIn(ckpt, ckpts)

        cfg2 = cfg.clone()
        cfg2.defrost()
        # load the last checkpoint from previous training
        cfg2.MODEL.WEIGHTS = os.path.join(tmp_dir, "last.ckpt")

        output_dir = os.path.join(tmp_dir, "output")
        out2 = main(cfg2, output_dir, GeneralizedRCNNTask, eval_only=True)
        accuracy = flatten_config_dict(out.accuracy)
        accuracy2 = flatten_config_dict(out2.accuracy)
        for k in accuracy:
            np.testing.assert_equal(accuracy[k], accuracy2[k])
