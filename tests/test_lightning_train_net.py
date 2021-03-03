#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import os
import tempfile
import unittest

import numpy as np
from d2go.config import CfgNode
from d2go.config.utils import flatten_config_dict
from d2go.runner.lightning_task import GeneralizedRCNNTask
from d2go.tests import meta_arch_helper as mah
from d2go.tests.helper import tempdir
from d2go.tools.lightning_train_net import main, FINAL_MODEL_CKPT


class TestLightningTrainNet(unittest.TestCase):
    def _get_cfg(self, tmp_dir) -> CfgNode:
        return mah.create_detection_cfg(GeneralizedRCNNTask, tmp_dir)

    @tempdir
    def test_train_net_main(self, root_dir):
        """ tests the main training entry point. """
        cfg = self._get_cfg(root_dir)
        # set distributed backend to none to avoid spawning child process,
        # which doesn't inherit the temporary dataset
        main(cfg, accelerator=None)

    @tempdir
    def test_checkpointing(self, tmp_dir):
        """ tests saving and loading from checkpoint. """
        cfg = self._get_cfg(tmp_dir)

        out = main(cfg, accelerator=None)
        ckpts = [file for file in os.listdir(tmp_dir) if file.endswith(".ckpt")]
        self.assertCountEqual(
            [
                "last.ckpt",
                FINAL_MODEL_CKPT,
            ],
            ckpts,
        )

        with tempfile.TemporaryDirectory() as tmp_dir2:
            cfg2 = cfg.clone()
            cfg2.defrost()
            cfg2.OUTPUT_DIR = tmp_dir2
            # load the last checkpoint from previous training
            cfg2.MODEL.WEIGHTS = os.path.join(tmp_dir, "last.ckpt")

            out2 = main(cfg2, accelerator=None, eval_only=True)
            accuracy = flatten_config_dict(out.accuracy)
            accuracy2 = flatten_config_dict(out2.accuracy)
            for k in accuracy:
                np.testing.assert_equal(accuracy[k], accuracy2[k])
