#!/usr/bin/env python3

import math
import os
import tempfile
import unittest

import d2go.runner.default_runner as default_runner
import mobile_cv.common.misc.iter_utils as iu
import mock
from d2go.config import auto_scale_world_size
from d2go.tools import train_net
from detectron2.checkpoint import DetectionCheckpointer

from . import helper, meta_arch_helper as mah


def _compare_dict_with_nan(dict1, dict2):
    def _equal(lhs, rhs):
        return lhs == rhs or (math.isnan(lhs) and math.isnan(rhs))

    return all(
        _equal(x.lhs, x.rhs) for x in iu.recursive_iterate(iu.create_pair(dict1, dict2))
    )


class TestToolsTrainNet(unittest.TestCase):
    def test_train_net_main(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            runner = default_runner.Detectron2GoRunner()
            cfg = mah.create_detection_cfg(runner, tmp_dir)

            with mock.patch(
                "d2go.setup.auto_scale_world_size",
                side_effect=auto_scale_world_size,
            ) as m:
                results = train_net.main(cfg, tmp_dir, runner, eval_only=False)
            self.assertEqual(m.call_count, 1)  # auto_scale_world_size should be called once in the main()
            self.assertIn("model_configs", results)
            self.assertTrue(os.path.isfile(list(results["model_configs"].values())[0]))

            results_eval = train_net.main(cfg, tmp_dir, runner, eval_only=True)
            self.assertTrue(
                _compare_dict_with_nan(results["accuracy"], results_eval["accuracy"])
            )
            default_runner._close_all_tbx_writers()

    def test_train_net_main_ema(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            runner = default_runner.Detectron2GoRunner()
            cfg = mah.create_detection_cfg(runner, tmp_dir)
            cfg.MODEL_EMA.ENABLED = True
            cfg.MODEL_EMA.DECAY = 0.0

            results = train_net.main(cfg, tmp_dir, runner, eval_only=False)

            # output dir for evaluation, specify this to force the train_net
            #   not to load the checkpoints, but load weights from MODEL.WEIGHTS
            eval_wd = os.path.join(tmp_dir, "eval_wd")
            os.makedirs(eval_wd)

            # make sure model ema weights are loaded properly
            cfg.defrost()
            cfg.merge_from_file(list(results["model_configs"].values())[0])
            cfg.OUTPUT_DIR = eval_wd
            results_eval = train_net.main(cfg, eval_wd, runner, eval_only=True)

            # one dataset, but with/without ema
            self.assertEqual(len(results["accuracy"]), 2)
            self.assertTrue(
                _compare_dict_with_nan(results["accuracy"], results_eval["accuracy"])
            )
            default_runner._close_all_tbx_writers()

    @helper.skip_if_no_gpu
    def test_train_net_main_ema_resume_cuda(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            runner = default_runner.Detectron2GoRunner()
            cfg = mah.create_detection_cfg(runner, tmp_dir)
            cfg.MODEL.DEVICE = "cuda"
            cfg.MODEL_EMA.ENABLED = True
            cfg.MODEL_EMA.DECAY = 0.0

            results = train_net.main(cfg, tmp_dir, runner, eval_only=False)

            # remove last two checkpoints, to simluate interrupt in training
            _remove_checkpoints(tmp_dir, 3)

            print("Resuming training...")
            # resume training
            results1 = train_net.main(cfg, tmp_dir, runner, eval_only=False)

            # one dataset, but with/without ema
            self.assertEqual(len(results["accuracy"]), 2)
            self.assertTrue(
                _compare_dict_with_nan(results["accuracy"], results1["accuracy"])
            )
            default_runner._close_all_tbx_writers()


def _remove_checkpoints(path, checkpoints_to_keep):
    assert checkpoints_to_keep > 0
    cps = DetectionCheckpointer(None, save_dir=path)
    all_checkpoints = cps.get_all_checkpoint_files()
    assert len(all_checkpoints) >= checkpoints_to_keep

    for idx in range(checkpoints_to_keep, len(all_checkpoints)):
        os.remove(all_checkpoints[idx])

    last_checkpoint_file = os.path.join(path, "last_checkpoint")
    with open(last_checkpoint_file, "w") as f:
        f.write(all_checkpoints[checkpoints_to_keep - 1])
