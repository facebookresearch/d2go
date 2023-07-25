#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved


import copy
import os
import tempfile
import unittest

import d2go.runner.default_runner as default_runner
import torch
from d2go.registry.builtin import META_ARCH_REGISTRY
from d2go.runner import create_runner
from d2go.runner.training_hooks import TRAINER_HOOKS_REGISTRY
from d2go.utils.testing import helper
from d2go.utils.testing.data_loader_helper import create_local_dataset
from detectron2.evaluation import COCOEvaluator, RotatedCOCOEvaluator
from detectron2.structures import Boxes, ImageList, Instances
from mobile_cv.arch.quantization.qconfig import (
    updateable_symmetric_moving_avg_minmax_config,
)
from torch.nn.parallel import DistributedDataParallel


@META_ARCH_REGISTRY.register()
class MetaArchForTest(torch.nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 4, kernel_size=3, stride=1, padding=1)
        self.bn = torch.nn.BatchNorm2d(4)
        self.relu = torch.nn.ReLU(inplace=True)
        self.avgpool = torch.nn.AdaptiveAvgPool2d((1, 1))

    @property
    def device(self):
        return self.conv.weight.device

    def forward(self, inputs):
        if not self.training:
            return self.inference(inputs)

        images = [x["image"] for x in inputs]
        images = ImageList.from_tensors(images, 1).to(self.device)
        ret = self.conv(images.tensor)
        ret = self.bn(ret)
        ret = self.relu(ret)
        ret = self.avgpool(ret)
        return {"loss": ret.norm()}

    def inference(self, inputs):
        instance = Instances((10, 10))
        instance.pred_boxes = Boxes(torch.tensor([[2.5, 2.5, 7.5, 7.5]]))
        instance.scores = torch.tensor([0.9])
        instance.pred_classes = torch.tensor([1], dtype=torch.int32)
        ret = [{"instances": instance}]
        return ret


@META_ARCH_REGISTRY.register()
class MetaArchForTestSingleValue(torch.nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.scale_weight = torch.nn.Parameter(torch.Tensor([1.0]))

    @property
    def device(self):
        return self.scale_weight.device

    def forward(self, inputs):
        if not self.training:
            return self.inference(inputs)

        ret = {"loss": self.scale_weight.norm() * 10.0}
        print(self.scale_weight)
        print(ret)
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
    cfg.MODEL.DEVICE = "cpu"
    cfg.MODEL.META_ARCHITECTURE = "MetaArchForTest"

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


class TestDefaultRunner(unittest.TestCase):
    def test_d2go_runner_build_model(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            ds_name = create_local_dataset(tmp_dir, 5, 10, 10)
            runner = default_runner.Detectron2GoRunner()
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

    def test_d2go_runner_train(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            ds_name = create_local_dataset(tmp_dir, 5, 10, 10)
            runner = default_runner.Detectron2GoRunner()
            cfg = _get_cfg(runner, tmp_dir, ds_name)

            model = runner.build_model(cfg)
            runner.do_train(cfg, model, resume=True)
            final_model_path = os.path.join(tmp_dir, "model_final.pth")
            self.assertTrue(os.path.isfile(final_model_path))
            default_runner._close_all_tbx_writers()

    def test_d2go_runner_test(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            ds_name = create_local_dataset(tmp_dir, 5, 10, 10)
            runner = default_runner.Detectron2GoRunner()
            cfg = _get_cfg(runner, tmp_dir, ds_name)

            model = runner.build_model(cfg)
            results = runner.do_test(cfg, model)
            self.assertEqual(results["default"][ds_name]["bbox"]["AP"], 10.0)
            default_runner._close_all_tbx_writers()

    def test_d2go_build_evaluator(self):
        for rotated, evaluator in [
            (True, RotatedCOCOEvaluator),
            (False, COCOEvaluator),
        ]:
            with tempfile.TemporaryDirectory() as tmp_dir:
                ds_name = create_local_dataset(tmp_dir, 5, 10, 10, is_rotated=rotated)
                runner = default_runner.Detectron2GoRunner()
                cfg = _get_cfg(runner, tmp_dir, ds_name)

                ds_evaluators = runner.get_evaluator(cfg, ds_name, tmp_dir)
                self.assertTrue(isinstance(ds_evaluators._evaluators[0], evaluator))

    def test_create_runner(self):
        runner = create_runner(
            ".".join(
                [
                    default_runner.Detectron2GoRunner.__module__,
                    default_runner.Detectron2GoRunner.__name__,
                ]
            )
        )
        self.assertTrue(isinstance(runner, default_runner.Detectron2GoRunner))

    @helper.enable_ddp_env()
    def test_d2go_runner_ema(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            ds_name = create_local_dataset(tmp_dir, 5, 10, 10)
            runner = default_runner.Detectron2GoRunner()
            cfg = _get_cfg(runner, tmp_dir, ds_name)
            cfg.MODEL.META_ARCHITECTURE = "MetaArchForTestSingleValue"
            cfg.MODEL_EMA.ENABLED = True
            cfg.MODEL_EMA.DECAY = 0.9
            cfg.MODEL_EMA.DECAY_WARM_UP_FACTOR = -1

            def _run_train(cfg):
                cfg = copy.deepcopy(cfg)
                model = runner.build_model(cfg)
                model = DistributedDataParallel(model, broadcast_buffers=False)
                runner.do_train(cfg, model, True)
                final_model_path = os.path.join(tmp_dir, "model_final.pth")
                trained_weights = torch.load(final_model_path)
                self.assertIn("ema_state", trained_weights)
                default_runner._close_all_tbx_writers()
                return final_model_path, model.module.ema_state

            def _run_test(cfg, final_path, gt_ema):
                cfg = copy.deepcopy(cfg)
                cfg.MODEL.WEIGHTS = final_path
                model = runner.build_model(cfg, eval_only=True)
                self.assertGreater(len(model.ema_state.state), 0)
                self.assertEqual(len(model.ema_state.state), len(gt_ema.state))
                self.assertTrue(
                    _compare_state_dict(
                        model.ema_state.state_dict(), gt_ema.state_dict()
                    )
                )
                results = runner.do_test(cfg, model)
                self.assertEqual(results["default"][ds_name]["bbox"]["AP"], 3.0)
                self.assertEqual(results["ema"][ds_name]["bbox"]["AP"], 9.0)
                default_runner._close_all_tbx_writers()

            def _run_build_model_with_ema_weight(cfg, final_path, gt_ema):
                cfg = copy.deepcopy(cfg)
                cfg.MODEL.WEIGHTS = final_path
                cfg.MODEL_EMA.USE_EMA_WEIGHTS_FOR_EVAL_ONLY = True
                model = runner.build_model(cfg, eval_only=True)
                self.assertTrue(
                    _compare_state_dict(model.state_dict(), gt_ema.state_dict())
                )

            final_model_path, gt_ema = _run_train(cfg)
            _run_test(cfg, final_model_path, gt_ema)
            _run_build_model_with_ema_weight(cfg, final_model_path, gt_ema)

    def test_d2go_runner_train_qat_hook_update_stat(self):
        """Check that the qat hook is used and updates stats"""

        @META_ARCH_REGISTRY.register()
        class MetaArchForTestQAT(MetaArchForTest):
            def prepare_for_quant(self, cfg):
                """Set the qconfig to updateable observers"""
                self.qconfig = updateable_symmetric_moving_avg_minmax_config
                return self

        def setup(tmp_dir):
            ds_name = create_local_dataset(tmp_dir, 5, 10, 10)
            runner = default_runner.Detectron2GoRunner()
            cfg = _get_cfg(runner, tmp_dir, ds_name)
            cfg.merge_from_list(
                (
                    ["MODEL.META_ARCHITECTURE", "MetaArchForTestQAT"]
                    + ["QUANTIZATION.QAT.ENABLED", "True"]
                    + ["QUANTIZATION.QAT.START_ITER", "0"]
                    + ["QUANTIZATION.QAT.ENABLE_OBSERVER_ITER", "0"]
                )
            )
            return runner, cfg

        # check observers have not changed their minmax vals (stats changed)
        with tempfile.TemporaryDirectory() as tmp_dir:
            runner, cfg = setup(tmp_dir)
            model = runner.build_model(cfg)
            runner.do_train(cfg, model, resume=True)
            observer = model.conv.activation_post_process.activation_post_process
            self.assertEqual(observer.min_val, torch.tensor(float("inf")))
            self.assertEqual(observer.max_val, torch.tensor(float("-inf")))
            self.assertNotEqual(observer.max_stat, torch.tensor(float("inf")))

        # check observer does not change if period is > max_iter
        with tempfile.TemporaryDirectory() as tmp_dir:
            runner, cfg = setup(tmp_dir)
            cfg.merge_from_list(
                (
                    ["QUANTIZATION.QAT.UPDATE_OBSERVER_STATS_PERIODICALLY", "True"]
                    + ["QUANTIZATION.QAT.UPDATE_OBSERVER_STATS_PERIOD", "10"]
                )
            )
            model = runner.build_model(cfg)
            runner.do_train(cfg, model, resume=True)
            observer = model.conv.activation_post_process.activation_post_process
            self.assertEqual(observer.min_val, torch.tensor(float("inf")))
            self.assertEqual(observer.max_val, torch.tensor(float("-inf")))
            self.assertNotEqual(observer.max_stat, torch.tensor(float("inf")))

        # check observer changes if period < max_iter
        with tempfile.TemporaryDirectory() as tmp_dir:
            runner, cfg = setup(tmp_dir)
            cfg.merge_from_list(
                (
                    ["QUANTIZATION.QAT.UPDATE_OBSERVER_STATS_PERIODICALLY", "True"]
                    + ["QUANTIZATION.QAT.UPDATE_OBSERVER_STATS_PERIOD", "1"]
                )
            )
            model = runner.build_model(cfg)
            runner.do_train(cfg, model, resume=True)
            observer = model.conv.activation_post_process.activation_post_process
            self.assertNotEqual(observer.min_val, torch.tensor(float("inf")))
            self.assertNotEqual(observer.max_val, torch.tensor(float("-inf")))
            self.assertNotEqual(observer.max_stat, torch.tensor(float("inf")))

        default_runner._close_all_tbx_writers()

    def test_d2go_runner_train_qat(self):
        """Make sure QAT runs"""

        @META_ARCH_REGISTRY.register()
        class MetaArchForTestQAT1(torch.nn.Module):
            def __init__(self, cfg):
                super().__init__()
                self.conv = torch.nn.Conv2d(3, 4, kernel_size=3, stride=1, padding=1)

            @property
            def device(self):
                return self.conv.weight.device

            def forward(self, inputs):
                images = [x["image"] for x in inputs]
                images = ImageList.from_tensors(images, 1)

                ret = self.conv(images.tensor)
                losses = {"loss": ret.norm()}

                # run the same conv again
                ret1 = self.conv(images.tensor)
                losses["ret1"] = ret1.norm()

                return losses

        def setup(tmp_dir, backend, qat_method):
            ds_name = create_local_dataset(tmp_dir, 5, 10, 10)
            runner = default_runner.Detectron2GoRunner()
            cfg = _get_cfg(runner, tmp_dir, ds_name)
            cfg.merge_from_list(
                (
                    ["MODEL.META_ARCHITECTURE", "MetaArchForTestQAT1"]
                    + ["QUANTIZATION.QAT.ENABLED", "True"]
                    + ["QUANTIZATION.QAT.START_ITER", "1"]
                    + ["QUANTIZATION.QAT.ENABLE_OBSERVER_ITER", "0"]
                    + ["QUANTIZATION.QAT.ENABLE_LEARNABLE_OBSERVER_ITER", "2"]
                    + ["QUANTIZATION.QAT.DISABLE_OBSERVER_ITER", "4"]
                    + ["QUANTIZATION.QAT.FREEZE_BN_ITER", "4"]
                    + ["QUANTIZATION.BACKEND", backend]
                    + ["QUANTIZATION.QAT.FAKE_QUANT_METHOD", qat_method]
                )
            )
            return runner, cfg

        # seems that fbgemm with learnable qat is not supported
        for backend, qat_method in [
            ("fbgemm", "default"),
            ("qnnpack", "default"),
            ("qnnpack", "learnable"),
        ]:
            with self.subTest(backend=backend, qat_method=qat_method):
                with tempfile.TemporaryDirectory() as tmp_dir:
                    runner, cfg = setup(tmp_dir, backend=backend, qat_method=qat_method)
                    model = runner.build_model(cfg)
                    print(model)
                    runner.do_train(cfg, model, resume=True)

            default_runner._close_all_tbx_writers()

    def test_d2go_runner_trainer_hooks(self):
        counts = 0

        @TRAINER_HOOKS_REGISTRY.register()
        def _check_hook_func(hooks, cfg):
            nonlocal counts
            counts = len(hooks)
            print(hooks)

        with tempfile.TemporaryDirectory() as tmp_dir:
            ds_name = create_local_dataset(tmp_dir, 5, 10, 10)
            runner = default_runner.Detectron2GoRunner()
            cfg = _get_cfg(runner, tmp_dir, ds_name)
            model = runner.build_model(cfg)
            runner.do_train(cfg, model, resume=True)

            default_runner._close_all_tbx_writers()

        self.assertGreater(counts, 0)


def _compare_state_dict(sd1, sd2, abs_error=1e-3):
    if len(sd1) != len(sd2):
        return False
    if set(sd1.keys()) != set(sd2.keys()):
        return False
    for name in sd1:
        if sd1[name].dtype == torch.float32:
            if torch.abs((sd1[name] - sd2[name])).max() > abs_error:
                return False
        elif (sd1[name] != sd2[name]).any():
            return False
    return True
