#!/usr/bin/env fbpython
# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

import os
import unittest
from typing import Dict, List

import torch

from d2go.config import CfgNode
from d2go.modeling import modeling_hook as mh
from d2go.registry.builtin import META_ARCH_REGISTRY
from d2go.runner.default_runner import Detectron2GoRunner
from d2go.trainer.activation_checkpointing import (
    ActivationCheckpointModelingHook,
    add_activation_checkpoint_configs,
)
from d2go.utils.testing.data_loader_helper import create_local_dataset
from d2go.utils.testing.helper import tempdir
from detectron2.structures import ImageList
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    CheckpointWrapper,
)


@META_ARCH_REGISTRY.register()
class MetaArchForTestAC(torch.nn.Module):
    def __init__(self, cfg: CfgNode) -> None:
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 4, kernel_size=3, stride=1, padding=1)
        self.bn = torch.nn.BatchNorm2d(4)
        self.relu = torch.nn.ReLU(inplace=True)
        self.linear = torch.nn.Linear(4, 4)
        self.avgpool = torch.nn.AdaptiveAvgPool2d((1, 1))

    @property
    def device(self) -> torch._C.device:
        return self.conv1.weight.device

    def forward(self, inputs: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        images = [x["image"] for x in inputs]
        images = ImageList.from_tensors(images, 1)
        ret = self.conv(images.tensor)
        ret = self.bn(ret)
        ret = self.relu(ret)
        ret = self.avgpool(ret)
        return {"loss": ret.norm()}


def _get_cfg(runner, output_dir, dataset_name):
    cfg = runner.get_default_cfg()
    cfg.MODEL.DEVICE = "cpu"
    cfg.MODEL.META_ARCHITECTURE = "MetaArchForTestAC"

    cfg.DATASETS.TRAIN = (dataset_name,)
    cfg.DATASETS.TEST = (dataset_name,)

    cfg.INPUT.MIN_SIZE_TRAIN = (10,)
    cfg.INPUT.MIN_SIZE_TEST = (10,)

    cfg.SOLVER.MAX_ITER = 3
    cfg.SOLVER.STEPS = []
    cfg.SOLVER.WARMUP_ITERS = 1
    cfg.SOLVER.CHECKPOINT_PERIOD = 3
    cfg.SOLVER.IMS_PER_BATCH = 2

    cfg.MODEL_EMA.ENABLED = True

    cfg.OUTPUT_DIR = output_dir

    return cfg


class TestActivationCheckpointing(unittest.TestCase):
    def test_ac_config(self) -> None:
        cfg = CfgNode()
        add_activation_checkpoint_configs(cfg)
        self.assertTrue(isinstance(cfg.ACTIVATION_CHECKPOINT, CfgNode))
        self.assertEqual(cfg.ACTIVATION_CHECKPOINT.REENTRANT, False)
        self.assertEqual(
            cfg.ACTIVATION_CHECKPOINT.AUTO_WRAP_POLICY, "always_wrap_policy"
        )
        self.assertEqual(cfg.ACTIVATION_CHECKPOINT.AUTO_WRAP_LAYER_CLS, [])

    def test_ac_modeling_hook_apply(self) -> None:
        """Check that the hook is registered"""
        self.assertTrue("ActivationCheckpointModelingHook" in mh.MODELING_HOOK_REGISTRY)

        cfg = CfgNode()
        add_activation_checkpoint_configs(cfg)
        ac_hook = ActivationCheckpointModelingHook(cfg)
        model = MetaArchForTestAC(cfg)
        ac_hook.apply(model)

        children = list(model.children())
        self.assertTrue(len(children) == 5)
        for child in children:
            self.assertTrue(isinstance(child, CheckpointWrapper))

    def test_ac_modeling_hook_autowrap(self) -> None:
        cfg = CfgNode()
        add_activation_checkpoint_configs(cfg)
        cfg.ACTIVATION_CHECKPOINT.AUTO_WRAP_POLICY = "layer_based_auto_wrap_policy"
        cfg.ACTIVATION_CHECKPOINT.AUTO_WRAP_LAYER_CLS = ["Conv2d", "BatchNorm2d"]
        ac_hook = ActivationCheckpointModelingHook(cfg)
        model = MetaArchForTestAC(cfg)
        ac_hook.apply(model)

        self.assertTrue(isinstance(model.conv, CheckpointWrapper))
        self.assertTrue(isinstance(model.bn, CheckpointWrapper))
        self.assertFalse(isinstance(model.linear, CheckpointWrapper))

    @tempdir
    def test_ac_runner(self, tmp_dir) -> None:
        ds_name = create_local_dataset(tmp_dir, 5, 10, 10)
        runner = Detectron2GoRunner()
        cfg = _get_cfg(runner, tmp_dir, ds_name)
        cfg.MODEL.MODELING_HOOKS = ["ActivationCheckpointModelingHook"]
        cfg.ACTIVATION_CHECKPOINT.AUTO_WRAP_POLICY = "layer_based_auto_wrap_policy"
        cfg.ACTIVATION_CHECKPOINT.AUTO_WRAP_LAYER_CLS = ["Conv2d", "BatchNorm2d"]
        cfg.MODEL_EMA.DECAY_WARM_UP_FACTOR = -1

        model = runner.build_model(cfg)
        runner.do_train(cfg, model, resume=False)
        self.assertTrue(os.path.exists(os.path.join(tmp_dir, "model_0000002.pth")))

        # resume training onto a non-AC-wrapped model
        cfg.MODEL.MODELING_HOOKS = []
        cfg.SOLVER.MAX_ITER = 6
        model = runner.build_model(cfg)
        runner.do_train(cfg, model, resume=True)
        self.assertTrue(os.path.exists(os.path.join(tmp_dir, "model_0000005.pth")))
