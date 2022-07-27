#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved


import copy
import unittest
from typing import List

import d2go.runner.default_runner as default_runner
import torch
from d2go.config import CfgNode
from d2go.modeling import modeling_hook as mh
from d2go.modeling.api import build_d2go_model, D2GoModelBuildResult
from d2go.registry.builtin import META_ARCH_REGISTRY, MODELING_HOOK_REGISTRY


@META_ARCH_REGISTRY.register()
class TestArch(torch.nn.Module):
    def __init__(self, cfg):
        super().__init__()

    def forward(self, x):
        return x * 2


# create a wrapper of the model that add 1 to the output
class PlusOneWrapper(torch.nn.Module):
    def __init__(self, model: torch.nn.Module):
        super().__init__()
        self.model = model

    def forward(self, x):
        return self.model(x) + 1


@MODELING_HOOK_REGISTRY.register()
class PlusOneHook(mh.ModelingHook):
    def __init__(self, cfg):
        super().__init__(cfg)

    def apply(self, model: torch.nn.Module) -> torch.nn.Module:
        return PlusOneWrapper(model)

    def unapply(self, model: torch.nn.Module) -> torch.nn.Module:
        assert isinstance(model, PlusOneWrapper)
        return model.model


# create a wrapper of the model that add 1 to the output
class TimesTwoWrapper(torch.nn.Module):
    def __init__(self, model: torch.nn.Module):
        super().__init__()
        self.model = model

    def forward(self, x):
        return self.model(x) * 2


@MODELING_HOOK_REGISTRY.register()
class TimesTwoHook(mh.ModelingHook):
    def __init__(self, cfg):
        super().__init__(cfg)

    def apply(self, model: torch.nn.Module) -> torch.nn.Module:
        return TimesTwoWrapper(model)

    def unapply(self, model: torch.nn.Module) -> torch.nn.Module:
        assert isinstance(model, TimesTwoWrapper)
        return model.model


class TestModelingHook(unittest.TestCase):
    def test_modeling_hook_simple(self):
        model = TestArch(None)
        hook = PlusOneHook(None)
        model_with_hook = hook.apply(model)
        self.assertEqual(model_with_hook(2), 5)
        original_model = hook.unapply(model_with_hook)
        self.assertEqual(model, original_model)

    def test_modeling_hook_cfg(self):
        """Create model with modeling hook using build_model"""
        cfg = CfgNode()
        cfg.MODEL = CfgNode()
        cfg.MODEL.DEVICE = "cpu"
        cfg.MODEL.META_ARCHITECTURE = "TestArch"
        cfg.MODEL.MODELING_HOOKS = ["PlusOneHook", "TimesTwoHook"]

        model_info: D2GoModelBuildResult = build_d2go_model(cfg)
        model: torch.nn.Module = model_info.model
        modeling_hooks: List[mh.ModelingHook] = model_info.modeling_hooks

        self.assertEqual(model(2), 10)
        self.assertEqual(len(modeling_hooks), 2)

        self.assertTrue(hasattr(model, "_modeling_hooks"))
        self.assertTrue(hasattr(model, "unapply_modeling_hooks"))
        orig_model = model.unapply_modeling_hooks()
        self.assertIsInstance(orig_model, TestArch)
        self.assertEqual(orig_model(2), 4)

    def test_modeling_hook_runner(self):
        """Create model with modeling hook from runner"""
        runner = default_runner.Detectron2GoRunner()
        cfg = runner.get_default_cfg()
        cfg.MODEL.DEVICE = "cpu"
        cfg.MODEL.META_ARCHITECTURE = "TestArch"
        cfg.MODEL.MODELING_HOOKS = ["PlusOneHook", "TimesTwoHook"]
        model = runner.build_model(cfg)
        self.assertEqual(model(2), 10)

        self.assertTrue(hasattr(model, "_modeling_hooks"))
        self.assertTrue(hasattr(model, "unapply_modeling_hooks"))
        orig_model = model.unapply_modeling_hooks()
        self.assertIsInstance(orig_model, TestArch)
        self.assertEqual(orig_model(2), 4)

        default_runner._close_all_tbx_writers()

    def test_modeling_hook_copy(self):
        """Create model with modeling hook, the model could be copied"""
        cfg = CfgNode()
        cfg.MODEL = CfgNode()
        cfg.MODEL.DEVICE = "cpu"
        cfg.MODEL.META_ARCHITECTURE = "TestArch"
        cfg.MODEL.MODELING_HOOKS = ["PlusOneHook", "TimesTwoHook"]

        model_info: D2GoModelBuildResult = build_d2go_model(cfg)
        model: torch.nn.Module = model_info.model
        modeling_hooks: List[mh.ModelingHook] = model_info.modeling_hooks

        self.assertEqual(model(2), 10)
        self.assertEqual(len(modeling_hooks), 2)

        model_copy = copy.deepcopy(model)

        orig_model = model.unapply_modeling_hooks()
        self.assertIsInstance(orig_model, TestArch)
        self.assertEqual(orig_model(2), 4)

        orig_model_copy = model_copy.unapply_modeling_hooks()
        self.assertEqual(orig_model_copy(2), 4)
