#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import unittest
from typing import List

import mock
import numpy as np
import torch
import torch.nn as nn
from d2go.config import CfgNode
from d2go.modeling.distillation import (
    _build_teacher,
    add_distillation_configs,
    BaseDistillationHelper,
    DistillationModelingHook,
    ExampleDistillationHelper,
    LabelDistillation,
    NoopPseudoLabeler,
    PseudoLabeler,
    RelabelTargetInBatch,
)
from d2go.modeling.meta_arch import modeling_hook as mh
from d2go.registry.builtin import (
    DISTILLATION_ALGORITHM_REGISTRY,
    DISTILLATION_HELPER_REGISTRY,
)
from d2go.utils.testing import helper
from mobile_cv.common.misc.file_utils import make_temp_directory


class DivideInputBy2(nn.Module):
    def forward(self, batched_inputs: List):
        """Divide all targets by 2 and batch output"""
        return [x / 2.0 for x in batched_inputs]


class DivideInputDictBy2(nn.Module):
    def forward(self, batched_inputs: List):
        """Divide all inputs by 2 and batch output

        Should be used with a pseudo labeler that will unpack the
        resulting tensor
        """
        output = []
        for d in batched_inputs:
            output.append(d["input"] / 2.0)
        return torch.stack(output)


class AddOne(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.Tensor([1]))

    def forward(self, x):
        return x + self.weight


class TestLabeler(PseudoLabeler):
    def __init__(self, teacher):
        self.teacher = teacher

    def label(self, x):
        return self.teacher(x)


@DISTILLATION_HELPER_REGISTRY.register()
class TestHelper(BaseDistillationHelper):
    def get_pseudo_labeler(self):
        """Run teacher model on inputs"""
        return TestLabeler(self.teacher)


class Noop(nn.Module):
    def forward(self, x):
        return x


def _get_input_data(n: int = 2, use_input_target: bool = False, requires_grad=False):
    """Return input data, dict if use_input_target is specified"""
    if not use_input_target:
        return torch.randn(n, requires_grad=requires_grad)

    return [
        {
            "input": torch.randn(1, requires_grad=requires_grad),
            "target": torch.randn(1),
        }
        for _ in range(n)
    ]


def _get_default_cfg():
    cfg = CfgNode()
    cfg.MODEL = CfgNode()
    cfg.MODEL.DEVICE = "cpu"
    cfg.MODEL.META_ARCHITECTURE = "TestArch"
    add_distillation_configs(cfg)
    cfg.DISTILLATION.ALGORITHM = "LabelDistillation"
    cfg.DISTILLATION.HELPER = "BaseDistillationHelper"
    cfg.DISTILLATION.TEACHER.TORCHSCRIPT_FNAME = ""
    cfg.DISTILLATION.TEACHER.DEVICE = ""
    return cfg


class TestDistillation(unittest.TestCase):
    def test_add_distillation_configs(self):
        """Check default config"""
        cfg = CfgNode()
        add_distillation_configs(cfg)
        self.assertTrue(isinstance(cfg.DISTILLATION.TEACHER, CfgNode))

    def test_build_teacher(self):
        """Check can build teacher using config"""
        # create torchscript
        model = DivideInputBy2()
        traced_model = torch.jit.trace(model, torch.randn(5))
        with make_temp_directory("tmp") as output_dir:
            fname = f"{output_dir}/tmp.pt"
            torch.jit.save(traced_model, fname)

            # create teacher
            cfg = _get_default_cfg()
            cfg.DISTILLATION.TEACHER.TORCHSCRIPT_FNAME = fname
            teacher = _build_teacher(cfg)
            batched_inputs = torch.randn(5)
            gt = batched_inputs / 2.0
            output = teacher(batched_inputs)
            torch.testing.assert_close(torch.Tensor(output), gt)

    @helper.skip_if_no_gpu
    def test_build_teacher_gpu(self):
        """Check teacher moved to cuda"""
        model = AddOne()
        traced_model = torch.jit.trace(model, torch.randn(5))
        with make_temp_directory("tmp") as output_dir:
            fname = f"{output_dir}/tmp.pt"
            torch.jit.save(traced_model, fname)

            # create teacher
            cfg = _get_default_cfg()
            cfg.MODEL.DEVICE = "cuda"
            cfg.DISTILLATION.TEACHER.TORCHSCRIPT_FNAME = fname
            teacher = _build_teacher(cfg)
            batched_inputs = torch.randn(5).to("cuda")
            gt = batched_inputs + torch.Tensor([1]).to("cuda")
            output = teacher(batched_inputs)
            torch.testing.assert_close(torch.Tensor(output), gt)


class TestPseudoLabeler(unittest.TestCase):
    def test_noop(self):
        """Check noop"""
        pseudo_labeler = NoopPseudoLabeler()
        x = np.random.randn(1)
        output = pseudo_labeler.label(x)
        self.assertEqual(x, output)

    def test_relabeltargetinbatch(self):
        """Check target is relabed using teacher"""
        teacher = DivideInputDictBy2()
        teacher.eval()
        teacher.device = torch.device("cpu")
        relabeler = RelabelTargetInBatch(teacher=teacher)
        batched_inputs = _get_input_data(n=2, use_input_target=True)
        gt = [{"input": d["input"], "target": d["input"] / 2.0} for d in batched_inputs]
        outputs = relabeler.label(batched_inputs)
        self.assertEqual(outputs, gt)


class TestDistillationHelper(unittest.TestCase):
    def test_registry(self):
        """Check base class in registry"""
        self.assertTrue("BaseDistillationHelper" in DISTILLATION_HELPER_REGISTRY)

    def test_base_distillation_helper(self):
        """Check base distillation helper returns input as output"""
        dh = BaseDistillationHelper(cfg=None, teacher=None)
        pseudo_labeler = dh.get_pseudo_labeler()
        self.assertTrue(isinstance(pseudo_labeler, NoopPseudoLabeler))

    def test_default_distillation_helper(self):
        """Default distillation uses teacher to relabel targets"""
        teacher = Noop()
        dh = ExampleDistillationHelper(cfg=None, teacher=teacher)
        pseudo_labeler = dh.get_pseudo_labeler()
        self.assertTrue(isinstance(pseudo_labeler, RelabelTargetInBatch))
        self.assertTrue(isinstance(pseudo_labeler.teacher, Noop))


class TestDistillationAlgorithm(unittest.TestCase):
    class LabelDistillationNoop(LabelDistillation, Noop):
        """Distillation should be used with dynamic mixin so we create
        a new class with mixin of a noop to test"""

        pass

    def test_registry(self):
        """Check distillation teacher in registry"""
        self.assertTrue("LabelDistillation" in DISTILLATION_ALGORITHM_REGISTRY)

    def test_label_distillation_inference(self):
        """Check inference defaults to student

        Use LabelDistillationNoop to set student model to noop
        """
        batched_inputs = _get_input_data(n=2)
        gt = batched_inputs.detach().clone()
        model = self.LabelDistillationNoop()
        model.dynamic_mixin_init(
            distillation_helper=TestHelper(cfg=None, teacher=DivideInputBy2()),
        )
        model.eval()
        output = model(batched_inputs)
        np.testing.assert_array_equal(output, gt)

    def test_label_distillation_training(self):
        """Check training uses pseudo labeler

        Distillation teacher should run the teacher model on the inputs and
        then pass to the noop
        """
        batched_inputs = _get_input_data(n=2, requires_grad=True)
        gt = [x / 2.0 for x in batched_inputs]
        model = self.LabelDistillationNoop()
        model.dynamic_mixin_init(
            distillation_helper=TestHelper(cfg=None, teacher=DivideInputBy2()),
        )
        model.train()
        output = model(batched_inputs)
        torch.testing.assert_close(output, gt)

        sum(output).backward()
        torch.testing.assert_close(batched_inputs.grad, torch.Tensor([0.5, 0.5]))


class TestDistillationModelingHook(unittest.TestCase):
    _build_teacher_ref = "d2go.modeling.distillation._build_teacher"

    def test_exists(self):
        """Check that the hook is registered"""
        self.assertTrue("DistillationModelingHook" in mh.MODELING_HOOK_REGISTRY)

    def test_init(self):
        """Check that we can build hook"""
        cfg = _get_default_cfg()
        with mock.patch(self._build_teacher_ref):
            DistillationModelingHook(cfg)

    def test_apply(self):
        """Check new model has distillation methods"""
        model = Noop()
        model.test_attr = "12345"
        cfg = _get_default_cfg()
        cfg.DISTILLATION.HELPER = "TestHelper"
        with mock.patch(self._build_teacher_ref):
            hook = DistillationModelingHook(cfg)
        hook.apply(model)
        # set teacher manually to override _build_teacher
        model.pseudo_labeler.teacher = DivideInputBy2()

        # check distillation attrs
        self.assertTrue(isinstance(model.distillation_helper, TestHelper))
        self.assertEqual(model._original_model_class, Noop)

        # check retains attrs
        self.assertTrue(hasattr(model, "test_attr"))
        self.assertEqual(model.test_attr, "12345")

        # check inference uses the baseline model which is a noop
        batched_inputs = _get_input_data(n=2)
        model.eval()
        gt = batched_inputs.detach().clone()
        output = model(batched_inputs)
        torch.testing.assert_close(output, gt)

        # check training uses the pseudo labeler
        model.train()
        gt = [x / 2.0 for x in batched_inputs]
        output = model(batched_inputs)
        torch.testing.assert_close(output, gt)

    def test_unapply(self):
        """Check removing distillation"""
        model = Noop()
        cfg = _get_default_cfg()
        with mock.patch(self._build_teacher_ref):
            hook = DistillationModelingHook(cfg)
        hook.apply(model)
        hook.unapply(model)

        for distillation_attr in [
            "distillation_helper",
            "_original_model_class",
        ]:
            self.assertFalse(hasattr(model, distillation_attr))

        # check forward is the original noop
        batched_inputs = _get_input_data(n=2)
        gt = batched_inputs.detach().clone()
        model.train()
        output = model(batched_inputs)
        torch.testing.assert_close(output, gt)
