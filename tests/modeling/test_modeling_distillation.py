#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import unittest
from typing import List
from unittest import mock

import numpy as np
import torch
import torch.nn as nn
from d2go.config import CfgNode
from d2go.modeling import modeling_hook as mh
from d2go.modeling.distillation import (
    _build_teacher,
    _set_device,
    BaseDistillationHelper,
    CachedLayer,
    compute_layer_losses,
    DefaultLossCombiner,
    DistillationModelingHook,
    DomainAdaptation,
    ExampleDistillationHelper,
    get_default_kd_image_classification_layer_losses,
    KnowledgeDistillation,
    LabelDistillation,
    LayerLossMetadata,
    NoopPseudoLabeler,
    PseudoLabeler,
    record_layers,
    register_layer_losses_and_to_device,
    RelabelTargetInBatch,
    set_cache_dict,
    unrecord_layers,
)
from d2go.registry.builtin import (
    DISTILLATION_ALGORITHM_REGISTRY,
    DISTILLATION_HELPER_REGISTRY,
    META_ARCH_REGISTRY,
)
from d2go.runner.config_defaults import add_distillation_configs
from d2go.runner.default_runner import BaseRunner
from d2go.utils.testing import helper
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.utils.file_io import PathManager
from mobile_cv.common.misc.file_utils import make_temp_directory
from mobile_cv.common.misc.mixin import dynamic_mixin, remove_dynamic_mixin


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


class DivideInputBy2OutputDict(nn.Module):
    def forward(self, batched_inputs: List):
        """Divide all targets by 2 and return dict output"""
        return {i: x / 2.0 for i, x in enumerate(batched_inputs)}


class TimesTable5OutputDict(nn.Module):
    def forward(self, batched_inputs: List):
        """Return first five entries of times table for each input with a dict output"""
        return {i: [x * i for i in range(1, 6)] for i, x in enumerate(batched_inputs)}


class ConstantStrOutput(nn.Module):
    def forward(self, batched_inputs: List):
        """Return some string"""
        return "Testing!"


class AddOne(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.Tensor([1]))

    def forward(self, x):
        return x + self.weight

    @property
    def device(self):
        return self.weight.device


class AddLayers(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer0 = AddOne()
        self.layer1 = AddOne()
        self.layer2 = AddOne()

    def forward(self, x):
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        if not self.training:
            return x
        return {"output": x}

    @property
    def device(self):
        return self.layer0.weight.device


class SimpleAdd(nn.Module):
    def forward(self, x, y):
        return x + y


class SimpleMul(nn.Module):
    def forward(self, x, y):
        return x * y


class TestLabeler(PseudoLabeler):
    def __init__(self, teacher):
        self.teacher = teacher

    def label(self, x):
        return self.teacher(x)


@META_ARCH_REGISTRY.register()
class TestMetaArchAddRand(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.weight = nn.Parameter(torch.rand(1))

    def forward(self, x):
        return x + self.weight


@DISTILLATION_HELPER_REGISTRY.register()
class TestHelper(BaseDistillationHelper):
    def get_pseudo_labeler(self):
        """Run teacher model on inputs"""
        return TestLabeler(self.teacher)

    def get_preprocess_student_input(self):
        return lambda x: x + 1

    def get_preprocess_teacher_input(self):
        return lambda x: x + 2

    def get_layer_losses(self, model=None):
        return [
            LayerLossMetadata(
                loss=SimpleAdd(),
                name="add",
                layer0="layer0",
                layer1="layer0",
            ),
            LayerLossMetadata(
                loss=SimpleMul(),
                name="mul",
                layer0="layer1",
                layer1="layer1",
            ),
        ]

    def get_combine_losses(self):
        return lambda d: {
            "output": d["output"] * 0.1,
            "add": d["add"] * 0.5,
            "mul": d["mul"] * 10.0,
        }


class TestDAHelper(BaseDistillationHelper):
    def get_preprocess_domain0_input(self):
        return lambda x: x["real"]

    def get_preprocess_domain1_input(self):
        return lambda x: x["synthetic"]

    def get_layer_losses(self, model=None):
        return [
            LayerLossMetadata(
                loss=SimpleAdd(),
                name="add",
                layer0="layer0",
                layer1="layer0",
            )
        ]

    def get_combine_losses(self):
        return lambda d0, d1, da, ta: {
            "real": d0["output"] * 0.1,
            "synthetic": d1["output"] * 0.5,
            "add": da["add"] * 10.0,
        }


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
    # model_ema.add_model_ema_configs(cfg)
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

        # check teacher model config is clone of student model
        self.assertEqual(cfg.DISTILLATION.TEACHER.CONFIG_FNAME, "")

    def test_build_teacher_torchscript(self):
        """Check can build teacher using torchscript fname in config"""
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
    def test_build_teacher_torchscript_gpu(self):
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

    def test_build_teacher_config(self):
        """Check build pytorch model using config"""
        # build model
        cfg = _get_default_cfg()
        cfg.MODEL.META_ARCHITECTURE = "TestMetaArchAddRand"
        gt_model = BaseRunner().build_model(cfg)
        with make_temp_directory("tmp") as output_dir:
            # save model
            checkpointer = DetectionCheckpointer(gt_model, save_dir=output_dir)
            checkpointer.save("checkpoint")
            cfg.MODEL.WEIGHTS = f"{output_dir}/checkpoint.pth"
            config_fname = f"{output_dir}/config.yaml"
            with PathManager.open(config_fname, "w") as f:
                f.write(cfg.dump())

            # load model and compare to gt
            cfg.DISTILLATION.TEACHER.TYPE = "config"
            cfg.DISTILLATION.TEACHER.CONFIG_FNAME = config_fname
            model = _build_teacher(cfg)
            self.assertEqual(gt_model.weight, model.weight)

    def test_build_teacher_none(self):
        """Check that we can ignore building the teacher"""
        # build model
        cfg = _get_default_cfg()
        cfg.MODEL.META_ARCHITECTURE = "TestMetaArchAddRand"
        cfg.DISTILLATION.TEACHER.TYPE = "no_teacher"
        model = _build_teacher(cfg)
        self.assertTrue(isinstance(model, nn.Module))

    def test_override_teacher_config_gpu_on_cpu(self):
        """Teacher cuda model can be run on cpu if specified in config"""
        # build model where teacher is specified on gpu but user overrides cpu
        cfg = _get_default_cfg()
        cfg.MODEL.META_ARCHITECTURE = "TestMetaArchAddRand"
        gt_model = BaseRunner().build_model(cfg)
        with make_temp_directory("tmp") as output_dir:
            # save model
            checkpointer = DetectionCheckpointer(gt_model, save_dir=output_dir)
            checkpointer.save("checkpoint")
            cfg.MODEL.WEIGHTS = f"{output_dir}/checkpoint.pth"
            cfg.MODEL.DEVICE = "cuda"
            config_fname = f"{output_dir}/config.yaml"
            with PathManager.open(config_fname, "w") as f:
                f.write(cfg.dump())

            # load model and compare to gt
            cfg.DISTILLATION.TEACHER.TYPE = "config"
            cfg.DISTILLATION.TEACHER.CONFIG_FNAME = config_fname
            cfg.DISTILLATION.TEACHER.DEVICE = "cpu"
            model = _build_teacher(cfg)
            self.assertEqual(gt_model.weight, model.weight)

    def test_set_device(self):
        """Check teacher device is set"""
        # without attr
        model = Noop()
        self.assertFalse(hasattr(model, "device"))
        device = torch.device("cpu")

        # without property
        model = _set_device(model, device)
        self.assertEqual(model.device, device)

        # with property
        model = AddOne()
        model = _set_device(model, device)
        self.assertEqual(model.device, device)

    def test_cached_layer_tensor(self):
        """Check cached layer saves layer output"""
        model = AddOne()
        cache = {}
        dynamic_mixin(
            model,
            CachedLayer,
            init_dict={"label": "test_layer", "cache": cache},
        )
        input = torch.randn(1)
        output = model(input)
        torch.testing.assert_close(output, cache["test_layer"])

    def test_cached_layer_list(self):
        """Check cached layer saves list"""
        model = DivideInputBy2()
        cache = {}
        dynamic_mixin(
            model,
            CachedLayer,
            init_dict={"label": "test_layer", "cache": cache},
        )
        input = [torch.randn(1) for _ in range(2)]
        output = model(input)
        torch.testing.assert_close(output, cache["test_layer"])

    def test_cached_layer_tuple(self):
        """Check cached layer saves list"""
        model = DivideInputBy2()
        cache = {}
        dynamic_mixin(
            model,
            CachedLayer,
            init_dict={"label": "test_layer", "cache": cache},
        )
        input = (torch.randn(1) for _ in range(2))
        output = model(input)
        torch.testing.assert_close(output, cache["test_layer"])

    def test_cached_layer_dict(self):
        """Check cached layer saves dict"""
        model = DivideInputBy2OutputDict()
        cache = {}
        dynamic_mixin(
            model,
            CachedLayer,
            init_dict={"label": "test_layer", "cache": cache},
        )
        input = [torch.randn(1) for _ in range(2)]
        output = model(input)
        torch.testing.assert_close(output, cache["test_layer"])

    def test_cached_layer_arbitrary(self):
        """Check cached layer saves arbitrary nested data structure"""
        model = TimesTable5OutputDict()
        cache = {}
        dynamic_mixin(
            model,
            CachedLayer,
            init_dict={"label": "test_layer", "cache": cache},
        )
        input = [torch.randn(1) for _ in range(2)]
        output = model(input)
        torch.testing.assert_close(output, cache["test_layer"])

    def test_cached_layer_unsupported(self):
        """Check cached layer doesn't save unsupported data type like strings"""
        model = ConstantStrOutput()
        cache = {}
        dynamic_mixin(
            model,
            CachedLayer,
            init_dict={"label": "test_layer", "cache": cache},
        )
        input = [torch.randn(1) for _ in range(2)]
        self.assertRaises(ValueError, model, input)

    def test_record_layers(self):
        """Check we can record specified layer"""
        model = AddLayers()
        cache = record_layers(model, ["", "layer0", "layer1", "layer2"])

        input = torch.Tensor([0])
        output = model(input)
        torch.testing.assert_close(cache["layer0"], torch.Tensor([1]))
        torch.testing.assert_close(cache["layer1"], torch.Tensor([2]))
        torch.testing.assert_close(cache["layer2"], torch.Tensor([3]))
        torch.testing.assert_close(cache[""], output)

    def test_unrecord_layers(self):
        """Check we can remove a recorded layer"""
        model = AddLayers()
        _ = record_layers(model, ["", "layer0", "layer1", "layer2"])
        unrecord_layers(model, ["", "layer0"])
        self.assertFalse(hasattr(model.layer0, "cache"))

    def test_compute_layer_losses(self):
        """Check iterating over loss dicts"""
        layer_losses = [
            LayerLossMetadata(
                loss=lambda x, y: x + y, name="add", layer0="l00", layer1="l10"
            ),
            LayerLossMetadata(
                loss=lambda x, y: x / y, name="div", layer0="l01", layer1="l11"
            ),
        ]
        layer0_cache = {"l00": torch.randn(1), "l01": torch.randn(1)}
        layer1_cache = {"l10": torch.randn(1), "l11": torch.randn(1)}
        output = compute_layer_losses(layer_losses, layer0_cache, layer1_cache)
        torch.testing.assert_close(
            output["add"], layer0_cache["l00"] + layer1_cache["l10"]
        )
        torch.testing.assert_close(
            output["div"], layer0_cache["l01"] / layer1_cache["l11"]
        )

    def test_set_cache_dict(self):
        """Check we can swap the cache dict used when recording layers"""
        model = AddLayers()
        cache = record_layers(model, ["", "layer0", "layer1", "layer2"])
        new_cache = {}
        set_cache_dict(model, new_cache)
        input = torch.Tensor([0])
        output = model(input)
        self.assertEqual(cache, {})
        torch.testing.assert_close(new_cache["layer0"], torch.Tensor([1]))
        torch.testing.assert_close(new_cache["layer1"], torch.Tensor([2]))
        torch.testing.assert_close(new_cache["layer2"], torch.Tensor([3]))
        torch.testing.assert_close(new_cache[""], output)

    def test_register_layer_losses(self):
        """Check losses can be registered to model"""
        model = AddOne()
        ll = [
            LayerLossMetadata(
                loss=SimpleAdd(),
                name="mul",
                layer0="layer1",
                layer1="layer1",
            ),
        ]
        registered_losses = register_layer_losses_and_to_device(ll, model)
        self.assertTrue(hasattr(model, "mul"))
        self.assertEqual(model.mul, registered_losses[0].loss)

    @helper.skip_if_no_gpu
    def test_register_layer_losses_and_to_device(self):
        """Check losses can be registered to model"""
        model = AddOne()
        model = model.to("cuda")
        ll = [
            LayerLossMetadata(
                loss=AddOne(),
                name="mul",
                layer0="layer1",
                layer1="layer1",
            ),
        ]
        register_layer_losses_and_to_device(ll, model)
        self.assertEqual(model.mul.device, model.device)


class TestPseudoLabeler(unittest.TestCase):
    def test_noop(self):
        """Check noop"""
        pseudo_labeler = NoopPseudoLabeler()
        x = np.random.randn(1)
        output = pseudo_labeler.label(x)
        torch.testing.assert_close(x, output)

    def test_relabeltargetinbatch(self):
        """Check target is relabed using teacher"""
        teacher = DivideInputDictBy2()
        teacher.eval()
        teacher.device = torch.device("cpu")
        relabeler = RelabelTargetInBatch(teacher=teacher)
        batched_inputs = _get_input_data(n=2, use_input_target=True)
        gt = [{"input": d["input"], "target": d["input"] / 2.0} for d in batched_inputs]
        outputs = relabeler.label(batched_inputs)
        torch.testing.assert_close(outputs, gt)


class TestDistillationHelper(unittest.TestCase):
    def test_registry(self):
        """Check base class in registry"""
        self.assertTrue("BaseDistillationHelper" in DISTILLATION_HELPER_REGISTRY)

    def test_base_distillation_helper(self):
        """Check base distillation helper returns input as output"""
        dh = BaseDistillationHelper(cfg=None, teacher=None)
        pseudo_labeler = dh.get_pseudo_labeler()
        self.assertTrue(isinstance(pseudo_labeler, NoopPseudoLabeler))

    def test_example_distillation_helper(self):
        """Example distillation uses teacher to relabel targets"""
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
        for algorithm in [
            "LabelDistillation",
            "KnowledgeDistillation",
            "DomainAdaptation",
        ]:
            self.assertTrue(algorithm in DISTILLATION_ALGORITHM_REGISTRY)

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

    def test_kd_inference(self):
        """Check inference defaults to student (and preprocessing)"""
        distillation_helper = TestHelper(cfg=CfgNode(), teacher=AddLayers())
        model = AddLayers()
        dynamic_mixin(
            model,
            KnowledgeDistillation,
            init_dict={"distillation_helper": distillation_helper},
        )
        model.eval()
        input = torch.randn(1)
        output = model(input)
        torch.testing.assert_close(output, input + 4.0)

    def test_kd_train(self):
        """Check train pass results in updated loss output"""
        distillation_helper = TestHelper(cfg=CfgNode(), teacher=AddLayers())
        model = AddLayers()
        dynamic_mixin(
            model,
            KnowledgeDistillation,
            init_dict={"distillation_helper": distillation_helper},
        )
        model.train()
        input = torch.randn(1)
        output = model(input)
        torch.testing.assert_close(output["output"], (input + 4.0) * 0.1)
        torch.testing.assert_close(output["add"], ((input + 2.0) + (input + 3.0)) * 0.5)
        torch.testing.assert_close(output["mul"], (input + 3.0) * (input + 4.0) * 10.0)

    def test_kd_remove_dynamic_mixin(self):
        """Check removing dynamic mixin removes cached layers"""
        distillation_helper = TestHelper(cfg=CfgNode(), teacher=AddLayers())
        model = AddLayers()
        dynamic_mixin(
            model,
            KnowledgeDistillation,
            init_dict={"distillation_helper": distillation_helper},
        )
        remove_dynamic_mixin(model)
        for module in model.modules():
            self.assertFalse(hasattr(module, "cache"))

    def test_da_inference(self):
        """Check inference defaults to student (and preprocessing)"""
        distillation_helper = TestDAHelper(cfg=CfgNode(), teacher=nn.Identity())
        model = AddLayers()
        dynamic_mixin(
            model,
            DomainAdaptation,
            init_dict={"distillation_helper": distillation_helper},
        )
        model.eval()
        input = {"real": torch.randn(1), "synthetic": torch.randn(1)}
        output = model(input)
        torch.testing.assert_close(output, input["real"] + 3.0)

    def test_da_train(self):
        """Check train pass results in updated loss output"""
        distillation_helper = TestDAHelper(cfg=CfgNode(), teacher=nn.Identity())
        model = AddLayers()
        dynamic_mixin(
            model,
            DomainAdaptation,
            init_dict={"distillation_helper": distillation_helper},
        )
        model.train()
        input = {"real": torch.randn(1), "synthetic": torch.randn(1)}
        output = model(input)
        self.assertEqual(set(output.keys()), {"real", "synthetic", "add"})
        torch.testing.assert_close(output["real"], (input["real"] + 3.0) * 0.1)
        torch.testing.assert_close(
            output["synthetic"], (input["synthetic"] + 3.0) * 0.5
        )
        torch.testing.assert_close(
            output["add"], ((input["real"] + 1.0) + (input["synthetic"] + 1.0)) * 10.0
        )

    def test_da_remove_dynamic_mixin(self):
        """Check removing dynamic mixin removes cached layers"""
        distillation_helper = TestHelper(cfg=CfgNode(), teacher=nn.Identity())
        model = AddLayers()
        dynamic_mixin(
            model,
            DomainAdaptation,
            init_dict={"distillation_helper": distillation_helper},
        )
        remove_dynamic_mixin(model)
        for module in model.modules():
            self.assertFalse(hasattr(module, "cache"))


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


class TestDistillationMiscTests(unittest.TestCase):
    def test_teacher_outside_updated_parameters(self):
        """
        Check that teacher values are ignored when updating student

        The teacher can often be referenced in the mixed in model. A common
        example is when the teacher is an attributed of the distillation
        helper.
         => DistillationModel.distillation_helper.teacher

        This raises the question of whether the teacher model will be affected
        by calls to the mixed in model:
            DisillationModel.train() => does teacher switch to training?
            setup_qat(DistillationModel) => will fuse occur on the teacher modules?

        The answer to these questions should be no as we want the teacher to remain static
        during training (unless specified). This is the case as long as teacher is an
        attribute of a non-module class (e.g., distillation_helper). This is because
        modules are registered in PyTorch as part of __setattr__. __setattr__ only checks
        if the value is a module or parameter. If the value is an object
        (e.g., distillation_helper) which contains modules, these modules are ignored.
        https://pytorch.org/docs/stable/_modules/torch/nn/modules/module.html#Module.register_parameter

        This unittest builds the teacher model and checks that only the student
        parameter is registered.
        """
        cfg = _get_default_cfg()
        cfg.MODEL.META_ARCHITECTURE = "TestMetaArchAddRand"
        prebuilt_teacher = BaseRunner().build_model(cfg)
        with make_temp_directory("tmp") as output_dir:
            checkpointer = DetectionCheckpointer(prebuilt_teacher, save_dir=output_dir)
            checkpointer.save("checkpoint")
            cfg.MODEL.WEIGHTS = f"{output_dir}/checkpoint.pth"
            config_fname = f"{output_dir}/config.yaml"
            with PathManager.open(config_fname, "w") as f:
                f.write(cfg.dump())
            cfg.DISTILLATION.TEACHER.TYPE = "config"
            cfg.DISTILLATION.TEACHER.CONFIG_FNAME = config_fname
            cfg.DISTILLATION.HELPER = "TestHelper"
            cfg.MODEL.MODELING_HOOKS = ["DistillationModelingHook"]
            distilled_model = BaseRunner().build_model(cfg)
            self.assertEqual(len(list(distilled_model.parameters())), 1)


class TestDistillationDefaults(unittest.TestCase):
    def test_kd_image_classification_layer_losses(self):
        """Check the default returns a list of layerlossmetadata"""
        layer_losses = get_default_kd_image_classification_layer_losses()
        self.assertTrue(isinstance(layer_losses, List))
        self.assertTrue(isinstance(layer_losses[0], LayerLossMetadata))

    def test_default_loss_combiner(self):
        """Check combiner multiplies loss by weights"""
        weights = {"a": torch.randn(1), "b": torch.randn(1)}
        combiner = DefaultLossCombiner(weights)
        input = {"a": 1.0, "b": 10.0}
        output = combiner(input)
        torch.testing.assert_close(output["a"], input["a"] * weights["a"])
        torch.testing.assert_close(output["b"], input["b"] * weights["b"])
