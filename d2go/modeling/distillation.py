#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

# This is the main definition of distillation components in D2Go. This
# includes:
#   DistillationModelingHook => how we update the student model to obtain
#     distillation methods and properties (e.g., override model.forward)
#   DistillationAlgorithm => how we define what occurs during distillation
#     (e.g., specific forward func, teacher weights updates)
#   DistillationHelper => main class users should use to customize their
#     distllation (e.g., define how to pseudo label inputs)
#
# We use two additional registries so that users can select their
# distillation algorithms in configs: DISILLATION_ALAGORITHM, DISTILLATION_HELPER

from abc import abstractmethod
from typing import List

import torch
import torch.nn as nn
from d2go.config import CfgNode as CN
from d2go.modeling import modeling_hook as mh
from d2go.registry.builtin import (
    DISTILLATION_ALGORITHM_REGISTRY,
    DISTILLATION_HELPER_REGISTRY,
)
from detectron2.utils.file_io import PathManager
from mobile_cv.common.misc.mixin import dynamic_mixin, remove_dynamic_mixin


def add_distillation_configs(_C: CN) -> None:
    """Add default parameters to config"""
    _C.DISTILLATION = CN()
    _C.DISTILLATION.ALGORITHM = "LabelDistillation"
    _C.DISTILLATION.HELPER = "BaseDistillationHelper"
    _C.DISTILLATION.TEACHER = CN()
    _C.DISTILLATION.TEACHER.TORCHSCRIPT_FNAME = ""
    _C.DISTILLATION.TEACHER.DEVICE = ""


class PseudoLabeler:
    @abstractmethod
    def label(self, x):
        """
        We expect all pseudolabelers to implement a func called label which
        will then be run on the input before passing the func output to the
        model

        This is typically something like running a teacher model on the input
        to generate new ground truth which we can use to override the input
        gt
        """
        pass


class NoopPseudoLabeler(PseudoLabeler):
    def label(self, x):
        return x


class RelabelTargetInBatch(PseudoLabeler):
    """Run the teacher model on the batched inputs, replace targets.

    We expect the batched_inputs to be a list of dicts:
        batched_inputs = [
            {"input": ..., "target": ...},
            {"input": ..., "target": ...},
            ...
        ]
    where there is a single label "target" that needs to be replaced

    The teacher can take this batch of inputs directly and return a tensor
    of size nchw where n corresopnds to the index of the input

    We return updated batched_inputs with the new target
        new_batched_inputs = [
            {"input": ..., "target": teacher_output[0, :]},
            {"input": ..., "target": teacher_output[1, :]},
            ...
        ]

    Note that the output of the teacher is a tensor of NCHW while we assume
    the target is CHW. Create a new pseudo_labeler if a different input
    output is needed.
    """

    def __init__(self, teacher: nn.Module):
        """Assume that a teacher is passed to the psuedolabeler

        As an example in distillation, the distillaiton helper should create
        or pass along a teacher to the psuedo labeler
        """
        self.teacher = teacher

    def label(self, batched_inputs: List) -> List:
        batched_inputs = [
            {"input": d["input"].to(self.teacher.device), "target": d["target"]}
            for d in batched_inputs
        ]
        with torch.no_grad():
            batched_outputs = self.teacher(batched_inputs)

        for i, input in enumerate(batched_inputs):
            input["target"] = batched_outputs[i, :]

        return batched_inputs


@DISTILLATION_HELPER_REGISTRY.register()
class BaseDistillationHelper:
    """Example of what distillation helper can provide

    Users should inherit this class and replace any functions with whatever they
    need in order to customize their distillation given a specific distililation
    algorithm (e.g., user wants to change the name of the label in the inputs).

    The distillation helper is an object passed to the distillation algorithm so
    any functionality in the helper can be accessed in the algorithm
    """

    def __init__(self, cfg: CN, teacher: nn.Module):
        self.cfg = cfg
        self.teacher = teacher

    def get_pseudo_labeler(self) -> PseudoLabeler:
        """
        pseudo_labeler should update the labels in batched_inputs with teacher model
        results

        This dummy psuedo_labeler returns the batched_inputs without modification
        """
        return NoopPseudoLabeler()


@DISTILLATION_HELPER_REGISTRY.register()
class ExampleDistillationHelper(BaseDistillationHelper):
    """
    This is an example of a user customizing distillation.

    We return a pseudo labeler that can be used with a specific project
    where the training input is a list of dicts with a label called target
    """

    def get_pseudo_labeler(self) -> PseudoLabeler:
        return RelabelTargetInBatch(self.teacher)


class BaseDistillationAlgorithm(nn.Module):
    """
    Base distillation algorithm

    All distillation algorithms will be initialized with the same inputs including the
    teacher model, distillation helper and student class. Require user to define forward
    which overrides student model forward.

    Note that the init is unused when we use mixin. We manually set these attributes in
    the modeling hook. However we keep the init to make it clear what attributes the
    class will contain.
    """

    def dynamic_mixin_init(
        self,
        distillation_helper: BaseDistillationHelper,
    ):
        # check if we might override user attrs with same name
        # add any new distillation method attrs to this list
        assert not hasattr(
            self, "distillation_helper"
        ), "Distillation attempting to override attribute that already exists: distillation_helper"
        self.distillation_helper = distillation_helper

    def remove_dynamic_mixin(self):
        del self.distillation_helper

    @abstractmethod
    def forward(self, *args, **kwargs):
        """User required to override forward to implement distillation"""

        # must call super to ensure student forward is used when calling the
        # super in the algorithm (i.e., DistillationAlgorithm.super().forward())
        # this is because distillation algorithms inherit this base class so
        # the MRO of the mixin class is something like:
        #   [DistillationAlgorithm, BaseDistillationAlgorithm, StudentModel]
        # DistillationAlgorithm forward uses super().forward to call the
        # student model but the BaseDistillationAlgorithm is the next class
        # in the MRO so we make sure to call super on BaseDistillationAlgorithm
        # so we can access the StudentModel forward.
        return super().forward(*args, **kwargs)


@DISTILLATION_ALGORITHM_REGISTRY.register()
class LabelDistillation(BaseDistillationAlgorithm):
    """Basic distillation uses a teacher model to generate new labels used
    by the student

    We modify the forward to replace the input labels with teacher outputs when
    the model is training and run the student at inference
    """

    def dynamic_mixin_init(self, distillation_helper: BaseDistillationHelper):
        """Init pseudo labeler"""
        super().dynamic_mixin_init(distillation_helper)
        self.pseudo_labeler = self.distillation_helper.get_pseudo_labeler()

    def remove_dynamic_mixin(self):
        super().remove_dynamic_mixin()
        del self.pseudo_labeler

    def forward(self, batched_inputs: List):
        """If training, overrides input labels with teacher outputs

        During inference, runs the student.

        Note: The "student" model can be accessed by calling super(). In order
        to run the student forward method, we call super().forward(input) as opposed
        to super()(input) as super objects are not callable. We avoid calling
        super().__call__(input) as this leads to infinite recursion. We can call
        super().forward(input) without worrying about ignoring hooks as we should
        be calling this model as model(input) which will then activate the hooks.
        """
        if not self.training:
            return super().forward(batched_inputs)

        new_batched_inputs = self.pseudo_labeler.label(batched_inputs)
        return super().forward(new_batched_inputs)


@mh.MODELING_HOOK_REGISTRY.register()
class DistillationModelingHook(mh.ModelingHook):
    """Wrapper hook that allows us to apply different distillation algorithms
    based on config

    This is meant to be used after creating a model:
        def build_model(cfg):
            model = d2_build_model(cfg)
            distillation_modeling_hook = DistillationModelingHook(cfg)
            d2go.modeling_hook.apply_modeling_hooks(model, distillation_modeling_hook)

    The udpated model will then be updated with a forward func that corresponds
    to the distillation method in the cfg as well as any new methods
    """

    def __init__(self, cfg):
        """
        Set the three major components
            distillation_algorithm_class => the distillation algorithm to be used, we
              only get the class as the apply() will mixin the class
            distillation_helper => user customization of the algorithm
            teacher => all distillation algorithms utilize an additional model to
              modify inputs
        """
        super().__init__(cfg)
        self.teacher = _build_teacher(cfg)
        self.distillation_algorithm_class = DISTILLATION_ALGORITHM_REGISTRY.get(
            cfg.DISTILLATION.ALGORITHM
        )
        self.distillation_helper = DISTILLATION_HELPER_REGISTRY.get(
            cfg.DISTILLATION.HELPER
        )(cfg, self.teacher)

    def apply(self, model: nn.Module) -> nn.Module:
        """Use dynamic mixin to apply the distillation class

        As opposed to wrapping the model, dynamic mixin allows us to override the
        model methods so that the model retains all existing attributes the user expects
        (e.g., if the user thinks their is an attr called model.my_attr then dynamic mixin
        retains that property). This has the advantage over directly overriding the model
        forward as we can still call the original model forward using super:

            old_model: MyModel
            new_model: MyDistillationClass = DistillationModelingHook(...).apply(old_model)

            class MyDistillationClass:
                def forward(self, ...):
                    # do some processing
                    ...
                    super().forward(...)  # call MyModel.forward
                    ...
        """
        dynamic_mixin(
            model,
            self.distillation_algorithm_class,
            init_dict={
                "distillation_helper": self.distillation_helper,
            },
        )
        return model

    def unapply(self, model: nn.Module) -> nn.Module:
        """Remove distillation class using dynamic mixin with saved original class"""
        remove_dynamic_mixin(model)
        return model


def _build_teacher(cfg):
    """Create teacher using config settings

    Only supports torchscript
    """
    assert (
        cfg.DISTILLATION.TEACHER.TORCHSCRIPT_FNAME
    ), "Only supports teacher loaded as torchscript"

    torchscript_fname = cfg.DISTILLATION.TEACHER.TORCHSCRIPT_FNAME
    with PathManager.open(torchscript_fname, "rb") as f:
        ts = torch.jit.load(f)

    # move teacher to same device as student unless specified
    device = torch.device(cfg.DISTILLATION.TEACHER.DEVICE or cfg.MODEL.DEVICE)
    ts = ts.to(device)
    ts.device = device
    ts.eval()
    return ts
