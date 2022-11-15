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
from dataclasses import dataclass
from typing import Dict, List, Set, Union

import torch
import torch.nn as nn
from d2go.config import CfgNode as CN
from d2go.modeling import modeling_hook as mh
from d2go.registry.builtin import (
    DISTILLATION_ALGORITHM_REGISTRY,
    DISTILLATION_HELPER_REGISTRY,
    MODELING_HOOK_REGISTRY,
)
from detectron2.utils.file_io import PathManager
from mobile_cv.common.misc.mixin import dynamic_mixin, remove_dynamic_mixin


def add_distillation_configs(_C: CN) -> None:
    """Add default parameters to config

    The TEACHER.CONFIG field allows us to build a PyTorch model using an
    existing config.  We can build any model that is normally supported by
    D2Go (e.g., FBNet) because we just use the same config
    """
    _C.DISTILLATION = CN()
    _C.DISTILLATION.ALGORITHM = "LabelDistillation"
    _C.DISTILLATION.HELPER = "BaseDistillationHelper"
    _C.DISTILLATION.TEACHER = CN()
    _C.DISTILLATION.TEACHER.TORCHSCRIPT_FNAME = ""
    _C.DISTILLATION.TEACHER.DEVICE = ""
    _C.DISTILLATION.TEACHER.TYPE = "torchscript"
    _C.DISTILLATION.TEACHER.CONFIG_FNAME = ""
    _C.DISTILLATION.TEACHER.RUNNER_NAME = "d2go.runner.GeneralizedRCNNRunner"
    _C.DISTILLATION.TEACHER.OVERWRITE_OPTS = []


@dataclass
class LayerLossMetadata:
    loss: nn.Module
    name: str
    layer0: str
    layer1: str


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


@MODELING_HOOK_REGISTRY.register()
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


def _build_teacher(cfg) -> nn.Module:
    """Create teacher using config settings

    Supports torchscript or creating pytorch model using config.
    """
    _validate_teacher_config(cfg)
    if cfg.DISTILLATION.TEACHER.TYPE == "torchscript":
        with PathManager.open(cfg.DISTILLATION.TEACHER.TORCHSCRIPT_FNAME, "rb") as f:
            model = torch.jit.load(f)
    elif cfg.DISTILLATION.TEACHER.TYPE == "config":
        from d2go.runner import import_runner
        from d2go.setup import create_cfg_from_cli

        # teacher config may be set to cuda
        # if user wants to run teacher on cpu only machine by specifying teacher.device,
        # need to override device to cpu before building model
        if cfg.DISTILLATION.TEACHER.DEVICE:
            cfg.DISTILLATION.TEACHER.OVERWRITE_OPTS.extend(
                ["MODEL.DEVICE", cfg.DISTILLATION.TEACHER.DEVICE]
            )

        teacher_cfg = create_cfg_from_cli(
            cfg.DISTILLATION.TEACHER.CONFIG_FNAME,
            cfg.DISTILLATION.TEACHER.OVERWRITE_OPTS,
            cfg.DISTILLATION.TEACHER.RUNNER_NAME,
        )
        runner = import_runner(cfg.DISTILLATION.TEACHER.RUNNER_NAME)()
        model = runner.build_model(teacher_cfg, eval_only=True)
    else:
        raise ValueError(f"Unexpected teacher type: {cfg.DISTILLATION.TEACHER.TYPE}")

    # move teacher to same device as student unless specified
    device = torch.device(cfg.DISTILLATION.TEACHER.DEVICE or cfg.MODEL.DEVICE)
    model = _set_device(model, device)
    model.eval()
    return model


def _set_device(model: nn.Module, device: torch.device) -> nn.Module:
    """Set the device of the model

    Some D2Go models have device as a property of the model (e.g., GeneralizedRCNN)
    whereas others are missing this attribute which is assumed by distillation
    to exist (e.g., we may call teacher.device to move inputs)

    This helper function guarantees that the model.device attribute exists
    and runs model.to(device)
    """
    model = model.to(device)
    if not hasattr(model, "device"):
        model.device = device
    return model


def _validate_teacher_config(cfg: CN) -> None:
    """We support torchscript or PyTorch checkpoint as teacher models

    If torchscript, need:
        * torchscript_filename
    If config, needs:
        * config_fname
    """
    if cfg.DISTILLATION.TEACHER.TYPE == "torchscript":
        assert (
            cfg.DISTILLATION.TEACHER.TORCHSCRIPT_FNAME
        ), "Trying to load torchscript model without fname"
    elif cfg.DISTILLATION.TEACHER.TYPE == "config":
        assert (
            cfg.DISTILLATION.TEACHER.CONFIG_FNAME
        ), "Trying to load D2Go teacher model without config"
    else:
        raise ValueError(
            f"Unrecognized DISTILLATION.TEACHER.TYPE: {cfg.DISTILLATION.TEACHER.TYPE}"
        )


class CachedLayer(nn.Module):
    """Cached layer records the output of a layer

    This is meant to be used with dynamic mixin. The layer overrides the forward
    of the original layer such that the input and the output is the same but
    the output of the layer is saved to a dict that can be retrieved later
    """

    def dynamic_mixin_init(
        self,
        label: str,
        cache: Dict[
            str, Union[torch.Tensor, List[torch.Tensor], Dict[str, torch.Tensor]]
        ],
    ):
        self.label = label
        self.cache = cache

    def remove_dynamic_mixin(self):
        del self.label
        del self.cache

    def forward(self, *args, **kwargs):
        """Run the original layer and save the output

        We clone the output to avoid the case where a subsequent module
        runs an inplace operation. However, this limits what the cache
        can support as we can only run clone on a tensor so we need to
        check the type of the output.

        Support of the output type is limited to:
          * tensor
          * List[tensor]
          * Dict[str, tensor]
        """
        output = super().forward(*args, **kwargs)
        if isinstance(output, torch.Tensor):
            self.cache[self.label] = output.clone()
        elif isinstance(output, List):
            cloned_output = []
            for x in output:
                if isinstance(x, torch.Tensor):
                    cloned_output.append(x.clone())
                else:
                    raise ValueError(f"Unexpected type to save: {type(x)}")
            self.cache[self.label] = cloned_output
        elif isinstance(output, Dict):
            cloned_output = {}
            for k, v in output.items():
                if isinstance(v, torch.Tensor):
                    cloned_output[k] = v.clone()
                else:
                    raise ValueError(f"Unexpected type to save: {type(v)}")
            self.cache[self.label] = cloned_output
        else:
            raise ValueError(f"Unexpected type to save: {type(output)}")
        return output


def record_layers(model: nn.Module, layer_names: Set[str]) -> Dict[str, torch.Tensor]:
    """Save the outputs of layer_names in model

    Iterates over all named layers in model, applies cached layer to layers in
    layer_names. Returns dict which is used by the cached layers.
    """
    cache = {}
    for name, module in model.named_modules():
        if name in layer_names:
            dynamic_mixin(
                module,
                CachedLayer,
                init_dict={"label": name, "cache": cache},
            )
    return cache


def unrecord_layers(model: nn.Module, layer_names: Set[str]) -> None:
    """Remove cached layers based on the layer_names"""
    for name, module in model.named_modules():
        if name in layer_names:
            remove_dynamic_mixin(module)


def compute_layer_losses(
    layer_losses: List[LayerLossMetadata],
    layer0_cache: Dict[str, torch.Tensor],
    layer1_cache: Dict[str, torch.Tensor],
) -> Dict[str, torch.Tensor]:
    """Compute loss over layers specified in layer_loss

    layer0_cache and layer1_cache should contain the data required to compute
    the losses specified in layer_loss
    """
    losses = {}
    for ll in layer_losses:
        if ll.layer0 not in layer0_cache:
            raise ValueError(f"Missing saved layer {ll.layer0} in layer0_cache")
        if ll.layer1 not in layer1_cache:
            raise ValueError(f"Missing saved layer {ll.layer1} in layer1_cache")

        losses[ll.name] = ll.loss(layer0_cache[ll.layer0], layer1_cache[ll.layer1])
    return losses
