#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

from abc import abstractmethod
from typing import List

import torch
from detectron2.utils.registry import Registry

MODELING_HOOK_REGISTRY = Registry("MODELING_HOOK")  # noqa F401 isort:skip
MODELING_HOOK_REGISTRY.__doc__ = """
Registry for modeling hook.

The registered object will be called with `obj(cfg)`
and expected to return a `ModelingHook` object.
"""


class ModelingHook(object):
    """Modeling hooks provide a way to modify the model during the model building
    process. It is simple but allows users to modify the model by creating wrapper,
    override member functions, adding additional components, and loss etc.. It
    could be used to implement features such as QAT, model transformation for training,
    distillation/semi-supervised learning, and customization for loading pre-trained
    weights.
    """

    def __init__(self, cfg):
        self.cfg = cfg

    @abstractmethod
    def apply(self, model: torch.nn.Module) -> torch.nn.Module:
        """This function will called during the model building process to modify
        the behavior of the input model.
        The created model will be
          model == create meta arch -> model_hook_1.apply(model) ->
            model_hook_2.apply(model) -> ...
        """
        pass

    @abstractmethod
    def unapply(self, model: torch.nn.Module) -> torch.nn.Module:
        """This function will be called when the users called model.get_exportable_model()
        after training. The main use case of the function is to remove the changes
        applied to the model in `apply`. The hooks will be called in reverse order
        as follow:
          model.get_exportable_model() == model_hook_N.unapply(model) ->
              model_hook_N-1.unapply(model) -> ... -> model_hook_1.unapply(model)
        """
        pass


def build_modeling_hooks(cfg, hook_names: List[str]) -> List[ModelingHook]:
    """Build the hooks from cfg"""
    ret = [MODELING_HOOK_REGISTRY.get(name)(cfg) for name in hook_names]
    return ret


def apply_modeling_hooks(
    model: torch.nn.Module, hooks: List[ModelingHook]
) -> torch.nn.Module:
    """Apply hooks on the model"""
    for hook in hooks:
        model = hook.apply(model)
    return model
