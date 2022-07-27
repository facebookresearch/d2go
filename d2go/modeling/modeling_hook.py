#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

from abc import abstractmethod
from typing import List, Tuple

import torch
from d2go.registry.builtin import MODELING_HOOK_REGISTRY


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
        """This function will be called when the users called model.unapply_modeling_hooks()
        after training. The main use case of the function is to remove the changes
        applied to the model in `apply`. The hooks will be called in reverse order
        as follow:
          model.unapply_modeling_hooks() == model_hook_N.unapply(model) ->
              model_hook_N-1.unapply(model) -> ... -> model_hook_1.unapply(model)
        """
        pass


def _build_modeling_hooks(cfg, hook_names: List[str]) -> List[ModelingHook]:
    """Build the hooks from cfg"""
    ret = [MODELING_HOOK_REGISTRY.get(name)(cfg) for name in hook_names]
    return ret


def _unapply_modeling_hook(
    model: torch.nn.Module, hooks: List[ModelingHook]
) -> torch.nn.Module:
    """Call unapply on the hooks in reversed order"""
    for hook in reversed(hooks):
        model = hook.unapply(model)
    return model


def _apply_modeling_hooks(
    model: torch.nn.Module, hooks: List[ModelingHook]
) -> torch.nn.Module:
    """Apply hooks on the model, users could call model.unapply_modeling_hooks()
    to return the model that removes all the hooks
    """
    if len(hooks) == 0:
        return model
    for hook in hooks:
        model = hook.apply(model)

    assert not hasattr(model, "_modeling_hooks")
    model._modeling_hooks = hooks

    def _unapply_modeling_hooks(self):
        assert hasattr(self, "_modeling_hooks")
        model = _unapply_modeling_hook(self, self._modeling_hooks)
        return model

    # add a function that could be used to unapply the modeling hooks
    assert not hasattr(model, "unapply_modeling_hooks")
    model.unapply_modeling_hooks = _unapply_modeling_hooks.__get__(model)

    return model


def build_and_apply_modeling_hooks(
    model: torch.nn.Module, cfg, hook_names: List[str]
) -> Tuple[torch.nn.Module, List[ModelingHook]]:
    """Build modeling hooks from cfg and apply hooks on the model. Users could
    call model.unapply_modeling_hooks() to return the model that removes all
    the hooks.
    """
    hooks = _build_modeling_hooks(cfg, hook_names)
    model = _apply_modeling_hooks(model, hooks)

    return model, hooks
