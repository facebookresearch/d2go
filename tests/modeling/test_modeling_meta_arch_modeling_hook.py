#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved


import unittest

import torch
from d2go.modeling.meta_arch import modeling_hook as mh


class TestArch(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x * 2


# create a wrapper of the model that add 1 to the output
class Wrapper(torch.nn.Module):
    def __init__(self, model: TestArch):
        super().__init__()
        self.model = model

    def forward(self, x):
        return self.model(x) + 1


class PlusOneHook(mh.ModelingHook):
    def __init__(self, cfg):
        super().__init__(cfg)

    def apply(self, model: torch.nn.Module) -> torch.nn.Module:
        return Wrapper(model)

    def unapply(self, model: torch.nn.Module) -> torch.nn.Module:
        assert isinstance(model, Wrapper)
        return model.model


class TestModelingHook(unittest.TestCase):
    def test_modeling_hook_simple(self):
        model = TestArch()
        hook = PlusOneHook(None)
        model_with_hook = hook.apply(model)
        self.assertEqual(model_with_hook(2), 5)
        original_model = hook.unapply(model_with_hook)
        self.assertEqual(model, original_model)
