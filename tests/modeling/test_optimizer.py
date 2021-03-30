#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved


import torch
import unittest
from d2go.optimizer import build_optimizer_mapper
import d2go.runner.default_runner as default_runner

class TestArch(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 4, kernel_size=3, stride=1, padding=1)
        self.bn = torch.nn.BatchNorm2d(4)
        self.relu = torch.nn.ReLU(inplace=True)
        self.avgpool = torch.nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        ret = self.conv(x)
        ret = self.bn(ret)
        ret = self.relu(ret)
        ret = self.avgpool(ret)
        return ret

def _test_each_optimizer(cfg):
    model = TestArch()
    optimizer = build_optimizer_mapper(cfg, model)
    optimizer.zero_grad()
    for _ in range(10):
        x = torch.rand(1, 3, 24, 24)
        y = model(x)
        loss = y.mean()
        loss.backward()
        optimizer.step()

class TestOptimizer(unittest.TestCase):

    def test_all_optimiers(self):
        runner = default_runner.Detectron2GoRunner()
        cfg = runner.get_default_cfg()
        multipliers = [None, [{'conv': 0.1}]]

        for optimizer_name in ["SGD", "AdamW"]:
            for mult in multipliers:
                cfg.SOLVER.OPTIMIZER = optimizer_name
                cfg.SOLVER.MULTIPLIERS = mult
                _test_each_optimizer(cfg)

    def test_full_model_grad_clipping(self):
        runner = default_runner.Detectron2GoRunner()
        cfg = runner.get_default_cfg()

        for optimizer_name in ["SGD", "AdamW"]:
            cfg.SOLVER.CLIP_GRADIENTS.CLIP_VALUE = 0.2
            cfg.SOLVER.CLIP_GRADIENTS.ENABLED = True
            cfg.SOLVER.CLIP_GRADIENTS.CLIP_TYPE = "full_model"
            cfg.SOLVER.OPTIMIZER = optimizer_name
            _test_each_optimizer(cfg)

