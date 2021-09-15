#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved


import random
import unittest

import d2go.runner.default_runner as default_runner
import torch
from d2go.optimizer import build_optimizer_mapper


class TestArch(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 4, kernel_size=5, stride=1, padding=1)
        self.bn = torch.nn.BatchNorm2d(4)
        self.relu = torch.nn.ReLU(inplace=True)
        self.avgpool = torch.nn.AdaptiveAvgPool2d((1, 1))
        self.linear = torch.nn.Linear(4, 1)

    def forward(self, x):
        ret = self.conv(x)
        ret = self.bn(ret)
        ret = self.relu(ret)
        ret = self.avgpool(ret)
        ret = torch.transpose(ret, 1, 3)
        ret = self.linear(ret)
        return ret


def _test_each_optimizer(cfg):
    print("Solver: " + str(cfg.SOLVER.OPTIMIZER))

    model = TestArch()
    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = build_optimizer_mapper(cfg, model)
    optimizer.zero_grad()

    random.seed(20210912)
    for _ in range(2500):
        target = torch.empty(1, 1, 1, 1).fill_(random.randint(0, 1))
        x = torch.add(torch.rand(1, 3, 16, 16), 2 * target)
        y_pred = model(x)
        loss = criterion(y_pred, target)
        loss.backward()
        optimizer.step()

    n_correct = 0
    for _ in range(200):
        target = torch.empty(1, 1, 1, 1).fill_(random.randint(0, 1))
        x = torch.add(torch.rand(1, 3, 16, 16), 2 * target)
        y_pred = torch.round(torch.sigmoid(model(x)))
        if y_pred == target:
            n_correct += 1

    print("Correct prediction rate {0}.".format(n_correct / 200))


class TestOptimizer(unittest.TestCase):
    def test_all_optimizers(self):
        runner = default_runner.Detectron2GoRunner()
        cfg = runner.get_default_cfg()
        multipliers = [None, [{"conv": 0.1}]]

        for optimizer_name in ["SGD", "AdamW", "SGD_MT", "AdamW_MT"]:
            for mult in multipliers:
                cfg.SOLVER.BASE_LR = 0.01
                cfg.SOLVER.OPTIMIZER = optimizer_name
                cfg.SOLVER.MULTIPLIERS = mult
                _test_each_optimizer(cfg)

    def test_full_model_grad_clipping(self):
        runner = default_runner.Detectron2GoRunner()
        cfg = runner.get_default_cfg()

        for optimizer_name in ["SGD", "AdamW", "SGD_MT", "AdamW_MT"]:
            cfg.SOLVER.BASE_LR = 0.02
            cfg.SOLVER.CLIP_GRADIENTS.CLIP_VALUE = 0.2
            cfg.SOLVER.CLIP_GRADIENTS.ENABLED = True
            cfg.SOLVER.CLIP_GRADIENTS.CLIP_TYPE = "full_model"
            cfg.SOLVER.OPTIMIZER = optimizer_name
            _test_each_optimizer(cfg)
