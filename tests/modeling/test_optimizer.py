#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved


import random
import unittest

import d2go.runner.default_runner as default_runner
import torch
from d2go.optimizer.build import build_optimizer_mapper
from d2go.utils.testing import helper


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


def _test_each_optimizer(cfg, cuda: bool = False):
    print("Solver: " + str(cfg.SOLVER.OPTIMIZER))
    device = "cuda:0" if cuda else "cpu"

    model = TestArch().to(device)
    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = build_optimizer_mapper(cfg, model)
    optimizer.zero_grad()

    random.seed(20210912)
    num_iter = 500
    for _ in range(num_iter):
        target = torch.empty(1, 1, 1, 1).fill_(random.randint(0, 1)).to(device)
        noise = torch.rand(1, 3, 16, 16).to(device)
        x = torch.add(noise, 2 * target)
        y_pred = model(x)
        loss = criterion(y_pred, target)
        loss.backward()
        optimizer.step()

    n_correct = 0
    n_eval = 100
    for _ in range(n_eval):
        target = torch.empty(1, 1, 1, 1).fill_(random.randint(0, 1)).to(device)
        x = torch.add(torch.rand(1, 3, 16, 16).to(device), 2 * target)
        y_pred = torch.round(torch.sigmoid(model(x)))
        if y_pred == target:
            n_correct += 1

    print("Correct prediction rate {0}.".format(n_correct / n_eval))


def _check_param_group(self, group, num_params=None, **kwargs):
    if num_params is not None:
        self.assertEqual(len(group["params"]), num_params)
    for key, val in kwargs.items():
        self.assertEqual(group[key], val)


def get_optimizer_cfg(
    lr,
    weight_decay=None,
    weight_decay_norm=None,
    weight_decay_bias=None,
    lr_mult=None,
):
    runner = default_runner.Detectron2GoRunner()
    cfg = runner.get_default_cfg()
    if lr is not None:
        cfg.SOLVER.BASE_LR = lr
    if weight_decay is not None:
        cfg.SOLVER.WEIGHT_DECAY = weight_decay
    if weight_decay_norm is not None:
        cfg.SOLVER.WEIGHT_DECAY_NORM = weight_decay_norm
    if weight_decay_bias is not None:
        cfg.SOLVER.WEIGHT_DECAY_BIAS = weight_decay_bias
    if lr_mult is not None:
        cfg.SOLVER.LR_MULTIPLIER_OVERWRITE = [lr_mult]
    return cfg


class TestOptimizer(unittest.TestCase):
    def test_create_optimizer_default(self):
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = torch.nn.Conv2d(3, 3, 1)
                self.bn = torch.nn.BatchNorm2d(3)

            def forward(self, x):
                return self.bn(self.conv(x))

        model = Model()
        cfg = get_optimizer_cfg(
            lr=1.0, weight_decay=1.0, weight_decay_norm=1.0, weight_decay_bias=1.0
        )
        optimizer = build_optimizer_mapper(cfg, model)
        self.assertEqual(len(optimizer.param_groups), 1)
        _check_param_group(
            self, optimizer.param_groups[0], num_params=4, weight_decay=1.0, lr=1.0
        )

    def test_create_optimizer_lr(self):
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.conv1 = torch.nn.Conv2d(3, 3, 1)
                self.conv2 = torch.nn.Conv2d(3, 3, 1)
                self.bn = torch.nn.BatchNorm2d(3)

            def forward(self, x):
                return self.bn(self.conv2(self.conv1(x)))

        model = Model()
        cfg = get_optimizer_cfg(
            lr=1.0,
            lr_mult={"conv1": 3.0, "conv2": 3.0},
            weight_decay=2.0,
            weight_decay_norm=2.0,
        )
        optimizer = build_optimizer_mapper(cfg, model)

        self.assertEqual(len(optimizer.param_groups), 2)

        _check_param_group(self, optimizer.param_groups[0], num_params=4, lr=3.0)
        _check_param_group(self, optimizer.param_groups[1], num_params=2, lr=1.0)

    def test_create_optimizer_weight_decay_norm(self):
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = torch.nn.Conv2d(3, 3, 1)
                self.bn = torch.nn.BatchNorm2d(3)

            def forward(self, x):
                return self.bn(self.conv(x))

        model = Model()
        cfg = get_optimizer_cfg(
            lr=1.0, weight_decay=1.0, weight_decay_norm=2.0, weight_decay_bias=1.0
        )
        optimizer = build_optimizer_mapper(cfg, model)

        self.assertEqual(len(optimizer.param_groups), 2)

        _check_param_group(
            self, optimizer.param_groups[0], num_params=2, lr=1.0, weight_decay=1.0
        )
        _check_param_group(
            self, optimizer.param_groups[1], num_params=2, lr=1.0, weight_decay=2.0
        )

    OPTIMIZER_NAMES_PART1 = ["SGD", "AdamW", "SGD_MT"]
    OPTIMIZER_NAMES_PART2 = ["AdamW_MT", "Adam"]

    def _test_optimizers_list(self, optimizers_list, fused: bool = False):
        runner = default_runner.Detectron2GoRunner()
        cfg = runner.get_default_cfg()
        multipliers = [None, [{"conv": 0.1}]]

        for optimizer_name in optimizers_list:
            for mult in multipliers:
                cfg.SOLVER.BASE_LR = 0.01
                cfg.SOLVER.FUSED = fused
                cfg.SOLVER.OPTIMIZER = optimizer_name
                cfg.SOLVER.MULTIPLIERS = mult
                _test_each_optimizer(cfg, cuda=fused)

    def test_all_optimizers_part_1(self):
        self._test_optimizers_list(self.OPTIMIZER_NAMES_PART1)

    def test_all_optimizers_part_2(self):
        self._test_optimizers_list(self.OPTIMIZER_NAMES_PART2)

    def _test_full_model_grad_clipping(self, optimizers_list):
        runner = default_runner.Detectron2GoRunner()
        cfg = runner.get_default_cfg()

        for optimizer_name in optimizers_list:
            cfg.SOLVER.BASE_LR = 0.02
            cfg.SOLVER.CLIP_GRADIENTS.CLIP_VALUE = 0.2
            cfg.SOLVER.CLIP_GRADIENTS.ENABLED = True
            cfg.SOLVER.CLIP_GRADIENTS.CLIP_TYPE = "full_model"
            cfg.SOLVER.OPTIMIZER = optimizer_name
            _test_each_optimizer(cfg)

    def test_full_model_grad_clipping_part1(self):
        self._test_full_model_grad_clipping(self.OPTIMIZER_NAMES_PART1)

    def test_full_model_grad_clipping_part2(self):
        self._test_full_model_grad_clipping(self.OPTIMIZER_NAMES_PART2)

    def test_create_optimizer_custom(self):
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = torch.nn.Conv2d(3, 3, 1)
                self.bn = torch.nn.BatchNorm2d(3)

            def forward(self, x):
                return self.bn(self.conv(x))

            def get_optimizer_param_groups(self, _opts):
                ret = [
                    {
                        "params": [self.conv.weight],
                        "lr": 10.0,
                    }
                ]
                return ret

        model = Model()
        cfg = get_optimizer_cfg(lr=1.0, weight_decay=1.0, weight_decay_norm=0.0)
        optimizer = build_optimizer_mapper(cfg, model)

        self.assertEqual(len(optimizer.param_groups), 3)

        _check_param_group(
            self, optimizer.param_groups[0], num_params=1, lr=10.0, weight_decay=1.0
        )
        _check_param_group(
            self, optimizer.param_groups[1], num_params=1, lr=1.0, weight_decay=1.0
        )
        _check_param_group(
            self, optimizer.param_groups[2], num_params=2, lr=1.0, weight_decay=0.0
        )

    @helper.enable_ddp_env()
    def test_create_optimizer_custom_ddp(self):
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = torch.nn.Conv2d(3, 3, 1)
                self.bn = torch.nn.BatchNorm2d(3)

            def forward(self, x):
                return self.bn(self.conv(x))

            def get_optimizer_param_groups(self, _opts):
                ret = [
                    {
                        "params": [self.conv.weight],
                        "lr": 10.0,
                    }
                ]
                return ret

        model = Model()
        model = torch.nn.parallel.DistributedDataParallel(model)
        cfg = get_optimizer_cfg(lr=1.0, weight_decay=1.0, weight_decay_norm=0.0)
        optimizer = build_optimizer_mapper(cfg, model)

        self.assertEqual(len(optimizer.param_groups), 3)

        _check_param_group(
            self, optimizer.param_groups[0], num_params=1, lr=10.0, weight_decay=1.0
        )
        _check_param_group(
            self, optimizer.param_groups[1], num_params=1, lr=1.0, weight_decay=1.0
        )
        _check_param_group(
            self, optimizer.param_groups[2], num_params=2, lr=1.0, weight_decay=0.0
        )
