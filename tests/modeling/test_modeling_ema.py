#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved


import copy
import itertools
import unittest

import d2go.runner.default_runner as default_runner
import torch
from d2go.modeling import ema
from d2go.utils.testing import helper


class TestArch(torch.nn.Module):
    def __init__(self, value=None, int_value=None):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 4, kernel_size=3, stride=1, padding=1)
        self.bn = torch.nn.BatchNorm2d(4)
        self.relu = torch.nn.ReLU(inplace=True)
        self.avgpool = torch.nn.AdaptiveAvgPool2d((1, 1))
        if value is not None:
            self.set_const_weights(value, int_value)

    def forward(self, x):
        ret = self.conv(x)
        ret = self.bn(ret)
        ret = self.relu(ret)
        ret = self.avgpool(ret)
        return ret

    def set_const_weights(self, value, int_value=None):
        if int_value is None:
            int_value = int(value)
        for x in itertools.chain(self.parameters(), self.buffers()):
            if x.dtype == torch.float32:
                x.data.fill_(value)
            else:
                x.data.fill_(int_value)


def _compare_state_dict(model1, model2, abs_error=1e-3):
    sd1 = model1.state_dict()
    sd2 = model2.state_dict()
    if len(sd1) != len(sd2):
        return False
    if set(sd1.keys()) != set(sd2.keys()):
        return False
    for name in sd1:
        if sd1[name].dtype == torch.float32:
            if torch.abs((sd1[name] - sd2[name])).max() > abs_error:
                return False
        elif (sd1[name] != sd2[name]).any():
            return False
    return True


class TestModelingModelEMA(unittest.TestCase):
    def test_emastate(self):
        model = TestArch()
        state = ema.EMAState.FromModel(model)
        # two for conv (conv.weight, conv.bias),
        # five for bn (bn.weight, bn.bias, bn.running_mean, bn.running_var, bn.num_batches_tracked)
        full_state = {
            "conv.weight",
            "conv.bias",
            "bn.weight",
            "bn.bias",
            "bn.running_mean",
            "bn.running_var",
            "bn.num_batches_tracked",
        }
        self.assertEqual(len(state.state), 7)
        self.assertTrue(set(state.state) == full_state)

        for _, val in state.state.items():
            self.assertFalse(val.requires_grad)

        model1 = TestArch()
        self.assertFalse(_compare_state_dict(model, model1))

        state.apply_to(model1)
        self.assertTrue(_compare_state_dict(model, model1))

        # test ema state that excludes buffers and frozen parameters
        model.conv.weight.requires_grad = False
        state1 = ema.EMAState.FromModel(model, include_frozen=False)
        # should exclude frozen parameter: conv.weight
        self.assertTrue(full_state - set(state1.state) == {"conv.weight"})

        state2 = ema.EMAState.FromModel(model, include_buffer=False)
        # should exclude buffers: bn.running_mean, bn.running_var, bn.num_batches_tracked
        self.assertTrue(
            full_state - set(state2.state)
            == {"bn.running_mean", "bn.running_var", "bn.num_batches_tracked"}
        )

        state3 = ema.EMAState.FromModel(
            model, include_frozen=False, include_buffer=False
        )
        # should exclude frozen param + buffers: conv.weight, bn.running_mean, bn.running_var, bn.num_batches_tracked
        self.assertTrue(set(state3.state) == {"conv.bias", "bn.weight", "bn.bias"})

    def test_emastate_saveload(self):
        model = TestArch()
        state = ema.EMAState.FromModel(model)

        model1 = TestArch()
        self.assertFalse(_compare_state_dict(model, model1))

        state1 = ema.EMAState()
        state1.load_state_dict(state.state_dict())
        state1.apply_to(model1)
        self.assertTrue(_compare_state_dict(model, model1))

    @helper.skip_if_no_gpu
    def test_emastate_crossdevice(self):
        model = TestArch()
        model.cuda()
        # state on gpu
        state = ema.EMAState.FromModel(model)
        self.assertEqual(state.device, torch.device("cuda:0"))
        # target model on cpu
        model1 = TestArch()
        state.apply_to(model1)
        self.assertEqual(next(model1.parameters()).device, torch.device("cpu"))
        self.assertTrue(_compare_state_dict(copy.deepcopy(model).cpu(), model1))

        # state on cpu
        state1 = ema.EMAState.FromModel(model, device="cpu")
        self.assertEqual(state1.device, torch.device("cpu"))
        # target model on gpu
        model2 = TestArch()
        model2.cuda()
        state1.apply_to(model2)
        self.assertEqual(next(model2.parameters()).device, torch.device("cuda:0"))
        self.assertTrue(_compare_state_dict(model, model2))

    def test_ema_updater(self):
        model = TestArch()
        state = ema.EMAState()

        updated_model = TestArch()

        updater = ema.EMAUpdater(state, decay=0.0)
        updater.init_state(model)
        for _ in range(3):
            cur = TestArch()
            updater.update(cur)
            state.apply_to(updated_model)
            # weight decay == 0.0, always use new model
            self.assertTrue(_compare_state_dict(updated_model, cur))

        updater = ema.EMAUpdater(state, decay=1.0)
        updater.init_state(model)
        for _ in range(3):
            cur = TestArch()
            updater.update(cur)
            state.apply_to(updated_model)
            # weight decay == 1.0, always use init model
            self.assertTrue(_compare_state_dict(updated_model, model))

    def test_ema_updater_decay(self):
        state = ema.EMAState()

        updater = ema.EMAUpdater(state, decay=0.7)
        updater.init_state(TestArch(1.0))
        gt_val = 1.0
        gt_val_int = 1
        for idx in range(3):
            updater.update(TestArch(float(idx)))
            updated_model = state.get_ema_model(TestArch())
            gt_val = gt_val * 0.7 + float(idx) * 0.3
            gt_val_int = int(gt_val_int * 0.7 + float(idx) * 0.3)
            self.assertTrue(
                _compare_state_dict(updated_model, TestArch(gt_val, gt_val_int))
            )


class TestModelingModelEMAHook(unittest.TestCase):
    def test_ema_hook(self):
        runner = default_runner.Detectron2GoRunner()
        cfg = runner.get_default_cfg()
        cfg.MODEL.DEVICE = "cpu"
        cfg.MODEL_EMA.ENABLED = True
        # use new model weights
        cfg.MODEL_EMA.DECAY = 0.0
        cfg.MODEL_EMA.DECAY_WARM_UP_FACTOR = -1

        model = TestArch()
        ema.may_build_model_ema(cfg, model)
        self.assertTrue(hasattr(model, "ema_state"))

        ema_hook = ema.EMAHook(cfg, model)
        ema_hook.before_train()
        ema_hook.before_step()
        model.set_const_weights(2.0)
        ema_hook.after_step()
        ema_hook.after_train()

        ema_checkpointers = ema.may_get_ema_checkpointer(cfg, model)
        self.assertEqual(len(ema_checkpointers), 1)

        out_model = TestArch()
        ema_checkpointers["ema_state"].apply_to(out_model)
        self.assertTrue(_compare_state_dict(out_model, model))
