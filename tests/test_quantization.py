#!/usr/bin/env python3

import copy
import unittest

import mobile_cv.arch.fbnet_v2.basic_blocks as bb
import torch
import torch.nn as nn
from common.utils_pytorch.model_utils import has_module
from d2go.config import CfgNode as CN
from d2go.modeling.quantization import (
    QATCheckpointer,
    add_quantization_default_configs,
    setup_qat_model,
)
from torch.quantization import DeQuantStub, QuantStub


class TestQuantization(unittest.TestCase):
    def test_setup_qat_model(self):
        """Check that setup model creates qat model"""
        cfg = CN()
        add_quantization_default_configs(cfg)
        cfg.QUANTIZATION.QAT.ENABLED = True
        model = bb.ConvBNRelu(1, 1, "conv", "bn", None)
        model.device = "cpu"
        qat_model = setup_qat_model(cfg, copy.deepcopy(model), enable_fake_quant=False)

        # check that the QAT model has observers
        self.assertTrue(QATCheckpointer._is_q_state_dict(qat_model.state_dict()))

        # qat should produce the same output as the fp32 model as fake_quant is disabled
        input = torch.randn(2, 1, 1, 1)
        gt = model(input)
        output = qat_model(input)
        self.assertTrue(torch.allclose(output, gt))

    def test_setup_qat_model_custom_qscheme(self):
        """Check that setup_qat will run custom_qscheme"""

        class TestMetaArch(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = nn.Conv2d(1, 1, 1, bias=False)
                self.device = "cpu"

            def forward(self, x):
                return self.conv(x)

            def prepare_for_quant(self, cfg):
                self.qconfig = torch.quantization.get_default_qat_qconfig(
                    cfg.QUANTIZATION.BACKEND
                )
                # optionally quantize the conv layer
                if cfg.QUANTIZATION.CUSTOM_QSCHEME == "test_qscheme":
                    self.conv = nn.Sequential(QuantStub(), self.conv, DeQuantStub())
                return self

        cfg = CN()
        add_quantization_default_configs(cfg)
        cfg.QUANTIZATION.QAT.ENABLED = True
        model = TestMetaArch()

        # model custom_qscheme
        cfg.QUANTIZATION.CUSTOM_QSCHEME = "test_qscheme"
        qat_model = setup_qat_model(cfg, copy.deepcopy(model), enable_fake_quant=False)
        self.assertTrue(QATCheckpointer._is_q_state_dict(qat_model.state_dict()))
        self.assertTrue(has_module(qat_model, QuantStub))

        # disable custom q_scheme
        cfg.QUANTIZATION.CUSTOM_QSCHEME = ""
        qat_model = setup_qat_model(cfg, copy.deepcopy(model), enable_fake_quant=False)
        self.assertTrue(QATCheckpointer._is_q_state_dict(qat_model.state_dict()))
        self.assertFalse(has_module(qat_model, QuantStub))
