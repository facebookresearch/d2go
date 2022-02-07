#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import unittest

import torch
from d2go.runner import create_runner
from detr.util.misc import nested_tensor_from_tensor_list
from fvcore.nn import flop_count_table, FlopCountAnalysis


class Tester(unittest.TestCase):
    @staticmethod
    def _set_detr_cfg(cfg, enc_layers, dec_layers, num_queries, dim_feedforward):
        cfg.MODEL.META_ARCHITECTURE = "Detr"
        cfg.MODEL.DETR.NUM_OBJECT_QUERIES = num_queries
        cfg.MODEL.DETR.ENC_LAYERS = enc_layers
        cfg.MODEL.DETR.DEC_LAYERS = dec_layers
        cfg.MODEL.DETR.DEEP_SUPERVISION = False
        cfg.MODEL.DETR.DIM_FEEDFORWARD = dim_feedforward  # 2048

    def _assert_model_output(self, model, scripted_model):
        x = nested_tensor_from_tensor_list(
            [torch.rand(3, 200, 200), torch.rand(3, 200, 250)]
        )
        out = model(x)
        out_script = scripted_model(x)
        self.assertTrue(out["pred_logits"].equal(out_script["pred_logits"]))
        self.assertTrue(out["pred_boxes"].equal(out_script["pred_boxes"]))

    def test_detr_res50_export(self):
        runner = create_runner("d2go.projects.detr.runner.DETRRunner")
        cfg = runner.get_default_cfg()
        cfg.MODEL.DEVICE = "cpu"
        # DETR
        self._set_detr_cfg(cfg, 6, 6, 100, 2048)
        # backbone
        cfg.MODEL.BACKBONE.NAME = "build_resnet_backbone"
        cfg.MODEL.RESNETS.DEPTH = 50
        cfg.MODEL.RESNETS.STRIDE_IN_1X1 = False
        cfg.MODEL.RESNETS.OUT_FEATURES = ["res2", "res3", "res4", "res5"]
        # build model
        model = runner.build_model(cfg).eval()
        model = model.detr
        scripted_model = torch.jit.script(model)
        self._assert_model_output(model, scripted_model)

    def test_detr_fbnet_export(self):
        runner = create_runner("d2go.projects.detr.runner.DETRRunner")
        cfg = runner.get_default_cfg()
        cfg.MODEL.DEVICE = "cpu"
        # DETR
        self._set_detr_cfg(cfg, 3, 3, 50, 256)
        # backbone
        cfg.MODEL.BACKBONE.NAME = "FBNetV2C4Backbone"
        cfg.MODEL.FBNET_V2.ARCH = "FBNetV3_A_dsmask_C5"
        cfg.MODEL.FBNET_V2.WIDTH_DIVISOR = 8
        cfg.MODEL.FBNET_V2.OUT_FEATURES = ["trunk4"]
        # build model
        model = runner.build_model(cfg).eval()
        model = model.detr
        print(model)
        scripted_model = torch.jit.script(model)
        self._assert_model_output(model, scripted_model)
        # print flops
        table = flop_count_table(FlopCountAnalysis(model, ([torch.rand(3, 224, 320)],)))
        print(table)
