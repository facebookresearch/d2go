#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved


import logging
import unittest

import torch
from d2go.runner import GeneralizedRCNNRunner
from detectron2.modeling import build_anchor_generator, build_backbone
from detectron2.modeling.proposal_generator import rpn

logger = logging.getLogger(__name__)


# overwrite configs if specified, otherwise default config is used
RPN_CFGS = {}


class TestRPNHeads(unittest.TestCase):
    def test_build_rpn_heads(self):
        """Make sure rpn heads run"""

        self.assertGreater(len(rpn.RPN_HEAD_REGISTRY._obj_map), 0)

        for name, builder in rpn.RPN_HEAD_REGISTRY._obj_map.items():
            logger.info("Testing {}...".format(name))
            cfg = GeneralizedRCNNRunner.get_default_cfg()
            if name in RPN_CFGS:
                cfg.merge_from_file(RPN_CFGS[name])

            backbone = build_backbone(cfg)
            backbone_shape = backbone.output_shape()
            rpn_input_shape = [backbone_shape[x] for x in cfg.MODEL.RPN.IN_FEATURES]
            rpn_head = builder(cfg, rpn_input_shape)

            in_channels = list(backbone_shape.values())[0].channels
            num_anchors = build_anchor_generator(cfg, rpn_input_shape).num_cell_anchors[
                0
            ]

            N, C_in, H, W = 2, in_channels, 24, 32
            input = torch.rand([N, C_in, H, W], dtype=torch.float32)
            LAYERS = len(cfg.MODEL.RPN.IN_FEATURES)
            out = rpn_head([input] * LAYERS)
            self.assertEqual(len(out), 2)
            logits, bbox_reg = out
            for idx in range(LAYERS):
                self.assertEqual(
                    logits[idx].shape,
                    torch.Size(
                        [input.shape[0], num_anchors, input.shape[2], input.shape[3]]
                    ),
                )
                self.assertEqual(
                    bbox_reg[idx].shape,
                    torch.Size(
                        [
                            logits[idx].shape[0],
                            num_anchors * 4,
                            logits[idx].shape[2],
                            logits[idx].shape[3],
                        ]
                    ),
                )

    def test_build_rpn_heads_with_rotated_anchor_generator(self):
        """Make sure rpn heads work with rotated anchor generator"""

        self.assertGreater(len(rpn.RPN_HEAD_REGISTRY._obj_map), 0)

        for name, builder in rpn.RPN_HEAD_REGISTRY._obj_map.items():
            logger.info("Testing {}...".format(name))
            cfg = GeneralizedRCNNRunner.get_default_cfg()
            if name in RPN_CFGS:
                cfg.merge_from_file(RPN_CFGS[name])

            cfg.MODEL.ANCHOR_GENERATOR.NAME = "RotatedAnchorGenerator"

            backbone = build_backbone(cfg)
            backbone_shape = backbone.output_shape()
            rpn_input_shape = [backbone_shape[x] for x in cfg.MODEL.RPN.IN_FEATURES]
            rpn_head = builder(cfg, rpn_input_shape)

            in_channels = list(backbone_shape.values())[0].channels
            anchor_generator = build_anchor_generator(cfg, rpn_input_shape)
            num_anchors = anchor_generator.num_cell_anchors[0]
            box_dim = anchor_generator.box_dim

            N, C_in, H, W = 2, in_channels, 24, 32
            input = torch.rand([N, C_in, H, W], dtype=torch.float32)
            LAYERS = len(cfg.MODEL.RPN.IN_FEATURES)
            out = rpn_head([input] * LAYERS)
            self.assertEqual(len(out), 2)
            logits, bbox_reg = out
            for idx in range(LAYERS):
                self.assertEqual(
                    logits[idx].shape,
                    torch.Size(
                        [input.shape[0], num_anchors, input.shape[2], input.shape[3]]
                    ),
                )
                self.assertEqual(
                    bbox_reg[idx].shape,
                    torch.Size(
                        [
                            logits[idx].shape[0],
                            num_anchors * box_dim,
                            logits[idx].shape[2],
                            logits[idx].shape[3],
                        ]
                    ),
                )


if __name__ == "__main__":
    unittest.main()
