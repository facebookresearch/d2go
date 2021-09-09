#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved


import unittest

import d2go.data.transforms.box_utils as bu
import numpy as np
import torch
from d2go.config import CfgNode
from d2go.data.transforms.build import build_transform_gen


def get_default_config():
    cfg = CfgNode()
    cfg.D2GO_DATA = CfgNode()
    cfg.D2GO_DATA.AUG_OPS = CfgNode()
    return cfg


class TestDataTransformsBoxUtils(unittest.TestCase):
    def test_min_box_ar(self):
        box_xywh = [4, 5, 10, 6]
        target_aspect_ratio = 1.0 / 2
        new_box = bu.get_min_box_aspect_ratio(box_xywh, target_aspect_ratio)
        self.assertArrayEqual(torch.Tensor([4, -2, 10, 20]), new_box)

    def test_get_box_from_mask(self):
        img_w, img_h = 8, 6
        mask = np.zeros([img_h, img_w])
        self.assertEqual(mask.shape, (img_h, img_w))
        mask[2:4, 3:6] = 1
        box = bu.get_box_from_mask(mask)
        self.assertEqual(box, (3, 2, 3, 2))

    def test_get_box_from_mask_union(self):
        img_w, img_h = 8, 6
        mask = np.zeros([img_h, img_w])
        self.assertEqual(mask.shape, (img_h, img_w))
        mask[2:4, 1:4] = 1
        mask[5:6, 4:8] = 1
        box = bu.get_box_from_mask(mask)
        self.assertEqual(box, (1, 2, 7, 4))

    def test_get_box_from_mask_empty(self):
        img_w, img_h = 8, 6
        mask = np.zeros([img_h, img_w])
        box = bu.get_box_from_mask(mask)
        self.assertIsNone(box)

    def test_scale_bbox_center(self):
        bbox = torch.Tensor([1, 2, 4, 5])
        out_bbox = bu.scale_bbox_center(bu.scale_bbox_center(bbox, 2.0), 0.5)
        self.assertArrayEqual(bbox, out_bbox)

    def assertArrayEqual(self, a1, a2):
        self.assertTrue(np.array_equal(a1, a2))

    def test_enlarge_bounding_box(self):
        default_cfg = get_default_config()

        default_cfg.D2GO_DATA.AUG_OPS.TRAIN = [
            'EnlargeBoundingBoxOp::{"fixed_pad": 20}',
            'EnlargeBoundingBoxOp::{"percentage": 0.2}',
        ]
        enlarge_box_tfm = build_transform_gen(default_cfg, is_train=True)

        boxes = np.array(
            [[91, 46, 144, 111]],
            dtype=np.float64,
        )
        transformed_bboxs = enlarge_box_tfm[0].apply_box(boxes)
        expected_bboxs = np.array(
            [[71, 26, 164, 131]],
            dtype=np.float64,
        )
        err_msg = "transformed_bbox = {}, expected {}".format(
            transformed_bboxs, expected_bboxs
        )
        self.assertTrue(np.allclose(transformed_bboxs, expected_bboxs), err_msg)

        boxes = np.array(
            [[91, 46, 144, 111]],
            dtype=np.float64,
        )
        transformed_bboxs = enlarge_box_tfm[1].apply_box(boxes)
        expected_bboxs = np.array(
            [[85.7, 39.5, 149.3, 117.5]],
            dtype=np.float64,
        )
        err_msg = "transformed_bbox = {}, expected {}".format(
            transformed_bboxs, expected_bboxs
        )
        self.assertTrue(np.allclose(transformed_bboxs, expected_bboxs), err_msg)

        boxes = np.array(
            [[[91, 46], [144, 111]]],
            dtype=np.float64,
        )
        transformed_bboxs = enlarge_box_tfm[1].apply_polygons(boxes)
        expected_bboxs = np.array(
            [[[85.7, 39.5], [149.3, 117.5]]],
            dtype=np.float64,
        )
        err_msg = "transformed_bbox = {}, expected {}".format(
            transformed_bboxs, expected_bboxs
        )
        self.assertTrue(np.allclose(transformed_bboxs, expected_bboxs), err_msg)

        dummy_data = np.array(
            [[91, 46, 144, 111]],
            dtype=np.float64,
        )
        dummy_data_out = enlarge_box_tfm[1].apply_image(dummy_data)
        expected_out = np.array(
            [[91, 46, 144, 111]],
            dtype=np.float64,
        )
        err_msg = "Apply image failed"
        self.assertTrue(np.allclose(dummy_data_out, expected_out), err_msg)

        default_cfg.D2GO_DATA.AUG_OPS.TRAIN = [
            'EnlargeBoundingBoxOp::{"fixed_pad": 20, "box_only": true}',
        ]
        enlarge_box_tfm = build_transform_gen(default_cfg, is_train=True)

        boxes = np.array([[91, 46, 144, 111]])
        transformed_bboxs = enlarge_box_tfm[0].apply_coords(boxes)
        err_msg = "transformed_bbox = {}, expected {}".format(transformed_bboxs, boxes)
        self.assertTrue(np.allclose(transformed_bboxs, boxes), err_msg)
