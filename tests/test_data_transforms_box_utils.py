#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved


import unittest

import d2go.data.transforms.box_utils as bu
import numpy as np
import torch


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
