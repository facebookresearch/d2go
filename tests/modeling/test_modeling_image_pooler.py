#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved


import unittest

import d2go.data.transforms.box_utils as bu
import d2go.modeling.image_pooler as image_pooler
import numpy as np
import torch
from d2go.utils.testing import rcnn_helper as rh
from detectron2.structures import Boxes


class TestModelingImagePooler(unittest.TestCase):
    def test_image_pooler(self):
        H, W = 8, 6
        image = torch.zeros(3, H, W)
        # xyxy
        boxes = torch.Tensor([[2, 3, 5, 7]])
        image[0, 3:7, 2:5] = 1
        image[1, 3:7, 2:5] = 2
        image[2, 3:7, 2:5] = 4

        img_pooler = image_pooler.ImagePooler(resize_short=6, resize_max=12).eval()
        pooled_img, pooled_box, transforms = img_pooler(image, boxes)

        # check pooled images
        self.assertEqual(pooled_img.shape, torch.Size([3, 8, 6]))
        self.assertArrayEqual(torch.unique(pooled_img[0, :, :]), [1])
        self.assertArrayEqual(torch.unique(pooled_img[1, :, :]), [2])
        self.assertArrayEqual(torch.unique(pooled_img[2, :, :]), [4])

        # check pooled boxes, in xyxy format
        self.assertArrayEqual(pooled_box, [[0, 0, 6, 8]])

        # inverse of transforms
        trans_inv = transforms.inverse()

        # inverse of boxes, xyxy
        inversed_box = trans_inv.apply_box(pooled_box)
        self.assertArrayEqual(inversed_box, boxes)

        pooled_sub_box = np.array([[2, 2, 4, 6]])
        inversed_sub_box = trans_inv.apply_box(pooled_sub_box)
        self.assertArrayEqual(inversed_sub_box, [[3, 4, 4, 6]])

    def test_image_pooler_scale_box(self):
        H, W = 8, 6
        image = torch.zeros(3, H, W)
        # xyxy
        boxes = torch.Tensor([[2, 3, 5, 7]])
        image[0, 3:7, 2:5] = 1
        image[1, 3:7, 2:5] = 2
        image[2, 3:7, 2:5] = 4

        img_pooler = image_pooler.ImagePooler(
            resize_type=None, box_scale_factor=4.0
        ).eval()
        pooled_img, pooled_box, transforms = img_pooler(image, boxes)

        # check pooled images
        self.assertEqual(pooled_img.shape, torch.Size([3, 8, 6]))
        self.assertArrayEqual(pooled_img, image)

        # check pooled boxes, in xyxy format, the box before scaling
        self.assertArrayEqual(pooled_box, [[2, 3, 5, 7]])

    def test_image_pooler_scale_box_large_crop_only(self):
        """Crop bbox"""
        H, W = 398, 224
        all_boxes = Boxes(torch.Tensor([[50, 40, 100, 80], [150, 60, 200, 120]]))
        image = rh.get_batched_inputs(1, (H, W), (H, W), all_boxes)[0]["image"]

        boxes = bu.get_box_union(all_boxes)
        self.assertArrayEqual(boxes.tensor, [[50, 40, 200, 120]])

        img_pooler = image_pooler.ImagePooler(
            resize_type=None, box_scale_factor=1.0
        ).eval()
        pooled_img, pooled_box, transforms = img_pooler(image, boxes.tensor)
        self.assertEqual(pooled_img.shape, torch.Size([3, 80, 150]))
        sub_boxes = rh.get_detected_instances_from_image([{"image": pooled_img}])[
            0
        ].pred_boxes
        self.assertArrayEqual(sub_boxes.tensor, [[0, 0, 50, 40], [100, 20, 150, 80]])

    def test_image_pooler_scale_box_large_crop_and_scale(self):
        """Crop bbox that is scaled"""
        H, W = 398, 224
        all_boxes = Boxes(torch.Tensor([[50, 40, 100, 80], [150, 60, 200, 120]]))
        image = rh.get_batched_inputs(1, (H, W), (H, W), all_boxes)[0]["image"]
        boxes = bu.get_box_union(all_boxes)

        img_pooler = image_pooler.ImagePooler(
            resize_type=None, box_scale_factor=1.2
        ).eval()
        pooled_img, pooled_box, transforms = img_pooler(image, boxes.tensor)
        self.assertEqual(pooled_img.shape, torch.Size([3, 96, 180]))

        # bbox with scaling in the original space
        orig_crop_box = transforms.inverse().apply_box(
            [0, 0, pooled_img.shape[2], pooled_img.shape[1]]
        )
        self.assertArrayEqual(orig_crop_box, [[35, 32, 215, 128]])

        sub_boxes = rh.get_detected_instances_from_image([{"image": pooled_img}])[
            0
        ].pred_boxes
        # gt_offset_xy = (50 - 35 = 15, 40 - 32 = 8)
        self.assertArrayEqual(sub_boxes.tensor, [[15, 8, 65, 48], [115, 28, 165, 88]])

    def test_image_pooler_scale_box_large_crop_scale_and_resize(self):
        """Crop bbox that is scaled, resize the cropped box"""
        H, W = 398, 224
        all_boxes = Boxes(torch.Tensor([[50, 40, 100, 80], [150, 60, 200, 120]]))
        image = rh.get_batched_inputs(1, (H, W), (H, W), all_boxes)[0]["image"]
        boxes = bu.get_box_union(all_boxes)

        img_pooler = image_pooler.ImagePooler(
            resize_type="resize_shortest",
            resize_short=48,
            resize_max=180,
            box_scale_factor=1.2,
        ).eval()
        pooled_img, pooled_box, transforms = img_pooler(image, boxes.tensor)
        self.assertEqual(pooled_img.shape, torch.Size([3, 48, 90]))

        # bbox with scaling in the original space
        orig_crop_box = transforms.inverse().apply_box(
            [0, 0, pooled_img.shape[2], pooled_img.shape[1]]
        )
        self.assertArrayEqual(orig_crop_box, [[35, 32, 215, 128]])

        # bbox without scaling in the original space
        orig_boxes = transforms.inverse().apply_box(pooled_box)
        self.assertArrayEqual(orig_boxes, boxes.tensor)

        sub_boxes = rh.get_detected_instances_from_image([{"image": pooled_img}])[
            0
        ].pred_boxes
        # [[7.5, 4, 32.5, 24], [57.5, 14, 82.5, 44]]
        self.assertArrayEqual(sub_boxes.tensor, [[7, 4, 33, 24], [57, 14, 83, 44]])

    def assertArrayEqual(self, a1, a2):
        self.assertTrue(np.array_equal(a1, a2))
