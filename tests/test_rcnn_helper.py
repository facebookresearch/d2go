#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved


import unittest

import numpy as np
import torch
from d2go.utils.testing import rcnn_helper as rh
from detectron2.structures import Boxes


class TestRCNNHelper(unittest.TestCase):
    def test_get_instances_from_image(self):
        boxes = Boxes(torch.Tensor([[50, 40, 100, 80], [150, 60, 200, 120]]))
        gt_kpts = torch.Tensor([75, 60, 1.0] * 21 + [175, 90, 1.0] * 21).reshape(
            2, 21, 3
        )
        batched_inputs = rh.get_batched_inputs(2, boxes=boxes)
        instances = rh.get_detected_instances_from_image(batched_inputs)
        self.assertEqual(len(instances), 2)
        self.assertArrayEqual(instances[0].pred_boxes.tensor, boxes.tensor)
        self.assertArrayEqual(instances[0].pred_keypoints, gt_kpts)

    def test_get_instances_from_image_scale_image(self):
        H, W = 398, 224
        all_boxes = Boxes(torch.Tensor([[50, 40, 100, 80], [150, 60, 200, 120]]))
        image = rh.get_batched_inputs(1, (H, W), (H, W), all_boxes)[0]["image"]

        boxes = rh.get_detected_instances_from_image([{"image": image}])[0].pred_boxes
        self.assertArrayEqual(boxes.tensor, all_boxes.tensor)

        # scale image by 0.5
        scale_image = torch.nn.functional.interpolate(
            torch.unsqueeze(image, 0),
            scale_factor=(0.5, 0.5),
            mode="bilinear",
            align_corners=False,
            recompute_scale_factor=False,
        )[0]
        sub_boxes = rh.get_detected_instances_from_image([{"image": scale_image}])[
            0
        ].pred_boxes
        self.assertArrayEqual(sub_boxes.tensor, [[25, 20, 50, 40], [75, 30, 100, 60]])

        # scale image by 0.75
        scale_image = torch.nn.functional.interpolate(
            torch.unsqueeze(image, 0),
            scale_factor=(0.75, 0.75),
            mode="bilinear",
            align_corners=False,
            recompute_scale_factor=False,
        )[0]
        sub_boxes = rh.get_detected_instances_from_image([{"image": scale_image}])[
            0
        ].pred_boxes
        # [[37.5, 30, 75, 60], [112.5, 45, 150, 90]])
        self.assertArrayEqual(sub_boxes.tensor, [[37, 30, 75, 60], [112, 45, 150, 90]])

    def test_mock_rcnn_inference(self):
        image_size = (1920, 1080)
        resize_size = (398, 224)
        scale_xy = (1080.0 / 224, 1920.0 / 398)

        gt_boxes = Boxes(torch.Tensor([[50, 40, 100, 80], [150, 60, 200, 120]]))
        gt_kpts = torch.Tensor([75, 60, 1.0] * 21 + [175, 90, 1.0] * 21).reshape(
            2, 21, 3
        )

        # create inputs
        batched_inputs = rh.get_batched_inputs(2, image_size, resize_size, gt_boxes)

        # create model
        model = rh.MockRCNNInference(image_size, resize_size)

        # run without post processing
        det_instances = model(batched_inputs, None, do_postprocess=False)

        self.assertArrayAllClose(
            det_instances[0].pred_boxes.tensor,
            gt_boxes.tensor,
            atol=1e-4,
        )
        self.assertArrayAllClose(
            det_instances[0].pred_keypoints,
            gt_kpts,
            atol=1e-4,
        )

        # run with post processing
        det_instances = model(batched_inputs, None, do_postprocess=True)

        gt_boxes_scaled = gt_boxes.clone()
        gt_boxes_scaled.scale(*scale_xy)

        gt_kpts_scaled = torch.Tensor(
            [75 * scale_xy[0], 60 * scale_xy[1], 1.0] * 21
            + [175 * scale_xy[0], 90 * scale_xy[1], 1.0] * 21
        ).reshape(2, 21, 3)

        self.assertArrayAllClose(
            det_instances[0]["instances"].pred_boxes.tensor,
            gt_boxes_scaled.tensor,
            atol=1e-4,
        )
        self.assertArrayAllClose(
            det_instances[0]["instances"].pred_keypoints,
            gt_kpts_scaled,
            atol=1e-4,
        )

    def assertArrayEqual(self, a1, a2):
        self.assertTrue(np.array_equal(a1, a2))

    def assertArrayAllClose(self, a1, a2, rtol=1.0e-5, atol=1.0e-8):
        self.assertTrue(np.allclose(a1, a2, rtol=rtol, atol=atol))
