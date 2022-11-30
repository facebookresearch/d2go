#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved


import unittest

import d2go.data.transforms.box_utils as bu
import numpy as np
import torch
from d2go.data.transforms import crop as tf_crop


class TestDataTransformsCrop(unittest.TestCase):
    def test_transform_crop_extent_transform(self):
        img_wh = (16, 11)

        sem_seg = np.zeros([img_wh[1], img_wh[0]], dtype=np.uint8)
        # h, w
        sem_seg[5, 4] = 1
        sem_seg[10, 13] = 1
        sem_seg[5:11, 4:14] = 1

        # src_rect: [x0, y0, x1, y1] in pixel coordinate, output_size: [h, w]
        trans = tf_crop.ExtentTransform(src_rect=[4, 5, 14, 11], output_size=[6, 10])
        out_mask = trans.apply_segmentation(sem_seg)
        self.assertArrayEqual(out_mask.shape, torch.Tensor([6, 10]))
        self.assertArrayEqual(np.unique(out_mask), torch.Tensor([1]))

        trans = tf_crop.ExtentTransform(src_rect=[3, 4, 15, 11], output_size=[7, 12])
        out_mask = trans.apply_segmentation(sem_seg)
        self.assertArrayEqual(out_mask.shape, torch.Tensor([7, 12]))
        self.assertArrayEqual(np.unique(out_mask), torch.Tensor([0, 1]))
        self.assertArrayEqual(np.unique(out_mask[1:, 1:-1]), torch.Tensor([1]))
        self.assertEqual(out_mask[:, 0].sum(), 0)
        self.assertArrayEqual(out_mask[0, :].sum(), 0)
        self.assertArrayEqual(out_mask[:, -1].sum(), 0)

    def test_transform_crop_random_crop_fixed_aspect_ratio(self):
        aug = tf_crop.RandomCropFixedAspectRatio([1.0 / 2])
        img_wh = (16, 11)

        img = np.ones([img_wh[1], img_wh[0], 3], dtype=np.uint8)
        sem_seg = np.zeros([img_wh[1], img_wh[0]], dtype=np.uint8)
        sem_seg[5, 4] = 1
        sem_seg[10, 13] = 1
        mask_xywh = bu.get_box_from_mask(sem_seg)
        self.assertArrayEqual(mask_xywh, torch.Tensor([4, 5, 10, 6]))

        trans = aug.get_transform(img, sem_seg)
        self.assertArrayEqual(trans.src_rect, torch.Tensor([4, -2, 14, 18]))
        self.assertArrayEqual(trans.output_size, torch.Tensor([20, 10]))

        out_img = trans.apply_image(img)
        self.assertArrayEqual(out_img.shape, torch.Tensor([20, 10, 3]))

        self.assertArrayEqual(np.unique(out_img[2:13, :, :]), torch.Tensor([1]))
        self.assertArrayEqual(np.unique(out_img[0:2, :, :]), torch.Tensor([0]))
        self.assertArrayEqual(np.unique(out_img[13:, :, :]), torch.Tensor([0]))

        out_mask = trans.apply_segmentation(sem_seg)
        self.assertArrayEqual(out_mask.shape, torch.Tensor([20, 10]))
        self.assertEqual(out_mask[7, 0], 1)
        self.assertEqual(out_mask[12, -1], 1)

    def test_transform_crop_random_crop_fixed_aspect_ratio_scale_offset(self):
        aug = tf_crop.RandomCropFixedAspectRatio(
            [1.0 / 2], scale_range=[0.5, 0.5], offset_scale_range=[-0.5, -0.5]
        )
        img_wh = (16, 11)

        img = np.ones([img_wh[1], img_wh[0], 3], dtype=np.uint8)
        sem_seg = np.zeros([img_wh[1], img_wh[0]], dtype=np.uint8)
        sem_seg[5, 4] = 1
        sem_seg[10, 13] = 1
        sem_seg[5:11, 4:14] = 1
        mask_xywh = bu.get_box_from_mask(sem_seg)
        self.assertArrayEqual(mask_xywh, torch.Tensor([4, 5, 10, 6]))

        trans = aug.get_transform(img, sem_seg)
        self.assertArrayEqual(trans.src_rect, torch.Tensor([1.5, 0.0, 6.5, 10.0]))
        self.assertArrayEqual(trans.output_size, torch.Tensor([10, 5]))

        out_img = trans.apply_image(img)
        self.assertArrayEqual(out_img.shape, torch.Tensor([10, 5, 3]))
        self.assertEqual(np.unique(out_img), 1)

        out_mask = trans.apply_segmentation(sem_seg)
        self.assertArrayEqual(out_mask.shape, torch.Tensor([10, 5]))
        self.assertEqual(np.unique(out_mask[6:, 3:]), 1)

    def test_transform_crop_random_crop_fixed_aspect_ratio_empty_mask(self):
        """The sem_mask is empty (the whole image is background)"""
        aug = tf_crop.RandomCropFixedAspectRatio([1.0 / 2])
        img_wh = (16, 11)

        img = np.ones([img_wh[1], img_wh[0], 3], dtype=np.uint8)
        sem_seg = np.zeros([img_wh[1], img_wh[0]], dtype=np.uint8)

        mask_xywh = bu.get_box_from_mask(sem_seg)
        self.assertEqual(mask_xywh, None)

        trans = aug.get_transform(img, sem_seg)
        self.assertIsInstance(trans, tf_crop.NoOpTransform)

        out_img = trans.apply_image(img)
        self.assertArrayEqual(out_img.shape, img.shape)

        out_mask = trans.apply_segmentation(sem_seg)
        self.assertArrayEqual(out_mask.shape, sem_seg.shape)

    def test_pad_transform(self):
        crop_w, crop_h = 4, 3
        full_w, full_h = 11, 9
        crop_x, crop_y = 5, 6
        trans = tf_crop.PadTransform(crop_x, crop_y, crop_w, crop_h, full_w, full_h)
        img = np.ones([crop_h, crop_w])
        trans_img = trans.apply_image(img)
        self.assertArrayEqual(trans_img.shape, [full_h, full_w])
        self.assertArrayEqual(np.unique(trans_img), [0, 1])
        full_img_gt = np.zeros([full_h, full_w])
        full_img_gt[crop_y : (crop_y + crop_h), crop_x : (crop_x + crop_w)] = 1
        self.assertArrayEqual(full_img_gt, trans_img)

    def test_crop_transform_inverse(self):
        crop_w, crop_h = 4, 3
        full_w, full_h = 11, 9
        crop_x, crop_y = 5, 6
        trans = tf_crop.InvertibleCropTransform(
            crop_x, crop_y, crop_w, crop_h, full_w, full_h
        )

        full_img_gt = np.zeros([full_h, full_w])
        full_img_gt[crop_y : (crop_y + crop_h), crop_x : (crop_x + crop_w)] = 1
        crop_img_gt = np.ones([crop_h, crop_w])

        self.assertArrayEqual(trans.apply_image(full_img_gt), crop_img_gt)
        self.assertArrayEqual(trans.inverse().apply_image(crop_img_gt), full_img_gt)
        self.assertArrayEqual(
            trans.inverse().inverse().apply_image(full_img_gt), crop_img_gt
        )

    def test_pad_border_divisible_transform(self):
        img_h, img_w = 10, 7
        divisibility = 8

        aug = tf_crop.PadBorderDivisible(divisibility)

        img = np.ones([img_h, img_w, 3]) * 3
        trans = aug.get_transform(img)
        pad_img = trans.apply_image(img)
        self.assertEqual(pad_img.shape, (16, 8, 3))
        inverse_img = trans.inverse().apply_image(pad_img)
        self.assertEqual(inverse_img.shape, (10, 7, 3))
        self.assertArrayEqual(img, inverse_img)

        mask = np.ones([img_h, img_w]) * 2
        pad_mask = trans.apply_segmentation(mask)
        self.assertEqual(pad_mask.shape, (16, 8))
        inverse_mask = trans.inverse().apply_segmentation(pad_mask)
        self.assertEqual(inverse_mask.shape, (10, 7))
        self.assertArrayEqual(mask, inverse_mask)

    def test_pad_to_square_augmentation(self):
        img_h, img_w = 5, 3

        aug = tf_crop.PadToSquare(pad_value=255)

        img = np.ones([img_h, img_w, 3])
        trans = aug.get_transform(img)
        pad_img = trans.apply_image(img)
        self.assertEqual(pad_img.shape, (5, 5, 3))

    def test_random_instance_crop(self):
        from detectron2.data import detection_utils as du
        from detectron2.data.transforms.augmentation import AugInput, AugmentationList
        from detectron2.structures import BoxMode

        aug = tf_crop.RandomInstanceCrop([1.0, 1.0])

        img_w, img_h = 10, 7
        annotations = [
            {
                "category_id": 0,
                "bbox": [1, 1, 4, 3],
                "bbox_mode": BoxMode.XYWH_ABS,
            },
            {
                "category_id": 0,
                "bbox": [2, 2, 4, 3],
                "bbox_mode": BoxMode.XYWH_ABS,
            },
            {
                "category_id": 0,
                "bbox": [6, 5, 3, 2],
                "bbox_mode": BoxMode.XYWH_ABS,
            },
        ]

        img = np.ones([img_h, img_w, 3]) * 3

        inputs = AugInput(image=img)
        # pass additional arguments
        inputs.annotations = annotations
        transforms = AugmentationList([aug])(inputs)

        self.assertIn(
            inputs.image.shape, [torch.Size([3, 4, 3]), torch.Size([2, 3, 3])]
        )

        # from dataset mapper unused annotations will be filtered out due to the
        #  iscrowd flag
        image_shape = inputs.image.shape[:2]
        annos = [
            du.transform_instance_annotations(
                obj,
                transforms,
                image_shape,
            )
            for obj in annotations
            if obj.get("iscrowd", 0) == 0
        ]
        instances = du.annotations_to_instances(annos, image_shape)
        filtered_instances = du.filter_empty_instances(instances)
        self.assertEqual(len(filtered_instances), 1)
        self.assertArrayEqual(
            filtered_instances.gt_boxes.tensor.tolist(),
            [[0, 0, image_shape[1], image_shape[0]]],
        )

    def assertArrayEqual(self, a1, a2):
        self.assertTrue(np.array_equal(a1, a2))
