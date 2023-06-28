#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved


import unittest

import numpy as np
from d2go.data.transforms import color_yuv as cy
from d2go.data.transforms.build import build_transform_gen
from d2go.runner import Detectron2GoRunner
from detectron2.data.transforms.augmentation import apply_augmentations


class TestDataTransformsColorYUV(unittest.TestCase):
    def test_yuv_color_transforms(self):
        default_cfg = Detectron2GoRunner.get_default_cfg()
        img = np.concatenate(
            [
                np.random.uniform(0, 1, size=(80, 60, 1)),
                np.random.uniform(-0.5, 0.5, size=(80, 60, 1)),
                np.random.uniform(-0.5, 0.5, size=(80, 60, 1)),
            ],
            axis=2,
        )

        default_cfg.D2GO_DATA.AUG_OPS.TRAIN = [
            'RandomContrastYUVOp::{"intensity_min": 0.3, "intensity_max": 0.5}',
        ]
        low_contrast_tfm = build_transform_gen(default_cfg, is_train=True)
        low_contrast, _ = apply_augmentations(low_contrast_tfm, img)

        default_cfg.D2GO_DATA.AUG_OPS.TRAIN = [
            'RandomSaturationYUVOp::{"intensity_min": 1.5, "intensity_max": 1.7}',
        ]
        high_saturation_tfm = build_transform_gen(default_cfg, is_train=True)
        high_saturation, _ = apply_augmentations(high_saturation_tfm, img)

        # Use pixel statistics to roughly check transformed images as expected
        # All channels have less variance
        self.assertLess(np.var(low_contrast[:, :, 0]), np.var(img[:, :, 0]))
        self.assertLess(np.var(low_contrast[:, :, 1]), np.var(img[:, :, 1]))
        self.assertLess(np.var(low_contrast[:, :, 2]), np.var(img[:, :, 2]))

        # 1st channel is unchanged (test w/ mean, var), 2nd + 3rd channels more variance
        self.assertAlmostEqual(np.mean(high_saturation[:, :, 0]), np.mean(img[:, :, 0]))
        self.assertAlmostEqual(np.var(high_saturation[:, :, 0]), np.var(img[:, :, 0]))
        self.assertGreater(np.var(high_saturation[:, :, 1]), np.var(img[:, :, 1]))
        self.assertGreater(np.var(high_saturation[:, :, 2]), np.var(img[:, :, 2]))

    def test_transform_color_yuv_rgbyuv_convert(self):
        image = np.arange(256).reshape(16, 16, 1).repeat(3, axis=2).astype(np.uint8)
        tf1 = cy.RGB2YUVBT601().get_transform(image)
        tf2 = cy.YUVBT6012RGB().get_transform(image)

        image_yuv = tf1.apply_image(image)
        image_rgb = tf2.apply_image(image_yuv)

        self.assertArrayEqual((image_rgb + 0.5).astype(np.uint8), image)

    def test_transform_color_yuv_rgbyuv_convert_invese(self):
        image = np.arange(256).reshape(16, 16, 1).repeat(3, axis=2).astype(np.uint8)

        tf = cy.RGB2YUVBT601().get_transform(image)

        image_yuv = tf.apply_image(image)
        image_rgb = tf.inverse().apply_image(image_yuv)

        self.assertArrayEqual((image_rgb + 0.5).astype(np.uint8), image)

    def assertArrayEqual(self, a1, a2):
        self.assertTrue(np.array_equal(a1, a2))
