#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved


import unittest

import numpy as np
from d2go.data.transforms.build import build_transform_gen
from d2go.runner import Detectron2GoRunner
from detectron2.data.transforms.augmentation import apply_augmentations


class TestDataTransformsBlur(unittest.TestCase):
    def test_gaussian_blur_transforms(self):
        default_cfg = Detectron2GoRunner.get_default_cfg()
        img = np.zeros((80, 60, 3)).astype(np.uint8)

        img[40, 30, :] = 255

        default_cfg.D2GO_DATA.AUG_OPS.TRAIN = [
            'RandomGaussianBlurOp::{"prob": 1.0, "k": 3, "sigma_range": [0.5, 0.5]}'
        ]
        tfm = build_transform_gen(default_cfg, is_train=True)
        trans_img, _ = apply_augmentations(tfm, img)

        self.assertEqual(img.shape, trans_img.shape)
        self.assertEqual(img.dtype, trans_img.dtype)

        self.assertEqual(trans_img[39, 29, 0], 3)
        self.assertEqual(trans_img[40, 29, 0], 21)
