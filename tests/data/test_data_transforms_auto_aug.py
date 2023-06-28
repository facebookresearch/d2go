#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved


import unittest

import numpy as np
from d2go.data.transforms.build import build_transform_gen
from d2go.runner import Detectron2GoRunner
from detectron2.data.transforms.augmentation import apply_augmentations


class TestDataTransformsAutoAug(unittest.TestCase):
    def test_rand_aug_transforms(self):
        default_cfg = Detectron2GoRunner.get_default_cfg()
        img = np.concatenate(
            [
                (np.random.uniform(0, 1, size=(80, 60, 1)) * 255).astype(np.uint8),
                (np.random.uniform(0, 1, size=(80, 60, 1)) * 255).astype(np.uint8),
                (np.random.uniform(0, 1, size=(80, 60, 1)) * 255).astype(np.uint8),
            ],
            axis=2,
        )

        default_cfg.D2GO_DATA.AUG_OPS.TRAIN = ['RandAugmentImageOp::{"num_ops": 20}']
        tfm = build_transform_gen(default_cfg, is_train=True)
        trans_img, _ = apply_augmentations(tfm, img)

        self.assertEqual(img.shape, trans_img.shape)
        self.assertEqual(img.dtype, trans_img.dtype)

    def test_trivial_aug_transforms(self):
        default_cfg = Detectron2GoRunner.get_default_cfg()
        img = np.concatenate(
            [
                (np.random.uniform(0, 1, size=(80, 60, 1)) * 255).astype(np.uint8),
            ],
            axis=2,
        )

        default_cfg.D2GO_DATA.AUG_OPS.TRAIN = ["TrivialAugmentWideImageOp"]
        tfm = build_transform_gen(default_cfg, is_train=True)
        trans_img, _ = apply_augmentations(tfm, img)

        self.assertEqual(img.shape, trans_img.shape)
        self.assertEqual(img.dtype, trans_img.dtype)

    def test_aug_mix_transforms(self):
        default_cfg = Detectron2GoRunner.get_default_cfg()
        img = np.concatenate(
            [
                (np.random.uniform(0, 1, size=(80, 60, 1)) * 255).astype(np.uint8),
                (np.random.uniform(0, 1, size=(80, 60, 1)) * 255).astype(np.uint8),
                (np.random.uniform(0, 1, size=(80, 60, 1)) * 255).astype(np.uint8),
            ],
            axis=2,
        )

        default_cfg.D2GO_DATA.AUG_OPS.TRAIN = ['AugMixImageOp::{"severity": 3}']
        tfm = build_transform_gen(default_cfg, is_train=True)
        trans_img, _ = apply_augmentations(tfm, img)

        self.assertEqual(img.shape, trans_img.shape)
        self.assertEqual(img.dtype, trans_img.dtype)
