#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved


import unittest

import numpy as np
from d2go.data.transforms.build import build_transform_gen
from d2go.runner import Detectron2GoRunner
from detectron2.data.transforms.augmentation import apply_transform_gens


class TestDataTransforms(unittest.TestCase):
    def test_build_transform_gen(self):
        default_cfg = Detectron2GoRunner.get_default_cfg()
        default_cfg.INPUT.MIN_SIZE_TRAIN = (30,)
        default_cfg.INPUT.MIN_SIZE_TEST = 30

        trans_train = build_transform_gen(default_cfg, is_train=True)
        trans_test = build_transform_gen(default_cfg, is_train=False)

        img = np.zeros((80, 60, 3))
        trans_img_train, tl_train = apply_transform_gens(trans_train, img)
        trans_img_test, tl_test = apply_transform_gens(trans_test, img)

        self.assertEqual(trans_img_train.shape, (40, 30, 3))
        self.assertEqual(trans_img_test.shape, (40, 30, 3))

    def test_build_transform_gen_resize_square(self):
        default_cfg = Detectron2GoRunner.get_default_cfg()
        default_cfg.INPUT.MIN_SIZE_TRAIN = (30,)
        default_cfg.INPUT.MIN_SIZE_TEST = 40
        default_cfg.D2GO_DATA.AUG_OPS.TRAIN = ["ResizeShortestEdgeSquareOp"]
        default_cfg.D2GO_DATA.AUG_OPS.TEST = ["ResizeShortestEdgeSquareOp"]

        trans_train = build_transform_gen(default_cfg, is_train=True)
        trans_test = build_transform_gen(default_cfg, is_train=False)

        img = np.zeros((80, 60, 3))
        trans_img_train, tl_train = apply_transform_gens(trans_train, img)
        trans_img_test, tl_test = apply_transform_gens(trans_test, img)

        self.assertEqual(trans_img_train.shape, (30, 30, 3))
        self.assertEqual(trans_img_test.shape, (40, 40, 3))
