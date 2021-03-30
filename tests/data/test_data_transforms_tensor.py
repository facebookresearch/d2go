#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved


import unittest

import numpy as np
import torch
from d2go.data.transforms import tensor as tensor_aug
from detectron2.data.transforms.augmentation import AugmentationList


class TestDataTransformsTensor(unittest.TestCase):
    def test_tensor_aug(self):
        """Data augmentation that that allows torch.Tensor as input"""

        img = torch.ones(3, 8, 6)
        augs = [tensor_aug.Tensor2Array(), tensor_aug.Array2Tensor()]

        inputs = tensor_aug.AugInput(image=img)
        transforms = AugmentationList(augs)(inputs)
        self.assertArrayEqual(img, inputs.image)

        # inverse is the same as itself
        out_img = transforms.inverse().apply_image(img)
        self.assertArrayEqual(img, out_img)

    def assertArrayEqual(self, a1, a2):
        self.assertTrue(np.array_equal(a1, a2))
