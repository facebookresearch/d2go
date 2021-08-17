#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved


import numpy as np
import unittest

from d2go.modeling.trimap import generate_boundary_weight_mask


class TestTrimap(unittest.TestCase):

    def test_generate_boundary_weight_mask(self):
        # smallest size mask to not give dilation too small warning
        mask_gt = np.zeros((115, 252))
        mask_gt[54:60, 123:128] = 1

        mask_dt = np.zeros((115, 252))
        mask_dt[55:61, 124:129] = 1

        weights = generate_boundary_weight_mask(mask_gt, mask_dt, non_bd_weight=0.1)
        unique_values = np.unique(weights)
        self.assertEqual(unique_values.size, 2)
        self.assertEqual(10 * unique_values[0], unique_values[1])

    def test_generate_boundary_weight_mask_from_negative_img_mask(self):
        # smallest size mask to not give dilation too small warning
        mask_gt = np.zeros((115, 252))
        mask_dt = np.zeros((115, 252))

        try:
            generate_boundary_weight_mask(mask_gt, mask_dt, non_bd_weight=0.0)
        except ZeroDivisionError:
            self.fail(
                "generate_boundary_weight_mask raised ZeroDivisionError unexpectedly")
