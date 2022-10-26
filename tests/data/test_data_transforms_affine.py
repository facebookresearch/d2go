#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import json
import unittest
from typing import Tuple

import cv2
import numpy as np
import torchvision.transforms as T
from d2go.data.transforms.build import build_transform_gen
from d2go.runner import Detectron2GoRunner
from detectron2.data.transforms import apply_augmentations


def generate_test_data(
    source_img: np.ndarray,
    angle: float = 0,
    translation: float = 0,
    scale: float = 1,
    shear: float = 0,
) -> Tuple[str, np.ndarray]:
    # Augmentation dictionary
    aug_dict = {
        "prob": 1.0,
        "angle_range": [angle, angle],
        "translation_range": [translation, translation],
        "scale_range": [scale, scale],
        "shear_range": [shear, shear],
    }
    aug_str = "RandomAffineOp::" + json.dumps(aug_dict)

    # Get image info
    img_sz = source_img.shape[0]
    center = [img_sz / 2, img_sz / 2]

    # Warp once to figure scale adjustment
    M_inv = T.functional._get_inverse_affine_matrix(
        center, angle, [0, 0], 1, [shear, shear]
    )
    M_inv.extend([0.0, 0.0, 1.0])
    M_inv = np.array(M_inv).reshape((3, 3))
    M = np.linalg.inv(M_inv)

    # Center in output patch
    img_corners = np.array(
        [
            [0, 0, img_sz - 1, img_sz - 1],
            [0, img_sz - 1, 0, img_sz - 1],
            [1, 1, 1, 1],
        ]
    )
    transformed_corners = M @ img_corners
    x_min = np.amin(transformed_corners[0])
    x_max = np.amax(transformed_corners[0])
    x_range = np.ceil(x_max - x_min)
    y_min = np.amin(transformed_corners[1])
    y_max = np.amax(transformed_corners[1])
    y_range = np.ceil(y_max - y_min)

    # Apply translation and scale after centering in output patch
    scale_adjustment = min(img_sz / x_range, img_sz / y_range)
    scale *= scale_adjustment

    # Test data output generation
    M_inv = T.functional._get_inverse_affine_matrix(
        center, angle, [translation, translation], scale, [shear, shear]
    )
    M_inv = np.array(M_inv).reshape((2, 3))

    exp_out_img = cv2.warpAffine(
        source_img,
        M_inv,
        (img_sz, img_sz),
        flags=cv2.WARP_INVERSE_MAP + cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )

    return aug_str, exp_out_img


class TestDataTransformsAffine(unittest.TestCase):
    def _check_array_close(self, test_output, exp_output):
        self.assertTrue(
            np.allclose(exp_output, test_output),
            f"Augmented image not the same, expecting\n{exp_output[:,:,0]} \n   got\n{test_output[:,:,0]} ",
        )

    def test_affine_transforms_angle(self):
        default_cfg = Detectron2GoRunner().get_default_cfg()

        img_sz = 11
        img = np.zeros((img_sz, img_sz, 3)).astype(np.uint8)
        img[((img_sz + 1) // 2) - 1, :, :] = 255

        for angle in [45, 90]:
            aug_str, exp_out_img = generate_test_data(img, angle=angle)

            default_cfg.D2GO_DATA.AUG_OPS.TRAIN = [aug_str]
            tfm = build_transform_gen(default_cfg, is_train=True)
            trans_img, _ = apply_augmentations(tfm, img)

            self._check_array_close(trans_img, exp_out_img)

    def test_affine_transforms_translation(self):
        default_cfg = Detectron2GoRunner().get_default_cfg()

        img_sz = 11
        img = np.zeros((img_sz, img_sz, 3)).astype(np.uint8)
        img[((img_sz + 1) // 2) - 1, :, :] = 255

        for translation in [0, 1, 2]:
            aug_str, exp_out_img = generate_test_data(img, translation=translation)

            default_cfg.D2GO_DATA.AUG_OPS.TRAIN = [aug_str]
            tfm = build_transform_gen(default_cfg, is_train=True)
            trans_img, _ = apply_augmentations(tfm, img)

            self._check_array_close(trans_img, exp_out_img)

    def test_affine_transforms_shear(self):
        default_cfg = Detectron2GoRunner().get_default_cfg()

        img_sz = 11
        img = np.zeros((img_sz, img_sz, 3)).astype(np.uint8)
        img[((img_sz + 1) // 2) - 1, :, :] = 255

        for shear in [0, 1, 2]:
            aug_str, exp_out_img = generate_test_data(img, shear=shear)

            default_cfg.D2GO_DATA.AUG_OPS.TRAIN = [aug_str]
            tfm = build_transform_gen(default_cfg, is_train=True)
            trans_img, _ = apply_augmentations(tfm, img)

            self._check_array_close(trans_img, exp_out_img)

    def test_affine_transforms_scale(self):
        default_cfg = Detectron2GoRunner().get_default_cfg()

        img_sz = 11
        img = np.zeros((img_sz, img_sz, 3)).astype(np.uint8)
        img[((img_sz + 1) // 2) - 1, :, :] = 255

        for scale in [0.9, 1, 1.1]:
            aug_str, exp_out_img = generate_test_data(img, scale=scale)

            default_cfg.D2GO_DATA.AUG_OPS.TRAIN = [aug_str]
            tfm = build_transform_gen(default_cfg, is_train=True)
            trans_img, _ = apply_augmentations(tfm, img)

            self._check_array_close(trans_img, exp_out_img)
