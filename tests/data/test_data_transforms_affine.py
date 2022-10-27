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
    fit_in_frame: bool = True,
    keep_aspect_ratio: bool = False,
) -> Tuple[str, np.ndarray]:
    # Augmentation dictionary
    aug_dict = {
        "prob": 1.0,
        "angle_range": [angle, angle],
        "translation_range": [translation, translation],
        "scale_range": [scale, scale],
        "shear_range": [shear, shear],
        "keep_aspect_ratio": keep_aspect_ratio,
        "fit_in_frame": fit_in_frame,
    }
    aug_str = "RandomAffineOp::" + json.dumps(aug_dict)

    # Get image info
    img_h, img_w = source_img.shape[0:2]
    center = [img_w / 2, img_h / 2]

    # Compute output_size
    max_size = max(img_w, img_h)
    out_w, out_h = (img_w, img_h) if keep_aspect_ratio else (max_size, max_size)

    if fit_in_frame:
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
                [0, 0, img_w - 1, img_w - 1],
                [0, img_h - 1, 0, img_h - 1],
                [1, 1, 1, 1],
            ]
        )
        new_corners = M @ img_corners
        x_range = np.ceil(np.amax(new_corners[0]) - np.amin(new_corners[0]))
        y_range = np.ceil(np.amax(new_corners[1]) - np.amin(new_corners[1]))

        # Apply translation and scale after centering in output patch
        scale_adjustment = min(out_w / x_range, out_h / y_range)
        scale *= scale_adjustment

    # Adjust output center location
    translation_t = [translation, translation]
    translation_adjustment = [(out_w - img_w) / 2, (out_h - img_h) / 2]
    translation_t[0] += translation_adjustment[0]
    translation_t[1] += translation_adjustment[1]

    # Test data output generation
    M_inv = T.functional._get_inverse_affine_matrix(
        center, angle, translation_t, scale, [shear, shear]
    )
    M_inv = np.array(M_inv).reshape((2, 3))

    exp_out_img = cv2.warpAffine(
        source_img,
        M_inv,
        (out_w, out_h),
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
        default_cfg = Detectron2GoRunner.get_default_cfg()

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
        default_cfg = Detectron2GoRunner.get_default_cfg()

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
        default_cfg = Detectron2GoRunner.get_default_cfg()

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
        default_cfg = Detectron2GoRunner.get_default_cfg()

        img_sz = 11
        img = np.zeros((img_sz, img_sz, 3)).astype(np.uint8)
        img[((img_sz + 1) // 2) - 1, :, :] = 255

        for scale in [0.9, 1, 1.1]:
            aug_str, exp_out_img = generate_test_data(img, scale=scale)

            default_cfg.D2GO_DATA.AUG_OPS.TRAIN = [aug_str]
            tfm = build_transform_gen(default_cfg, is_train=True)
            trans_img, _ = apply_augmentations(tfm, img)

            self._check_array_close(trans_img, exp_out_img)

    def test_affine_transforms_angle_non_square(self):
        default_cfg = Detectron2GoRunner.get_default_cfg()

        img_sz = 11
        img = np.zeros((img_sz, img_sz - 2, 3)).astype(np.uint8)
        img[((img_sz + 1) // 2) - 1, :, :] = 255

        for keep_aspect_ratio in [False, True]:
            aug_str, exp_out_img = generate_test_data(
                img, angle=45, keep_aspect_ratio=keep_aspect_ratio
            )

            default_cfg.D2GO_DATA.AUG_OPS.TRAIN = [aug_str]
            tfm = build_transform_gen(default_cfg, is_train=True)
            trans_img, _ = apply_augmentations(tfm, img)

            self._check_array_close(trans_img, exp_out_img)

    def test_affine_transforms_angle_no_fit_to_frame(self):
        default_cfg = Detectron2GoRunner.get_default_cfg()

        img_sz = 11
        img = np.zeros((img_sz, img_sz, 3)).astype(np.uint8)
        img[((img_sz + 1) // 2) - 1, :, :] = 255

        aug_str, exp_out_img = generate_test_data(img, angle=45, fit_in_frame=False)

        default_cfg.D2GO_DATA.AUG_OPS.TRAIN = [aug_str]
        tfm = build_transform_gen(default_cfg, is_train=True)
        trans_img, _ = apply_augmentations(tfm, img)

        self._check_array_close(trans_img, exp_out_img)
