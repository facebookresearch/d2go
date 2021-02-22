#!/usr/bin/env python3

import unittest

import numpy as np
from detectron2.data.transforms import apply_augmentations, AugInput
from d2go.data.transforms.build import build_transform_gen
from d2go.data.transforms.blur import LocalizedBoxMotionBlur
from d2go.runner import Detectron2GoRunner


def _assertArrayUnequal(a1, a2):
    assert not np.array_equal(a1, a2)

def _assertArrayEqual(a1, a2):
    assert np.array_equal(a1, a2)


class TestDataTransformsBlur(unittest.TestCase):

    def test_local_box_blur_transforms(self):
        default_cfg = Detectron2GoRunner().get_default_cfg()
        img = np.random.uniform(0, 1, size=(100, 100, 3))
        dataset_dict = {
            "annotations": [
                {"bbox": [0, 7, 20, 10]},
                {"bbox": [50, 60, 20, 10]},
            ]
        }
        inputs = AugInput(image=img)
        inputs.annotations = dataset_dict["annotations"]

        default_cfg.D2GO_DATA.AUG_OPS.TRAIN = [
            'RandomLocalizedBoxMotionBlurOp::{"k": [3, 7], "direction": [-1, 1], "angle": [0, 360]}',
        ]
        tfm = build_transform_gen(default_cfg, is_train=True)
        self.assertEqual(len(tfm), 1)
        self.assertTrue(isinstance(tfm[0], LocalizedBoxMotionBlur))
        transformed_input, _  = apply_augmentations(tfm, inputs)
        output = transformed_input.image

        # Check boxes regions have changed.
        for ann in dataset_dict["annotations"]:
            x, y, w, h = ann["bbox"]
            img_region = img[y:y+h, x:x+w]
            output_region = output[y:y+h, x:x+w]
            _assertArrayUnequal(img_region, output_region)

        # Check non-box regions are the same.
        non_box_output = np.array(output)
        non_box_img = np.array(img)
        for ann in dataset_dict["annotations"]:
            x, y, w, h = ann["bbox"]
            # Zero out the box regions
            non_box_output[y:y+h, x:x+w] = 0
            non_box_img[y:y+h, x:x+w] = 0
        _assertArrayEqual(non_box_img, non_box_output)
