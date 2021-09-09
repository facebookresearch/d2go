#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

from typing import List, Tuple, Dict

import detectron2.data.transforms.augmentation as aug
import numpy as np
from detectron2.config import CfgNode
from detectron2.data.transforms import NoOpTransform, Transform

from .build import TRANSFORM_OP_REGISTRY, _json_load


class LocalizedBoxMotionBlurTransform(Transform):
    """Transform to blur provided bounding boxes from an image."""

    def __init__(
        self,
        bounding_boxes: List[List[int]],
        k: Tuple[float, float] = (7, 15),
        angle: Tuple[float, float] = (0, 360),
        direction: Tuple[float, float] = (-1.0, 1.0),
    ):
        import imgaug.augmenters as iaa

        super().__init__()
        self._set_attributes(locals())
        self.aug = iaa.MotionBlur(k, angle, direction, 1)

    def apply_image(self, img: np.ndarray) -> np.ndarray:
        bbox_regions = [img[y : y + h, x : x + w] for x, y, w, h in self.bounding_boxes]
        blurred_boxes = self.aug.augment_images(bbox_regions)
        new_img = np.array(img)
        for (x, y, w, h), blurred in zip(self.bounding_boxes, blurred_boxes):
            new_img[y : y + h, x : x + w] = blurred
        return new_img

    def apply_segmentation(self, segmentation: np.ndarray) -> np.ndarray:
        """Apply no transform on the full-image segmentation."""
        return segmentation

    def apply_coords(self, coords: np.ndarray):
        """Apply no transform on the coordinates."""
        return coords

    def inverse(self) -> Transform:
        """The inverse is a No-op, only for geometric transforms."""
        return NoOpTransform()


class LocalizedBoxMotionBlur(aug.Augmentation):
    """
    Performs faked motion blur on bounding box annotations in an image.
    Randomly selects motion blur parameters from the ranges `k`, `angle`, `direction`.
    """

    def __init__(
        self,
        prob: float = 0.5,
        k: Tuple[float, float] = (7, 15),
        angle: Tuple[float, float] = (0, 360),
        direction: Tuple[float, float] = (-1.0, 1.0),
    ):
        super().__init__()
        self._init(locals())

    def _validate_bbox_xywh_within_bounds(
        self, bbox: List[int], img_h: int, img_w: int
    ):
        x, y, w, h = bbox
        assert x >= 0, f"Invalid x {x}"
        assert y >= 0, f"Invalid y {x}"
        assert y + h <= img_h, f"Invalid right {x+w} (img width {img_w})"
        assert y + h <= img_h, f"Invalid bottom {y+h} (img height {img_h})"

    def get_transform(self, image: np.ndarray, annotations: List[Dict]) -> Transform:
        do_tfm = self._rand_range() < self.prob
        if do_tfm:
            return self._get_blur_transform(image, annotations)
        else:
            return NoOpTransform()

    def _get_blur_transform(
        self, image: np.ndarray, annotations: List[Dict]
    ) -> Transform:
        """
        Return a `Transform` that simulates motion blur within the image's bounding box regions.
        """
        img_h, img_w = image.shape[:2]
        bboxes = [ann["bbox"] for ann in annotations]
        # Debug
        for bbox in bboxes:
            self._validate_bbox_xywh_within_bounds(bbox, img_h, img_w)

        return LocalizedBoxMotionBlurTransform(
            bboxes,
            k=self.k,
            angle=self.angle,
            direction=self.direction,
        )


# example repr: "LocalizedBoxMotionBlurOp::{'prob': 0.5, 'k': [3,7], 'angle': [0, 360]}"
@TRANSFORM_OP_REGISTRY.register()
def RandomLocalizedBoxMotionBlurOp(
    cfg: CfgNode, arg_str: str, is_train: bool
) -> List[Transform]:
    assert is_train
    kwargs = _json_load(arg_str) if arg_str is not None else {}
    assert isinstance(kwargs, dict)
    return [LocalizedBoxMotionBlur(**kwargs)]


class MotionBlurTransform(Transform):
    def __init__(
        self,
        k: Tuple[float, float] = (7, 15),
        angle: Tuple[float, float] = (0, 360),
        direction: Tuple[float, float] = (-1.0, 1.0),
    ):
        """
        Args:
           will apply the specified blur to the image
        """
        import imgaug.augmenters as iaa

        super().__init__()
        self._set_attributes(locals())
        self.aug = iaa.MotionBlur(k, angle, direction, 1)

    def apply_image(self, img: np.ndarray) -> np.ndarray:
        img = self.aug.augment_image(img)
        return img

    def apply_segmentation(self, segmentation: np.ndarray) -> np.ndarray:
        return segmentation

    def apply_coords(self, coords: np.ndarray) -> np.ndarray:
        return coords


class RandomMotionBlur(aug.Augmentation):
    """
    Apply random motion blur.
    """

    def __init__(
        self,
        prob: float = 0.5,
        k: Tuple[float, float] = (3, 7),
        angle: Tuple[float, float] = (0, 360),
        direction: Tuple[float, float] = (-1.0, 1.0),
    ):
        """
        Args:
            prob (float): probability of applying transform
            k (tuple): refer to `iaa.MotionBlur`
            angle (tuple): refer to `iaa.MotionBlur`
            direction (tuple): refer to `iaa.MotionBlur`
        """
        super().__init__()
        # Turn all locals into member variables.
        self._init(locals())

    def get_transform(self, img: np.ndarray) -> Transform:
        do = self._rand_range() < self.prob
        if do:
            return MotionBlurTransform(self.k, self.angle, self.direction)
        else:
            return NoOpTransform()


# example repr: "RandomMotionBlurOp::{'prob': 0.5, 'k': [3,7], 'angle': [0, 360]}"
@TRANSFORM_OP_REGISTRY.register()
def RandomMotionBlurOp(cfg: CfgNode, arg_str: str, is_train: bool) -> List[Transform]:
    assert is_train
    kwargs = _json_load(arg_str) if arg_str is not None else {}
    assert isinstance(kwargs, dict)
    return [RandomMotionBlur(**kwargs)]
