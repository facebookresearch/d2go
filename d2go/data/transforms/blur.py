#!/usr/bin/env python3

import detectron2.data.transforms.augmentation as aug
from detectron2.data.transforms import NoOpTransform, Transform
import numpy as np
from .build import TRANSFORM_OP_REGISTRY, _json_load


class LocalizedBoxMotionBlurTransform(Transform):
    """ Transform to blur provided bounding boxes from an image. """
    def __init__(self, bounding_boxes, k=(7, 15), angle=(0, 360), direction=(-1.0, 1.0)):
        import imgaug.augmenters as iaa

        super().__init__()
        self._set_attributes(locals())
        self.aug = iaa.MotionBlur(k, angle, direction, 1)

    def apply_image(self, img):
        bbox_regions = [img[y:y+h, x:x+w] for x, y, w, h in self.bounding_boxes]
        blurred_boxes = self.aug.augment_images(bbox_regions)
        new_img = np.array(img)
        for (x, y, w, h), blurred in zip(self.bounding_boxes, blurred_boxes):
            new_img[y:y+h, x:x+w] = blurred
        return new_img

    def apply_segmentation(self, segmentation):
        """ Apply no transform on the full-image segmentation. """
        return segmentation

    def apply_coords(self, coords):
        """ Apply no transform on the coordinates. """
        return coords

    def inverse(self) -> Transform:
        """ The inverse is a No-op, only for geometric transforms. """
        return NoOpTransform()

class LocalizedBoxMotionBlur(aug.Augmentation):
    """
    Performs faked motion blur on bounding box annotations in an image.
    Randomly selects motion blur parameters from the ranges `k`, `angle`, `direction`.
    """

    def __init__(self, prob=0.5, k=(7, 15), angle=(0, 360), direction=(-1.0, 1.0)):
        super().__init__()
        self._init(locals())

    def _validate_bbox_xywh_within_bounds(self, bbox, img_h, img_w):
        x, y, w, h = bbox
        assert x >= 0, f"Invalid x {x}"
        assert y >= 0, f"Invalid y {x}"
        assert y+h <= img_h, f"Invalid right {x+w} (img width {img_w})"
        assert y+h <= img_h, f"Invalid bottom {y+h} (img height {img_h})"

    def get_transform(self, image, annotations):
        do_tfm = self._rand_range() < self.prob
        if do_tfm:
            return self._get_blur_transform(image, annotations)
        else:
            return NoOpTransform()

    def _get_blur_transform(self, image, annotations):
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
def RandomLocalizedBoxMotionBlurOp(cfg, arg_str, is_train):
    assert is_train
    kwargs = _json_load(arg_str) if arg_str is not None else {}
    assert isinstance(kwargs, dict)
    return [LocalizedBoxMotionBlur(**kwargs)]


class MotionBlurTransform(Transform):
    def __init__(self, k=(7, 15), angle=(0, 360), direction=(-1.0, 1.0)):
        """
        Args:
           will apply the specified blur to the image
        """
        import imgaug.augmenters as iaa

        super().__init__()
        self._set_attributes(locals())
        self.aug = iaa.MotionBlur(k, angle, direction, 1)

    def apply_image(self, img):
        img = self.aug.augment_image(img)
        return img

    def apply_segmentation(self, segmentation):
        return segmentation

    def apply_coords(self, coords):
        return coords


class RandomMotionBlur(aug.Augmentation):
    """
        Apply random motion blur.
    """

    def __init__(self, prob=0.5, k=(3, 7), angle=(0, 360), direction=(-1.0, 1.0)):
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

    def get_transform(self, img):
        do = self._rand_range() < self.prob
        if do:
            return MotionBlurTransform(self.k, self.angle, self.direction)
        else:
            return NoOpTransform()


# example repr: "RandomMotionBlurOp::{'prob': 0.5, 'k': [3,7], 'angle': [0, 360]}"
@TRANSFORM_OP_REGISTRY.register()
def RandomMotionBlurOp(cfg, arg_str, is_train):
    assert is_train
    kwargs = _json_load(arg_str) if arg_str is not None else {}
    assert isinstance(kwargs, dict)
    return [RandomMotionBlur(**kwargs)]
