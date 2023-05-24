#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved


from typing import Callable, List, Union

import detectron2.data.transforms.augmentation as aug
import numpy as np
from d2go.data.transforms.build import _json_load, TRANSFORM_OP_REGISTRY
from detectron2.config import CfgNode
from detectron2.data import detection_utils as du
from detectron2.data.transforms.transform import Transform
from fvcore.transforms.transform import BlendTransform


class InvertibleColorTransform(Transform):
    """
    Generic wrapper for invertible photometric transforms.
    These transformations should only affect the color space and
        not the coordinate space of the image (e.g. annotation
        coordinates such as bounding boxes should not be changed)
    """

    def __init__(self, op: Callable, inverse_op: Callable):
        """
        Args:
            op (Callable): operation to be applied to the image,
                which takes in an ndarray and returns an ndarray.
        """
        if not callable(op):
            raise ValueError("op parameter should be callable")
        if not callable(inverse_op):
            raise ValueError("inverse_op parameter should be callable")
        super().__init__()
        self._set_attributes(locals())

    def apply_image(self, img: np.ndarray) -> np.ndarray:
        return self.op(img)

    def apply_coords(self, coords: np.ndarray) -> np.ndarray:
        return coords

    def inverse(self) -> Transform:
        return InvertibleColorTransform(self.inverse_op, self.op)

    def apply_segmentation(self, segmentation: np.ndarray) -> np.ndarray:
        return segmentation


class RandomContrastYUV(aug.Augmentation):
    """
    Randomly transforms contrast for images in YUV format.
    See similar:
        detectron2.data.transforms.RandomContrast,
        detectron2.data.transforms.RandomBrightness
    """

    def __init__(self, intensity_min: float, intensity_max: float):
        super().__init__()
        self._init(locals())

    def get_transform(self, img: np.ndarray) -> Transform:
        w = np.random.uniform(self.intensity_min, self.intensity_max)
        pure_gray = np.zeros_like(img)
        pure_gray[:, :, 0] = 0.5
        return BlendTransform(src_image=pure_gray, src_weight=1 - w, dst_weight=w)


class RandomSaturationYUV(aug.Augmentation):
    """
    Randomly transforms saturation for images in YUV format.
    See similar: detectron2.data.transforms.RandomSaturation
    """

    def __init__(self, intensity_min: float, intensity_max: float):
        super().__init__()
        self._init(locals())

    def get_transform(self, img: np.ndarray) -> Transform:
        assert (
            len(img.shape) == 3 and img.shape[-1] == 3
        ), f"Expected (H, W, 3), image shape {img.shape}"
        w = np.random.uniform(self.intensity_min, self.intensity_max)
        grayscale = np.zeros_like(img)
        grayscale[:, :, 0] = img[:, :, 0]
        return BlendTransform(src_image=grayscale, src_weight=1 - w, dst_weight=w)


def convert_rgb_to_yuv_bt601(image: np.ndarray) -> np.ndarray:
    """Convert RGB image in (H, W, C) to YUV format
    image: range 0 ~ 255
    """
    image = image / 255.0
    image = np.dot(image, np.array(du._M_RGB2YUV).T)
    return image


def convery_yuv_bt601_to_rgb(image: np.ndarray) -> np.ndarray:
    return du.convert_image_to_rgb(image, "YUV-BT.601")


class RGB2YUVBT601(aug.Augmentation):
    def __init__(self):
        super().__init__()
        self.trans = InvertibleColorTransform(
            convert_rgb_to_yuv_bt601, convery_yuv_bt601_to_rgb
        )

    def get_transform(self, image) -> Transform:
        return self.trans


class YUVBT6012RGB(aug.Augmentation):
    def __init__(self):
        super().__init__()
        self.trans = InvertibleColorTransform(
            convery_yuv_bt601_to_rgb, convert_rgb_to_yuv_bt601
        )

    def get_transform(self, image) -> Transform:
        return self.trans


def build_func(
    cfg: CfgNode, arg_str: str, is_train: bool, obj
) -> List[Union[aug.Augmentation, Transform]]:
    assert is_train
    kwargs = _json_load(arg_str) if arg_str is not None else {}
    assert isinstance(kwargs, dict)
    return [obj(**kwargs)]


@TRANSFORM_OP_REGISTRY.register()
def RandomContrastYUVOp(
    cfg: CfgNode, arg_str: str, is_train: bool
) -> List[Union[aug.Augmentation, Transform]]:
    return build_func(cfg, arg_str, is_train, obj=RandomContrastYUV)


@TRANSFORM_OP_REGISTRY.register()
def RandomSaturationYUVOp(
    cfg: CfgNode, arg_str: str, is_train: bool
) -> List[Union[aug.Augmentation, Transform]]:
    return build_func(cfg, arg_str, is_train, obj=RandomSaturationYUV)


@TRANSFORM_OP_REGISTRY.register()
def RGB2YUVBT601Op(
    cfg: CfgNode, arg_str: str, is_train: bool
) -> List[Union[aug.Augmentation, Transform]]:
    return build_func(cfg, arg_str, is_train, obj=RGB2YUVBT601)


@TRANSFORM_OP_REGISTRY.register()
def YUVBT6012RGBOp(
    cfg: CfgNode, arg_str: str, is_train: bool
) -> List[Union[aug.Augmentation, Transform]]:
    return build_func(cfg, arg_str, is_train, obj=YUVBT6012RGB)
