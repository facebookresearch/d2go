#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved


from typing import List, Optional, Union

import detectron2.data.transforms.augmentation as aug
import numpy as np
import torchvision.transforms as tvtf
from d2go.data.transforms.tensor import Array2Tensor, Tensor2Array
from detectron2.config import CfgNode
from fvcore.transforms.transform import Transform

from .build import _json_load, TRANSFORM_OP_REGISTRY


class ToTensorWrapper:
    def __init__(self, transform):
        self.a2t = Array2Tensor(preserve_dtype=True)
        self.transform = transform
        self.t2a = Tensor2Array()

    def __call__(self, img: np.ndarray):
        return self.t2a.apply_image(self.transform(self.a2t.apply_image(img)))


class RandAugmentImage(Transform):
    """Rand Augment transform, only support image transformation"""

    def __init__(
        self,
        num_ops: int = 2,
        magnitude: int = 9,
        num_magnitude_bins: int = 31,
        interpolation: tvtf.functional.InterpolationMode = tvtf.functional.InterpolationMode.NEAREST,
        fill: Optional[List[float]] = None,
    ):
        transform = tvtf.RandAugment(
            num_ops, magnitude, num_magnitude_bins, interpolation, fill
        )
        self.transform = ToTensorWrapper(transform)

    def apply_image(self, img: np.ndarray) -> np.array:
        assert (
            img.dtype == np.uint8
        ), f"Only uint8 image format is supported, got {img.dtype}"
        return self.transform(img)

    def apply_coords(self, coords: np.ndarray) -> np.ndarray:
        raise NotImplementedError()

    def apply_segmentation(self, segmentation: np.ndarray) -> np.ndarray:
        raise NotImplementedError()


class TrivialAugmentWideImage(Transform):
    """TrivialAugmentWide transform, only support image transformation"""

    def __init__(
        self,
        num_magnitude_bins: int = 31,
        interpolation: tvtf.functional.InterpolationMode = tvtf.functional.InterpolationMode.NEAREST,
        fill: Optional[List[float]] = None,
    ):
        transform = tvtf.TrivialAugmentWide(num_magnitude_bins, interpolation, fill)
        self.transform = ToTensorWrapper(transform)

    def apply_image(self, img: np.ndarray) -> np.array:
        assert (
            img.dtype == np.uint8
        ), f"Only uint8 image format is supported, got {img.dtype}"
        return self.transform(img)

    def apply_coords(self, coords: np.ndarray) -> np.ndarray:
        raise NotImplementedError()

    def apply_segmentation(self, segmentation: np.ndarray) -> np.ndarray:
        raise NotImplementedError()


# example repr: 'RandAugmentImageOp::{"magnitude": 9}'
@TRANSFORM_OP_REGISTRY.register()
def RandAugmentImageOp(
    cfg: CfgNode, arg_str: str, is_train: bool
) -> List[Union[aug.Augmentation, Transform]]:
    assert is_train
    kwargs = _json_load(arg_str) if arg_str is not None else {}
    assert isinstance(kwargs, dict)
    return [RandAugmentImage(**kwargs)]


# example repr: 'TrivialAugmentWideImageOp::{"num_magnitude_bins": 31}'
@TRANSFORM_OP_REGISTRY.register()
def TrivialAugmentWideImageOp(
    cfg: CfgNode, arg_str: str, is_train: bool
) -> List[Union[aug.Augmentation, Transform]]:
    assert is_train
    kwargs = _json_load(arg_str) if arg_str is not None else {}
    assert isinstance(kwargs, dict)
    return [TrivialAugmentWideImage(**kwargs)]
