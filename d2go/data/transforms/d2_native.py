#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved


import logging
from typing import List, Optional, Union

import detectron2.data.transforms.augmentation as aug
from d2go.data.transforms.build import _json_load, TRANSFORM_OP_REGISTRY
from detectron2.config import CfgNode
from detectron2.data import transforms as d2T
from detectron2.projects.point_rend import ColorAugSSDTransform

logger = logging.getLogger(__name__)


D2_RANDOM_TRANSFORMS = {
    "RandomBrightness": d2T.RandomBrightness,
    "RandomContrast": d2T.RandomContrast,
    "RandomCrop": d2T.RandomCrop,
    "RandomRotation": d2T.RandomRotation,
    "RandomExtent": d2T.RandomExtent,
    "RandomFlip": d2T.RandomFlip,
    "RandomSaturation": d2T.RandomSaturation,
    "RandomLighting": d2T.RandomLighting,
    "RandomResize": d2T.RandomResize,
    "FixedSizeCrop": d2T.FixedSizeCrop,
    "ResizeScale": d2T.ResizeScale,
    "MinIoURandomCrop": d2T.MinIoURandomCrop,
}


def build_func(
    cfg: CfgNode, arg_str: str, is_train: bool, name: str
) -> List[Union[aug.Augmentation, d2T.Transform]]:
    assert is_train, "Random augmentation is for training only"
    kwargs = _json_load(arg_str) if arg_str is not None else {}
    assert isinstance(kwargs, dict)
    return [D2_RANDOM_TRANSFORMS[name](**kwargs)]


# example 1: RandomFlipOp
# example 2: RandomFlipOp::{}
# example 3: RandomFlipOp::{"prob":0.5}
# example 4: RandomBrightnessOp::{"intensity_min":1.0, "intensity_max":2.0}
@TRANSFORM_OP_REGISTRY.register()
def RandomBrightnessOp(
    cfg: CfgNode, arg_str: str, is_train: bool
) -> List[Union[aug.Augmentation, d2T.Transform]]:
    return build_func(cfg, arg_str, is_train, name="RandomBrightness")


@TRANSFORM_OP_REGISTRY.register()
def RandomContrastOp(
    cfg: CfgNode, arg_str: str, is_train: bool
) -> List[Union[aug.Augmentation, d2T.Transform]]:
    return build_func(cfg, arg_str, is_train, name="RandomContrast")


@TRANSFORM_OP_REGISTRY.register()
def RandomCropOp(
    cfg: CfgNode, arg_str: str, is_train: bool
) -> List[Union[aug.Augmentation, d2T.Transform]]:
    return build_func(cfg, arg_str, is_train, name="RandomCrop")


@TRANSFORM_OP_REGISTRY.register()
def RandomRotation(
    cfg: CfgNode, arg_str: str, is_train: bool
) -> List[Union[aug.Augmentation, d2T.Transform]]:
    return build_func(cfg, arg_str, is_train, name="RandomRotation")


@TRANSFORM_OP_REGISTRY.register()
def RandomExtentOp(
    cfg: CfgNode, arg_str: str, is_train: bool
) -> List[Union[aug.Augmentation, d2T.Transform]]:
    return build_func(cfg, arg_str, is_train, name="RandomExtent")


@TRANSFORM_OP_REGISTRY.register()
def RandomFlipOp(
    cfg: CfgNode, arg_str: str, is_train: bool
) -> List[Union[aug.Augmentation, d2T.Transform]]:
    return build_func(cfg, arg_str, is_train, name="RandomFlip")


@TRANSFORM_OP_REGISTRY.register()
def RandomSaturationOp(
    cfg: CfgNode, arg_str: str, is_train: bool
) -> List[Union[aug.Augmentation, d2T.Transform]]:
    return build_func(cfg, arg_str, is_train, name="RandomSaturation")


@TRANSFORM_OP_REGISTRY.register()
def RandomLightingOp(
    cfg: CfgNode, arg_str: str, is_train: bool
) -> List[Union[aug.Augmentation, d2T.Transform]]:
    return build_func(cfg, arg_str, is_train, name="RandomLighting")


@TRANSFORM_OP_REGISTRY.register()
def RandomSSDColorAugOp(
    cfg: CfgNode, arg_str: str, is_train: bool
) -> List[Union[aug.Augmentation, d2T.Transform]]:
    assert is_train
    kwargs = _json_load(arg_str) if arg_str is not None else {}
    assert isinstance(kwargs, dict)
    assert "img_format" not in kwargs
    return [ColorAugSSDTransform(img_format=cfg.INPUT.FORMAT, **kwargs)]


# example repr: ResizeScaleOp::{"min_scale": 0.1, "max_scale": 2.0, "target_height": 1024, "target_width": 1024}
@TRANSFORM_OP_REGISTRY.register()
def ResizeScaleOp(
    cfg: CfgNode, arg_str: Optional[str], is_train: bool
) -> List[aug.Augmentation]:
    return build_func(cfg, arg_str, is_train, name="ResizeScale")


@TRANSFORM_OP_REGISTRY.register()
def MinIoURandomCropOp(
    cfg: CfgNode, arg_str: Optional[str], is_train: bool
) -> List[aug.Augmentation]:
    return build_func(cfg, arg_str, is_train, name="MinIoURandomCrop")


# example repr: FixedSizeCropOp::{"crop_size": [1024, 1024]}
@TRANSFORM_OP_REGISTRY.register()
def FixedSizeCropOp(
    cfg: CfgNode, arg_str: Optional[str], is_train: bool
) -> List[aug.Augmentation]:
    return build_func(cfg, arg_str, is_train, name="FixedSizeCrop")


# example repr: RandomResizeOp::{"shape_list": [[224, 224], [256, 256], [320, 320]]}
@TRANSFORM_OP_REGISTRY.register()
def RandomResizeOp(
    cfg: CfgNode, arg_str: Optional[str], is_train: bool
) -> List[aug.Augmentation]:
    return build_func(cfg, arg_str, is_train, name="RandomResize")
