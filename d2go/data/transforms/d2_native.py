#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved


import logging

from .build import TRANSFORM_OP_REGISTRY, _json_load
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
}


def build_func(cfg, arg_str, is_train, name):
    assert is_train
    kwargs = _json_load(arg_str) if arg_str is not None else {}
    assert isinstance(kwargs, dict)
    return [D2_RANDOM_TRANSFORMS[name](**kwargs)]


# example 1: RandomFlipOp
# example 2: RandomFlipOp::{}
# example 3: RandomFlipOp::{"prob":0.5}
# example 4: RandomBrightnessOp::{"intensity_min":1.0, "intensity_max":2.0}
@TRANSFORM_OP_REGISTRY.register()
def RandomBrightnessOp(cfg, arg_str, is_train):
    return build_func(cfg, arg_str, is_train, name="RandomBrightness")


@TRANSFORM_OP_REGISTRY.register()
def RandomContrastOp(cfg, arg_str, is_train):
    return build_func(cfg, arg_str, is_train, name="RandomContrast")


@TRANSFORM_OP_REGISTRY.register()
def RandomCropOp(cfg, arg_str, is_train):
    return build_func(cfg, arg_str, is_train, name="RandomCrop")


@TRANSFORM_OP_REGISTRY.register()
def RandomRotation(cfg, arg_str, is_train):
    return build_func(cfg, arg_str, is_train, name="RandomRotation")


@TRANSFORM_OP_REGISTRY.register()
def RandomExtentOp(cfg, arg_str, is_train):
    return build_func(cfg, arg_str, is_train, name="RandomExtent")


@TRANSFORM_OP_REGISTRY.register()
def RandomFlipOp(cfg, arg_str, is_train):
    return build_func(cfg, arg_str, is_train, name="RandomFlip")


@TRANSFORM_OP_REGISTRY.register()
def RandomSaturationOp(cfg, arg_str, is_train):
    return build_func(cfg, arg_str, is_train, name="RandomSaturation")


@TRANSFORM_OP_REGISTRY.register()
def RandomLightingOp(cfg, arg_str, is_train):
    return build_func(cfg, arg_str, is_train, name="RandomLighting")


@TRANSFORM_OP_REGISTRY.register()
def RandomSSDColorAugOp(cfg, arg_str, is_train):
    assert is_train
    kwargs = _json_load(arg_str) if arg_str is not None else {}
    assert isinstance(kwargs, dict)
    assert "img_format" not in kwargs
    return [ColorAugSSDTransform(img_format=cfg.INPUT.FORMAT, **kwargs)]
