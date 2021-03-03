#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved


import json
import logging

from detectron2.data import transforms as d2T
from detectron2.utils.registry import Registry


logger = logging.getLogger(__name__)


TRANSFORM_OP_REGISTRY = Registry("D2GO_TRANSFORM_REGISTRY")


def _json_load(arg_str):
    try:
        return json.loads(arg_str)
    except json.decoder.JSONDecodeError as e:
        logger.warning("Can't load arg_str: {}".format(arg_str))
        raise e


# example repr: "ResizeShortestEdgeOp"
@TRANSFORM_OP_REGISTRY.register()
def ResizeShortestEdgeOp(cfg, arg_str, is_train):
    if is_train:
        min_size = cfg.INPUT.MIN_SIZE_TRAIN
        max_size = cfg.INPUT.MAX_SIZE_TRAIN
        sample_style = cfg.INPUT.MIN_SIZE_TRAIN_SAMPLING
    else:
        min_size = cfg.INPUT.MIN_SIZE_TEST
        max_size = cfg.INPUT.MAX_SIZE_TEST
        sample_style = "choice"
    if sample_style == "range":
        assert (
            len(min_size) == 2
        ), "more than 2 ({}) min_size(s) are provided for ranges".format(len(min_size))

    tfm_gens = []
    if not min_size == 0:  # set to zero to disable resize
        tfm_gens.append(d2T.ResizeShortestEdge(min_size, max_size, sample_style))
    return tfm_gens


# example repr: "ResizeShortestEdgeSquareOp"
@TRANSFORM_OP_REGISTRY.register()
def ResizeShortestEdgeSquareOp(cfg, arg_str, is_train):
    """ Resize the input to square using INPUT.MIN_SIZE_TRAIN or INPUT.MIN_SIZE_TEST
        without keeping aspect ratio
    """
    if is_train:
        min_size = cfg.INPUT.MIN_SIZE_TRAIN
        assert (
            isinstance(min_size, (list, tuple)) and len(min_size) == 1
        ), "Only a signle size is supported"
        min_size = min_size[0]
    else:
        min_size = cfg.INPUT.MIN_SIZE_TEST

    tfm_gens = []
    if not min_size == 0:  # set to zero to disable resize
        tfm_gens.append(d2T.Resize(shape=[min_size, min_size]))
    return tfm_gens


@TRANSFORM_OP_REGISTRY.register()
def ResizeOp(cfg, arg_str, is_train):
    kwargs = _json_load(arg_str) if arg_str is not None else {}
    assert isinstance(kwargs, dict)
    return [d2T.Resize(**kwargs)]


_TRANSFORM_REPR_SEPARATOR = "::"


def parse_tfm_gen_repr(tfm_gen_repr):
    if tfm_gen_repr.count(_TRANSFORM_REPR_SEPARATOR) == 0:
        return tfm_gen_repr, None
    elif tfm_gen_repr.count(_TRANSFORM_REPR_SEPARATOR) == 1:
        return tfm_gen_repr.split(_TRANSFORM_REPR_SEPARATOR)
    else:
        raise ValueError(
            "Can't to parse transform repr name because of multiple separator found."
            " Offending name: {}"
        )


def build_transform_gen(cfg, is_train):
    """
    This function builds a list of TransformGen or Transform objects using the a list of
    strings from cfg.D2GO_DATA.AUG_OPS.TRAIN/TEST. Each string (aka. `tfm_gen_repr`)
    will be split into `name` and `arg_str` (separated by "::"); the `name`
    will be used to lookup the registry while `arg_str` will be used as argument.

    Each function in registry needs to take `cfg`, `arg_str` and `is_train` as
    input, and return a list of TransformGen or Transform objects.
    """
    tfm_gen_repr_list = (
        cfg.D2GO_DATA.AUG_OPS.TRAIN if is_train else cfg.D2GO_DATA.AUG_OPS.TEST
    )
    tfm_gens = [
        TRANSFORM_OP_REGISTRY.get(name)(cfg, arg_str, is_train)
        for name, arg_str in [
            parse_tfm_gen_repr(tfm_gen_repr) for tfm_gen_repr in tfm_gen_repr_list
        ]
    ]
    assert all(isinstance(gens, list) for gens in tfm_gens)
    tfm_gens = [gen for gens in tfm_gens for gen in gens]
    assert all(isinstance(gen, (d2T.Transform, d2T.TransformGen)) for gen in tfm_gens)

    return tfm_gens
