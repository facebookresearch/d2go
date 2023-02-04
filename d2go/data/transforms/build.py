#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved


import json
import logging
from typing import Dict, List, Optional, Tuple

from detectron2.config import CfgNode
from detectron2.data import transforms as d2T
from detectron2.utils.registry import Registry


logger = logging.getLogger(__name__)


TRANSFORM_OP_REGISTRY = Registry("D2GO_TRANSFORM_REGISTRY")


def _json_load(arg_str: str) -> Dict:
    try:
        return json.loads(arg_str)
    except json.decoder.JSONDecodeError as e:
        logger.warning("Can't load arg_str: {}".format(arg_str))
        raise e


# example repr: "ResizeShortestEdgeOp"
@TRANSFORM_OP_REGISTRY.register()
def ResizeShortestEdgeOp(
    cfg: CfgNode, arg_str: str, is_train: bool
) -> List[d2T.Transform]:
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
def ResizeShortestEdgeSquareOp(
    cfg: CfgNode, arg_str: str, is_train: bool
) -> List[d2T.Transform]:
    """Resize the input to square using INPUT.MIN_SIZE_TRAIN or INPUT.MIN_SIZE_TEST
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
def ResizeOp(cfg: CfgNode, arg_str: str, is_train: bool) -> List[d2T.Transform]:
    kwargs = _json_load(arg_str) if arg_str is not None else {}
    assert isinstance(kwargs, dict)
    return [d2T.Resize(**kwargs)]


_TRANSFORM_REPR_SEPARATOR = "::"


def parse_tfm_gen_repr(tfm_gen_repr: str) -> Tuple[str, Optional[str]]:
    if tfm_gen_repr.count(_TRANSFORM_REPR_SEPARATOR) == 0:
        return tfm_gen_repr, None
    else:
        # Split only after first delimiter, to allow for:
        #  - nested transforms, e.g:
        #   'SomeTransformOp::{"args": ["SubTransform2Op::{\\"param1\\": 0, \\"param2\\": false}", "SubTransform2Op::{\\"param1\\": 0.8}"], "other_args": 2}'
        #  - list of transforms, e.g.:
        #   ["SubTransform2Op::{\\"param1\\": 0, \\"param2\\": false}", "SubTransform2Op::{\\"param1\\": 0.8}"]
        # TODO(T144470024): Support recursive parsing. For now, it's user responsibility to ensure the nested transforms are parsed correctly.
        return tfm_gen_repr.split(_TRANSFORM_REPR_SEPARATOR, 1)


def build_transform_gen(
    cfg: CfgNode, is_train: bool, tfm_gen_repr_list: Optional[List[str]] = None
) -> List[d2T.Transform]:
    """
    This function builds a list of TransformGen or Transform objects using a list of
    strings (`tfm_gen_repr_list). If list is not provided, cfg.D2GO_DATA.AUG_OPS.TRAIN/TEST is used.
    Each string (aka. `tfm_gen_repr`) will be split into `name` and `arg_str` (separated by "::");
    the `name` will be used to lookup the registry while `arg_str` will be used as argument.

    Each function in registry needs to take `cfg`, `arg_str` and `is_train` as
    input, and return a list of TransformGen or Transform objects.
    """
    tfm_gen_repr_list = tfm_gen_repr_list or (
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
