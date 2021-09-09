#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import logging
import re

import torch.nn as nn
from detectron2.layers import FrozenBatchNorm2d

logger = logging.getLogger(__name__)


def add_model_freezing_configs(_C):
    _C.MODEL.FROZEN_LAYER_REG_EXP = []


def set_requires_grad(model, reg_exps, value):
    total_num_parameters = 0
    unmatched_parameters = []
    unmatched_parameter_names = []
    matched_parameters = []
    matched_parameter_names = []
    for name, parameter in model.named_parameters():
        total_num_parameters += 1
        matched = False
        for frozen_layers_regex in reg_exps:
            if re.match(frozen_layers_regex, name):
                matched = True
                parameter.requires_grad = value
                matched_parameter_names.append(name)
                matched_parameters.append(parameter)
                break
        if not matched:
            unmatched_parameter_names.append(name)
            unmatched_parameters.append(parameter)
    logger.info(
        "Matched layers (require_grad={}): {}".format(value, matched_parameter_names)
    )
    logger.info("Unmatched layers: {}".format(unmatched_parameter_names))
    return matched_parameter_names, unmatched_parameter_names


def _freeze_matched_bn(module, name, reg_exps, matched_names, unmatched_names):
    """
    Recursive function to freeze bn layers that match specified regular expressions.
    """

    res = module

    # Base case: current module is a leaf node
    if len(list(module.children())) == 0:
        if isinstance(module, nn.modules.batchnorm._BatchNorm):
            matched = False
            for frozen_layers_regex in reg_exps:
                if re.match(frozen_layers_regex, name):
                    matched = True
                    matched_names.append(name)
                    # Convert to frozen batch norm
                    res = FrozenBatchNorm2d.convert_frozen_batchnorm(module)
            if not matched:
                unmatched_names.append(name)
        return res

    # Recursion: current module has children
    for child_name, child in module.named_children():
        _name = name + "." + child_name if name != "" else child_name
        new_child = _freeze_matched_bn(
            child, _name, reg_exps, matched_names, unmatched_names
        )
        if new_child is not child:
            res.add_module(child_name, new_child)

    return res


def freeze_matched_bn(module, reg_exps):
    """
    Convert matching batchnorm layers in module into FrozenBatchNorm2d.

    Args:
        module: nn.Module
        reg_exps: list of regular expressions to match

    Returns:
        If module is an instance of batchnorm and it matches the reg exps,
        returns a new FrozenBatchNorm2d module.
        Otherwise, in-place converts the matching batchnorm child modules to FrozenBatchNorm2d
        and returns the main module.
    """

    matched_names = []
    unmatched_names = []
    res = _freeze_matched_bn(module, "", reg_exps, matched_names, unmatched_names)

    logger.info("Matched BN layers are frozen: {}".format(matched_names))
    logger.info("Unmatched BN layers: {}".format(unmatched_names))

    return res
