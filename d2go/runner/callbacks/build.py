#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

from detectron2.utils.registry import Registry
from d2go.config import CfgNode

CALLBACK_REGISTRY = Registry("D2GO_CALLBACK_REGISTRY")

def build_quantization_callback(cfg: CfgNode):
    return CALLBACK_REGISTRY.get(cfg.QUANTIZATION.NAME).from_config(cfg)
