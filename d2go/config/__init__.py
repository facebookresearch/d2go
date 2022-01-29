#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved


# forward the namespace to avoid `d2go.config.config`
from .config import (
    CONFIG_CUSTOM_PARSE_REGISTRY,
    CONFIG_SCALING_METHOD_REGISTRY,
    CfgNode,
    auto_scale_world_size,
    reroute_config_path,
    temp_defrost,
)


__all__ = [
    "CONFIG_CUSTOM_PARSE_REGISTRY",
    "CONFIG_SCALING_METHOD_REGISTRY",
    "CfgNode",
    "auto_scale_world_size",
    "reroute_config_path",
    "temp_defrost",
]
