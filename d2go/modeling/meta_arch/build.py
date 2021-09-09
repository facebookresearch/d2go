#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved


from d2go.utils.misc import _log_api_usage
from detectron2.modeling import build_model as d2_build_model


def build_model(cfg):
    """
    Build the whole model architecture, defined by ``cfg.MODEL.META_ARCHITECTURE``.
    Note that it does not load any weights from ``cfg``.
    """
    meta_arch = cfg.MODEL.META_ARCHITECTURE
    model = d2_build_model(cfg)
    _log_api_usage("modeling.meta_arch." + meta_arch)
    return model
