#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved


from d2go.modeling.meta_arch import modeling_hook as mh
from d2go.utils.misc import _log_api_usage
from detectron2.modeling import build_model as d2_build_model


def build_model(cfg):
    """
    Build the whole model architecture, defined by ``cfg.MODEL.META_ARCHITECTURE``.
    Note that it does not load any weights from ``cfg``.
    """
    meta_arch = cfg.MODEL.META_ARCHITECTURE
    model = d2_build_model(cfg)

    # apply modeling hooks
    # some custom projects bypass d2go's default config so may not have the
    # MODELING_HOOKS key
    if hasattr(cfg.MODEL, "MODELING_HOOKS"):
        hook_names = cfg.MODEL.MODELING_HOOKS
        model = mh.build_and_apply_modeling_hooks(model, cfg, hook_names)

    _log_api_usage("modeling.meta_arch." + meta_arch)
    return model
