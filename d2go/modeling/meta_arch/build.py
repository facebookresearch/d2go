#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import torch

from d2go.modeling.meta_arch import modeling_hook as mh
from d2go.registry.builtin import META_ARCH_REGISTRY
from d2go.utils.misc import _log_api_usage
from detectron2.modeling import META_ARCH_REGISTRY as D2_META_ARCH_REGISTRY


def build_model(cfg):
    """
    Build the whole model architecture, defined by ``cfg.MODEL.META_ARCHITECTURE``.
    Note that it does not load any weights from ``cfg``.
    """

    # initialize the meta-arch and cast to the device
    meta_arch = cfg.MODEL.META_ARCHITECTURE
    # NOTE: during transition we also check if meta_arch is registered as D2 MetaArch
    # TODO: remove this check after Sep 2022.
    if meta_arch not in META_ARCH_REGISTRY and meta_arch in D2_META_ARCH_REGISTRY:
        raise KeyError(
            f"Can't find '{meta_arch}' in D2Go's META_ARCH_REGISTRY, although it is in"
            f" D2's META_ARCH_REGISTRY, now D2Go uses its own registry, please register"
            f" it in D2Go's META_ARCH_REGISTRY."
        )
    model = META_ARCH_REGISTRY.get(meta_arch)(cfg)
    model.to(torch.device(cfg.MODEL.DEVICE))

    # apply modeling hooks
    # some custom projects bypass d2go's default config so may not have the
    # MODELING_HOOKS key
    if hasattr(cfg.MODEL, "MODELING_HOOKS"):
        hook_names = cfg.MODEL.MODELING_HOOKS
        model = mh.build_and_apply_modeling_hooks(model, cfg, hook_names)

    _log_api_usage("modeling.meta_arch." + meta_arch)
    return model
