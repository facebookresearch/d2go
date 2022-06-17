#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved


import logging
from functools import lru_cache

from d2go.modeling.meta_arch.rcnn import GeneralizedRCNNPatch
from d2go.modeling.meta_arch.semantic_seg import SemanticSegmentorPatch
from d2go.registry.builtin import META_ARCH_REGISTRY
from detectron2.modeling import (
    GeneralizedRCNN,
    META_ARCH_REGISTRY as D2_META_ARCH_REGISTRY,
    SemanticSegmentor,
)

logger = logging.getLogger(__name__)


@lru_cache()  # only call once
def patch_d2_meta_arch():
    """
    Register meta-archietectures that are registered in D2's registry, also convert D2's
    meta-arch into D2Go's meta-arch.

    D2Go requires interfaces like prepare_for_export/prepare_for_quant from meta-arch in
    order to do export/quant, this function applies the monkey patch to the original
    D2's meta-archs.
    """

    def _check_and_set(cls_obj, method_name, method_func):
        if hasattr(cls_obj, method_name):
            assert getattr(cls_obj, method_name) == method_func
        else:
            setattr(cls_obj, method_name, method_func)

    def _apply_patch(dst_cls, src_cls):
        assert hasattr(src_cls, "METHODS_TO_PATCH")
        for method_name in src_cls.METHODS_TO_PATCH:
            assert hasattr(src_cls, method_name)
            _check_and_set(dst_cls, method_name, getattr(src_cls, method_name))

    _apply_patch(GeneralizedRCNN, GeneralizedRCNNPatch)
    _apply_patch(SemanticSegmentor, SemanticSegmentorPatch)
    # TODO: patch other meta-archs defined in D2

    for name, meta_arch_class in D2_META_ARCH_REGISTRY:
        logger.info(f"Re-register the D2 meta-arch in D2Go: {meta_arch_class}")
        META_ARCH_REGISTRY.register(name, meta_arch_class)
