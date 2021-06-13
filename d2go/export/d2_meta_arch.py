#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved


import logging
from functools import lru_cache

from d2go.modeling.meta_arch.rcnn import GeneralizedRCNNPatch
from d2go.modeling.meta_arch.semantic_seg import SemanticSegmentorPatch
from detectron2.modeling import GeneralizedRCNN, SemanticSegmentor

logger = logging.getLogger(__name__)


@lru_cache()  # only call once
def patch_d2_meta_arch():
    """
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
