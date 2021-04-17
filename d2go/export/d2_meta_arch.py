#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved


import logging
from functools import lru_cache

import torch
from d2go.export.api import PredictorExportConfig
from d2go.utils.prepare_for_export import d2_meta_arch_prepare_for_export
from detectron2.export.caffe2_modeling import (
    META_ARCH_CAFFE2_EXPORT_TYPE_MAP,
    convert_batched_inputs_to_c2_format,
)
from detectron2.export.shared import get_pb_arg_vali, get_pb_arg_vals
from detectron2.modeling import META_ARCH_REGISTRY, GeneralizedRCNN
from detectron2.modeling.postprocessing import detector_postprocess
from mobile_cv.arch.utils import fuse_utils
from mobile_cv.arch.utils.quantize_utils import (
    wrap_non_quant_group_norm,
    wrap_quant_subclass,
    QuantWrapper,
)
from mobile_cv.predictor.api import FuncInfo
from torch.quantization.quantize_fx import prepare_fx, prepare_qat_fx, convert_fx

logger = logging.getLogger(__name__)


@lru_cache()  # only call once
def patch_d2_meta_arch():
    # HACK: inject prepare_for_export for all D2's meta-arch
    for cls_obj in META_ARCH_REGISTRY._obj_map.values():
        if cls_obj.__module__.startswith("detectron2."):
            if hasattr(cls_obj, "prepare_for_export"):
                assert cls_obj.prepare_for_export == d2_meta_arch_prepare_for_export
            else:
                cls_obj.prepare_for_export = d2_meta_arch_prepare_for_export

            if hasattr(cls_obj, "prepare_for_quant"):
                assert cls_obj.prepare_for_quant == d2_meta_arch_prepare_for_quant
            else:
                cls_obj.prepare_for_quant = d2_meta_arch_prepare_for_quant

            if hasattr(cls_obj, "prepare_for_quant_convert"):
                assert (
                    cls_obj.prepare_for_quant_convert
                    == d2_meta_arch_prepare_for_quant_convert
                )
            else:
                cls_obj.prepare_for_quant_convert = (
                    d2_meta_arch_prepare_for_quant_convert
                )


def _apply_eager_mode_quant(cfg, model):

    if isinstance(model, GeneralizedRCNN):
        """ Wrap each quantized part of the model to insert Quant and DeQuant in-place """

        # Wrap backbone and proposal_generator
        model.backbone = wrap_quant_subclass(
            model.backbone, n_inputs=1, n_outputs=len(model.backbone._out_features)
        )
        model.proposal_generator.rpn_head = wrap_quant_subclass(
            model.proposal_generator.rpn_head,
            n_inputs=len(cfg.MODEL.RPN.IN_FEATURES),
            n_outputs=len(cfg.MODEL.RPN.IN_FEATURES) * 2,
        )
        # Wrap the roi_heads, box_pooler is not quantized
        if hasattr(model.roi_heads, "box_head"):
            model.roi_heads.box_head = wrap_quant_subclass(
                model.roi_heads.box_head,
                n_inputs=1,
                n_outputs=1,
            )
        # for faster_rcnn_R_50_C4
        if hasattr(model.roi_heads, "res5"):
            model.roi_heads.res5 = wrap_quant_subclass(
                model.roi_heads.res5,
                n_inputs=1,
                n_outputs=1,
            )

        model.roi_heads.box_predictor = wrap_quant_subclass(
            model.roi_heads.box_predictor, n_inputs=1, n_outputs=2
        )
        # Optionally wrap keypoint and mask heads, pools are not quantized
        if hasattr(model.roi_heads, "keypoint_head"):
            model.roi_heads.keypoint_head = wrap_quant_subclass(
                model.roi_heads.keypoint_head,
                n_inputs=1,
                n_outputs=1,
                wrapped_method_name="layers",
            )
        if hasattr(model.roi_heads, "mask_head"):
            model.roi_heads.mask_head = wrap_quant_subclass(
                model.roi_heads.mask_head,
                n_inputs=1,
                n_outputs=1,
                wrapped_method_name="layers",
            )

        # StandardROIHeadsWithSubClass uses a subclass head
        if hasattr(model.roi_heads, "subclass_head"):
            q_subclass_head = QuantWrapper(model.roi_heads.subclass_head)
            model.roi_heads.subclass_head = q_subclass_head
    else:
        raise NotImplementedError(
            "Eager mode for {} is not supported".format(type(model))
        )

    # TODO: wrap the normalizer and make it quantizable

    # NOTE: GN is not quantizable, assuming all GN follows a quantized conv,
    # wrap them with dequant-quant
    model = wrap_non_quant_group_norm(model)
    return model


def _fx_quant_prepare(self, cfg):
    prep_fn = prepare_qat_fx if self.training else prepare_fx
    qconfig = {"": self.qconfig}
    self.backbone = prep_fn(
        self.backbone,
        qconfig,
        {"preserved_attributes": ["size_divisibility"]},
    )
    self.proposal_generator.rpn_head.rpn_feature = prep_fn(
        self.proposal_generator.rpn_head.rpn_feature, qconfig
    )
    self.proposal_generator.rpn_head.rpn_regressor.cls_logits = prep_fn(
        self.proposal_generator.rpn_head.rpn_regressor.cls_logits, qconfig
    )
    self.proposal_generator.rpn_head.rpn_regressor.bbox_pred = prep_fn(
        self.proposal_generator.rpn_head.rpn_regressor.bbox_pred, qconfig
    )
    self.roi_heads.box_head.roi_box_conv = prep_fn(
        self.roi_heads.box_head.roi_box_conv, qconfig
    )
    self.roi_heads.box_head.avgpool = prep_fn(self.roi_heads.box_head.avgpool, qconfig)
    self.roi_heads.box_predictor.cls_score = prep_fn(
        self.roi_heads.box_predictor.cls_score, qconfig
    )
    self.roi_heads.box_predictor.bbox_pred = prep_fn(
        self.roi_heads.box_predictor.bbox_pred, qconfig
    )


def d2_meta_arch_prepare_for_quant(self, cfg):
    model = self
    torch.backends.quantized.engine = cfg.QUANTIZATION.BACKEND
    model.qconfig = (
        torch.quantization.get_default_qat_qconfig(cfg.QUANTIZATION.BACKEND)
        if model.training
        else torch.quantization.get_default_qconfig(cfg.QUANTIZATION.BACKEND)
    )
    logger.info("Setup the model with qconfig:\n{}".format(model.qconfig))

    # Modify the model for eager mode
    if cfg.QUANTIZATION.EAGER_MODE:
        model = _apply_eager_mode_quant(cfg, model)
        model = fuse_utils.fuse_model(model, inplace=True)
    else:
        _fx_quant_prepare(model, cfg)

    return model


def d2_meta_arch_prepare_for_quant_convert(self, cfg):
    if cfg.QUANTIZATION.EAGER_MODE:
        raise NotImplementedError()

    self.backbone = convert_fx(
        self.backbone,
        convert_custom_config_dict={"preserved_attributes": ["size_divisibility"]},
    )
    self.proposal_generator.rpn_head.rpn_feature = convert_fx(
        self.proposal_generator.rpn_head.rpn_feature
    )
    self.proposal_generator.rpn_head.rpn_regressor.cls_logits = convert_fx(
        self.proposal_generator.rpn_head.rpn_regressor.cls_logits
    )
    self.proposal_generator.rpn_head.rpn_regressor.bbox_pred = convert_fx(
        self.proposal_generator.rpn_head.rpn_regressor.bbox_pred
    )
    self.roi_heads.box_head.roi_box_conv = convert_fx(
        self.roi_heads.box_head.roi_box_conv
    )
    self.roi_heads.box_head.avgpool = convert_fx(self.roi_heads.box_head.avgpool)
    self.roi_heads.box_predictor.cls_score = convert_fx(
        self.roi_heads.box_predictor.cls_score
    )
    self.roi_heads.box_predictor.bbox_pred = convert_fx(
        self.roi_heads.box_predictor.bbox_pred
    )
    return self
