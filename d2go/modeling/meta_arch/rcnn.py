#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved


import logging

import torch
from d2go.export.api import PredictorExportConfig
from d2go.utils.export_utils import (
    D2Caffe2MetaArchPreprocessFunc,
    D2Caffe2MetaArchPostprocessFunc,
    D2RCNNTracingWrapper,
)
from detectron2.export.caffe2_modeling import META_ARCH_CAFFE2_EXPORT_TYPE_MAP
from detectron2.modeling import GeneralizedRCNN
from detectron2.projects.point_rend import PointRendMaskHead
from detectron2.utils.registry import Registry
from mobile_cv.arch.utils import fuse_utils
from mobile_cv.arch.utils.quantize_utils import (
    wrap_non_quant_group_norm,
    wrap_quant_subclass,
    QuantWrapper,
)
from mobile_cv.predictor.api import FuncInfo
from torch.quantization.quantize_fx import prepare_fx, prepare_qat_fx, convert_fx

logger = logging.getLogger(__name__)


# NOTE: Customized heads are often used in the GeneralizedRCNN, this leads to the needs
# for also customizating export/quant APIs, therefore registries are provided for easy
# override without creating new meta-archs. For other less general meta-arch, this type
# of registries might be over-kill.
RCNN_PREPARE_FOR_EXPORT_REGISTRY = Registry("RCNN_PREPARE_FOR_EXPORT")
RCNN_PREPARE_FOR_QUANT_REGISTRY = Registry("RCNN_PREPARE_FOR_QUANT")
RCNN_PREPARE_FOR_QUANT_CONVERT_REGISTRY = Registry("RCNN_PREPARE_FOR_QUANT_CONVERT")


class GeneralizedRCNNPatch:
    METHODS_TO_PATCH = [
        "prepare_for_export",
        "prepare_for_quant",
        "prepare_for_quant_convert",
    ]

    def prepare_for_export(self, cfg, *args, **kwargs):
        func = RCNN_PREPARE_FOR_EXPORT_REGISTRY.get(cfg.RCNN_PREPARE_FOR_EXPORT)
        return func(self, cfg, *args, **kwargs)

    def prepare_for_quant(self, cfg, *args, **kwargs):
        func = RCNN_PREPARE_FOR_QUANT_REGISTRY.get(cfg.RCNN_PREPARE_FOR_QUANT)
        return func(self, cfg, *args, **kwargs)

    def prepare_for_quant_convert(self, cfg, *args, **kwargs):
        func = RCNN_PREPARE_FOR_QUANT_CONVERT_REGISTRY.get(
            cfg.RCNN_PREPARE_FOR_QUANT_CONVERT
        )
        return func(self, cfg, *args, **kwargs)


@RCNN_PREPARE_FOR_EXPORT_REGISTRY.register()
def default_rcnn_prepare_for_export(self, cfg, inputs, predictor_type):

    if "torchscript" in predictor_type and "@tracing" in predictor_type:
        return PredictorExportConfig(
            model=D2RCNNTracingWrapper(self),
            data_generator=D2RCNNTracingWrapper.generator_trace_inputs,
            run_func_info=FuncInfo.gen_func_info(
                D2RCNNTracingWrapper.RunFunc, params={}
            ),
        )

    if cfg.MODEL.META_ARCHITECTURE in META_ARCH_CAFFE2_EXPORT_TYPE_MAP:
        C2MetaArch = META_ARCH_CAFFE2_EXPORT_TYPE_MAP[cfg.MODEL.META_ARCHITECTURE]
        c2_compatible_model = C2MetaArch(cfg, self)

        preprocess_info = FuncInfo.gen_func_info(
            D2Caffe2MetaArchPreprocessFunc,
            params=D2Caffe2MetaArchPreprocessFunc.get_params(cfg, c2_compatible_model),
        )
        postprocess_info = FuncInfo.gen_func_info(
            D2Caffe2MetaArchPostprocessFunc,
            params=D2Caffe2MetaArchPostprocessFunc.get_params(cfg, c2_compatible_model),
        )

        preprocess_func = preprocess_info.instantiate()

        return PredictorExportConfig(
            model=c2_compatible_model,
            # Caffe2MetaArch takes a single tuple as input (which is the return of
            # preprocess_func), data_generator requires all positional args as a tuple.
            data_generator=lambda x: (preprocess_func(x),),
            preprocess_info=preprocess_info,
            postprocess_info=postprocess_info,
        )

    raise NotImplementedError("Can't determine prepare_for_tracing!")


def _apply_eager_mode_quant(cfg, model):

    if isinstance(model, GeneralizedRCNN):
        """Wrap each quantized part of the model to insert Quant and DeQuant in-place"""

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
        prepare_custom_config_dict={
            "preserved_attributes": ["size_divisibility"],
            # keep the output of backbone quantized, to avoid
            # redundant dequant
            # TODO: output of backbone is a dict and currently this will keep all output
            # quantized, when we fix the implementation of "output_quantized_idxs"
            # we'll need to change this
            "output_quantized_idxs": [0],
        },
    )
    self.proposal_generator.rpn_head.rpn_feature = prep_fn(
        self.proposal_generator.rpn_head.rpn_feature,
        qconfig,
        prepare_custom_config_dict={
            # rpn_feature expecting quantized input, this is used to avoid redundant
            # quant
            "input_quantized_idxs": [0]
        },
    )
    self.proposal_generator.rpn_head.rpn_regressor.cls_logits = prep_fn(
        self.proposal_generator.rpn_head.rpn_regressor.cls_logits, qconfig
    )
    self.proposal_generator.rpn_head.rpn_regressor.bbox_pred = prep_fn(
        self.proposal_generator.rpn_head.rpn_regressor.bbox_pred, qconfig
    )
    self.roi_heads.box_head.roi_box_conv = prep_fn(
        self.roi_heads.box_head.roi_box_conv,
        qconfig,
        prepare_custom_config_dict={
            "output_quantized_idxs": [0],
        },
    )
    self.roi_heads.box_head.avgpool = prep_fn(
        self.roi_heads.box_head.avgpool,
        qconfig,
        prepare_custom_config_dict={"input_quantized_idxs": [0]},
    )
    self.roi_heads.box_predictor.cls_score = prep_fn(
        self.roi_heads.box_predictor.cls_score,
        qconfig,
        prepare_custom_config_dict={"input_quantized_idxs": [0]},
    )
    self.roi_heads.box_predictor.bbox_pred = prep_fn(
        self.roi_heads.box_predictor.bbox_pred,
        qconfig,
        prepare_custom_config_dict={"input_quantized_idxs": [0]},
    )


@RCNN_PREPARE_FOR_QUANT_REGISTRY.register()
def default_rcnn_prepare_for_quant(self, cfg):
    model = self
    torch.backends.quantized.engine = cfg.QUANTIZATION.BACKEND
    model.qconfig = (
        torch.quantization.get_default_qat_qconfig(cfg.QUANTIZATION.BACKEND)
        if model.training
        else torch.quantization.get_default_qconfig(cfg.QUANTIZATION.BACKEND)
    )
    if (
        hasattr(model, "roi_heads")
        and hasattr(model.roi_heads, "mask_head")
        and isinstance(model.roi_heads.mask_head, PointRendMaskHead)
    ):
        model.roi_heads.mask_head.qconfig = None
    logger.info("Setup the model with qconfig:\n{}".format(model.qconfig))

    # Modify the model for eager mode
    if cfg.QUANTIZATION.EAGER_MODE:
        model = _apply_eager_mode_quant(cfg, model)
        model = fuse_utils.fuse_model(model, inplace=True)
    else:
        _fx_quant_prepare(model, cfg)

    return model


@RCNN_PREPARE_FOR_QUANT_CONVERT_REGISTRY.register()
def default_rcnn_prepare_for_quant_convert(self, cfg):
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
