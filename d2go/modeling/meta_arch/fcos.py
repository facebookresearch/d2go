#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import logging

import torch.nn as nn
from d2go.config import CfgNode as CN
from d2go.export.api import PredictorExportConfig
from d2go.modeling.meta_arch.rcnn import D2RCNNInferenceWrapper
from d2go.quantization.qconfig import set_backend_and_create_qconfig
from d2go.registry.builtin import META_ARCH_REGISTRY
from detectron2.config import configurable
from detectron2.layers.batch_norm import CycleBatchNormList
from detectron2.modeling.backbone import build_backbone
from detectron2.modeling.backbone.fpn import FPN
from detectron2.modeling.meta_arch.fcos import FCOS as d2_FCOS, FCOSHead
from detectron2.utils.registry import Registry
from mobile_cv.arch.utils import fuse_utils
from mobile_cv.arch.utils.quantize_utils import (
    wrap_non_quant_group_norm,
    wrap_quant_subclass,
)

from mobile_cv.predictor.api import FuncInfo


logger = logging.getLogger(__name__)

# Registry to store custom export logic
FCOS_PREPARE_FOR_EXPORT_REGISTRY = Registry("FCOS_PREPARE_FOR_EXPORT")


class FCOSInferenceWrapper(nn.Module):
    def __init__(
        self,
        model,
    ):
        super().__init__()
        self.model = model

    def forward(self, image):
        inputs = [{"image": image}]
        return self.model.forward(inputs)[0]["instances"]


def add_fcos_configs(cfg):
    cfg.MODEL.FCOS = CN()
    # the number of foreground classes.
    cfg.MODEL.FCOS.NUM_CLASSES = 80
    cfg.MODEL.FCOS.IN_FEATURES = ["p3", "p4", "p5", "p6", "p7"]
    cfg.MODEL.FCOS.NUM_CONVS = 4
    cfg.MODEL.FCOS.HEAD_NORM = "GN"

    # inference parameters
    cfg.MODEL.FCOS.SCORE_THRESH_TEST = 0.04
    cfg.MODEL.FCOS.TOPK_CANDIDATES_TEST = 1000
    cfg.MODEL.FCOS.NMS_THRESH_TEST = 0.6

    # Focal loss parameters
    cfg.MODEL.FCOS.FOCAL_LOSS_ALPHA = 0.25
    cfg.MODEL.FCOS.FOCAL_LOSS_GAMMA = 2.0

    # Export method
    cfg.FCOS_PREPARE_FOR_EXPORT = "default_fcos_prepare_for_export"


# Re-register D2's meta-arch in D2Go with updated APIs
@META_ARCH_REGISTRY.register()
class FCOS(d2_FCOS):
    """
    Implement config->argument translation for FCOS model.
    """

    @configurable
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @classmethod
    def from_config(cls, cfg):
        backbone = build_backbone(cfg)
        backbone_shape = backbone.output_shape()
        try:
            feature_shapes = [backbone_shape[f] for f in cfg.MODEL.FCOS.IN_FEATURES]
        except KeyError:
            raise KeyError(
                f"Available keys: {backbone_shape.keys()}.  Requested keys: {cfg.MODEL.FCOS.IN_FEATURES}"
            )
        head = FCOSHead(
            input_shape=feature_shapes,
            num_classes=cfg.MODEL.FCOS.NUM_CLASSES,
            conv_dims=[feature_shapes[0].channels] * cfg.MODEL.FCOS.NUM_CONVS,
            norm=cfg.MODEL.FCOS.HEAD_NORM,
        )
        return {
            "backbone": backbone,
            "head": head,
            "pixel_mean": cfg.MODEL.PIXEL_MEAN,
            "pixel_std": cfg.MODEL.PIXEL_STD,
            "num_classes": cfg.MODEL.FCOS.NUM_CLASSES,
            "head_in_features": cfg.MODEL.FCOS.IN_FEATURES,
            # Loss parameters:
            "focal_loss_alpha": cfg.MODEL.FCOS.FOCAL_LOSS_ALPHA,
            "focal_loss_gamma": cfg.MODEL.FCOS.FOCAL_LOSS_GAMMA,
            # Inference parameters:
            "test_score_thresh": cfg.MODEL.FCOS.SCORE_THRESH_TEST,
            "test_topk_candidates": cfg.MODEL.FCOS.TOPK_CANDIDATES_TEST,
            "test_nms_thresh": cfg.MODEL.FCOS.NMS_THRESH_TEST,
            "max_detections_per_image": cfg.TEST.DETECTIONS_PER_IMAGE,
        }

    # HACK: default FCOS export shares the same prepare functions w/ RCNN under certain constrains
    def prepare_for_export(self, cfg, *args, **kwargs):
        func = FCOS_PREPARE_FOR_EXPORT_REGISTRY.get(cfg.FCOS_PREPARE_FOR_EXPORT)
        return func(self, cfg, *args, **kwargs)

    def prepare_for_quant(self, cfg, *args, **kwargs):
        """Wrap each quantized part of the model to insert Quant and DeQuant in-place"""

        model = self
        qconfig = set_backend_and_create_qconfig(
            cfg, is_train=cfg.QUANTIZATION.QAT.ENABLED
        )
        logger.info("Setup the model with qconfig:\n{}".format(qconfig))
        model.backbone.qconfig = qconfig
        model.head.qconfig = qconfig

        # Wrap the backbone based on the architecture type
        if isinstance(model.backbone, FPN):
            # Same trick in RCNN's _apply_eager_mode_quant
            model.backbone.bottom_up = wrap_quant_subclass(
                model.backbone.bottom_up,
                n_inputs=1,
                n_outputs=len(model.backbone.bottom_up._out_features),
            )
        else:
            model.backbone = wrap_quant_subclass(
                model.backbone, n_inputs=1, n_outputs=len(model.backbone._out_features)
            )

        def unpack_cyclebatchnormlist(module):
            # HACK: This function flattens CycleBatchNormList for quantization purpose
            if isinstance(module, CycleBatchNormList):
                if len(module) > 1:
                    # TODO: add quantization support of CycleBatchNormList
                    raise NotImplementedError(
                        "CycleBatchNormList w/ more than one element cannot be quantized"
                    )
                else:
                    num_channel = module.weight.size(0)
                    new_module = nn.BatchNorm2d(num_channel, affine=True)
                    new_module.weight = module.weight
                    new_module.bias = module.bias
                    new_module.running_mean = module[0].running_mean
                    new_module.running_var = module[0].running_var
                    module = new_module
            else:
                for name, child in module.named_children():
                    new_child = unpack_cyclebatchnormlist(child)
                    if new_child is not child:
                        module.add_module(name, new_child)
            return module

        model.head = unpack_cyclebatchnormlist(model.head)

        # Wrap the FCOS head
        model.head = wrap_quant_subclass(
            model.head,
            n_inputs=len(cfg.MODEL.FCOS.IN_FEATURES),
            n_outputs=len(cfg.MODEL.FCOS.IN_FEATURES) * 3,
        )

        model = fuse_utils.fuse_model(
            model,
            is_qat=cfg.QUANTIZATION.QAT.ENABLED,
            inplace=True,
        )
        model = wrap_non_quant_group_norm(model)
        return model


@FCOS_PREPARE_FOR_EXPORT_REGISTRY.register()
def default_fcos_prepare_for_export(self, cfg, inputs, predictor_type):
    pytorch_model = self

    preprocess_info = FuncInfo.gen_func_info(
        D2RCNNInferenceWrapper.Preprocess, params={}
    )
    preprocess_func = preprocess_info.instantiate()

    return PredictorExportConfig(
        model=FCOSInferenceWrapper(pytorch_model),
        data_generator=lambda x: (preprocess_func(x),),
        model_export_method=predictor_type,
        preprocess_info=preprocess_info,
        postprocess_info=FuncInfo.gen_func_info(
            D2RCNNInferenceWrapper.Postprocess,
            params={"detector_postprocess_done_in_model": True},
        ),
    )
