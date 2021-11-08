#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved


import logging

import torch
import torch.nn as nn
from caffe2.proto import caffe2_pb2
from d2go.export.api import PredictorExportConfig
from d2go.utils.qat_utils import get_qat_qconfig
from detectron2.export.caffe2_modeling import (
    META_ARCH_CAFFE2_EXPORT_TYPE_MAP,
    convert_batched_inputs_to_c2_format,
)
from detectron2.export.shared import get_pb_arg_vali, get_pb_arg_vals
from detectron2.modeling import GeneralizedRCNN
from detectron2.modeling.postprocessing import detector_postprocess
from detectron2.projects.point_rend import PointRendMaskHead
from detectron2.utils.registry import Registry
from mobile_cv.arch.utils import fuse_utils
from mobile_cv.arch.utils.quantize_utils import (
    wrap_non_quant_group_norm,
    wrap_quant_subclass,
    QuantWrapper,
)
from mobile_cv.predictor.api import FuncInfo
from torch.ao.quantization.quantize_fx import prepare_fx, prepare_qat_fx, convert_fx

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
    if (
        "@c2_ops" in predictor_type
        or "caffe2" in predictor_type
        or "onnx" in predictor_type
    ):
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
        model_export_kwargs = {}
        if "torchscript" in predictor_type:
            model_export_kwargs["force_disable_tracing_adapter"] = True

        return PredictorExportConfig(
            model=c2_compatible_model,
            # Caffe2MetaArch takes a single tuple as input (which is the return of
            # preprocess_func), data_generator requires all positional args as a tuple.
            data_generator=lambda x: (preprocess_func(x),),
            model_export_method=predictor_type.replace("@c2_ops", "", 1),
            model_export_kwargs=model_export_kwargs,
            preprocess_info=preprocess_info,
            postprocess_info=postprocess_info,
        )

    else:
        preprocess_info = FuncInfo.gen_func_info(
            D2RCNNInferenceWrapper.Preprocess, params={}
        )
        preprocess_func = preprocess_info.instantiate()
        return PredictorExportConfig(
            model=D2RCNNInferenceWrapper(self),
            data_generator=lambda x: (preprocess_func(x),),
            preprocess_info=preprocess_info,
            postprocess_info=FuncInfo.gen_func_info(
                D2RCNNInferenceWrapper.Postprocess, params={}
            ),
        )


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
        get_qat_qconfig(
            cfg.QUANTIZATION.BACKEND, cfg.QUANTIZATION.QAT.FAKE_QUANT_METHOD
        )
        if model.training
        else torch.ao.quantization.get_default_qconfig(cfg.QUANTIZATION.BACKEND)
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


class D2Caffe2MetaArchPreprocessFunc(object):
    def __init__(self, size_divisibility, device):
        self.size_divisibility = size_divisibility
        self.device = device

    def __call__(self, inputs):
        data, im_info = convert_batched_inputs_to_c2_format(
            inputs, self.size_divisibility, self.device
        )
        return (data, im_info)

    @staticmethod
    def get_params(cfg, model):
        fake_predict_net = caffe2_pb2.NetDef()
        model.encode_additional_info(fake_predict_net, None)
        size_divisibility = get_pb_arg_vali(fake_predict_net, "size_divisibility", 0)
        device = get_pb_arg_vals(fake_predict_net, "device", b"cpu").decode("ascii")
        return {
            "size_divisibility": size_divisibility,
            "device": device,
        }


class D2Caffe2MetaArchPostprocessFunc(object):
    def __init__(self, external_input, external_output, encoded_info):
        self.external_input = external_input
        self.external_output = external_output
        self.encoded_info = encoded_info

    def __call__(self, inputs, tensor_inputs, tensor_outputs):
        encoded_info = self.encoded_info.encode("ascii")
        fake_predict_net = caffe2_pb2.NetDef().FromString(encoded_info)
        meta_architecture = get_pb_arg_vals(fake_predict_net, "meta_architecture", None)
        meta_architecture = meta_architecture.decode("ascii")

        model_class = META_ARCH_CAFFE2_EXPORT_TYPE_MAP[meta_architecture]
        convert_outputs = model_class.get_outputs_converter(fake_predict_net, None)
        c2_inputs = tensor_inputs
        c2_results = dict(zip(self.external_output, tensor_outputs))
        return convert_outputs(inputs, c2_inputs, c2_results)

    @staticmethod
    def get_params(cfg, model):
        # NOTE: the post processing has different values for different meta
        # architectures, here simply relying Caffe2 meta architecture to encode info
        # into a NetDef and storing it as whole.
        fake_predict_net = caffe2_pb2.NetDef()
        model.encode_additional_info(fake_predict_net, None)
        encoded_info = fake_predict_net.SerializeToString().decode("ascii")

        # HACK: Caffe2MetaArch's post processing requires the blob name of model output,
        # this information is missed for torchscript. There's no easy way to know this
        # unless using NamedTuple for tracing.
        external_input = ["data", "im_info"]
        if cfg.MODEL.META_ARCHITECTURE == "GeneralizedRCNN":
            external_output = ["bbox_nms", "score_nms", "class_nms"]
            if cfg.MODEL.MASK_ON:
                external_output.extend(["mask_fcn_probs"])
            if cfg.MODEL.KEYPOINT_ON:
                if cfg.EXPORT_CAFFE2.USE_HEATMAP_MAX_KEYPOINT:
                    external_output.extend(["keypoints_out"])
                else:
                    external_output.extend(["kps_score"])
        else:
            raise NotImplementedError("")

        return {
            "external_input": external_input,
            "external_output": external_output,
            "encoded_info": encoded_info,
        }


class D2RCNNInferenceWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, image):
        """
        This function describes what happends during the tracing. Note that the output
        contains non-tensor, therefore the TracingAdaptedTorchscriptExport must be used in
        order to convert the output back from flattened tensors.
        """
        inputs = [{"image": image}]
        return self.model.inference(inputs, do_postprocess=False)[0]

    @staticmethod
    class Preprocess(object):
        """
        This function describes how to covert orginal input (from the data loader)
        to the inputs used during the tracing (i.e. the inputs of forward function).
        """

        def __call__(self, batch):
            assert len(batch) == 1, "only support single batch"
            return batch[0]["image"]

    class Postprocess(object):
        def __call__(self, batch, inputs, outputs):
            """
            This function describes how to run the predictor using exported model. Note
            that `tracing_adapter_wrapper` runs the traced model under the hood and
            behaves exactly the same as the forward function.
            """
            assert len(batch) == 1, "only support single batch"
            width, height = batch[0]["width"], batch[0]["height"]
            r = detector_postprocess(outputs, height, width)
            return [{"instances": r}]
