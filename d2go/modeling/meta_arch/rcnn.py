#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved


import inspect
import logging

import torch
import torch.nn as nn
from d2go.export.api import PredictorExportConfig
from d2go.quantization.qconfig import set_backend_and_create_qconfig
from detectron2.modeling import GeneralizedRCNN
from detectron2.modeling.backbone.fpn import FPN
from detectron2.modeling.postprocessing import detector_postprocess
from detectron2.projects.point_rend import PointRendMaskHead
from detectron2.utils.registry import Registry
from mobile_cv.arch.utils import fuse_utils
from mobile_cv.arch.utils.quantize_utils import (
    QuantWrapper,
    wrap_non_quant_group_norm,
    wrap_quant_subclass,
)
from mobile_cv.predictor.api import FuncInfo
from torch.ao.quantization import convert
from torch.ao.quantization.quantize_fx import convert_fx, prepare_fx, prepare_qat_fx

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
        "_cast_model_to_device",
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

    def _cast_model_to_device(self, device):
        return _cast_detection_model(self, device)


@RCNN_PREPARE_FOR_EXPORT_REGISTRY.register()
def default_rcnn_prepare_for_export(self, cfg, inputs, predictor_type):
    pytorch_model = self

    if (
        "@c2_ops" in predictor_type
        or "caffe2" in predictor_type
        or "onnx" in predictor_type
    ):
        from detectron2.export.caffe2_modeling import META_ARCH_CAFFE2_EXPORT_TYPE_MAP

        C2MetaArch = META_ARCH_CAFFE2_EXPORT_TYPE_MAP[cfg.MODEL.META_ARCHITECTURE]
        c2_compatible_model = C2MetaArch(cfg, pytorch_model)

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
        do_postprocess = cfg.RCNN_EXPORT.INCLUDE_POSTPROCESS
        preprocess_info = FuncInfo.gen_func_info(
            D2RCNNInferenceWrapper.Preprocess, params={}
        )
        preprocess_func = preprocess_info.instantiate()
        return PredictorExportConfig(
            model=D2RCNNInferenceWrapper(
                pytorch_model,
                do_postprocess=do_postprocess,
            ),
            data_generator=lambda x: (preprocess_func(x),),
            model_export_method=predictor_type,
            preprocess_info=preprocess_info,
            postprocess_info=FuncInfo.gen_func_info(
                D2RCNNInferenceWrapper.Postprocess,
                params={"detector_postprocess_done_in_model": do_postprocess},
            ),
        )


def _apply_eager_mode_quant(cfg, model):

    if isinstance(model, GeneralizedRCNN):
        """Wrap each quantized part of the model to insert Quant and DeQuant in-place"""

        # Wrap backbone and proposal_generator
        if isinstance(model.backbone, FPN):
            # HACK: currently the quantization won't pick up D2's the Conv2d, which is
            # used by D2's default FPN (same as FBNetV2FPN), this causes problem if we
            # warpping entire backbone as whole. The current solution is only quantizing
            # bottom_up and leaving other parts un-quantized. TODO (T109761730): However
            # we need to re-visit this if using other (fbnet-based) FPN module since the
            # new FPN module might be pikced by quantization.
            model.backbone.bottom_up = wrap_quant_subclass(
                model.backbone.bottom_up,
                n_inputs=1,
                n_outputs=len(model.backbone.bottom_up._out_features),
            )
        else:
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
    assert not isinstance(self.backbone, FPN), "FPN is not supported in FX mode"
    # TODO[quant-example-inputs]: Expose example_inputs as argument
    # Note: this is used in quantization for all submodules
    example_inputs = (torch.rand(1, 3, 3, 3),)
    self.backbone = prep_fn(
        self.backbone,
        qconfig,
        example_inputs,
        prepare_custom_config_dict={
            "preserved_attributes": ["size_divisibility", "padding_constraints"],
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
        example_inputs,
        prepare_custom_config_dict={
            # rpn_feature expecting quantized input, this is used to avoid redundant
            # quant
            "input_quantized_idxs": [0]
        },
    )
    self.proposal_generator.rpn_head.rpn_regressor.cls_logits = prep_fn(
        self.proposal_generator.rpn_head.rpn_regressor.cls_logits,
        qconfig,
        example_inputs,
    )
    self.proposal_generator.rpn_head.rpn_regressor.bbox_pred = prep_fn(
        self.proposal_generator.rpn_head.rpn_regressor.bbox_pred,
        qconfig,
        example_inputs,
    )
    self.roi_heads.box_head.roi_box_conv = prep_fn(
        self.roi_heads.box_head.roi_box_conv,
        qconfig,
        example_inputs,
        prepare_custom_config_dict={
            "output_quantized_idxs": [0],
        },
    )
    self.roi_heads.box_head.avgpool = prep_fn(
        self.roi_heads.box_head.avgpool,
        qconfig,
        example_inputs,
        prepare_custom_config_dict={"input_quantized_idxs": [0]},
    )
    self.roi_heads.box_predictor.cls_score = prep_fn(
        self.roi_heads.box_predictor.cls_score,
        qconfig,
        example_inputs,
        prepare_custom_config_dict={"input_quantized_idxs": [0]},
    )
    self.roi_heads.box_predictor.bbox_pred = prep_fn(
        self.roi_heads.box_predictor.bbox_pred,
        qconfig,
        example_inputs,
        prepare_custom_config_dict={"input_quantized_idxs": [0]},
    )


@RCNN_PREPARE_FOR_QUANT_REGISTRY.register()
def default_rcnn_prepare_for_quant(self, cfg):
    model = self
    model.qconfig = set_backend_and_create_qconfig(cfg, is_train=model.training)
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
        model = fuse_utils.fuse_model(
            model,
            is_qat=cfg.QUANTIZATION.QAT.ENABLED,
            inplace=True,
        )
    else:
        _fx_quant_prepare(model, cfg)

    return model


@RCNN_PREPARE_FOR_QUANT_CONVERT_REGISTRY.register()
def default_rcnn_prepare_for_quant_convert(self, cfg):
    if cfg.QUANTIZATION.EAGER_MODE:
        convert(self, inplace=True)
    else:
        assert not isinstance(self.backbone, FPN), "FPN is not supported in FX mode"
        self.backbone = convert_fx(
            self.backbone,
            convert_custom_config_dict={
                "preserved_attributes": ["size_divisibility", "padding_constraints"]
            },
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
        from detectron2.export.caffe2_modeling import (
            convert_batched_inputs_to_c2_format,
        )

        data, im_info = convert_batched_inputs_to_c2_format(
            inputs, self.size_divisibility, self.device
        )
        return (data, im_info)

    @staticmethod
    def get_params(cfg, model):
        from caffe2.proto import caffe2_pb2
        from detectron2.export.shared import get_pb_arg_vali, get_pb_arg_vals

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
        from caffe2.proto import caffe2_pb2
        from detectron2.export.caffe2_modeling import META_ARCH_CAFFE2_EXPORT_TYPE_MAP
        from detectron2.export.shared import get_pb_arg_vals

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
        from caffe2.proto import caffe2_pb2

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
    def __init__(
        self,
        model,
        do_postprocess=False,
    ):
        super().__init__()
        self.model = model
        self.do_postprocess = do_postprocess

    def forward(self, image):
        """
        This function describes what happends during the tracing. Note that the output
        contains non-tensor, therefore the TracingAdaptedTorchscriptExport must be used in
        order to convert the output back from flattened tensors.
        """
        if self.do_postprocess:
            inputs = [
                {
                    "image": image,
                    # NOTE: the width/height is not available since the model takes a
                    # single image tensor as input. Therefore even though post-process
                    # is specified, the wrapped model doesn't resize the output to its
                    # original width/height.
                    # TODO: If this is needed, we might make the model take extra
                    # width/height info like the C2-style inputs.
                }
            ]
            return self.model.forward(inputs)[0]["instances"]
        else:
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
        def __init__(self, detector_postprocess_done_in_model=False):
            """
            Args:
                detector_postprocess_done_in_model (bool): whether `detector_postprocess`
                has already applied in the D2RCNNInferenceWrapper
            """
            self.detector_postprocess_done_in_model = detector_postprocess_done_in_model

        def __call__(self, batch, inputs, outputs):
            """
            This function describes how to run the predictor using exported model. Note
            that `tracing_adapter_wrapper` runs the traced model under the hood and
            behaves exactly the same as the forward function.
            """
            assert len(batch) == 1, "only support single batch"
            width, height = batch[0]["width"], batch[0]["height"]
            if self.detector_postprocess_done_in_model:
                image_shape = batch[0]["image"].shape  # chw
                if image_shape[1] != height or image_shape[2] != width:
                    raise NotImplementedError(
                        f"Image tensor (shape: {image_shape}) doesn't match the"
                        f" input width ({width}) height ({height}). Since post-process"
                        f" has been done inside the torchscript without width/height"
                        f" information, can't recover the post-processed output to "
                        f"orignail resolution."
                    )
                return [{"instances": outputs}]
            else:
                r = detector_postprocess(outputs, height, width)
                return [{"instances": r}]


# TODO: model.to(device) might not work for detection meta-arch, this function is the
# workaround, in general, we might need a meta-arch API for this if needed.
def _cast_detection_model(model, device):
    # check model is an instance of one of the meta arch
    from detectron2.export.caffe2_modeling import Caffe2MetaArch
    from detectron2.modeling import META_ARCH_REGISTRY

    if isinstance(model, Caffe2MetaArch):
        model._wrapped_model = _cast_detection_model(model._wrapped_model, device)
        return model

    assert isinstance(model, tuple(META_ARCH_REGISTRY._obj_map.values()))
    model.to(device)
    # cast normalizer separately
    if hasattr(model, "normalizer") and not (
        hasattr(model, "pixel_mean") and hasattr(model, "pixel_std")
    ):
        pixel_mean = inspect.getclosurevars(model.normalizer).nonlocals["pixel_mean"]
        pixel_std = inspect.getclosurevars(model.normalizer).nonlocals["pixel_std"]
        pixel_mean = pixel_mean.to(device)
        pixel_std = pixel_std.to(device)
        model.normalizer = lambda x: (x - pixel_mean) / pixel_std
    return model
