#!/usr/bin/env python3

import logging
from functools import lru_cache

import torch
from caffe2.proto import caffe2_pb2
from d2go.export.api import PredictorExportConfig
from detectron2.export.caffe2_modeling import (
    META_ARCH_CAFFE2_EXPORT_TYPE_MAP,
    convert_batched_inputs_to_c2_format,
)
from detectron2.export.shared import get_pb_arg_vali, get_pb_arg_vals
from detectron2.modeling import META_ARCH_REGISTRY, GeneralizedRCNN
from mobile_cv.arch.utils import fuse_utils
from mobile_cv.arch.utils.quantize_utils import (
    wrap_non_quant_group_norm,
    wrap_quant_subclass,
    QuantWrapper,
)
from mobile_cv.predictor.api import FuncInfo

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
        model.roi_heads.box_head = wrap_quant_subclass(
            model.roi_heads.box_head,
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


def d2_meta_arch_prepare_for_quant(self, cfg):
    model = self

    # Modify the model for eager mode
    if cfg.QUANTIZATION.EAGER_MODE:
        model = _apply_eager_mode_quant(cfg, model)

    model = fuse_utils.fuse_model(model, inplace=True)

    torch.backends.quantized.engine = cfg.QUANTIZATION.BACKEND
    model.qconfig = (
        torch.quantization.get_default_qat_qconfig(cfg.QUANTIZATION.BACKEND)
        if model.training
        else torch.quantization.get_default_qconfig(cfg.QUANTIZATION.BACKEND)
    )
    logger.info("Setup the model with qconfig:\n{}".format(model.qconfig))

    return model


def d2_meta_arch_prepare_for_export(self, cfg, inputs, export_scheme):

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
        # this information is missed for torchscript. There'no easy way to know this
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
