#!/usr/bin/env python3

import sys
import json
import importlib
import dataclasses
from caffe2.proto import caffe2_pb2
from detectron2.export.caffe2_modeling import (
    META_ARCH_CAFFE2_EXPORT_TYPE_MAP,
    convert_batched_inputs_to_c2_format,
)
from detectron2.export.shared import get_pb_arg_vali, get_pb_arg_vals
from detectron2.modeling.postprocessing import detector_postprocess


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


def dataclass_object_dump(ob):
    datacls = type(ob)
    if not dataclasses.is_dataclass(datacls):
        raise TypeError(f"Expected dataclass instance, got '{datacls!r}' object")
    mod = sys.modules.get(datacls.__module__)
    if mod is None or not hasattr(mod, datacls.__qualname__):
        raise ValueError(f"Can't resolve '{datacls!r}' reference")
    ref = f"{datacls.__module__}.{datacls.__qualname__}"
    fields = (f.name for f in dataclasses.fields(ob))
    return {**{f: getattr(ob, f) for f in fields}, "__dataclass__": ref}



def dataclass_object_load(d):
    ref = d.pop("__dataclass__", None)
    if ref is None:
        return d
    try:
        modname, hasdot, qualname = ref.rpartition(".")
        module = importlib.import_module(modname)
        datacls = getattr(module, qualname)
        if not dataclasses.is_dataclass(datacls) or not isinstance(datacls, type):
            raise ValueError
        return datacls(**d)
    except (ModuleNotFoundError, ValueError, AttributeError, TypeError):
        raise ValueError(f"Invalid dataclass reference {ref!r}") from None


class D2TracingAdapterPreprocessFunc(object):
    def __call__(self, inputs):
        assert len(inputs) == 1, "only support single batch"
        return inputs[0]["image"]


class D2TracingAdapterPostFunc(object):
    def __init__(self, outputs_schema_json):
        self.outputs_schema = json.loads(
            outputs_schema_json, object_hook=dataclass_object_load
        )

    def __call__(self, inputs, tensor_inputs, tensor_outputs):
        results_per_image = self.outputs_schema(tensor_outputs)

        assert len(inputs) == 1, "only support single batch"
        width, height = inputs[0]["width"], inputs[0]["height"]
        r = detector_postprocess(results_per_image, height, width)
        return [{"instances": r}]
