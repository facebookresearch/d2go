#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved


import json
import logging

import torch
from d2go.export.api import PredictorExportConfig
from detectron2.export.caffe2_modeling import META_ARCH_CAFFE2_EXPORT_TYPE_MAP
from mobile_cv.predictor.api import FuncInfo
from detectron2.export.flatten import TracingAdapter
from detectron2.export.torchscript_patch import patch_builtin_len
from d2go.utils.export_utils import (D2Caffe2MetaArchPreprocessFunc,
        D2Caffe2MetaArchPostprocessFunc, D2TracingAdapterPreprocessFunc, D2TracingAdapterPostFunc,
        dataclass_object_dump)

logger = logging.getLogger(__name__)


def d2_meta_arch_prepare_for_export(self, cfg, inputs, export_scheme):

    if "torchscript" in export_scheme and "@tracing" in export_scheme:

        def inference_func(model, image):
            inputs = [{"image": image}]
            return model.inference(inputs, do_postprocess=False)[0]

        def data_generator(x):
            return (x[0]["image"],)

        image = data_generator(inputs)[0]
        wrapper = TracingAdapter(self, image, inference_func)
        wrapper.eval()

        # HACK: outputs_schema can only be obtained after running tracing, but
        # PredictorExportConfig requires a pre-defined postprocessing function, this
        # causes tracing to run twice.
        logger.info("tracing the model to get outputs_schema ...")
        with torch.no_grad(), patch_builtin_len():
            _ = torch.jit.trace(wrapper, (image,))
        outputs_schema_json = json.dumps(
            wrapper.outputs_schema, default=dataclass_object_dump
        )

        return PredictorExportConfig(
            model=wrapper,
            data_generator=data_generator,
            preprocess_info=FuncInfo.gen_func_info(
                D2TracingAdapterPreprocessFunc, params={}
            ),
            postprocess_info=FuncInfo.gen_func_info(
                D2TracingAdapterPostFunc,
                params={"outputs_schema_json": outputs_schema_json},
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

