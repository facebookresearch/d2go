#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved


import logging

from d2go.export.api import PredictorExportConfig
from d2go.utils.export_utils import (
    D2Caffe2MetaArchPreprocessFunc,
    D2Caffe2MetaArchPostprocessFunc,
    D2RCNNTracingWrapper,
)
from detectron2.export.caffe2_modeling import META_ARCH_CAFFE2_EXPORT_TYPE_MAP
from mobile_cv.predictor.api import FuncInfo

logger = logging.getLogger(__name__)


def d2_meta_arch_prepare_for_export(self, cfg, inputs, predictor_type):

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
