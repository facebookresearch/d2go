#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved


"""
API for exporting a pytorch model to a predictor, the predictor contains model(s) in
deployable format and predifined functions as glue code. The exported predictor should
generate same output as the original pytorch model. (See predictor/api.py for details of
predictor)

This API defines two customizable methods for the pytorch model:
    prepare_for_export (required by the default export_predictor): returns
        PredictorExportConfig which tells information about how export the predictor.
    export_predictor (optional): the implementation of export process. The default
        implementation is provided to cover the majority of use cases where the
        individual model(s) can be exported in standard way.

NOTE:
    1: There's a difference between predictor type and model type. model type
        refers to predifined deployable format such as caffe2, torchscript(_int8),
        while the predictor type can be anything that "export_predictor" can
        recognize.
    2: The standard model exporting methods are provided by the library code, they're
        meant to be modularized and can be used by customized export_predictor as well.
"""

import json
import logging
import os
from typing import Any, Callable, Dict, NamedTuple, Optional, Union

import torch
import torch.nn as nn
import torch.quantization.quantize_fx
from d2go.modeling.quantization import post_training_quantize
from fvcore.common.file_io import PathManager
from mobile_cv.arch.utils import fuse_utils
from mobile_cv.predictor.api import FuncInfo, ModelInfo, PredictorInfo
from mobile_cv.predictor.builtin_functions import (
    IdentityPostprocess,
    IdentityPreprocess,
    NaiveRunFunc,
)


logger = logging.getLogger(__name__)


class PredictorExportConfig(NamedTuple):
    """
    Storing information for exporting a predictor.

    Args:
        model (any nested iterable structure of nn.Module): the model(s) to be exported
            (via tracing/onnx or scripting). This can be sub-model(s) when the predictor
            consists of multiple models in deployable format, and/or pre/post processing
            is excluded due to requirement of tracing or hardward incompatibility.
        data_generator (Callable): a function to generate all data needed for tracing,
            such that data = data_generator(x), the returned data has the same nested
            structure as model. The data for each model will be treated as positional
            arguments, i.e. model(*data).
        model_export_kwargs (Dict): additional kwargs when exporting each sub-model, it
            follows the same nested structure as the model, and may contains information
            such as scriptable.

        preprocess_info (FuncInfo): info for predictor's preprocess
        postprocess_info (FuncInfo): info for predictor's postprocess
        run_func_info (FuncInfo): info for predictor's run_fun
    """

    model: Union[nn.Module, Any]
    # Shall we save data_generator in the predictor? This might be necessary when data
    # is needed, eg. running benchmark for sub models
    data_generator: Optional[Callable] = None
    model_export_kwargs: Optional[Union[Dict, Any]] = None

    preprocess_info: FuncInfo = FuncInfo.gen_func_info(IdentityPreprocess, params={})
    postprocess_info: FuncInfo = FuncInfo.gen_func_info(IdentityPostprocess, params={})
    run_func_info: FuncInfo = FuncInfo.gen_func_info(NaiveRunFunc, params={})


def convert_and_export_predictor(
    cfg, pytorch_model, predictor_type, output_dir, data_loader
):
    """
    Entry point for convert and export model. This involes two steps:
        - convert: converting the given `pytorch_model` to another format, currently
            mainly for quantizing the model.
        - export: exporting the converted `pytorch_model` to predictor. This step
            should not alter the behaviour of model.
    """
    if "int8" in predictor_type:
        if not cfg.QUANTIZATION.QAT.ENABLED:
            logger.info(
                "The model is not quantized during training, running post"
                " training quantization ..."
            )
            pytorch_model = post_training_quantize(cfg, pytorch_model, data_loader)
            # only check bn exists in ptq as qat still has bn inside fused ops
            assert not fuse_utils.check_bn_exist(pytorch_model)
        logger.info(f"Converting quantized model {cfg.QUANTIZATION.BACKEND}...")
        if cfg.QUANTIZATION.EAGER_MODE:
            # TODO(future diff): move this logic to prepare_for_quant_convert
            pytorch_model = torch.quantization.convert(pytorch_model, inplace=False)
        else:  # FX graph mode quantization
            if hasattr(pytorch_model, 'prepare_for_quant_convert'):
                pytorch_model = pytorch_model.prepare_for_quant_convert(cfg)
            else:
                # TODO(future diff): move this to a default function
                pytorch_model = torch.quantization.quantize_fx.convert_fx(pytorch_model)

        logger.info("Quantized Model:\n{}".format(pytorch_model))

    return export_predictor(cfg, pytorch_model, predictor_type, output_dir, data_loader)


def export_predictor(cfg, pytorch_model, predictor_type, output_dir, data_loader):
    """
    Interface for exporting a pytorch model to predictor of given type. This function
    can be override to arhieve customized exporting procedure, eg. using non-default
    optimization passes, composing traced models, etc.

    Args:
        cfg (CfgNode): the config
        pytorch_model (nn.Module): a pytorch model, mostly also a meta-arch
        predictor_type (str): a string which specifies the type of predictor, note that
            the definition of type is interpreted by "export_predictor", the default
            implementation uses the deployable model format (eg. caffe2_fp32,
            torchscript_int8) as predictor type.
        output_dir (str): the parent directory where the predictor will be saved
        data_loader: data loader for the pytorch model

    Returns:
        predictor_path (str): the directory of exported predictor, a sub-directory of
            "output_dir"
    """
    # predictor exporting can be customized by implement "export_predictor" of meta-arch
    if hasattr(pytorch_model, "export_predictor"):
        return pytorch_model.export_predictor(
            cfg, predictor_type, output_dir, data_loader
        )
    else:
        return default_export_predictor(
            cfg, pytorch_model, predictor_type, output_dir, data_loader
        )


def default_export_predictor(
    cfg, pytorch_model, predictor_type, output_dir, data_loader
):
    # The default implementation acts based on the PredictorExportConfig returned by
    # calling "prepare_for_export". It'll export all sub models in standard way
    # according to the "predictor_type".
    assert hasattr(pytorch_model, "prepare_for_export"), pytorch_model
    inputs = next(iter(data_loader))
    export_config = pytorch_model.prepare_for_export(
        cfg, inputs, export_scheme=predictor_type
    )

    predictor_path = os.path.join(output_dir, predictor_type)
    PathManager.mkdirs(predictor_path)

    # TODO: also support multiple models from nested dict in the default implementation
    assert isinstance(export_config.model, nn.Module), "Currently support single model"
    model = export_config.model
    input_args = (
        export_config.data_generator(inputs)
        if export_config.data_generator is not None
        else None
    )
    model_export_kwargs = export_config.model_export_kwargs or {}
    # the default implementation assumes model type is the same as the predictor type
    model_type = predictor_type
    model_path = predictor_path  # maye be sub dir for multipe models

    standard_model_export(
        model,
        model_type=model_type,
        save_path=model_path,
        input_args=input_args,
        **model_export_kwargs,
    )
    model_rel_path = os.path.relpath(model_path, predictor_path)

    # assemble predictor
    predictor_info = PredictorInfo(
        model=ModelInfo(path=model_rel_path, type=model_type),
        preprocess_info=export_config.preprocess_info,
        postprocess_info=export_config.postprocess_info,
        run_func_info=export_config.run_func_info,
    )
    with PathManager.open(
        os.path.join(predictor_path, "predictor_info.json"), "w"
    ) as f:
        json.dump(predictor_info.to_dict(), f, indent=4)

    return predictor_path


# TODO: determine if saving data should be part of standard_model_export or not.
# TODO: determine how to support PTQ, option 1): do everything inside this function,
# drawback: needs data loader; no customization. option 2): do calibration outside,
# and only do tracing inside (same as fp32 torchscript model).
# TODO: define the supported model types, current caffe2/torchscript/torchscript_int8
# is not enough.
# TODO: determine if registry is needed (probably not since we only need to support
# a few known formats) as libarary code.
def standard_model_export(model, model_type, save_path, input_args, **kwargs):
    if model_type.startswith("torchscript"):
        from d2go.export.torchscript import trace_and_save_torchscript
        trace_and_save_torchscript(model, input_args, save_path, **kwargs)
    elif model_type == "caffe2":
        from d2go.export.caffe2 import export_caffe2
        # TODO: export_caffe2 depends on D2, need to make a copy of the implemetation
        # TODO: support specifying optimization pass via kwargs
        export_caffe2(model, input_args[0], save_path, **kwargs)
    else:
        raise NotImplementedError("Incorrect model_type: {}".format(model_type))
