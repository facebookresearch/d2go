#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved


"""
API for exporting a pytorch model to a predictor, the predictor contains model(s) in
deployable format and predefined functions as glue code. The exported predictor should
generate same output as the original pytorch model. (See predictor/api.py for details of
predictor)

This API defines customizable methods for the pytorch model:
    prepare_for_export (required by the default export_predictor): returns
        PredictorExportConfig which tells information about how export the predictor.

NOTE:
    1: There's a difference between predictor type and model type. model type
        refers to predefined deployable format such as caffe2, torchscript(_int8),
        while the predictor type can be anything that "export_predictor" can
        recognize.
    2: The standard model exporting methods are provided by the library code, they're
        meant to be modularized and can be used by customized export_predictor as well.
"""

import json
import logging
import os
from typing import Iterable

import torch.nn as nn
from d2go.config import CfgNode
from d2go.export.api import ModelExportMethod, ModelExportMethodRegistry
from d2go.quantization.modeling import (
    convert_to_quantized_model,
    post_training_quantize,
)
from detectron2.utils.file_io import PathManager
from mobile_cv.arch.utils import fuse_utils
from mobile_cv.predictor.api import ModelInfo, PredictorInfo


logger = logging.getLogger(__name__)


def is_predictor_quantized(predictor_type: str) -> bool:
    return "int8" in predictor_type or "quant" in predictor_type


def convert_model(
    cfg: CfgNode,
    pytorch_model: nn.Module,
    predictor_type: str,
    data_loader: Iterable,
):
    """Converts pytorch model to pytorch model (fuse for fp32, fake quantize for int8)"""
    return (
        convert_quantized_model(cfg, pytorch_model, data_loader)
        if is_predictor_quantized(predictor_type)
        else _convert_fp_model(cfg, pytorch_model, data_loader)
    )


def convert_quantized_model(
    cfg: CfgNode, pytorch_model: nn.Module, data_loader: Iterable
) -> nn.Module:
    if not cfg.QUANTIZATION.QAT.ENABLED:
        # For PTQ, converts pytorch model to fake-quantized pytorch model. For QAT, the
        # built pytorch model is already fake-quantized.
        logger.info(
            "The model is not quantized during training, running post"
            " training quantization ..."
        )

        pytorch_model = post_training_quantize(cfg, pytorch_model, data_loader)
        # only check bn exists in ptq as qat still has bn inside fused ops
        if fuse_utils.check_bn_exist(pytorch_model):
            logger.warn("Post training quantized model has bn inside fused ops")
    logger.info("Converting quantized model...")

    # convert the fake-quantized model to int8 model
    pytorch_model = convert_to_quantized_model(cfg, pytorch_model)
    logger.info(f"Quantized Model:\n{pytorch_model}")
    return pytorch_model


def _convert_fp_model(
    cfg: CfgNode, pytorch_model: nn.Module, data_loader: Iterable
) -> nn.Module:
    """Converts floating point predictor"""
    if not isinstance(cfg, CfgNode) or (not cfg.QUANTIZATION.QAT.ENABLED):
        # Do not fuse model again for QAT model since it will remove observer statistics (e.g. min_val, max_val)
        pytorch_model = fuse_utils.fuse_model(pytorch_model)
        logger.info(f"Fused Model:\n{pytorch_model}")
        if fuse_utils.count_bn_exist(pytorch_model) > 0:
            logger.warning("BN existed in pytorch model after fusing.")
    return pytorch_model


def convert_and_export_predictor(
    cfg,
    pytorch_model,
    predictor_type,
    output_dir,
    data_loader,
):
    """
    Entry point for convert and export model. This involves two steps:
        - convert: converting the given `pytorch_model` to another format, currently
            mainly for quantizing the model.
        - export: exporting the converted `pytorch_model` to predictor. This step
            should not alter the behaviour of model.
    """
    pytorch_model = convert_model(cfg, pytorch_model, predictor_type, data_loader)
    return export_predictor(cfg, pytorch_model, predictor_type, output_dir, data_loader)


def export_predictor(cfg, pytorch_model, predictor_type, output_dir, data_loader):
    """
    Interface for exporting a pytorch model to predictor of given type. This function
    can be override to achieve customized exporting procedure, eg. using non-default
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
    return default_export_predictor(
        cfg, pytorch_model, predictor_type, output_dir, data_loader
    )


def _export_single_model(
    predictor_path,
    model,
    input_args,
    save_path,
    model_export_method,
    model_export_kwargs,
):
    assert isinstance(model, nn.Module), model
    # model_export_method either inherits ModelExportMethod or is a key in the registry
    model_export_method_str = None
    if isinstance(model_export_method, str):
        model_export_method_str = model_export_method
        model_export_method = ModelExportMethodRegistry.get(model_export_method)
    assert issubclass(model_export_method, ModelExportMethod), model_export_method
    logger.info(f"Using model export method: {model_export_method}")

    load_kwargs = model_export_method.export(
        model=model,
        input_args=input_args,
        save_path=save_path,
        export_method=model_export_method_str,
        **model_export_kwargs,
    )
    assert isinstance(load_kwargs, dict)
    model_rel_path = os.path.relpath(save_path, predictor_path)
    return ModelInfo(
        path=model_rel_path,
        export_method=f"{model_export_method.__module__}.{model_export_method.__qualname__}",
        load_kwargs=load_kwargs,
    )


def default_export_predictor(
    cfg, pytorch_model, predictor_type, output_dir, data_loader
):
    # The default implementation acts based on the PredictorExportConfig returned by
    # calling "prepare_for_export". It'll export all sub models in standard way
    # according to the "predictor_type".
    assert hasattr(pytorch_model, "prepare_for_export"), pytorch_model
    inputs = next(iter(data_loader))
    export_config = pytorch_model.prepare_for_export(cfg, inputs, predictor_type)
    model_inputs = (
        export_config.data_generator(inputs)
        if export_config.data_generator is not None
        else (inputs,)
    )

    predictor_path = os.path.join(output_dir, predictor_type)
    PathManager.mkdirs(predictor_path)

    predictor_init_kwargs = {
        "preprocess_info": export_config.preprocess_info,
        "postprocess_info": export_config.postprocess_info,
        "run_func_info": export_config.run_func_info,
    }

    if isinstance(export_config.model, dict):
        models_info = {}
        for name, model in export_config.model.items():
            save_path = os.path.join(predictor_path, name)
            model_export_kwargs = (
                {}
                if export_config.model_export_kwargs is None
                else export_config.model_export_kwargs[name]
            )
            if hasattr(cfg, "QUANTIZATION") and cfg.QUANTIZATION.RECIPE is not None:
                model_export_kwargs["recipe"] = cfg.QUANTIZATION.RECIPE
            model_info = _export_single_model(
                predictor_path=predictor_path,
                model=model,
                input_args=model_inputs[name] if model_inputs is not None else None,
                save_path=save_path,
                model_export_method=(
                    predictor_type
                    if export_config.model_export_method is None
                    else export_config.model_export_method[name]
                ),
                model_export_kwargs=model_export_kwargs,
            )
            models_info[name] = model_info
        predictor_init_kwargs["models"] = models_info
    else:
        save_path = predictor_path  # for single model exported files are put under `predictor_path` together with predictor_info.json
        model_export_kwargs = (
            {}
            if export_config.model_export_kwargs is None
            else export_config.model_export_kwargs
        )
        if hasattr(cfg, "QUANTIZATION") and cfg.QUANTIZATION.RECIPE is not None:
            model_export_kwargs["recipe"] = cfg.QUANTIZATION.RECIPE
        model_info = _export_single_model(
            predictor_path=predictor_path,
            model=export_config.model,
            input_args=model_inputs,
            save_path=save_path,
            model_export_method=export_config.model_export_method or predictor_type,
            model_export_kwargs=model_export_kwargs,
        )
        predictor_init_kwargs["model"] = model_info

    # assemble predictor
    predictor_info = PredictorInfo(**predictor_init_kwargs)
    with PathManager.open(
        os.path.join(predictor_path, "predictor_info.json"), "w"
    ) as f:
        json.dump(predictor_info.to_dict(), f, indent=4)

    return predictor_path
