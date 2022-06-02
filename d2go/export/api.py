#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import json
import sys
from abc import ABC, abstractmethod
from typing import Callable, Dict, NamedTuple, Optional, Union

import torch.nn as nn
from mobile_cv.common.misc.file_utils import make_temp_directory
from mobile_cv.common.misc.registry import Registry
from mobile_cv.predictor.api import FuncInfo
from mobile_cv.predictor.builtin_functions import (
    IdentityPostprocess,
    IdentityPreprocess,
    NaiveRunFunc,
)

if sys.version_info >= (3, 8):
    from typing import final
else:
    # If final decorator not available when using older python version, replace with the
    # dummy implementation that does nothing.
    def final(func):
        return func


class PredictorExportConfig(NamedTuple):
    """
    Storing information for exporting a predictor.

    Args:
        model (any nested iterable structure of nn.Module): the model(s) to be exported
            (via tracing/onnx or scripting). This can be sub-model(s) when the predictor
            consists of multiple models in deployable format, and/or pre/post processing
            is excluded due to requirement of tracing or hardware incompatibility.
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

    model: Union[nn.Module, Dict[str, nn.Module]]
    data_generator: Optional[Callable] = None
    model_export_method: Optional[Union[str, Dict[str, str]]] = None
    model_export_kwargs: Optional[Union[Dict, Dict[str, Dict]]] = None

    preprocess_info: FuncInfo = FuncInfo.gen_func_info(IdentityPreprocess, params={})
    postprocess_info: FuncInfo = FuncInfo.gen_func_info(IdentityPostprocess, params={})
    run_func_info: FuncInfo = FuncInfo.gen_func_info(NaiveRunFunc, params={})


class ModelExportMethod(ABC):
    """
    Base class for "model export method". Each model export method can export a pytorch
    model to a certain deployable format, such as torchscript or caffe2. It consists
    with `export` and `load` methods.
    """

    @classmethod
    @abstractmethod
    def export(cls, model, input_args, save_path, export_method, **export_kwargs):
        """
        Export the model to deployable format.

        Args:
            model (nn.Module): a pytorch model to export
            input_args (Tuple[Any]): inputs of model, called as model(*input_args)
            save_path (str): directory where the model will be exported
            export_method (str): string name for the export method
            export_kwargs (Dict): additional parameters for exporting model defined
                by each model export method.
        Return:
            load_kwargs (Dict): additional information (besides save_path) needed in
                order to load the exported model. This needs to be JSON serializable.
        """
        pass

    @classmethod
    @abstractmethod
    def load(cls, save_path, **load_kwargs):
        """
        Load the exported model back for inference.

        Args:
            save_path (str): directory where the model is stored.
            load_kwargs (Dict): addtional information to load the exported model.
        Returns:
            model (nn.Module): a nn.Module (often time a wrapper for non torchscript
                types like caffe2), it works the same as the original pytorch model,
                i.e. getting the same output when called as model(*input_args)
        """
        pass

    @classmethod
    @final
    def test_export_and_load(
        cls, model, input_args, export_method, export_kwargs, output_checker
    ):
        """
        Illustrate the life-cycle of export and load, used for testing.
        """
        with make_temp_directory("test_export_and_load") as save_path:
            # run the orginal model
            assert isinstance(model, nn.Module), model
            assert isinstance(input_args, (list, tuple)), input_args
            original_output = model(*input_args)
            # export the model
            model.eval()  # TODO: decide where eval() should be called
            load_kwargs = cls.export(
                model, input_args, save_path, export_method, **export_kwargs
            )
            # sanity check for load_kwargs
            assert isinstance(load_kwargs, dict), load_kwargs
            assert json.dumps(load_kwargs), load_kwargs
            # loaded model back
            loaded_model = cls.load(save_path, **load_kwargs)
            # run the loaded model
            assert isinstance(loaded_model, nn.Module), loaded_model
            new_output = loaded_model(*input_args)
            # compare outputs
            output_checker(new_output, original_output)


ModelExportMethodRegistry = Registry("ModelExportMethod", allow_override=True)
