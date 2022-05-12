from typing import Callable, Dict, NamedTuple, Optional, Union

import torch.nn as nn
from mobile_cv.predictor.api import FuncInfo
from mobile_cv.predictor.builtin_functions import (
    IdentityPostprocess,
    IdentityPreprocess,
    NaiveRunFunc,
)


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
