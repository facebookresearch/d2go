#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
#!/usr/bin/python
import errno
import importlib
import inspect
import logging
import math
import os
import pickle
import re
import signal
import sys
import tempfile
import threading
import time
import traceback
import typing
import warnings
import zipfile
from contextlib import contextmanager
from functools import partial
from functools import wraps
from random import random
from typing import (
    Any,
    Callable,
    Iterable,
    List,
    Mapping,
    NamedTuple,
    Optional,
    Tuple,
    Type,
    TypeVar,
    Union,
)

import detectron2.utils.comm as comm
import pkg_resources
import six
import torch
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from detectron2.engine import (
    DefaultTrainer,
    default_argument_parser,
    default_setup,
    hooks,
    launch,
)
from detectron2.evaluation import (
    CityscapesInstanceEvaluator,
    CityscapesSemSegEvaluator,
    COCOEvaluator,
    COCOPanopticEvaluator,
    DatasetEvaluators,
    LVISEvaluator,
    PascalVOCDetectionEvaluator,
    SemSegEvaluator,
    verify_results,
)

T = TypeVar("T")
CallbackMapping = Mapping[Callable, Optional[Iterable[Any]]]
FuncType = Callable[..., Any]
F = TypeVar("F", bound=FuncType)
RT = TypeVar("RT")
NT = TypeVar("T", bound=NamedTuple)

from detectron2.utils.events import TensorboardXWriter


class MultipleFunctionCallError(Exception):
    pass


def run_once(
    raise_on_multiple: bool = False,
    # pyre-fixme[34]: `Variable[T]` isn't present in the function's parameters.
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """
    A decorator to wrap a function such that it only ever runs once
    Useful, for example, with exit handlers that could be run via atexit or
    via a signal handler. The decorator will cache the result of the first call
    and return it on subsequent calls. If `raise_on_multiple` is set, any call
    to the function after the first one will raise a
    `MultipleFunctionCallError`.
    """

    def decorator(func: Callable[..., T]) -> (Callable[..., T]):
        signal: List[T] = []

        @wraps(func)
        def wrapper(*args, **kwargs) -> T:
            if signal:
                if raise_on_multiple:
                    raise MultipleFunctionCallError(
                        "Function %s was called multiple times" % func.__name__
                    )
                return signal[0]
            signal.append(func(*args, **kwargs))
            return signal[0]

        return wrapper

    return decorator


class retryable(object):
    """Fake retryable function"""

    def __init__(self, num_tries=1, sleep_time=0.1):
        pass

    def __call__(self, func: F) -> F:
        return func


# pyre-fixme[3]: Return type must be annotated.
# pyre-fixme[2]: Parameter must be annotated.
def get_dir_path(relative_path):
    """Return a path for a directory in this package, extracting if necessary

    For an entire directory within the par file (zip, fastzip) or lpar
    structure, this function will check to see if the contents are extracted;
    extracting each file that has not been extracted.  It returns the path of
    a directory containing the expected contents, making sure permissions are
    correct.

    Returns a string path, throws exeption on error
    """
    return os.path.dirname(importlib.import_module(relative_path).__file__)


# copy util function for oss
def alias(x, name, is_backward=False):
    if not torch.onnx.is_in_onnx_export():
        return x
    assert isinstance(x, torch.Tensor)
    return torch.ops._caffe2.AliasWithName(x, name, is_backward=is_backward)


class D2Trainer(DefaultTrainer):
    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        """
        Create evaluator(s) for a given dataset.
        This uses the special metadata "evaluator_type" associated with each builtin dataset.
        For your own dataset, you can simply create an evaluator manually in your
        script and do not have to worry about the hacky if-else logic here.
        """
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        evaluator_list = []
        evaluator_type = MetadataCatalog.get(dataset_name).evaluator_type
        if evaluator_type in ["sem_seg", "coco_panoptic_seg"]:
            evaluator_list.append(
                SemSegEvaluator(
                    dataset_name,
                    distributed=True,
                    output_dir=output_folder,
                )
            )
        if evaluator_type in ["coco", "coco_panoptic_seg"]:
            evaluator_list.append(COCOEvaluator(dataset_name, output_dir=output_folder))
        if evaluator_type == "coco_panoptic_seg":
            evaluator_list.append(COCOPanopticEvaluator(dataset_name, output_folder))
        if evaluator_type == "cityscapes_instance":
            assert (
                torch.cuda.device_count() >= comm.get_rank()
            ), "CityscapesEvaluator currently do not work with multiple machines."
            return CityscapesInstanceEvaluator(dataset_name)
        if evaluator_type == "cityscapes_sem_seg":
            assert (
                torch.cuda.device_count() >= comm.get_rank()
            ), "CityscapesEvaluator currently do not work with multiple machines."
            return CityscapesSemSegEvaluator(dataset_name)
        elif evaluator_type == "pascal_voc":
            return PascalVOCDetectionEvaluator(dataset_name)
        elif evaluator_type == "lvis":
            return LVISEvaluator(dataset_name, output_dir=output_folder)
        if len(evaluator_list) == 0:
            raise NotImplementedError(
                "no Evaluator for the dataset {} with the type {}".format(
                    dataset_name, evaluator_type
                )
            )
        elif len(evaluator_list) == 1:
            return evaluator_list[0]
        return DatasetEvaluators(evaluator_list)
