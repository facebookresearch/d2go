#!/usr/bin/env python3

import importlib
from typing import Type

from .default_runner import BaseRunner, Detectron2GoRunner, GeneralizedRCNNRunner

# TODO: remove importing DensePoseRunner and OcrRunner
from .densepose_runner import DensePoseRunner
from .ocr_runner import OcrRunner
from .xray_detection_beta.runner.default_runner import XRayDetectionRunner

# NOTE: WARNING: This code path will be hit by all projects, be careful about making
# the import here especially if it uses third-part library, which can slow down the
# initialization or cause unintended side effects.
from . import argos  # noqa

def get_class(class_full_name: str) -> Type:
    """Imports and returns the task class."""
    runner_module_name, runner_class_name = class_full_name.rsplit(".", 1)
    runner_module = importlib.import_module(runner_module_name)
    runner_class = getattr(runner_module, runner_class_name)
    return runner_class


def create_runner(class_full_name: str, *args, **kwargs) -> BaseRunner:
    """Constructs a runner instance of the given class."""
    runner_class = get_class(class_full_name)
    return runner_class(*args, **kwargs)
