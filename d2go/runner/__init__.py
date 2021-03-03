#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved


import importlib
from typing import Type

from .default_runner import BaseRunner, Detectron2GoRunner, GeneralizedRCNNRunner

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
