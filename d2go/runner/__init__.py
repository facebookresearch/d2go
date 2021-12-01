#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved


import importlib
from typing import Type, Union, Optional

from pytorch_lightning import LightningModule

from .default_runner import BaseRunner, GeneralizedRCNNRunner


def create_runner(
    class_full_name: Optional[str], *args, **kwargs
) -> Union[BaseRunner, Type[LightningModule]]:
    """Constructs a runner instance if class is a d2go runner. Returns class
    type if class is a Lightning module.
    """
    if class_full_name is None:
        runner_class = GeneralizedRCNNRunner
    else:
        runner_module_name, runner_class_name = class_full_name.rsplit(".", 1)
        runner_module = importlib.import_module(runner_module_name)
        runner_class = getattr(runner_module, runner_class_name)
    if issubclass(runner_class, LightningModule):
        # Return runner class for Lightning module since it requires config
        # to construct
        return runner_class
    return runner_class(*args, **kwargs)
