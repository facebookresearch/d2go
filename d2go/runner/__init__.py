#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved


import importlib
from typing import Optional, Type, Union

from d2go.runner.api import RunnerV2Mixin
from d2go.runner.default_runner import (
    BaseRunner,
    Detectron2GoRunner,
    GeneralizedRCNNRunner,
)
from d2go.runner.lightning_task import DefaultTask
from d2go.runner.training_hooks import TRAINER_HOOKS_REGISTRY

__all__ = [
    "RunnerV2Mixin",
    "BaseRunner",
    "Detectron2GoRunner",
    "GeneralizedRCNNRunner",
    "TRAINER_HOOKS_REGISTRY",
    "create_runner",
    "import_runner",
]


# TODO: remove this function
def create_runner(
    class_full_name: Optional[str], *args, **kwargs
) -> Union[BaseRunner, Type[DefaultTask]]:
    """Constructs a runner instance if class is a d2go runner. Returns class
    type if class is a Lightning module.
    """
    if class_full_name is None:
        runner_class = GeneralizedRCNNRunner
    else:
        runner_class = import_runner(class_full_name)
    if issubclass(runner_class, DefaultTask):
        # Return runner class for Lightning module since it requires config
        # to construct
        return runner_class
    return runner_class(*args, **kwargs)


def import_runner(
    class_full_name: str, check: bool = True
) -> Type[Union[BaseRunner, DefaultTask]]:
    runner_module_name, runner_class_name = class_full_name.rsplit(".", 1)
    runner_module = importlib.import_module(runner_module_name)
    runner_class = getattr(runner_module, runner_class_name)

    if check and not (
        issubclass(runner_class, BaseRunner) ^ issubclass(runner_class, DefaultTask)
    ):
        raise ValueError(
            f"The runner must be subclass of either `BaseRunner` or `DefaultTaks`,"
            f" found: {runner_class}"
        )

    return runner_class
