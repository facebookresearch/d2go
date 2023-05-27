#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved


import logging
import os
import warnings
from contextlib import contextmanager
from typing import Any, Callable, Dict, Iterator, Optional

import detectron2.utils.comm as comm
import torch
from d2go.config.config import CfgNode
from d2go.utils.tensorboard_log_util import get_tensorboard_log_dir  # noqa: forwarding
from detectron2.utils.file_io import PathManager
from tabulate import tabulate

logger = logging.getLogger(__name__)

# Subdirectory with model configurations dumped by the training binary.
TRAINED_MODEL_CONFIGS_DIR: str = "trained_model_configs"


def check_version(library, min_version, warning_only=False):
    """Check the version of the library satisfies the provided minimum version.
    An exception is thrown if the check does not pass.
    Parameters
    ----------
    min_version : str
        Minimum version
    warning_only : bool
        Printing a warning instead of throwing an exception.
    """
    from distutils.version import LooseVersion

    version = library.__version__
    bad_version = LooseVersion(version) < LooseVersion(min_version)
    if bad_version:
        msg = (
            f"Installed {library.__name__} version {version} does not satisfy the "
            f"minimum required version {min_version}"
        )
        if warning_only:
            warnings.warn(msg)
        else:
            raise AssertionError(msg)
        return False
    return True


def metrics_dict_to_metrics_table(dic):
    assert isinstance(dic, dict)
    ret = []
    for key in sorted(dic.keys()):
        value = dic[key]
        if isinstance(value, dict):
            for sub_metrics in metrics_dict_to_metrics_table(value):
                ret.append([key] + sub_metrics)
        else:
            ret.append([key, value])
    return ret


def print_metrics_table(metrics_dic):
    metrics_table = metrics_dict_to_metrics_table(metrics_dic)
    metrics_tabulate = tabulate(
        metrics_table,
        tablefmt="pipe",
        headers=["model", "dataset", "task", "metric", "score"],
    )
    logger.info("Metrics table: \n" + metrics_tabulate)


def dump_trained_model_configs(
    output_dir: str, trained_cfgs: Dict[str, CfgNode]
) -> Dict[str, str]:
    """Writes trained model config files to output_dir.

    Args:
        output_dir: output file directory.
        trained_cfgs: map from model name to the config of trained model.

    Returns:
        A map of model name to model config path.
    """
    trained_model_configs = {}
    trained_model_config_dir = os.path.join(output_dir, TRAINED_MODEL_CONFIGS_DIR)
    PathManager.mkdirs(trained_model_config_dir)
    for name, trained_cfg in trained_cfgs.items():
        config_file = os.path.join(trained_model_config_dir, "{}.yaml".format(name))
        trained_model_configs[name] = config_file
        if comm.is_main_process():
            logger.info("Dumping trained config file: {}".format(config_file))
            with PathManager.open(config_file, "w") as f:
                f.write(trained_cfg.dump())
        comm.synchronize()
        logger.info("Finished dumping trained config file")
    return trained_model_configs


def save_binary_outputs(filename: str, outputs: Any) -> None:
    """Helper function to serialize and save function outputs in binary format."""
    with PathManager.open(filename, "wb") as f:
        torch.save(outputs, f)


def load_binary_outputs(filename: str) -> Any:
    """Helper function to load and deserialize function outputs saved in binary format."""
    with PathManager.open(filename, "rb") as f:
        return torch.load(f)


@contextmanager
def mode(net: torch.nn.Module, training: bool) -> Iterator[torch.nn.Module]:
    """Temporarily switch to training/evaluation mode."""
    istrain = net.training
    try:
        net.train(training)
        yield net
    finally:
        net.train(istrain)


def _log_api_usage(identifier: str):
    """
    Internal function used to log the usage of different d2go components
    inside facebook's infra.
    """
    torch._C._log_api_usage_once("d2go." + identifier)


def _log_api_usage_on_main_process(identifier: str):
    """
    Log the usage of d2go API only on the main process.
    """
    if comm.is_main_process():
        _log_api_usage(identifier)
