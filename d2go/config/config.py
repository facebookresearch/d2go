#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved


import contextlib
import logging

import mock
import yaml
from d2go.utils.helper import reroute_config_path
from detectron2.config import CfgNode as _CfgNode
from fvcore.common.registry import Registry

logger = logging.getLogger(__name__)


class CfgNode(_CfgNode):
    @classmethod
    def cast_from_other_class(cls, other_cfg):
        """Cast an instance of other CfgNode to D2Go's CfgNode (or its subclass)"""
        new_cfg = CfgNode(other_cfg)
        # copy all fields inside __dict__, this will preserve fields like __deprecated_keys__
        for k, v in other_cfg.__dict__.items():
            new_cfg.__dict__[k] = v
        return new_cfg

    def merge_from_file(self, cfg_filename: str, *args, **kwargs):
        cfg_filename = reroute_config_path(cfg_filename)
        with reroute_load_yaml_with_base():
            return super().merge_from_file(cfg_filename, *args, **kwargs)

    @staticmethod
    def load_yaml_with_base(filename: str, *args, **kwargs):
        with reroute_load_yaml_with_base():
            return _CfgNode.load_yaml_with_base(filename, *args, **kwargs)

    def __hash__(self):
        # dump follows alphabetical order, thus good for hash use
        return hash(self.dump())


@contextlib.contextmanager
def temp_defrost(cfg):
    is_frozen = cfg.is_frozen()
    if is_frozen:
        cfg.defrost()
    yield cfg
    if is_frozen:
        cfg.freeze()


@contextlib.contextmanager
def reroute_load_yaml_with_base():
    BASE_KEY = "_BASE_"
    _safe_load = yaml.safe_load
    _unsafe_load = yaml.unsafe_load

    def mock_safe_load(f):
        cfg = _safe_load(f)
        if BASE_KEY in cfg:
            cfg[BASE_KEY] = reroute_config_path(cfg[BASE_KEY])
        return cfg

    def mock_unsafe_load(f):
        cfg = _unsafe_load(f)
        if BASE_KEY in cfg:
            cfg[BASE_KEY] = reroute_config_path(cfg[BASE_KEY])
        return cfg

    with mock.patch("yaml.safe_load", side_effect=mock_safe_load):
        with mock.patch("yaml.unsafe_load", side_effect=mock_unsafe_load):
            yield


CONFIG_SCALING_METHOD_REGISTRY = Registry("CONFIG_SCALING_METHOD")


def auto_scale_world_size(cfg, new_world_size):
    """
    Usually the config file is written for a specific number of devices, this method
    scales the config (in-place!) according to the actual world size using the
    pre-registered scaling methods specified as cfg.SOLVER.AUTO_SCALING_METHODS.

    Note for registering scaling methods:
        - The method will only be called when scaling is needed. It won't be called
            if SOLVER.REFERENCE_WORLD_SIZE is 0 or equal to target world size. Thus
            cfg.SOLVER.REFERENCE_WORLD_SIZE will always be positive.
        - The method updates cfg in-place, no return is required.
        - No need for changing SOLVER.REFERENCE_WORLD_SIZE.

    Args:
        cfg (CfgNode): original config which contains SOLVER.REFERENCE_WORLD_SIZE and
            SOLVER.AUTO_SCALING_METHODS.
        new_world_size: the target world size
    """

    old_world_size = cfg.SOLVER.REFERENCE_WORLD_SIZE
    if old_world_size == 0 or old_world_size == new_world_size:
        return cfg

    original_cfg = cfg.clone()
    frozen = original_cfg.is_frozen()
    cfg.defrost()

    assert len(cfg.SOLVER.AUTO_SCALING_METHODS) > 0, cfg.SOLVER.AUTO_SCALING_METHODS
    for scaling_method in cfg.SOLVER.AUTO_SCALING_METHODS:
        logger.info("Applying auto scaling method: {}".format(scaling_method))
        CONFIG_SCALING_METHOD_REGISTRY.get(scaling_method)(cfg, new_world_size)

    assert (
        cfg.SOLVER.REFERENCE_WORLD_SIZE == cfg.SOLVER.REFERENCE_WORLD_SIZE
    ), "Runner's scale_world_size shouldn't change SOLVER.REFERENCE_WORLD_SIZE"
    cfg.SOLVER.REFERENCE_WORLD_SIZE = new_world_size

    if frozen:
        cfg.freeze()

    from d2go.config.utils import get_cfg_diff_table

    table = get_cfg_diff_table(cfg, original_cfg)
    logger.info("Auto-scaled the config according to the actual world size: \n" + table)
