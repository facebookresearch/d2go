#!/usr/bin/env python3

import contextlib
import logging
import os

import mock
import yaml
from detectron2.config import CfgNode as _CfgNode
from fvcore.common.registry import Registry

logger = logging.getLogger(__name__)


def get_cfg_diff_table(cfg, original_cfg):
    # print out the differences
    from d2go.config import CfgNode
    def _find_all_keys(obj):
        assert isinstance(obj, CfgNode)
        ret = []
        for key in sorted(obj.keys()):
            value = obj[key]
            if isinstance(value, CfgNode):
                for sub_key in _find_all_keys(value):
                    ret.append("{}.{}".format(key, sub_key))
            else:
                ret.append(key)
        return ret

    def _get_value(obj, full_key):
        for k in full_key.split("."):
            obj = obj[k]
        return obj

    all_old_keys = _find_all_keys(original_cfg)
    all_new_keys = _find_all_keys(cfg)
    assert all_old_keys == all_new_keys

    diff_table = []
    for full_key in all_new_keys:
        old_value = _get_value(original_cfg, full_key)
        new_value = _get_value(cfg, full_key)
        if old_value != new_value:
            diff_table.append([full_key, old_value, new_value])

    from tabulate import tabulate
    table = tabulate(
        diff_table,
        tablefmt="pipe",
        headers=["config key", "old value", "new value"],
    )
    return table

def reroute_config_path(path: str) -> str:
    """
    Supporting rerouting the config files for convenience:
        d2go:// -> mobile-vision/d2go/...
        detectron2go:// -> mobile-vision/d2go/configs/...
        detectron2:// -> vision/fair/detectron2/configs/...
        flow:// -> fblearner/flow/projects/mobile_vision/detectron2go/...
        mv_experimental:// -> mobile-vision/experimental/...
            (see //mobile-vision/experimental:mv_experimental_d2go_yaml_files)

    Those config are considered as code, so they'll reflect your current checkout,
        try using canary if you have local changes.
    """

    import pkg_resources
    from d2go.utils.helper import get_dir_path

    if path.startswith("d2go://"):
        rel_path = path[len("d2go://") :]
        import d2go

        package_path = get_dir_path(d2go.__name__)
        return os.path.join(package_path, rel_path)
    elif path.startswith("detectron2go://"):
        rel_path = path[len("detectron2go://") :]
        import d2go

        package_path = get_dir_path(os.path.join(d2go.__name__, "configs"))
        return os.path.join(package_path, rel_path)
    elif path.startswith("detectron2://"):
        rel_path = path[len("detectron2://") :]
        config_in_resource = pkg_resources.resource_filename(
            "detectron2.tools.train_net", "configs/{}".format(rel_path)
        )
        return config_in_resource
    elif path.startswith("mv_experimental://"):
        rel_path = path[len("mv_experimental://") :]
        # pyre-fixme[21]: Could not find module `mv_experimental_d2go_yaml_files`.
        import mv_experimental_d2go_yaml_files

        package_path = get_dir_path(mv_experimental_d2go_yaml_files.__name__)
        return os.path.join(package_path, rel_path)
    elif path.startswith("flow://"):
        rel_path = path[len("flow://") :]
        # pyre-fixme[21]: Could not find module
        #  `fblearner.flow.projects.mobile_vision.detectron2go`.
        import fblearner.flow.projects.mobile_vision.detectron2go as d2go_flow

        # pyre-fixme[16]: Module `mobile_vision` has no attribute `detectron2go`.
        return pkg_resources.resource_filename(d2go_flow.__name__, rel_path)

    return path


class CfgNode(_CfgNode):
    def merge_from_file(self, cfg_filename: str, *args, **kwargs):
        cfg_filename = reroute_config_path(cfg_filename)
        with reroute_load_yaml_with_base():
            return super().merge_from_file(cfg_filename, *args, **kwargs)

    @staticmethod
    def load_yaml_with_base(filename: str, *args, **kwargs):
        with reroute_load_yaml_with_base():
            return _CfgNode.load_yaml_with_base(filename, *args, **kwargs)

    def merge_from_other_cfg(self, cfg_other):
        # NOTE: D24397488 changes default MODEL.FBNET_V2.ARCH_DEF from "" to [], change
        # the value to be able to load old full configs.
        # TODO: remove this by end of 2020.
        if cfg_other.get("MODEL", {}).get("FBNET_V2", {}).get("ARCH_DEF", None) == "":
            import logging
            logger = logging.getLogger(__name__)
            logger.warning("Default value for MODEL.FBNET_V2.ARCH_DEF has changed to []")
            cfg_other.MODEL.FBNET_V2.ARCH_DEF = []
        return super().merge_from_other_cfg(cfg_other)

    def __hash__(self):
        # dump follows alphabetical order, thus good for hash use
        return hash(self.dump())


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

    table = get_cfg_diff_table(cfg, original_cfg)
    logger.info("Auto-scaled the config according to the actual world size: \n" + table)
