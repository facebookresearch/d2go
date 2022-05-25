#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import os
from enum import Enum
from typing import Any, Dict, List

import pkg_resources
from mobile_cv.common.misc.oss_utils import fb_overwritable


@fb_overwritable()
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
    assert isinstance(path, str), path

    if path.startswith("d2go://"):
        rel_path = path[len("d2go://") :]
        config_in_resource = pkg_resources.resource_filename("d2go", rel_path)
        return config_in_resource
    elif path.startswith("detectron2go://"):
        rel_path = path[len("detectron2go://") :]
        config_in_resource = pkg_resources.resource_filename(
            "d2go", os.path.join("configs", rel_path)
        )
        return config_in_resource
    elif path.startswith("detectron2://"):
        rel_path = path[len("detectron2://") :]
        config_in_resource = pkg_resources.resource_filename(
            "detectron2.model_zoo", os.path.join("configs", rel_path)
        )
        return config_in_resource

    return path


def _flatten_config_dict(x, reorder, prefix):
    if not isinstance(x, dict):
        return {prefix: x}

    d = {}
    for k in sorted(x.keys()) if reorder else x.keys():
        v = x[k]
        new_key = f"{prefix}.{k}" if prefix else k
        d.update(_flatten_config_dict(v, reorder, new_key))
    return d


def flatten_config_dict(dic, reorder=True):
    """
    Flattens nested dict into single layer dict, for example:
        flatten_config_dict({
            "MODEL": {
                "FBNET_V2": {
                    "ARCH_DEF": "val0",
                    "ARCH": "val1:,
                },
            }
        })
        => {"MODEL.FBNET_V2.ARCH_DEF": "val0", "MODEL.FBNET_V2.ARCH": "val1"}

    Args:
        dic (dict or CfgNode): a nested dict whose keys are strings.
        reorder (bool): if True, the returned dict will be sorted according to the keys;
            otherwise original order will be preserved.

    Returns:
        dic: a single-layer dict
    """
    return _flatten_config_dict(dic, reorder=reorder, prefix="")


def config_dict_to_list_str(config_dict: Dict) -> List[str]:
    """Creates a list of str given configuration dict

    This can be useful to generate pretraining or overwrite opts
    in D2Go when a user has config_dict
    """
    d = flatten_config_dict(config_dict)
    str_list = []
    for k, v in d.items():
        str_list.append(k)
        str_list.append(str(v))
    return str_list


def get_from_flattened_config_dict(dic, flattened_key, default=None):
    """
    Reads out a value from the nested config dict using flattened config key (i.e. all
    keys from each level put together with "." separator), the default value is returned
    if the flattened key doesn't exist.
    e.g. if the config dict is
        MODEL:
            TEST:
            SCORE_THRESHOLD: 0.7
        Then to access the value of SCORE_THRESHOLD, this API should be called

        >> score_threshold = get_from_flattened_config_dict(cfg, "MODEL.TEST.SCORE_THRESHOLD")
    """
    for k in flattened_key.split("."):
        if k not in dic:
            return default
        dic = dic[k]
    return dic


def get_cfg_diff_table(cfg, original_cfg):
    """
    Print the different of two config dicts side-by-side in a table
    """

    all_old_keys = list(flatten_config_dict(original_cfg, reorder=True).keys())
    all_new_keys = list(flatten_config_dict(cfg, reorder=True).keys())
    assert all_old_keys == all_new_keys

    diff_table = []
    for full_key in all_new_keys:
        old_value = get_from_flattened_config_dict(original_cfg, full_key)
        new_value = get_from_flattened_config_dict(cfg, full_key)
        if old_value != new_value:
            diff_table.append([full_key, old_value, new_value])

    from tabulate import tabulate

    table = tabulate(
        diff_table,
        tablefmt="pipe",
        headers=["config key", "old value", "new value"],
    )
    return table


def get_diff_cfg(old_cfg, new_cfg):
    """
    outputs a CfgNode containing keys, values appearing in new_cfg and not in old_cfg.
    If `new_allowed` is not set, then new keys will throw a KeyError
    old_cfg: CfgNode, the original config, usually the dafulat
    new_cfg: CfgNode, the full config being passed by the user

    if new allowed is not set on new_cfg, key error is raised
    returns: CfgNode, a config containing only key, value changes between old_cfg and new_cfg

    example:
        Cfg1:
            SYSTEM:
                NUM_GPUS: 2
            TRAIN:
                SCALES: (1, 2)
            DATASETS:
                train_2017:
                    17: 1
                    18: 1
        Cfg2:
            SYSTEM:
                NUM_GPUS: 2
            TRAIN:
                SCALES: (4, 5, 8)
            DATASETS:
                train_2017:
                    17: 1
                    18: 1

        get_diff_cfg(Cfg1, Cfg2) gives:
            TRAIN:
                SCALES: (8, 16, 32)
    """

    def get_diff_cfg_rec(old_cfg, new_cfg, out):
        for key in new_cfg.keys():
            if key not in old_cfg.keys() and old_cfg.is_new_allowed():
                out[key] = new_cfg[key]
            elif old_cfg[key] != new_cfg[key]:
                if type(new_cfg[key]) is type(out):
                    out[key] = out.__class__()
                    out[key] = get_diff_cfg_rec(old_cfg[key], new_cfg[key], out[key])
                else:

                    out[key] = new_cfg[key]
        return out

    out = new_cfg.__class__()
    return get_diff_cfg_rec(old_cfg, new_cfg, out)


def namedtuple_to_dict(obj: Any):
    """Convert NamedTuple or dataclass to dict so it can be used as config"""
    res = {}
    for k, v in obj.__dict__.items():
        if isinstance(v, Enum):
            # in case of enum, serialize the enum value
            res[k] = v.value
        else:
            res[k] = v
    return res
