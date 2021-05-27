#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

from typing import Dict, List


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
