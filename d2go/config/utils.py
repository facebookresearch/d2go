#!/usr/bin/env python3

import contextlib
import copy
import json
import shlex
from typing import Dict, List


@contextlib.contextmanager
def temp_defrost(cfg):
    is_frozen = cfg.is_frozen()
    if is_frozen:
        cfg.defrost()
    yield cfg
    if is_frozen:
        cfg.freeze()


def str_wrap_fbnet_arch_def(d: Dict, inplace=False) -> Dict:
    """Replaces MODEL.FBNET_V2.ARCH_DEF with wrapped json string

    Searches the input dict to see if it contains MODEL.FBNET_V2.ARCH_DEF
    and replaces the value with a wrapped json string

    The json string is created because FBNet builder runs json.loads on the
    archdef. The json string needs to be wrapped in another string because
    CfgNode runs literal_eval in order to check whether it should continue
    to create CfgNodes if the value is a dict.

        arch_def = {...}                               # {...}
        arch_def = json.dumps(arch_def)                # '{...}'
        arch_def = strwrap(arch_def)                   # '"{...}"'

        CfgNode(arch_def) => literal_eval(arch_def)    # '{...}'
        FBNetBuilder(arch_def) => json.loads(arch_def) # {...}

    Example:
        config = {"MODEL": {"FBNET_V2": {"ARCH_DEF": [1, 1, 1]}}}
        str_wrap_fbnet_arch_def(config)
          => {"MODEL": {"FBNET_V2": {"ARCH_DEF": '''"[1, 1, 1]"'''}}}
    """
    if not inplace:
        d = copy.deepcopy(d)

    try:
        archdef = d["MODEL"]["FBNET_V2"]["ARCH_DEF"]
        # MODEL.FBNET_V2.ARCH_DEF needs to be json str
        archdef = json.dumps(archdef)
        # CfgNode runs literal_eval when merging so wrap around str
        archdef = shlex.quote(archdef)
        d["MODEL"]["FBNET_V2"]["ARCH_DEF"] = archdef
    except KeyError:
        pass

    return d


def flatten_config_dict(x, prefix=""):
    """Flattens config dict into single layer dict

    Example:
        flatten_config_dict({
            MODEL: {
                FBNET_V2: {
                    ARCH_DEF: "val0"
                }
            }
        })
        => {"MODEL.FBNET_V2.ARCH_DEF": "val0"}
    """
    if not isinstance(x, dict):
        return {prefix: x}

    d = {}
    for k, v in x.items():
        new_key = f"{prefix}.{k}" if prefix else k
        d.update(flatten_config_dict(v, new_key))
    return d


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
