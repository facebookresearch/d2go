#!/usr/bin/env python3

import logging
import os

from tabulate import tabulate

from .tensorboard_log_util import get_tensorboard_log_dir

logger = logging.getLogger(__name__)


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
