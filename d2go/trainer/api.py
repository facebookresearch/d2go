#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved


"""
Trainer APIs on which D2Go's binary can build on top.
"""

from dataclasses import dataclass
from typing import Any, Dict, Optional

from d2go.evaluation.api import AccuracyDict, MetricsDict

# TODO (T127368935) Split to TrainNetOutput and TestNetOutput
@dataclass
class TrainNetOutput:
    accuracy: AccuracyDict[Any]
    metrics: MetricsDict[Any]
    # Optional, because we use None to distinguish "not used" from
    # empty model configs. With T127368935, this should be reverted to dict.
    model_configs: Optional[Dict[str, str]]
    # TODO (T127368603): decide if `tensorboard_log_dir` should be part of output
    tensorboard_log_dir: Optional[str] = None


def do_train():
    pass


def do_test():
    pass
