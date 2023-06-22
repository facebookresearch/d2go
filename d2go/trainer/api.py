#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved


"""
Trainer APIs on which D2Go's binary can build on top.
"""

from dataclasses import dataclass
from typing import Dict, Optional

from d2go.evaluation.api import AccuracyDict, MetricsDict


@dataclass
class TrainNetOutput:
    accuracy: AccuracyDict[float]
    metrics: MetricsDict[float]
    model_configs: Dict[str, str]
    # TODO (T127368603): decide if `tensorboard_log_dir` should be part of output
    tensorboard_log_dir: Optional[str] = None


@dataclass
class TestNetOutput:
    accuracy: AccuracyDict[float]
    metrics: MetricsDict[float]
    # TODO (T127368603): decide if `tensorboard_log_dir` should be part of output
    tensorboard_log_dir: Optional[str] = None


@dataclass
class EvaluatorOutput:
    accuracy: AccuracyDict[float]
    metrics: MetricsDict[float]


def do_train():
    pass


def do_test():
    pass
