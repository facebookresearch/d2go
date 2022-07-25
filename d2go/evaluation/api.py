#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

from typing import Dict, TypeVar, Union

T = TypeVar("T")

# "accuracy" in D2Go is defined by a 4-level dictionary in the order of:
# model_tag -> dataset -> task -> metrics
AccuracyDict = Dict[str, Dict[str, Dict[str, Dict[str, T]]]]

# "metric" in D2Go is a nested dictionary, which may have arbitrary levels.
MetricsDict = Union[Dict[str, "MetricsDict"], T]
