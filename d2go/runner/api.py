#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

from typing import final

from d2go.config import CfgNode


class RunnerV2Mixin(object):
    """
    Interface for (V2) Runner:

        - `get_default_cfg` is not a runner method anymore.
    """

    @classmethod
    @final
    def get_default_cfg(cls) -> CfgNode:
        raise NotImplementedError("")
