#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved


# NOTE: making necessary imports to register with Registery
from . import backbone  # noqa
from . import meta_arch  # noqa
from . import modeldef  # noqa

# namespace forwarding
from .meta_arch.build import build_model

__all__ = [
    "build_model",
]
