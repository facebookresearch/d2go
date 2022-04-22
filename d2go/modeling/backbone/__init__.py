#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

# @fb-only: from . import fb  # noqa 
from . import fbnet_v2


# Explicitly expose all registry-based modules
__all__ = [
    "fbnet_v2",
    # @fb-only: "fb", 
]
