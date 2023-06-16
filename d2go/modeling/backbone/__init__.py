#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

from d2go.modeling.backbone import fbnet_v2 as _fbnet_v2  # noqa
# @fb-only: from d2go.modeling.backbone import fb as _fb  # isort:skip  # noqa 


# Explicitly expose all registry-based modules
__all__ = [
    "fbnet_v2",
    # @fb-only: "fb", 
]
