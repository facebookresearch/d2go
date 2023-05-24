#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved


# import all modules to make sure Registry works
# @fb-only: from d2go.data.transforms import fb  # isort:skip  # noqa 
from d2go.data.transforms import (  # noqa
    affine,
    auto_aug,
    blur,
    box_utils,
    color_yuv,
    crop,
    d2_native,
)
