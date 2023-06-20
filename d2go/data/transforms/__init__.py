#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved


# Populating registreis
from d2go.data.transforms import (  # noqa
    affine as _affine,
    auto_aug,
    blur as _blur,
    box_utils as _box_utils,
    color_yuv as _color_yuv,
    crop as _crop,
    d2_native as _d2_native,
)
# @fb-only: from d2go.data.transforms import fb as _fb  # isort:skip  # noqa 
