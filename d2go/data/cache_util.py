#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import os

from mobile_cv.common.misc.oss_utils import fb_overwritable


@fb_overwritable()
def _cache_json_file(json_file):
    # TODO: entirely rely on PathManager for caching
    json_file = os.fspath(json_file)
    return json_file
