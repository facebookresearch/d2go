#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved


import logging
import os
from functools import lru_cache

from mobile_cv.common.misc.oss_utils import fb_overwritable


@fb_overwritable()
def get_tensorboard_log_dir(output_dir):
    return output_dir
