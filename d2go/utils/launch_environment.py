#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

from mobile_cv.common.misc.oss_utils import fb_overwritable


@fb_overwritable()
def get_model_zoo_storage_prefix() -> str:
    return "https://mobile-cv.s3-us-west-2.amazonaws.com/d2go/models/"


@fb_overwritable()
def get_launch_environment():
    return "local"


MODEL_ZOO_STORAGE_PREFIX = get_model_zoo_storage_prefix()
