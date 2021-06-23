#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

from d2go.utils.misc import fb_overwritable


@fb_overwritable()
def get_launch_environment():
    return "local"
