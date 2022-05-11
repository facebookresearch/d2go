#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

from mobile_cv.common.misc.py import dynamic_import


def fb_overwritable():
    """Decorator on function that has alternative internal implementation"""
    try:
        import d2go.utils.fb.open_source_canary  # noqa

        is_oss = False
    except ImportError:
        is_oss = True

    def deco(oss_func):
        if is_oss:
            return oss_func
        else:
            oss_module = oss_func.__module__
            fb_module = oss_module + "_fb"  # xxx.py -> xxx_fb.py
            fb_func = dynamic_import("{}.{}".format(fb_module, oss_func.__name__))
            return fb_func

    return deco
