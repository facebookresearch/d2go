#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved


from functools import wraps
from tempfile import TemporaryDirectory

import torch
import torch.distributed as dist


def skip_if_no_gpu(func):
    """Decorator that can be used to skip GPU tests on non-GPU machines."""
    func.skip_if_no_gpu = True

    @wraps(func)
    def wrapper(*args, **kwargs):
        if not torch.cuda.is_available():
            return
        if torch.cuda.device_count() <= 0:
            return

        return func(*args, **kwargs)

    return wrapper


def enable_ddp_env(func):
    @wraps(func)
    def wrapper(*args, **kwargs):

        dist.init_process_group(
            "gloo",
            rank=0,
            world_size=1,
            init_method="file:///tmp/detectron2go_test_ddp_init",
        )
        ret = func(*args, **kwargs)
        dist.destroy_process_group()
        return ret

    return wrapper


def tempdir(func):
    """ A decorator for creating a tempory directory that is cleaned up after function execution. """

    @wraps(func)
    def wrapper(self, *args, **kwargs):
        with TemporaryDirectory() as temp:
            return func(self, temp, *args, **kwargs)

    return wrapper
