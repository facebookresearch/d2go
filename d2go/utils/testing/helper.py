#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved


import importlib
import os
import socket
import uuid
from functools import wraps
from tempfile import TemporaryDirectory
from typing import Optional

import torch
from d2go.distributed import distributed_worker, DistributedParams


def get_resource_path(file: Optional[str] = None):
    path_list = [
        os.path.dirname(importlib.import_module("d2go.tests").__file__),
        "resources",
    ]
    if file is not None:
        path_list.append(file)

    return os.path.join(*path_list)


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


def enable_ddp_env(backend="gloo"):
    def _enable_ddp_env(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            def find_free_port() -> str:
                s = socket.socket()
                s.bind(("localhost", 0))  # Bind to a free port provided by the host.
                return str(s.getsockname()[1])

            os.environ["MASTER_ADDR"] = "localhost"
            os.environ["MASTER_PORT"] = find_free_port()

            return distributed_worker(
                main_func=func,
                args=args,
                kwargs=kwargs,
                backend=backend,
                init_method="file:///tmp/detectron2go_test_ddp_init_{}".format(
                    uuid.uuid4().hex
                ),
                dist_params=DistributedParams(
                    local_rank=0,
                    machine_rank=0,
                    global_rank=0,
                    num_processes_per_machine=1,
                    world_size=1,
                ),
                return_save_file=None,  # don't save file
            )

        return wrapper

    return _enable_ddp_env


def tempdir(func):
    """A decorator for creating a tempory directory that is cleaned up after function execution."""

    @wraps(func)
    def wrapper(self, *args, **kwargs):
        with TemporaryDirectory() as temp:
            return func(self, temp, *args, **kwargs)

    return wrapper
