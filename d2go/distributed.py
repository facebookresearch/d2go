#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved


"""
Extend the mobile_cv.torch.utils_pytorch.distributed_helper to add D2/D2Go specific
features, functions in this module share the same signatures as the ones from mobile_cv.
"""

import logging
from datetime import timedelta
from typing import Any, Callable, Dict, Optional, Tuple

import detectron2.utils.comm as d2_comm
import mobile_cv.torch.utils_pytorch.comm as mcv_comm
import torch
from d2go.config import CfgNode, temp_defrost
from d2go.utils.launch_environment import get_launch_environment
from mobile_cv.torch.utils_pytorch.distributed_helper import (
    DEFAULT_TIMEOUT,
    DistributedParams,
    enable_dist_process_groups,
    launch as _launch,
    save_return_deco,
)

logger = logging.getLogger(__name__)


# BC-compatible
def get_local_rank():
    return mcv_comm.get_local_rank()


# BC-compatible
def get_num_processes_per_machine():
    return mcv_comm.get_local_size()


# Modify mobile_cv's `default_distributed_worker` to also setup D2's comm module
def distributed_worker(
    main_func: Callable,
    args: Tuple[Any, ...],
    kwargs: Dict[str, Any],
    backend: str,
    init_method: Optional[str] = None,
    dist_params: Optional[DistributedParams] = None,
    return_save_file: Optional[str] = None,
    timeout: timedelta = DEFAULT_TIMEOUT,
):
    dist_params = dist_params or DistributedParams.from_environ()
    with enable_dist_process_groups(backend, init_method, dist_params, timeout):
        d2_comm._LOCAL_PROCESS_GROUP = mcv_comm._LOCAL_PROCESS_GROUP
        # Now the D2's comm module should be fully functional
        deco = save_return_deco(return_save_file, dist_params.global_rank)
        return deco(main_func)(*args, **kwargs)


def launch(
    main_func: Callable,
    num_processes_per_machine: int,
    num_machines: int = 1,
    machine_rank: int = 0,
    dist_url: Optional[str] = None,
    backend: str = "NCCL",
    always_spawn: bool = False,
    launch_method: str = "multiprocessing",
    timeout: timedelta = DEFAULT_TIMEOUT,
    args: Tuple[Any, ...] = (),
    kwargs: Dict[str, Any] = None,
) -> Dict[int, Any]:
    """
    D2Go's specialized launch method, it does a few more things on top of mcv's launch:
        - Automatically convert GPU to CPU if CUDA is not available.
        - Add D2Go-specific initialziation in the _distributed_worker.
    """

    if get_launch_environment() == "local" and not torch.cuda.is_available():
        assert len(args) > 0, args
        cfg = args[0]
        if isinstance(cfg, CfgNode) and cfg.MODEL.DEVICE == "cuda":
            logger.warning(
                "Detected that CUDA is not available on this machine, set MODEL.DEVICE"
                " to cpu and backend to GLOO"
            )
            with temp_defrost(cfg):
                cfg.MODEL.DEVICE = "cpu"
        backend = "GLOO"

    return _launch(
        main_func=main_func,
        num_processes_per_machine=num_processes_per_machine,
        num_machines=num_machines,
        machine_rank=machine_rank,
        dist_url=dist_url,
        backend=backend,
        always_spawn=always_spawn,
        launch_method=launch_method,
        timeout=timeout,
        args=args,
        kwargs=kwargs,
        _distributed_worker=distributed_worker,
    )
