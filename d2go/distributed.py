#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved


"""
Similar to detectron2.engine.launch, may support a few more things:
    - support for get_local_rank.
    - support other backends like GLOO.
"""

import logging
import tempfile

import detectron2.utils.comm as comm
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from d2go.config import CfgNode, temp_defrost
from d2go.utils.launch_environment import get_launch_environment


logger = logging.getLogger(__name__)


_LOCAL_RANK = 0
_NUM_PROCESSES_PER_MACHINE = 1


def _set_local_rank(local_rank):
    global _LOCAL_RANK
    _LOCAL_RANK = local_rank


def _set_num_processes_per_machine(num_processes):
    global _NUM_PROCESSES_PER_MACHINE
    _NUM_PROCESSES_PER_MACHINE = num_processes


def get_local_rank():
    return _LOCAL_RANK


def get_num_processes_per_machine():
    return _NUM_PROCESSES_PER_MACHINE


def launch(
    main_func,
    num_processes_per_machine,
    num_machines=1,
    machine_rank=0,
    dist_url=None,
    backend="NCCL",
    always_spawn=False,
    args=(),
):
    logger.info(
        f"Launch with num_processes_per_machine: {num_processes_per_machine},"
        f" num_machines: {num_machines}, machine_rank: {machine_rank},"
        f" dist_url: {dist_url}, backend: {backend}."
    )

    if get_launch_environment() == "local" and not torch.cuda.is_available():
        assert len(args) > 0, args
        cfg = args[0]
        assert isinstance(cfg, CfgNode)
        if cfg.MODEL.DEVICE == "cuda":
            logger.warning(
                "Detected that CUDA is not available on this machine, set MODEL.DEVICE"
                " to cpu and backend to GLOO"
            )
            with temp_defrost(cfg):
                cfg.MODEL.DEVICE = "cpu"
            backend = "GLOO"

    if backend == "NCCL":
        assert (
            num_processes_per_machine <= torch.cuda.device_count()
        ), "num_processes_per_machine is greater than device count: {} vs {}".format(
            num_processes_per_machine, torch.cuda.device_count()
        )

    world_size = num_machines * num_processes_per_machine
    if world_size > 1 or always_spawn:
        # https://github.com/pytorch/pytorch/pull/14391
        # TODO prctl in spawned processes
        prefix = f"detectron2go_{main_func.__module__}.{main_func.__name__}_return"
        with tempfile.NamedTemporaryFile(prefix=prefix, suffix=".pth") as f:
            return_file = f.name
            mp.spawn(
                _distributed_worker,
                nprocs=num_processes_per_machine,
                args=(
                    main_func,
                    world_size,
                    num_processes_per_machine,
                    machine_rank,
                    dist_url,
                    backend,
                    return_file,
                    args,
                ),
                daemon=False,
            )
            if machine_rank == 0:
                return torch.load(return_file)
    else:
        return main_func(*args)


def _distributed_worker(
    local_rank,
    main_func,
    world_size,
    num_processes_per_machine,
    machine_rank,
    dist_url,
    backend,
    return_file,
    args,
):
    assert backend in ["NCCL", "GLOO"]
    _set_local_rank(local_rank)
    _set_num_processes_per_machine(num_processes_per_machine)

    # NOTE: this is wrong if using different number of processes across machine
    global_rank = machine_rank * num_processes_per_machine + local_rank
    try:
        dist.init_process_group(
            backend=backend,
            init_method=dist_url,
            world_size=world_size,
            rank=global_rank,
        )
    except Exception as e:
        logger.error("Process group URL: {}".format(dist_url))
        raise e
    # synchronize is needed here to prevent a possible timeout after calling
    # init_process_group
    # See: https://github.com/facebookresearch/maskrcnn-benchmark/issues/172
    comm.synchronize()
    if backend in ["NCCL"]:
        torch.cuda.set_device(local_rank)

    # Setup the local process group (which contains ranks within the same machine)
    assert comm._LOCAL_PROCESS_GROUP is None
    num_machines = world_size // num_processes_per_machine
    for i in range(num_machines):
        ranks_on_i = list(
            range(i * num_processes_per_machine, (i + 1) * num_processes_per_machine)
        )
        pg = dist.new_group(ranks_on_i)
        if i == machine_rank:
            comm._LOCAL_PROCESS_GROUP = pg

    ret = main_func(*args)
    if global_rank == 0:
        logger.info(
            "Save {}.{} return to: {}".format(
                main_func.__module__, main_func.__name__, return_file
            )
        )
        torch.save(ret, return_file)
