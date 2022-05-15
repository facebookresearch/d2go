#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved


import argparse
import logging
import os
import time
from typing import Optional

import detectron2.utils.comm as comm
import torch
from d2go.config import (
    auto_scale_world_size,
    CfgNode,
    reroute_config_path,
    temp_defrost,
)
from d2go.config.utils import get_diff_cfg
from d2go.distributed import get_local_rank, get_num_processes_per_machine
from d2go.runner import BaseRunner, create_runner
from d2go.utils.helper import run_once
from d2go.utils.launch_environment import get_launch_environment
from detectron2.utils.collect_env import collect_env_info
from detectron2.utils.file_io import PathManager
from detectron2.utils.logger import setup_logger as _setup_logger
from detectron2.utils.serialize import PicklableWrapper
from mobile_cv.common.misc.py import FolderLock, MultiprocessingPdb, post_mortem_if_fail

logger = logging.getLogger(__name__)


def basic_argument_parser(
    distributed=True,
    requires_config_file=True,
    requires_output_dir=True,
):
    """Basic cli tool parser for Detectron2Go binaries"""
    parser = argparse.ArgumentParser(description="PyTorch Object Detection Training")
    parser.add_argument(
        "--runner",
        type=str,
        default="d2go.runner.GeneralizedRCNNRunner",
        help="Full class name, i.e. (package.)module.class",
    )
    parser.add_argument(
        "--config-file",
        help="path to config file",
        default="",
        required=requires_config_file,
        metavar="FILE",
    )
    parser.add_argument(
        "--output-dir",
        help="When given, this will override the OUTPUT_DIR in the config-file",
        required=requires_output_dir,
        default=None,
        type=str,
    )
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )

    if distributed:
        parser.add_argument(
            "--num-processes", type=int, default=1, help="number of gpus per machine"
        )
        parser.add_argument("--num-machines", type=int, default=1)
        parser.add_argument(
            "--machine-rank",
            type=int,
            default=0,
            help="the rank of this machine (unique per machine)",
        )
        parser.add_argument(
            "--dist-url", default="file:///tmp/d2go_dist_file_{}".format(time.time())
        )
        parser.add_argument("--dist-backend", type=str, default="NCCL")

    if not requires_config_file:
        # NOTE if not passing yaml file, user should explicitly set the
        # following args, and use `opts` for non-common usecase.
        parser.add_argument(
            "--datasets",
            type=str,
            nargs="+",
            required=True,
            help="cfg.DATASETS.TEST",
        )
        parser.add_argument(
            "--min_size",
            type=int,
            required=True,
            help="cfg.INPUT.MIN_SIZE_TEST",
        )
        parser.add_argument(
            "--max_size",
            type=int,
            required=True,
            help="cfg.INPUT.MAX_SIZE_TEST",
        )
        return parser

    return parser


def create_cfg_from_cli_args(args, default_cfg):
    """
    Instead of loading from defaults.py, this binary only includes necessary
    configs building from scratch, and overrides them from args. There're two
    levels of config:
        _C: the config system used by this binary, which is a sub-set of training
            config, override by configurable_cfg. It can also be override by
            args.opts for convinience.
        configurable_cfg: common configs that user should explicitly specify
            in the args.
    """

    _C = CfgNode()
    _C.INPUT = default_cfg.INPUT
    _C.DATASETS = default_cfg.DATASETS
    _C.DATALOADER = default_cfg.DATALOADER
    _C.TEST = default_cfg.TEST
    if hasattr(default_cfg, "D2GO_DATA"):
        _C.D2GO_DATA = default_cfg.D2GO_DATA
    if hasattr(default_cfg, "TENSORBOARD"):
        _C.TENSORBOARD = default_cfg.TENSORBOARD

    # NOTE configs below might not be necessary, but must add to make code work
    _C.MODEL = CfgNode()
    _C.MODEL.META_ARCHITECTURE = default_cfg.MODEL.META_ARCHITECTURE
    _C.MODEL.MASK_ON = default_cfg.MODEL.MASK_ON
    _C.MODEL.KEYPOINT_ON = default_cfg.MODEL.KEYPOINT_ON
    _C.MODEL.LOAD_PROPOSALS = default_cfg.MODEL.LOAD_PROPOSALS
    assert _C.MODEL.LOAD_PROPOSALS is False, "caffe2 model doesn't support"

    _C.OUTPUT_DIR = args.output_dir

    configurable_cfg = [
        "DATASETS.TEST",
        args.datasets,
        "INPUT.MIN_SIZE_TEST",
        args.min_size,
        "INPUT.MAX_SIZE_TEST",
        args.max_size,
    ]

    cfg = _C.clone()
    cfg.merge_from_list(configurable_cfg)
    cfg.merge_from_list(args.opts)

    return cfg


def prepare_for_launch(args):
    """
    Load config, figure out working directory, create runner.
        - when args.config_file is empty, returned cfg will be the default one
        - returned output_dir will always be non empty, args.output_dir has higher
            priority than cfg.OUTPUT_DIR.
    """
    logger.info(args)
    runner = create_runner(args.runner)

    cfg = runner.get_default_cfg()

    if args.config_file:
        with PathManager.open(reroute_config_path(args.config_file), "r") as f:
            print("Loaded config file {}:\n{}".format(args.config_file, f.read()))
        cfg.merge_from_file(args.config_file)
        cfg.merge_from_list(args.opts)
    else:
        cfg = create_cfg_from_cli_args(args, default_cfg=cfg)
    cfg.freeze()

    assert args.output_dir or args.config_file
    output_dir = args.output_dir or cfg.OUTPUT_DIR
    return cfg, output_dir, runner


def maybe_override_output_dir(cfg: CfgNode, output_dir: str):
    if cfg.OUTPUT_DIR != output_dir:
        with temp_defrost(cfg):
            logger.warning(
                "Override cfg.OUTPUT_DIR ({}) to be the same as output_dir {}".format(
                    cfg.OUTPUT_DIR, output_dir
                )
            )
            cfg.OUTPUT_DIR = output_dir


def setup_after_launch(
    cfg: CfgNode,
    output_dir: str,
    runner: Optional[BaseRunner] = None,
    _scale_world_size: bool = True,  # HACK: temporarily allow lightning_train_net to by pass this.
):
    """
    Binary-level setup after entering DDP, including
        - creating working directory
        - setting up logger
        - logging environment
        - printing and dumping config
        - (optional) initializing runner
    """

    create_dir_on_global_main_process(output_dir)
    setup_loggers(output_dir)
    log_system_info()

    cfg.freeze()
    maybe_override_output_dir(cfg, output_dir)
    logger.info("Running with full config:\n{}".format(cfg))
    dump_cfg(cfg, os.path.join(output_dir, "config.yaml"))

    if runner:
        logger.info("Initializing runner ...")
        runner = initialize_runner(runner, cfg)
        logger.info("Running with runner: {}".format(runner))

    # save the diff config
    if runner:
        default_cfg = runner.get_default_cfg()
        dump_cfg(
            get_diff_cfg(default_cfg, cfg),
            os.path.join(output_dir, "diff_config.yaml"),
        )
    else:
        # TODO: support getting default_cfg without runner.
        pass

    # scale the config after dumping so that dumped config files keep original world size
    if _scale_world_size:
        auto_scale_world_size(cfg, new_world_size=comm.get_world_size())


def setup_logger(
    module_name: str,
    output_dir: str,
    abbrev_name: Optional[str] = None,
    color: Optional[bool] = None,
) -> logging.Logger:
    if not color:
        color = get_launch_environment() == "local"
    if not abbrev_name:
        abbrev_name = module_name

    logger = _setup_logger(
        output_dir,
        distributed_rank=comm.get_rank(),
        color=color,
        name=module_name,
        abbrev_name=abbrev_name,
    )

    # NOTE: the root logger might has been configured by other applications,
    # since this already sub-top level, just don't propagate to root.
    logger.propagate = False

    return logger


@run_once()
def setup_loggers(output_dir):
    setup_logger("detectron2", output_dir, abbrev_name="d2")
    setup_logger("fvcore", output_dir)
    setup_logger("d2go", output_dir)
    setup_logger("mobile_cv", output_dir)

    # NOTE: all above loggers have FileHandler pointing to the same file as d2_logger.
    # Those files are opened upon creation, but it seems fine in 'a' mode.


def log_system_info():
    num_processes = get_num_processes_per_machine()
    logger.info(
        "Using {} processes per machine. Rank of current process: {}".format(
            num_processes, comm.get_rank()
        )
    )
    wf_id = os.getenv("WORKFLOW_RUN_ID", None)
    if wf_id is not None:
        logger.info("FBLearner Flow Run ID: {}".format(wf_id))
    logger.info("Environment info:\n" + collect_env_info())
    try:
        from detectron2.fb.utils import print_fbcode_info

        print_fbcode_info()
    except ImportError:
        pass


def dump_cfg(cfg: CfgNode, path: str) -> None:
    if comm.is_main_process():
        with PathManager.open(path, "w") as f:
            f.write(cfg.dump())
        logger.info("Full config saved to {}".format(path))


def create_dir_on_global_main_process(path: str) -> None:
    if comm.get_rank() == 0 and path:
        PathManager.mkdirs(path)
    # Add a barrier to make sure the existance of the dir for non-master process
    comm.synchronize()


def initialize_runner(runner: BaseRunner, cfg: CfgNode) -> BaseRunner:
    assert runner is not None, "now always requires a runner instance"
    runner._initialize(cfg)
    return runner


def caffe2_global_init(logging_print_net_summary=0, num_threads=None):
    if num_threads is None:
        if get_num_processes_per_machine() > 1:
            # by default use single thread when DDP with multiple processes
            num_threads = 1
        else:
            # GlobalInit will clean PyTorch's num_threads and set it to 1,
            # thus keep PyTorch's default value to make it truly default.
            num_threads = torch.get_num_threads()

    if not get_local_rank() == 0:
        logging_print_net_summary = 0  # only enable for local main process

    from caffe2.python import workspace

    workspace.GlobalInit(
        [
            "caffe2",
            "--caffe2_log_level=2",
            "--caffe2_logging_print_net_summary={}".format(logging_print_net_summary),
            "--caffe2_omp_num_threads={}".format(num_threads),
            "--caffe2_mkl_num_threads={}".format(num_threads),
        ]
    )
    logger.info("Using {} threads after GlobalInit".format(torch.get_num_threads()))


def post_mortem_if_fail_for_main(main_func):
    def new_main_func(cfg, output_dir, *args, **kwargs):
        pdb_ = (
            MultiprocessingPdb(FolderLock(output_dir))
            if comm.get_world_size() > 1
            else None  # fallback to use normal pdb for single process
        )
        return post_mortem_if_fail(pdb_)(main_func)(cfg, output_dir, *args, **kwargs)

    return PicklableWrapper(new_main_func)
