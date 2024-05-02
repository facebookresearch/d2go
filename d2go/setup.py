#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved


import argparse
import logging
import os
import time
from typing import Callable, List, Optional, Tuple, Type, TypeVar, Union

import detectron2.utils.comm as comm
import torch
from d2go.config import (
    auto_scale_world_size,
    CfgNode,
    load_full_config_from_file,
    reroute_config_path,
    temp_defrost,
)
from d2go.config.utils import get_diff_cfg
from d2go.distributed import (
    D2GoSharedContext,
    get_local_rank,
    get_num_processes_per_machine,
)
from d2go.runner import import_runner
from d2go.runner.api import RunnerV2Mixin
from d2go.runner.default_runner import BaseRunner
from d2go.runner.lightning_task import DefaultTask
from d2go.utils.helper import run_once
from d2go.utils.launch_environment import get_launch_environment
from d2go.utils.logging import initialize_logging, replace_print_with_logging
from detectron2.utils.collect_env import collect_env_info
from detectron2.utils.env import seed_all_rng
from detectron2.utils.file_io import PathManager
from detectron2.utils.logger import setup_logger as _setup_logger
from detectron2.utils.serialize import PicklableWrapper
from mobile_cv.common.misc.py import FolderLock, MultiprocessingPdb, post_mortem_if_fail

# @manual=//torchtnt/utils:device
from torchtnt.utils.device import set_float32_precision

# @manual=//torchtnt/utils:env
from torchtnt.utils.env import seed

logger = logging.getLogger(__name__)

_RT = TypeVar("_RT")


@run_once()
def setup_root_logger(logging_level: int = logging.INFO) -> None:
    """
    Sets up the D2Go root logger. When a new logger is created, it lies in a tree.
    If the logger being used does not have a specific level being specified, it
    will default to using its parent logger. In this case, by setting the root
    logger level to debug, or what is given, we change the default behaviour
    for all loggers.

    See https://docs.python.org/3/library/logging.html for a more in-depth
    description
    """
    initialize_logging(logging_level)
    replace_print_with_logging()


def basic_argument_parser(
    distributed=True,
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
    parser.add_argument(
        "--save-return-file",
        help="When given, the main function outputs will be serialized and saved to this file",
        default=None,
        type=str,
    )
    parser.add_argument(
        "--disable-post-mortem",
        action="store_true",
        help="whether to NOT connect pdb on failure, which only works locally",
    )

    if distributed:
        parser.add_argument(
            "--num-processes", type=int, default=1, help="number of gpus per machine"
        )
        parser.add_argument("--num-machines", type=int, default=1)
        parser.add_argument("--run-as-worker", type=bool, default=False)
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

    return parser


def build_basic_cli_args(
    config_path: Optional[str] = None,
    output_dir: Optional[str] = None,
    runner_name: Optional[str] = None,
    save_return_file: Optional[str] = None,
    num_processes: Optional[Union[int, str]] = None,
    num_machines: Optional[Union[int, str]] = None,
    machine_rank: Optional[Union[int, str]] = None,
    dist_url: Optional[str] = None,
    dist_backend: Optional[str] = None,
    disable_post_mortem: bool = False,
    run_as_worker: bool = False,
    # Evaluator args below
    predictor_path: Optional[str] = None,
    num_threads: Optional[int] = None,
    caffe2_engine: Optional[int] = None,
    caffe2_logging_print_net_summary: Optional[int] = None,
) -> List[str]:
    """
    Returns parameters in the form of CLI arguments for the binary using
    basic_argument_parser to set up its argument parser.

    For the parameters definition and meaning, see basic_argument_parser.
    """
    args: List[str] = []
    if config_path is not None:
        args += ["--config-file", config_path]
    if output_dir is not None:
        args += ["--output-dir", output_dir]
    if runner_name is not None:
        args += ["--runner", runner_name]
    if save_return_file is not None:
        args += ["--save-return-file", str(save_return_file)]
    if disable_post_mortem:
        args += ["--disable-post-mortem"]
    if run_as_worker:
        args += ["--run-as-worker", str(run_as_worker)]
    if num_processes is not None:
        args += ["--num-processes", str(num_processes)]
    if num_machines is not None:
        args += ["--num-machines", str(num_machines)]
    if machine_rank is not None:
        args += ["--machine-rank", str(machine_rank)]
    if dist_url is not None:
        args += ["--dist-url", str(dist_url)]
    if dist_backend is not None:
        args += ["--dist-backend", str(dist_backend)]
    if predictor_path is not None:
        args += ["--predictor-path", predictor_path]
    if num_threads is not None:
        args += ["--num-threads", int(num_threads)]
    if caffe2_engine is not None:
        args += ["--caffe2-engine", int(caffe2_engine)]
    if caffe2_logging_print_net_summary is not None:
        args += [
            "--caffe2_logging_print_net_summary",
            str(caffe2_logging_print_net_summary),
        ]
    return args


def create_cfg_from_cli(
    config_file: str,
    overwrites: Optional[List[str]],
    runner_class: Union[None, str, Type[BaseRunner], Type[DefaultTask]],
) -> CfgNode:
    """
    Centralized function to load config object from config file. It currently supports:
        - YACS based config (return yacs's CfgNode)
    """
    config_file = reroute_config_path(config_file)
    with PathManager.open(config_file, "r") as f:
        # TODO: switch to logger, note that we need to initilaize logger outside of main
        # for running locally.
        print("Loaded config file {}:\n{}".format(config_file, f.read()))

    if isinstance(runner_class, str):
        print(f"Importing runner: {runner_class} ...")
        runner_class = import_runner(runner_class)
    if runner_class is None or issubclass(runner_class, RunnerV2Mixin):
        # Runner-less API
        cfg = load_full_config_from_file(config_file)
    else:
        # backward compatible for old API
        cfg = runner_class.get_default_cfg()
        cfg.merge_from_file(config_file)

    cfg.merge_from_list(overwrites or [])
    cfg.freeze()

    return cfg


def prepare_for_launch(
    args,
) -> Tuple[CfgNode, str, str]:
    """
    Load config, figure out working directory, create runner.
        - when args.config_file is empty, returned cfg will be the default one
        - returned output_dir will always be non empty, args.output_dir has higher
            priority than cfg.OUTPUT_DIR.
    """
    logger.info(args)

    cfg = create_cfg_from_cli(
        config_file=args.config_file,
        overwrites=args.opts,
        runner_class=args.runner,
    )

    # overwrite the output_dir based on config if output is not set via cli
    assert args.output_dir or args.config_file
    output_dir = args.output_dir or cfg.OUTPUT_DIR

    return cfg, output_dir, args.runner


def maybe_override_output_dir(cfg: CfgNode, output_dir: str):
    if cfg.OUTPUT_DIR != output_dir:
        with temp_defrost(cfg):
            logger.warning(
                "Override cfg.OUTPUT_DIR ({}) to be the same as output_dir {}".format(
                    cfg.OUTPUT_DIR, output_dir
                )
            )
            cfg.OUTPUT_DIR = output_dir


def setup_before_launch(
    cfg: CfgNode,
    output_dir: str,
    runner_class: Union[None, str, Type[BaseRunner], Type[DefaultTask]],
) -> Union[None, D2GoSharedContext]:
    """
    Setup logic before spawning workers. Including:
        - Shared context initilization to be passed to all workers
    """
    if isinstance(runner_class, str):
        logger.info(f"Importing runner: {runner_class} ...")
        runner_class = import_runner(runner_class)

    if hasattr(runner_class, "create_shared_context"):
        return runner_class.create_shared_context(cfg)
    return None


def setup_after_launch(
    cfg: CfgNode,
    output_dir: str,
    runner_class: Union[None, str, Type[BaseRunner], Type[DefaultTask]],
) -> Union[None, BaseRunner, Type[DefaultTask]]:
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

    if isinstance(runner_class, str):
        logger.info(f"Importing runner: {runner_class} ...")
        runner_class = import_runner(runner_class)

    if issubclass(runner_class, DefaultTask):
        # TODO(T123679504): merge this with runner code path to return runner instance
        logger.info(f"Importing lightning task: {runner_class} ...")
        runner = runner_class
    elif issubclass(runner_class, BaseRunner):
        logger.info(f"Initializing runner: {runner_class} ...")
        runner = runner_class()
        runner = initialize_runner(runner, cfg)
        logger.info("Running with runner: {}".format(runner))
    else:
        assert runner_class is None, f"Unsupported runner class: {runner_class}"
        runner = None

    # save the diff config
    default_cfg = (
        runner_class.get_default_cfg()
        if runner_class and not issubclass(runner_class, RunnerV2Mixin)
        else cfg.get_default_cfg()
    )
    dump_cfg(
        get_diff_cfg(default_cfg, cfg),
        os.path.join(output_dir, "diff_config.yaml"),
    )

    # scale the config after dumping so that dumped config files keep original world size
    auto_scale_world_size(cfg, new_world_size=comm.get_world_size())

    # avoid random pytorch and CUDA algorithms during the training
    if cfg.SOLVER.DETERMINISTIC:
        logging.warning("Using deterministic training for the reproducibility")

        # tf32
        set_float32_precision("highest")
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False

        # pytorch deterministic
        torch.set_deterministic_debug_mode(2)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True)
        torch.utils.deterministic.fill_uninitialized_memory = True
        # reference: https://docs.nvidia.com/cuda/cublas/index.html#cublasApi_reproducibility
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

    if cfg.SEED > 0:
        seed_all_rng(cfg.SEED)

    return runner


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
        enable_propagation=True,
        configure_stdout=False,
    )

    return logger


@run_once()
def setup_loggers(output_dir):
    # Setup logging in each of the distributed processes.
    setup_root_logger()
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


def post_mortem_if_fail_for_main(main_func: Callable[..., _RT]) -> Callable[..., _RT]:
    def new_main_func(cfg, output_dir, *args, **kwargs) -> _RT:
        pdb_ = (
            MultiprocessingPdb(FolderLock(output_dir))
            if comm.get_world_size() > 1
            else None  # fallback to use normal pdb for single process
        )
        return post_mortem_if_fail(pdb_)(main_func)(cfg, output_dir, *args, **kwargs)

    return PicklableWrapper(new_main_func)
