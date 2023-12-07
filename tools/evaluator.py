#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

"""
Binary to evaluate predictor-based model (consist of models in deployable format such
torchscript, caffe2, etc.) using Detectron2Go system (dataloading, evaluation, etc).
"""

import logging
import sys
from typing import Callable, List, Optional, Type, Union

import torch
from d2go.config import CfgNode
from d2go.distributed import launch
from d2go.quantization.qconfig import smart_decode_backend
from d2go.runner import BaseRunner
from d2go.setup import (
    basic_argument_parser,
    build_basic_cli_args,
    caffe2_global_init,
    post_mortem_if_fail_for_main,
    prepare_for_launch,
    setup_after_launch,
    setup_before_launch,
    setup_root_logger,
)
from d2go.trainer.api import EvaluatorOutput
from d2go.utils.mast import gather_mast_errors, mast_error_handler

from d2go.utils.misc import print_metrics_table, save_binary_outputs
from mobile_cv.predictor.api import create_predictor

logger = logging.getLogger("d2go.tools.caffe2_evaluator")


def main(
    cfg: CfgNode,
    output_dir: str,
    runner_class: Union[str, Type[BaseRunner]],
    # binary specific optional arguments
    predictor_path: str,
    num_threads: Optional[int] = None,
    caffe2_engine: Optional[int] = None,
    caffe2_logging_print_net_summary: int = 0,
) -> EvaluatorOutput:
    # FIXME: Ideally the quantization backend should be encoded in the torchscript model
    # or the predictor, and be used automatically during the inference, without user
    # manually setting the global variable.
    torch.backends.quantized.engine = smart_decode_backend(cfg.QUANTIZATION.BACKEND)
    print("run with quantized engine: ", torch.backends.quantized.engine)

    runner = setup_after_launch(cfg, output_dir, runner_class)
    caffe2_global_init(caffe2_logging_print_net_summary, num_threads)

    predictor = create_predictor(predictor_path)
    metrics = runner.do_test(cfg, predictor)
    print_metrics_table(metrics)
    runner.cleanup()
    return EvaluatorOutput(
        accuracy=metrics,
        metrics=metrics,
    )


def wrapped_main(*args, **kwargs) -> Callable:
    return mast_error_handler(main)(*args, **kwargs)


def run_with_cmdline_args(args):
    cfg, output_dir, runner_name = prepare_for_launch(args)
    shared_context = setup_before_launch(cfg, output_dir, runner_name)
    main_func = (
        wrapped_main
        if args.disable_post_mortem
        else post_mortem_if_fail_for_main(wrapped_main)
    )
    outputs = launch(
        main_func,
        args.num_processes,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        backend="GLOO",
        always_spawn=False,
        shared_context=shared_context,
        args=(cfg, output_dir, runner_name),
        kwargs={
            "predictor_path": args.predictor_path,
            "num_threads": args.num_threads,
            "caffe2_engine": args.caffe2_engine,
            "caffe2_logging_print_net_summary": args.caffe2_logging_print_net_summary,
        },
    )
    # Only save results from global rank 0 for consistency.
    if args.save_return_file is not None and args.machine_rank == 0:
        save_binary_outputs(args.save_return_file, outputs[0])


def build_cli_args(
    eval_only: bool = False,
    resume: bool = False,
    **kwargs,
) -> List[str]:
    """Returns parameters in the form of CLI arguments for evaluator binary.

    For the list of non-train_net-specific parameters, see build_basic_cli_args."""
    args = build_basic_cli_args(**kwargs)
    if eval_only:
        args += ["--eval-only"]
    if resume:
        args += ["--resume"]
    return args


def cli(args=None):
    parser = basic_argument_parser()
    parser.add_argument(
        "--predictor-path",
        type=str,
        help="Path (a directory) to the exported model that will be evaluated",
    )
    # === performance config ===========================================================
    parser.add_argument(
        "--num-threads",
        type=int,
        default=None,
        help="Number of omp/mkl threads (per process) to use in Caffe2's GlobalInit",
    )
    parser.add_argument(
        "--caffe2-engine",
        type=str,
        default=None,
        help="If set, engine of all ops will be set by this value",
    )
    parser.add_argument(
        "--caffe2_logging_print_net_summary",
        type=int,
        default=0,
        help="Control the --caffe2_logging_print_net_summary in GlobalInit",
    )
    args = sys.argv[1:] if args is None else args
    run_with_cmdline_args(parser.parse_args(args))


if __name__ == "__main__":
    setup_root_logger()
    gather_mast_errors(cli())
