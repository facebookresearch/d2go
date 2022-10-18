#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

"""
Binary to evaluate predictor-based model (consist of models in deployable format such
torchscript, caffe2, etc.) using Detectron2Go system (dataloading, evaluation, etc).
"""

import logging
import sys
from dataclasses import dataclass
from typing import Optional, Type, Union

import torch
from d2go.config import CfgNode
from d2go.distributed import launch
from d2go.evaluation.api import AccuracyDict, MetricsDict
from d2go.quantization.qconfig import smart_decode_backend
from d2go.runner import BaseRunner
from d2go.setup import (
    basic_argument_parser,
    caffe2_global_init,
    post_mortem_if_fail_for_main,
    prepare_for_launch,
    setup_after_launch,
)
from d2go.utils.misc import print_metrics_table
from mobile_cv.predictor.api import create_predictor

logger = logging.getLogger("d2go.tools.caffe2_evaluator")


@dataclass
class EvaluatorOutput:
    accuracy: AccuracyDict[float]
    metrics: MetricsDict[float]


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
    return EvaluatorOutput(
        accuracy=metrics,
        metrics=metrics,
    )


def run_with_cmdline_args(args):
    cfg, output_dir, runner_name = prepare_for_launch(args)
    main_func = main if args.disable_post_mortem else post_mortem_if_fail_for_main(main)
    launch(
        main_func,
        args.num_processes,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        backend="GLOO",
        always_spawn=False,
        args=(cfg, output_dir, runner_name),
        kwargs={
            "predictor_path": args.predictor_path,
            "num_threads": args.num_threads,
            "caffe2_engine": args.caffe2_engine,
            "caffe2_logging_print_net_summary": args.caffe2_logging_print_net_summary,
        },
    )


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
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    cli()
