#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

"""
Binary to evaluate predictor-based model (consist of models in deployable format such
torchscript, caffe2, etc.) using Detectron2Go system (dataloading, evaluation, etc).
"""

import torch
import logging

from d2go.distributed import launch
from d2go.setup import (
    basic_argument_parser,
    caffe2_global_init,
    post_mortem_if_fail_for_main,
    prepare_for_launch,
    setup_after_launch,
)
from d2go.utils.misc import print_metrics_table
from mobile_cv.common.misc.py import post_mortem_if_fail
from mobile_cv.predictor.api import create_predictor

logger = logging.getLogger("d2go.tools.caffe2_evaluator")


def main(
    cfg,
    output_dir,
    runner,
    # binary specific optional arguments
    predictor_path,
    num_threads=None,
    caffe2_engine=None,
    caffe2_logging_print_net_summary=0,
):
    torch.backends.quantized.engine = cfg.QUANTIZATION.BACKEND
    print("run with quantized engine: ", torch.backends.quantized.engine)

    setup_after_launch(cfg, output_dir, runner)
    caffe2_global_init(caffe2_logging_print_net_summary, num_threads)

    predictor = create_predictor(predictor_path)
    metrics = runner.do_test(cfg, predictor)
    print_metrics_table(metrics)
    return {
        "accuracy": metrics,
        "metrics": metrics,
    }


@post_mortem_if_fail()
def run_with_cmdline_args(args):
    cfg, output_dir, runner = prepare_for_launch(args)
    launch(
        post_mortem_if_fail_for_main(main),
        args.num_processes,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        backend="GLOO",
        always_spawn=False,
        args=(
            cfg,
            output_dir,
            runner,
            # binary specific optional arguments
            args.predictor_path,
            args.num_threads,
            args.caffe2_engine,
            args.caffe2_logging_print_net_summary,
        ),
    )


if __name__ == "__main__":
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
    run_with_cmdline_args(parser.parse_args())
