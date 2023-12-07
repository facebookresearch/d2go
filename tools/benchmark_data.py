#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Tool for benchmarking data loading
"""

import logging
import time
from dataclasses import dataclass
from typing import Type, Union

import detectron2.utils.comm as comm
import numpy as np
from d2go.config import CfgNode
from d2go.distributed import get_num_processes_per_machine, launch
from d2go.evaluation.api import AccuracyDict, MetricsDict
from d2go.runner import BaseRunner
from d2go.setup import (
    basic_argument_parser,
    post_mortem_if_fail_for_main,
    prepare_for_launch,
    setup_after_launch,
)
from d2go.utils.misc import print_metrics_table
from detectron2.fb.env import get_launch_environment
from detectron2.utils.logger import log_every_n_seconds
from fvcore.common.history_buffer import HistoryBuffer

logger = logging.getLogger("d2go.tools.benchmark_data")


@dataclass
class BenchmarkDataOutput:
    accuracy: AccuracyDict[float]
    metrics: MetricsDict[float]


def main(
    cfg: CfgNode,
    output_dir: str,
    runner_class: Union[str, Type[BaseRunner]],
    is_train: bool = True,
) -> BenchmarkDataOutput:
    runner = setup_after_launch(cfg, output_dir, runner_class)

    if is_train:
        data_loader = runner.build_detection_train_loader(cfg)
    else:
        assert len(cfg.DATASETS.TEST) > 0, cfg.DATASETS.TEST
        data_loader = runner.build_detection_test_loader(cfg, cfg.DATASETS.TEST[0])

    TOTAL_BENCHMARK_TIME = (
        100 if get_launch_environment() == "local" else 600
    )  # run benchmark for 10 min
    LOGGING_METER_WINDOW_SIZE = 20
    LOGGING_METER_TIME_INTERVAL = 5
    WARMUP_ITERS = 5

    # initialize
    time_per_iter = HistoryBuffer(max_length=10000)
    total_time = 0

    start = time.time()
    for no, batch in enumerate(data_loader):
        data_time = time.time() - start
        time_per_iter.update(data_time)
        total_time += data_time

        if no == 0:
            logger.info("Show the first batch as example:\n{}".format(batch))

        # Assume batch size is constant
        batch_size = cfg.SOLVER.IMS_PER_BATCH // comm.get_world_size()
        assert len(batch) * batch_size

        median = time_per_iter.median(window_size=LOGGING_METER_WINDOW_SIZE)
        avg = time_per_iter.avg(window_size=LOGGING_METER_WINDOW_SIZE)
        log_every_n_seconds(
            logging.INFO,
            "iter: {};"
            " recent per-iter seconds: {:.4f} (avg) {:.4f} (median);"
            " recent per-image seconds: {:.4f} (avg) {:.4f} (median).".format(
                no,
                avg,
                median,
                avg / batch_size,
                median / batch_size,
            ),
            n=LOGGING_METER_TIME_INTERVAL,
        )

        # Synchronize between processes, exit when all processes are running for enough
        # time. This mimic the loss.backward(), the logged time doesn't include the time
        # for synchronize.
        finished = comm.all_gather(total_time >= TOTAL_BENCHMARK_TIME)
        if all(x for x in finished):
            logger.info("Benchmarking finished after {} seconds".format(total_time))
            break

        start = time.time()

    dataset_name = ":".join(cfg.DATASETS.TRAIN) if is_train else cfg.DATASETS.TEST[0]
    time_per_iter = [x[0] for x in time_per_iter.values()]
    time_per_iter = time_per_iter[
        min(WARMUP_ITERS, max(len(time_per_iter) - WARMUP_ITERS, 0)) :
    ]
    results = {
        "environment": {
            "num_workers": cfg.DATALOADER.NUM_WORKERS,
            "world_size": comm.get_world_size(),
            "processes_per_machine": get_num_processes_per_machine(),
        },
        "main_processes_stats": {
            "batch_size_per_process": batch_size,
            "per_iter_avg": np.average(time_per_iter),
            "per_iter_p1": np.percentile(time_per_iter, 1, interpolation="nearest"),
            "per_iter_p10": np.percentile(time_per_iter, 10, interpolation="nearest"),
            "per_iter_p50": np.percentile(time_per_iter, 50, interpolation="nearest"),
            "per_iter_p90": np.percentile(time_per_iter, 90, interpolation="nearest"),
            "per_iter_p99": np.percentile(time_per_iter, 99, interpolation="nearest"),
            "per_image_avg": np.average(time_per_iter) / batch_size,
            "per_image_p1": np.percentile(time_per_iter, 1, interpolation="nearest")
            / batch_size,
            "per_image_p10": np.percentile(time_per_iter, 10, interpolation="nearest")
            / batch_size,
            "per_image_p50": np.percentile(time_per_iter, 50, interpolation="nearest")
            / batch_size,
            "per_image_p90": np.percentile(time_per_iter, 90, interpolation="nearest")
            / batch_size,
            "per_image_p99": np.percentile(time_per_iter, 99, interpolation="nearest")
            / batch_size,
        },
        "data_processes_stats": {},  # TODO: add worker stats
    }
    # Metrics follows the hierarchy of: name -> dataset -> task -> metrics -> number
    metrics = {"_name_": {dataset_name: results}}
    print_metrics_table(metrics)

    runner.cleanup()
    return BenchmarkDataOutput(
        accuracy=metrics,
        metrics=metrics,
    )


def run_with_cmdline_args(args):
    cfg, output_dir, runner_name = prepare_for_launch(args)
    main_func = main if args.disable_post_mortem else post_mortem_if_fail_for_main(main)
    launch(
        main_func,
        num_processes_per_machine=args.num_processes,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        backend=args.dist_backend,
        args=(cfg, output_dir, runner_name),
        kwargs={
            "is_train": args.is_train,
        },
    )


if __name__ == "__main__":
    parser = basic_argument_parser(requires_output_dir=True)
    parser.add_argument(
        "--is-train",
        type=bool,
        default=True,
        help="data loader is train",
    )
    run_with_cmdline_args(parser.parse_args())
