#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

"""
Detection Training Script.
"""

import logging
import sys
from typing import List, Type, Union

import detectron2.utils.comm as comm
from d2go.config import CfgNode
from d2go.distributed import launch
from d2go.runner import BaseRunner
from d2go.setup import (
    basic_argument_parser,
    build_basic_cli_args,
    post_mortem_if_fail_for_main,
    prepare_for_launch,
    setup_after_launch,
    setup_before_launch,
    setup_root_logger,
)
from d2go.trainer.api import TestNetOutput, TrainNetOutput
from d2go.trainer.fsdp import is_fsdp_enabled
from d2go.utils.misc import (
    dump_trained_model_configs,
    print_metrics_table,
    save_binary_outputs,
)
from detectron2.engine.defaults import create_ddp_model

from torch.distributed.elastic.multiprocessing.errors import (
    _NOT_AVAILABLE,
    ChildFailedError,
    get_error_handler,
)

logger = logging.getLogger("d2go.tools.train_net")


def main(
    cfg: CfgNode,
    output_dir: str,
    runner_class: Union[str, Type[BaseRunner]],
    eval_only: bool = False,
    resume: bool = True,  # NOTE: always enable resume when running on cluster
) -> Union[TrainNetOutput, TestNetOutput]:

    logger.info("Starting main")
    error_handler = get_error_handler()
    logger.debug(f">>>>>>> Error handler is: {type(error_handler)=}, {error_handler=}")
    error_handler.initialize()
    logger.debug("Error handler has been initialized")

    try:  # Main error handler starts here...
        logger.debug(f"Entered main for d2go, {runner_class=}")
        runner = setup_after_launch(cfg, output_dir, runner_class)

        model = runner.build_model(cfg)
        logger.info("Model:\n{}".format(model))

        if eval_only:
            checkpointer = runner.build_checkpointer(cfg, model, save_dir=output_dir)
            # checkpointer.resume_or_load() will skip all additional checkpointable
            # which may not be desired like ema states
            if resume and checkpointer.has_checkpoint():
                checkpoint = checkpointer.resume_or_load(
                    cfg.MODEL.WEIGHTS, resume=resume
                )
            else:
                checkpoint = checkpointer.load(cfg.MODEL.WEIGHTS)
            train_iter = checkpoint.get("iteration", None)
            model.eval()
            metrics = runner.do_test(cfg, model, train_iter=train_iter)
            print_metrics_table(metrics)
            return TestNetOutput(
                accuracy=metrics,
                metrics=metrics,
            )

        # Use DDP if FSDP is not enabled
        # TODO (T142223289): rewrite ddp wrapping as modeling hook
        if not is_fsdp_enabled(cfg):
            model = create_ddp_model(
                model,
                fp16_compression=cfg.MODEL.DDP_FP16_GRAD_COMPRESS,
                device_ids=None
                if cfg.MODEL.DEVICE == "cpu"
                else [comm.get_local_rank()],
                broadcast_buffers=False,
                find_unused_parameters=cfg.MODEL.DDP_FIND_UNUSED_PARAMETERS,
            )

        logger.info("Starting train..")
        trained_cfgs = runner.do_train(cfg, model, resume=resume)

        final_eval = cfg.TEST.FINAL_EVAL
        if final_eval:
            # run evaluation after training in the same processes
            metrics = runner.do_test(cfg, model)
            print_metrics_table(metrics)
        else:
            metrics = {}

        # dump config files for trained models
        trained_model_configs = dump_trained_model_configs(cfg.OUTPUT_DIR, trained_cfgs)
        return TrainNetOutput(
            # for e2e_workflow
            accuracy=metrics,
            # for unit_workflow
            model_configs=trained_model_configs,
            metrics=metrics,
        )
    except ChildFailedError as e:
        logger.info(f"Got a ChildFailedError: {e=}")
        rank, failure = e.get_first_failure()
        if failure.error_file != _NOT_AVAILABLE:
            error_handler.dump_error_file(failure.error_file, failure.exitcode)
        else:
            logger.info(
                (
                    f"local_rank {rank} FAILED with no error file."
                    f" Decorate your entrypoint fn with @record for traceback info."
                    f" See: https://pytorch.org/docs/stable/elastic/errors.html"
                )
            )
            raise
    except Exception as e:
        logger.info(f"Caught a generic exception: {e=}")
        error_handler.record_exception(e)
        raise


def run_with_cmdline_args(args):
    cfg, output_dir, runner_name = prepare_for_launch(args)
    shared_context = setup_before_launch(cfg, output_dir, runner_name)

    main_func = main if args.disable_post_mortem else post_mortem_if_fail_for_main(main)
    outputs = launch(
        main_func,
        num_processes_per_machine=args.num_processes,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        backend=args.dist_backend,
        shared_context=shared_context,
        args=(cfg, output_dir, runner_name),
        kwargs={
            "eval_only": args.eval_only,
            "resume": args.resume,
        },
    )

    # Only save results from global rank 0 for consistency.
    if args.save_return_file is not None and args.machine_rank == 0:
        save_binary_outputs(args.save_return_file, outputs[0])


def cli(args=None):
    logger.info(f"Inside CLI, {args=}")
    parser = basic_argument_parser(requires_output_dir=False)
    parser.add_argument(
        "--eval-only", action="store_true", help="perform evaluation only"
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="whether to attempt to resume from the checkpoint directory",
    )
    args = sys.argv[1:] if args is None else args
    run_with_cmdline_args(parser.parse_args(args))


def build_cli_args(
    eval_only: bool = False,
    resume: bool = False,
    **kwargs,
) -> List[str]:
    """Returns parameters in the form of CLI arguments for train_net binary.

    For the list of non-train_net-specific parameters, see build_basic_cli_args."""
    args = build_basic_cli_args(**kwargs)
    if eval_only:
        args += ["--eval-only"]
    if resume:
        args += ["--resume"]
    return args


if __name__ == "__main__":
    setup_root_logger()
    logger.info("Starting CLI application")
    try:
        cli()
    finally:
        logging.info("Entering final reply file generation step")
        import glob
        import os
        import shutil

        torchx_reply_files = glob.glob("/tmp/torchx_*/**/*.json", recursive=True)
        logger.info(
            f"Found the following reply files on this host: {torchx_reply_files}"
        )
        first_reply_file = None
        first_reply_file_st = float("Inf")
        for f in torchx_reply_files:
            if (mtime := os.stat(f).st_mtime) < first_reply_file_st:
                first_reply_file = f
                first_reply_file_st = mtime
        if first_reply_file and os.environ.get("MAST_HPC_TASK_FAILURE_REPLY_FILE"):
            logger.info(
                f'Copying {first_reply_file=} to {os.environ["MAST_HPC_TASK_FAILURE_REPLY_FILE"]}'
            )
            shutil.copyfile(
                first_reply_file, os.environ["MAST_HPC_TASK_FAILURE_REPLY_FILE"]
            )
