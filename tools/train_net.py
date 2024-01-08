#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

"""
Detection Training Script.
"""

import logging
import sys
from typing import Callable, Dict, List, Type, Union

import detectron2.utils.comm as comm
from d2go.config import CfgNode
from d2go.distributed import distributed_worker, launch
from d2go.runner import BaseRunner
from d2go.runner.config_defaults import preprocess_cfg
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
from d2go.utils.mast import gather_mast_errors, mast_error_handler
from d2go.utils.misc import (
    dump_trained_model_configs,
    print_metrics_table,
    save_binary_outputs,
)
from detectron2.engine.defaults import create_ddp_model
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

logger = logging.getLogger("d2go.tools.train_net")
# Make sure logging is set up centrally even for e.g. dataloading workers which
# have entry points outside of D2Go.
setup_root_logger()


TrainOrTestNetOutput = Union[TrainNetOutput, TestNetOutput]


def main(
    cfg: CfgNode,
    output_dir: str,
    runner_class: Union[str, Type[BaseRunner]],
    eval_only: bool = False,
    resume: bool = True,  # NOTE: always enable resume when running on cluster
) -> TrainOrTestNetOutput:
    logger.debug(f"Entered main for d2go, {runner_class=}")
    runner = setup_after_launch(cfg, output_dir, runner_class)

    model = runner.build_model(cfg)
    logger.info("Model:\n{}".format(model))

    if eval_only:
        checkpointer = runner.build_checkpointer(cfg, model, save_dir=output_dir)
        # checkpointer.resume_or_load() will skip all additional checkpointable
        # which may not be desired like ema states
        if resume and checkpointer.has_checkpoint():
            checkpoint = checkpointer.resume_or_load(cfg.MODEL.WEIGHTS, resume=resume)
        else:
            checkpoint = checkpointer.load(cfg.MODEL.WEIGHTS)
        train_iter = checkpoint.get("iteration", None)
        model.eval()
        metrics = runner.do_test(cfg, model, train_iter=train_iter)
        print_metrics_table(metrics)
        runner.cleanup()
        return TestNetOutput(
            accuracy=metrics,
            metrics=metrics,
        )

    # Use DDP if FSDP is not enabled
    # TODO (T142223289): rewrite ddp wrapping as modeling hook
    if not isinstance(model, FSDP):
        model = create_ddp_model(
            model,
            fp16_compression=cfg.MODEL.DDP_FP16_GRAD_COMPRESS,
            device_ids=None if cfg.MODEL.DEVICE == "cpu" else [comm.get_local_rank()],
            broadcast_buffers=False,
            find_unused_parameters=cfg.MODEL.DDP_FIND_UNUSED_PARAMETERS,
            gradient_as_bucket_view=cfg.MODEL.DDP_GRADIENT_AS_BUCKET_VIEW,
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
    runner.cleanup()
    return TrainNetOutput(
        # for e2e_workflow
        accuracy=metrics,
        # for unit_workflow
        model_configs=trained_model_configs,
        metrics=metrics,
    )


def wrapped_main(*args, **kwargs) -> Callable[..., TrainOrTestNetOutput]:
    return mast_error_handler(main)(*args, **kwargs)


def run_with_cmdline_args(args):
    cfg, output_dir, runner_name = prepare_for_launch(args)
    cfg = preprocess_cfg(cfg)
    shared_context = setup_before_launch(cfg, output_dir, runner_name)

    main_func = (
        wrapped_main
        if args.disable_post_mortem
        else post_mortem_if_fail_for_main(wrapped_main)
    )

    if args.run_as_worker:
        logger.info("Running as worker")
        result: TrainOrTestNetOutput = distributed_worker(
            main_func,
            args=(cfg, output_dir, runner_name),
            kwargs={
                "eval_only": args.eval_only,
                "resume": args.resume,
            },
            backend=args.dist_backend,
            init_method=None,  # init_method is env by default
            dist_params=None,
            return_save_file=None,
            shared_context=shared_context,
        )
    else:
        outputs: Dict[int, TrainOrTestNetOutput] = launch(
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
        # The indices of outputs are global ranks of all workers on this node, here we
        # use the local master result.
        result: TrainOrTestNetOutput = outputs[args.machine_rank * args.num_processes]

    # Only save result from global rank 0 for consistency.
    if args.save_return_file is not None and args.machine_rank == 0:
        logger.info(f"Operator result: {result}")
        logger.info(f"Writing result to {args.save_return_file}.")
        save_binary_outputs(args.save_return_file, result)


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


def invoke_main() -> None:
    gather_mast_errors(cli())


if __name__ == "__main__":
    invoke_main()  # pragma: no cover
