#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

"""
Binary to convert pytorch detectron2go model to a predictor, which contains model(s) in
deployable format (such as torchscript, caffe2, ...)
"""

import copy
import logging
import sys
from dataclasses import dataclass
from typing import Any, Dict, List, Type, Union

import mobile_cv.lut.lib.pt.flops_utils as flops_utils
from d2go.config import CfgNode, temp_defrost
from d2go.distributed import set_shared_context
from d2go.export.exporter import convert_and_export_predictor
from d2go.runner import BaseRunner
from d2go.setup import (
    basic_argument_parser,
    post_mortem_if_fail_for_main,
    prepare_for_launch,
    setup_after_launch,
    setup_before_launch,
    setup_root_logger,
)


logger = logging.getLogger("d2go.tools.export")


@dataclass
class ExporterOutput:
    predictor_paths: Dict[str, str]
    accuracy_comparison: Dict[str, Any]


def main(
    cfg: CfgNode,
    output_dir: str,
    runner_class: Union[str, Type[BaseRunner]],
    # binary specific optional arguments
    predictor_types: List[str],
    device: str = "cpu",
    compare_accuracy: bool = False,
    skip_if_fail: bool = False,
    skip_model_weights: bool = False,
) -> ExporterOutput:
    if compare_accuracy:
        raise NotImplementedError(
            "compare_accuracy functionality isn't currently supported."
        )
        # NOTE: dict for metrics of all exported models (and original pytorch model)
        # ret["accuracy_comparison"] = accuracy_comparison

    cfg = copy.deepcopy(cfg)
    with temp_defrost(cfg):
        if skip_model_weights:
            cfg.merge_from_list(["MODEL.WEIGHTS", ""])

    runner = setup_after_launch(cfg, output_dir, runner_class)

    with temp_defrost(cfg):
        cfg.merge_from_list(["MODEL.DEVICE", device])

    model = runner.build_model(cfg, eval_only=True)
    # NOTE: train dataset is used to avoid leakage since the data might be used for
    # running calibration for quantization. test_loader is used to make sure it follows
    # the inference behaviour (augmentation will not be applied).
    datasets = list(cfg.DATASETS.TRAIN)
    data_loader = runner.build_detection_test_loader(cfg, datasets)

    logger.info("Running the pytorch model and print FLOPS ...")
    first_batch = next(iter(data_loader))
    input_args = (first_batch,)
    flops_utils.print_model_flops(model, input_args)

    predictor_paths: Dict[str, str] = {}
    for typ in predictor_types:
        # convert_and_export_predictor might alter the model, copy before calling it
        pytorch_model = copy.deepcopy(model)
        try:
            predictor_path = convert_and_export_predictor(
                cfg,
                pytorch_model,
                typ,
                output_dir,
                data_loader,
            )
            logger.info(f"Predictor type {typ} has been exported to {predictor_path}")
            predictor_paths[typ] = predictor_path
        except Exception as e:
            logger.exception(f"Export {typ} predictor failed: {e}")
            if not skip_if_fail:
                raise e

    runner.cleanup()
    return ExporterOutput(
        predictor_paths=predictor_paths,
        accuracy_comparison={},
    )


def run_with_cmdline_args(args):
    cfg, output_dir, runner_name = prepare_for_launch(args)
    shared_context = setup_before_launch(cfg, output_dir, runner_name)
    if shared_context is not None:
        set_shared_context(shared_context)

    main_func = main if args.disable_post_mortem else post_mortem_if_fail_for_main(main)
    return main_func(
        cfg,
        output_dir,
        runner_name,
        # binary specific optional arguments
        predictor_types=args.predictor_types,
        device=args.device,
        compare_accuracy=args.compare_accuracy,
        skip_if_fail=args.skip_if_fail,
        skip_model_weights=args.skip_model_weights,
    )


def get_parser():
    parser = basic_argument_parser(distributed=False)
    parser.add_argument(
        "--predictor-types",
        type=str,
        nargs="+",
        help="List of strings specify the types of predictors to export",
    )
    parser.add_argument(
        "--device", default="cpu", help="the device to export the model on"
    )
    parser.add_argument(
        "--compare-accuracy",
        action="store_true",
        help="If true, all exported models and the original pytorch model will be"
        " evaluated on cfg.DATASETS.TEST",
    )
    parser.add_argument(
        "--skip-if-fail",
        action="store_true",
        default=False,
        help="If set, suppress the exception for failed exporting and continue to"
        " export the next type of model",
    )
    parser.add_argument("--skip-model-weights", action="store_true")

    return parser


def cli(args=None):
    args = sys.argv[1:] if args is None else args
    run_with_cmdline_args(get_parser().parse_args(args))


if __name__ == "__main__":
    setup_root_logger()
    cli()
