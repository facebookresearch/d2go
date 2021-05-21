#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

"""
Binary to convert pytorch detectron2go model to a predictor, which contains model(s) in
deployable format (such as torchscript, caffe2, ...)
"""

import copy
import json
import logging
import os
import tempfile
import typing
from typing import Optional

import mobile_cv.lut.lib.pt.flops_utils as flops_utils
import torch
from d2go.config import temp_defrost, CfgNode
from d2go.export.api import convert_and_export_predictor
from d2go.setup import (
    basic_argument_parser,
    prepare_for_launch,
    setup_after_launch,
)
from iopath.common.file_io import PathManager
from iopath.fb.manifold import ManifoldPathHandler
from mobile_cv.common.misc.py import post_mortem_if_fail

path_manager = PathManager()
path_manager.register_handler(ManifoldPathHandler())


logger = logging.getLogger("d2go.tools.export")

INFERNCE_CONFIG_FILENAME = "inference_config.json"
MOBILE_OPTIMIZED_BUNDLE_FILENAME = "mobile_optimized_bundled.ptl"


def write_model_with_config(
    output_dir: str, model_jit_path: str, inference_config: Optional[CfgNode] = None
):
    """
    Writes the sdk inference config along with model file and saves the model
    with configuration at ${output_dir}/mobile_optimized_bundled.ptl
    """
    model_jit_local_path = path_manager.get_local_path(model_jit_path)
    model = torch.jit.load(model_jit_local_path)
    extra_files = {}
    if inference_config:
        extra_files = {
            INFERNCE_CONFIG_FILENAME: json.dumps(inference_config.as_flattened_dict())
        }

    bundled_model_path = os.path.join(output_dir, MOBILE_OPTIMIZED_BUNDLE_FILENAME)

    with tempfile.NamedTemporaryFile() as temp_file:
        model._save_for_lite_interpreter(temp_file.name, _extra_files=extra_files)
        path_manager.copy_from_local(temp_file.name, bundled_model_path, overwrite=True)

    logger.info(f"Saved bundled model to: {bundled_model_path}")


def _add_inference_config(
    predictor_paths: typing.Dict[str, str],
    inference_config: Optional[CfgNode],
):
    """Adds inference config in _extra_files as json and writes the bundled model"""
    if inference_config is None:
        return

    for _, export_dir in predictor_paths.items():
        model_jit_path = os.path.join(export_dir, "model.jit")
        write_model_with_config(export_dir, model_jit_path, inference_config)


def main(
    cfg,
    output_dir,
    runner,
    # binary specific optional arguments
    predictor_types: typing.List[str],
    compare_accuracy: bool = False,
    skip_if_fail: bool = False,
    inference_config: Optional[CfgNode] = None,
):
    cfg = copy.deepcopy(cfg)
    setup_after_launch(cfg, output_dir, runner)

    with temp_defrost(cfg):
        cfg.merge_from_list(["MODEL.DEVICE", "cpu"])
    model = runner.build_model(cfg, eval_only=True)

    # NOTE: train dataset is used to avoid leakage since the data might be used for
    # running calibration for quantization. test_loader is used to make sure it follows
    # the inference behaviour (augmentation will not be applied).
    datasets = cfg.DATASETS.TRAIN[0]
    data_loader = runner.build_detection_test_loader(cfg, datasets)

    logger.info("Running the pytorch model and print FLOPS ...")
    first_batch = next(iter(data_loader))
    input_args = (first_batch,)
    flops_utils.print_model_flops(model, input_args)

    predictor_paths: typing.Dict[str, str] = {}
    for typ in predictor_types:
        # convert_and_export_predictor might alter the model, copy before calling it
        pytorch_model = copy.deepcopy(model)
        try:
            predictor_path = convert_and_export_predictor(
                cfg, pytorch_model, typ, output_dir, data_loader
            )
            logger.info(f"Predictor type {typ} has been exported to {predictor_path}")
            predictor_paths[typ] = predictor_path
        except Exception as e:
            logger.warning(f"Export {typ} predictor failed: {e}")
            if not skip_if_fail:
                raise e

    _add_inference_config(predictor_paths, inference_config)

    ret = {"predictor_paths": predictor_paths, "accuracy_comparison": {}}
    if compare_accuracy:
        raise NotImplementedError()
        # NOTE: dict for metrics of all exported models (and original pytorch model)
        # ret["accuracy_comparison"] = accuracy_comparison

    return ret


@post_mortem_if_fail()
def run_with_cmdline_args(args):
    cfg, output_dir, runner = prepare_for_launch(args)
    inference_config = None
    if args.inference_config_file:
        inference_config = CfgNode(
            CfgNode.load_yaml_with_base(args.inference_config_file)
        )

    return main(
        cfg,
        output_dir,
        runner,
        # binary specific optional arguments
        predictor_types=args.predictor_types,
        compare_accuracy=args.compare_accuracy,
        skip_if_fail=args.skip_if_fail,
        inference_config=inference_config,
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
    parser.add_argument(
        "--inference-config-file",
        type=str,
        default=None,
        help="Inference config file containing the model parameters for c++ sdk pipeline",
    )
    return parser


def cli():
    run_with_cmdline_args(get_parser().parse_args())


if __name__ == "__main__":
    cli()
