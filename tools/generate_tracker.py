#!/usr/bin/env python3
"""
Binary to convert torchscript detection model to tracker.

This is a temporary converter until we figure out how to do this
with the new predictor api.
"""

import argparse
import copy
import logging
import os
import sys

import torch
from d2go.config import temp_defrost
from d2go.export.d2_meta_arch import d2_meta_arch_prepare_for_quant
from d2go.modeling.meta_arch.traceable_tracker import (
    DetectAndTrack,
    TraceableTracker,
)
from d2go.modeling.quantization import post_training_quantize
from d2go.setup import (
    basic_argument_parser,
    prepare_for_launch,
    setup_after_launch,
)
from mobile_cv.arch.utils import fuse_utils
from torch.utils.mobile_optimizer import optimize_for_mobile


logger = logging.getLogger("d2go.tools.generate_tracker")


def quantize(model, cfg, data_loader):
    """Run post training quantization and conversion"""
    logger.info(f"PTQ using backend {cfg.QUANTIZATION.BACKEND}...")
    first_batch = next(iter(data_loader))
    tensor_inputs = model.get_caffe2_inputs(first_batch)
    model = d2_meta_arch_prepare_for_quant(model, cfg)
    quant_model = post_training_quantize(cfg, model, [tensor_inputs])
    assert not fuse_utils.check_bn_exist(quant_model)
    quant_model = torch.quantization.convert(quant_model, inplace=False)
    return quant_model


def trace(detector, tracker, data_loader):
    """Trace the detect and track model

    Return the traced models and data
    """
    # get data
    first_batch = next(iter(data_loader))
    inputs_detect = detector.get_caffe2_inputs(first_batch)

    # trace detector and use output to create tracker input
    with torch.no_grad():
        script_detector = torch.jit.trace(detector, (inputs_detect,))
    rois = script_detector(inputs_detect)[0]

    # use the initial roi and add the index bc this is needed to trace
    rois = torch.cat((torch.Tensor([[0.0]]), rois[:1, :]), axis=1)
    inputs_track = (*inputs_detect, rois)

    # trace tracker
    with torch.no_grad():
        script_tracker = torch.jit.trace(tracker, (inputs_track,))

    # combined detectandtrack
    dt = DetectAndTrack(script_detector, script_tracker)
    script_dt = torch.jit.script(dt)
    inputs_dt = (
        (*inputs_detect, torch.Tensor([False]), torch.Tensor([])),
        (*inputs_detect, torch.Tensor([True]), rois),
    )

    return {
        "detect": (script_detector, inputs_detect),
        "track": (script_tracker, inputs_track),
        "detectandtrack": (script_dt, inputs_dt),
    }


def save_models(models_data, output_dir, save_for_lite_interpreter=False):
    """Save the torchscript models and their data"""
    os.makedirs(output_dir, exist_ok=True)
    for k, (model, data) in models_data.items():
        model_file = os.path.join(output_dir, f"{k}.jit")
        logger.info(f"Saving {k} model to {model_file} ...")
        if save_for_lite_interpreter:
            model._save_for_lite_interpreter(model_file)
        else:
            model.save(model_file)

        data_file = os.path.join(output_dir, f"{k}_data.pth")
        logger.info(f"Saving {k} data to {data_file} ...")
        torch.save(data, data_file)

        oplist_file = os.path.join(output_dir, f"{k}_oplist.txt")
        oplist = torch.jit.export_opnames(model)
        logger.info(f"Saving {k} op_list to {oplist_file} ...")
        with open(oplist_file, "w") as f:
            for op in oplist:
                f.write(f"{op}\n")


def run(args, script_args):
    cfg, output_dir, runner = prepare_for_launch(args)
    cfg = copy.deepcopy(cfg)
    setup_after_launch(cfg, output_dir, runner)
    with temp_defrost(cfg):
        cfg.merge_from_list(["MODEL.DEVICE", "cpu"])

    # build traceable detection model
    detector = runner.build_traceable_model(cfg)

    # prepare data
    dataset = cfg.DATASETS.TEST[0]
    data_loader = runner.build_detection_test_loader(cfg, dataset)

    # run quantization
    quant_detector = quantize(detector, cfg, data_loader)

    # build tracker
    # NOTE: the detector could be modified by the traceable tracker but
    # these modifications (e.g., patch roi_head) has already been done
    # to the detector, it should be the same
    quant_tracker = TraceableTracker(cfg, quant_detector._wrapped_model)

    # build traced models
    traced_models_data = trace(quant_detector, quant_tracker, data_loader)

    # run mobile optimization
    if script_args.mobile:
        logger.info("Running mobile optimization ...")
        traced_models_data = {
            k: (optimize_for_mobile(model), data)
            for k, (model, data) in traced_models_data.items()
        }

    # save models
    save_models(traced_models_data, args.output_dir, script_args.mobile)


def parse_args(args_list):
    parser = basic_argument_parser(distributed=False)
    parser_script = argparse.ArgumentParser(
        description="PyTorch Object Detection Training"
    )
    parser_script.add_argument("--mobile", action="store_true")
    script_args, args_list = parser_script.parse_known_args(args_list)
    args = parser.parse_args(args_list)
    return args, script_args


if __name__ == "__main__":
    args, script_args = parse_args(sys.argv[1:])
    run(args, script_args)
