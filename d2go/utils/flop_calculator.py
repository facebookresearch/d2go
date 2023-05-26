#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import copy
import logging
import os
import traceback

import detectron2.utils.comm as comm
import mobile_cv.lut.lib.pt.flops_utils as flops_utils
import torch
from d2go.utils.helper import run_once
from detectron2.utils.analysis import FlopCountAnalysis
from detectron2.utils.file_io import PathManager
from detectron2.utils.registry import Registry
from fvcore.nn import flop_count_str, flop_count_table


PROFILER_REGISTRY = Registry("PROFILER")


logger = logging.getLogger(__name__)


@torch.no_grad()
def dump_flops_info(model, inputs, output_dir, use_eval_mode=True):
    """
    Dump flops information about model, using the given model inputs.
    Information are dumped to output_dir using various flop counting tools
    in different formats. Only a simple table is printed to terminal.

    Args:
        inputs: a tuple of positional arguments used to call model with.
        use_eval_mode: turn the model into eval mode for flop counting. Otherwise,
            will use the original mode. It's recommended to use eval mode, because
            training mode typically follows a different codepath.
    """
    if not comm.is_main_process():
        return
    logger.info("Evaluating model's number of parameters and FLOPS")

    try:
        model = copy.deepcopy(model)
    except Exception:
        logger.info("Failed to deepcopy the model and skip FlopsEstimation.")
        return

    # Delete other forward_pre_hooks so they are not simultaneously called.
    # The keys are wrapped in a list to avoid mutating ordered_dict during iteration.
    # See https://github.com/pytorch/pytorch/issues/49739 for more details.
    for hook_key in list(model._forward_pre_hooks.keys()):
        logger.warning(f"Forward hook with key {hook_key} was removed in flop counter.")
        model._forward_pre_hooks.pop(hook_key)

    if use_eval_mode:
        model.eval()
    inputs = copy.deepcopy(inputs)

    # 1. using mobile_cv flop counter
    try:
        fest = flops_utils.FlopsEstimation(model)
        with fest.enable():
            model(*inputs)
            fest.add_flops_info()
            model_str = str(model)
        output_file = os.path.join(output_dir, "flops_str_mobilecv.txt")
        with PathManager.open(output_file, "w") as f:
            f.write(model_str)
            logger.info(f"Flops info written to {output_file}")
    except Exception:
        logger.exception("Failed to estimate flops using mobile_cv's FlopsEstimation")

    # 2. using d2/fvcore's flop counter
    output_file = os.path.join(output_dir, "flops_str_fvcore.txt")
    try:
        flops = FlopCountAnalysis(model, inputs)

        # 2.1: dump as model str
        model_str = flop_count_str(flops)
        with PathManager.open(output_file, "w") as f:
            f.write(model_str)
            logger.info(f"Flops info written to {output_file}")

        # 2.2: dump as table
        flops_table = flop_count_table(flops, max_depth=10)
        output_file = os.path.join(output_dir, "flops_table_fvcore.txt")
        with PathManager.open(output_file, "w") as f:
            f.write(flops_table)
            logger.info(f"Flops table (full version) written to {output_file}")

        # 2.3: print a table with a shallow depth
        flops_table = flop_count_table(flops, max_depth=3)
        logger.info("Flops table:\n" + flops_table)
    except Exception:
        with PathManager.open(output_file, "w") as f:
            traceback.print_exc(file=f)
        logger.warning(
            "Failed to estimate flops using detectron2's FlopCountAnalysis. "
            f"Error written to {output_file}."
        )
        flops = float("nan")
    return flops


def add_flop_printing_hook(
    model,
    output_dir: str,
):
    """
    Add a pytorch module forward hook that will print/save flops of the whole model
    at the first time the model is called.

    Args:
        output_dir: directory to save more detailed flop info
    """

    def hook(module, input):
        handle.remove()
        dump_flops_info(module, input, output_dir)
        return input

    handle = model.register_forward_pre_hook(hook)


@PROFILER_REGISTRY.register()
def default_flop_counter(model, cfg):
    from torch.distributed.fsdp.fully_sharded_data_parallel import (
        FullyShardedDataParallel as FSDP,
    )

    # TODO: deepcopy() not supported for FSDP yet (https://github.com/pytorch/pytorch/issues/82070), so we disable flop counter for now
    if isinstance(model, FSDP):
        logger.warn(
            "Default flop counter is disabled because it's not supported for FSDP yet. "
        )
        return

    return add_flop_printing_hook(model, cfg.OUTPUT_DIR)


# NOTE: the logging can be too long and messsy when printing flops multiple
# times, especially when running eval during training, thus using `run_once`
# to limit it. `dump_flops_info` can log flops more concisely.
@run_once()
def add_print_flops_callback(cfg, model, disable_after_callback=True):
    def _print_flops_callback(self, model, model_data):
        self.add_flops_info()
        logger.info("Callback: model flops info:\n{}".format(model))

        def _guess_batch_size():
            # Inputs are meta-arch dependent, the most general solution will be
            # adding a function like `get_batch_size()` to each meta arch
            ret = 1
            try:
                model_input_shapes = model_data(model)["input_shapes"]
                assert isinstance(model_input_shapes, list)
                assert len(model_input_shapes) > 0
                # assuming the first input is a list of images
                ret = len(model_input_shapes[0])
            except Exception:
                ret = cfg.SOLVER.IMS_PER_BATCH // comm.get_world_size()
                logger.warning(
                    "Could not get batch size, compute from"
                    f" `cfg.SOLVER.IMS_PER_BATCH`={ret}"
                )
                pass

            return ret

        nparams, nflops = self.get_flops()
        batch_size = _guess_batch_size()
        nflops_single = nflops / batch_size
        logger.info(
            f"Model parameters (M): {nparams}, "
            f"MFlops (batch_size={batch_size}): {nflops}, "
            f"MFlops (batch_size=1): {nflops_single}"
        )

        if disable_after_callback:
            self.set_enable(False)

    fest = flops_utils.FlopsEstimation(model).set_callback(_print_flops_callback)
    logger.info("Added callback to log flops info after the first inference")
    fest.set_enable(True)
    return fest


def attach_profiler(profiler_name):
    return PROFILER_REGISTRY.get(profiler_name)


def attach_profilers(cfg, model):
    for profiler in cfg.PROFILERS:
        attach_profiler(profiler)(model, cfg)
