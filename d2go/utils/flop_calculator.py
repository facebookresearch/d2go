#!/usr/bin/env python3
import copy
import logging

import detectron2.utils.comm as comm
import mobile_cv.lut.lib.pt.flops_utils as flops_utils
from d2go.utils.helper import run_once


logger = logging.getLogger(__name__)


def print_flops(model, first_batch):
    logger.info("Evaluating model's number of parameters and FLOPS")
    model_flops = copy.deepcopy(model)
    model_flops.eval()
    fest = flops_utils.FlopsEstimation(model_flops)
    with fest.enable():
        model_flops(first_batch)
        fest.add_flops_info()
        model_str = str(model_flops)
        logger.info(model_str)
    return model_str


# NOTE: the logging can be too long and messsy when printing flops multiple
# times, especially when running eval during training, thus using `run_once`
# to limit it. TODO: log the flops more concisely.
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
