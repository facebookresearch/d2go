#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved


import logging
import os

import detectron2.utils.comm as comm
import torch
from d2go.utils.visualization import VisualizerWrapper
from detectron2.utils.file_io import PathManager


logger = logging.getLogger(__name__)


def get_rel_loss_checker(rel_thres=1.0):
    def _loss_delta_exceeds_thresh(prev_loss, loss):
        if prev_loss is None:
            return True
        prev_sum = sum(prev_loss.values())
        cur_sum = sum(loss.values())
        if prev_sum <= 0:
            return True
        if (cur_sum - prev_sum) / prev_sum >= rel_thres:
            return False
        return True

    return _loss_delta_exceeds_thresh


class TrainImageWriter(object):
    def __init__(self, cfg, tbx_writer, max_count=5):
        """max_count: max number of data written to tensorboard, additional call
        will be ignored
        """
        self.visualizer = VisualizerWrapper(cfg)
        self.writer = tbx_writer
        self.max_count = max_count
        self.counter = 0

    def __call__(self, all_data):
        if self.max_count > 0 and self.counter >= self.max_count:
            return

        data = all_data["data"]
        step = all_data["step"]

        for idx, cur_data in enumerate(data):
            name = f"train_abnormal_losses/{step}/img_{idx}/{cur_data['file_name']}"
            vis_img = self.visualizer.visualize_train_input(cur_data)
            self.writer._writer.add_image(name, vis_img, step, dataformats="HWC")
        logger.warning(
            "Train images with bad losses written to tensorboard 'train_abnormal_losses'"
        )
        self.counter += 1


class FileWriter(object):
    def __init__(self, output_dir, max_count=5):
        """max_count: max number of data written to tensorboard, additional call
        will be ignored
        """
        self.output_dir = output_dir
        self.max_count = max_count
        self.counter = 0

    def __call__(self, all_data):
        if self.max_count > 0 and self.counter >= self.max_count:
            return

        output_dir = self.output_dir
        step = all_data["step"]
        losses = all_data["losses"]

        file_name = f"train_abnormal_losses_{step}_{comm.get_rank()}.pth"
        out_file = os.path.join(output_dir, file_name)
        with PathManager.open(out_file, "wb") as fp:
            torch.save(all_data, fp)
        logger.warning(
            f"Iteration {step} has bad losses {losses}. "
            f"all information saved to {out_file}."
        )
        self.counter += 1


def get_writers(cfg, tbx_writer):
    writers = [TrainImageWriter(cfg, tbx_writer), FileWriter(cfg.OUTPUT_DIR)]
    return writers


class AbnormalLossChecker(object):
    def __init__(self, start_iter, writers, valid_loss_checker=None):
        self.valid_loss_checker = valid_loss_checker or get_rel_loss_checker()
        self.writers = writers or []
        assert isinstance(self.writers, list)

        self.prev_index = start_iter
        self.prev_loss = None

    def check_step(self, losses, data=None, model=None):
        with torch.no_grad():
            is_valid = self.valid_loss_checker(self.prev_loss, losses)
        if not is_valid:
            self._write_invalid_info(losses, data, model)
        self.prev_index += 1
        self.prev_loss = losses
        return is_valid

    def _write_invalid_info(self, losses, data, model):
        all_info = {
            "losses": losses,
            "data": data,
            "model": getattr(model, "module", model),
            "prev_loss": self.prev_loss,
            "step": self.prev_index + 1,
        }

        for writer in self.writers:
            writer(all_info)


class AbnormalLossCheckerWrapper(torch.nn.Module):
    def __init__(self, model, checker):
        super().__init__()
        self.checker = checker
        self.model = model
        self.training = model.training

    def forward(self, x):
        losses = self.model(x)
        self.checker.check_step(losses, data=x, model=self.model)
        return losses
