#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved


import logging
import os
import re
import time

from detectron2.checkpoint import DetectionCheckpointer
from detectron2.utils.file_io import PathManager
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint


logger = logging.getLogger(__name__)


def fetch_checkpoints_till_final(checkpoint_dir):
    """
    A generator that yields all checkpoint paths under the given directory, it'll
        keep refreshing until model_final is found.
    """

    MIN_SLEEP_INTERVAL = 1.0  # in seconds
    MAX_SLEEP_INTERVAL = 60.0  # in seconds
    sleep_interval = MIN_SLEEP_INTERVAL

    finished_checkpoints = set()

    def _add_and_log(path):
        finished_checkpoints.add(path)
        logger.info("Found checkpoint: {}".format(path))
        return path

    def _log_and_sleep(sleep_interval):
        logger.info(
            "Sleep {} seconds while waiting for model_final.pth".format(sleep_interval)
        )
        time.sleep(sleep_interval)
        return min(sleep_interval * 2, MAX_SLEEP_INTERVAL)

    def _get_lightning_checkpoints(path: str):
        return [
            os.path.join(path, x)
            for x in PathManager.ls(path)
            if x.endswith(ModelCheckpoint.FILE_EXTENSION)
            and not x.startswith(ModelCheckpoint.CHECKPOINT_NAME_LAST)
        ]

    while True:
        if not PathManager.exists(checkpoint_dir):
            sleep_interval = _log_and_sleep(sleep_interval)
            continue

        checkpoint_paths = DetectionCheckpointer(
            None, save_dir=checkpoint_dir
        ).get_all_checkpoint_files()

        checkpoint_paths = [
            cpt_path
            for cpt_path in checkpoint_paths
            if os.path.basename(cpt_path).startswith("model")
        ]

        checkpoint_paths.extend(_get_lightning_checkpoints(checkpoint_dir))

        final_model_path = None
        periodic_checkpoints = []

        for path in sorted(checkpoint_paths):
            if path.endswith("model_final.pth") or path.endswith("model_final.ckpt"):
                final_model_path = path
                continue

            if path.endswith(ModelCheckpoint.FILE_EXTENSION):
                # Lightning checkpoint
                model_iter = int(
                    re.findall(
                        r"(?<=step=)\d+(?={})".format(ModelCheckpoint.FILE_EXTENSION),
                        path,
                    )[0]
                )
            else:
                model_iter = int(re.findall(r"(?<=model_)\d+(?=\.pth)", path)[0])
            periodic_checkpoints.append((path, model_iter))

        periodic_checkpoints = [
            pc for pc in periodic_checkpoints if pc[0] not in finished_checkpoints
        ]
        periodic_checkpoints = sorted(periodic_checkpoints, key=lambda x: x[1])
        for pc in periodic_checkpoints:
            yield _add_and_log(pc[0])
            sleep_interval = MIN_SLEEP_INTERVAL

        if final_model_path is None:
            sleep_interval = _log_and_sleep(sleep_interval)
        else:
            yield _add_and_log(final_model_path)
            break
