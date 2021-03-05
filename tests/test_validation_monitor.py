#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved


import unittest
from pathlib import Path

from d2go.utils.validation_monitor import fetch_checkpoints_till_final
from detectron2.utils.file_io import PathManager
from mobile_cv.common.misc.file_utils import make_temp_directory
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint


def create_file(filename):
    with PathManager.open(filename, "w") as _:
        pass


class TestValidationMonitor(unittest.TestCase):
    def test_fetch_checkpoints_local(self):
        with make_temp_directory("test") as output_dir:
            output_dir = Path(output_dir)
            for i in range(5):
                create_file(output_dir / f"model_{i}.pth")
            create_file(output_dir / "model_final.pth")
            checkpoints = list(fetch_checkpoints_till_final(output_dir))
            assert len(checkpoints) == 6

    def test_fetch_lightning_checkpoints_local(self):
        with make_temp_directory("test") as output_dir:
            output_dir = Path(output_dir)
            ext = ModelCheckpoint.FILE_EXTENSION
            for i in range(5):
                create_file(output_dir / f"step={i}{ext}")
            create_file(output_dir / f"model_final{ext}")
            create_file(output_dir / f"{ModelCheckpoint.CHECKPOINT_NAME_LAST}{ext}")
            checkpoints = list(fetch_checkpoints_till_final(output_dir))
            self.assertEqual(len(checkpoints), 6)
