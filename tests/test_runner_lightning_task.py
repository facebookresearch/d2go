#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved


import os
import tempfile
import unittest
from copy import deepcopy
from typing import Dict

import d2go.runner.default_runner as default_runner
import pytorch_lightning as pl  # type: ignore
import torch
from d2go.config import CfgNode
from d2go.runner.lightning_task import GeneralizedRCNNTask
from detectron2.utils.events import EventStorage
from torch import Tensor

from d2go.tests import meta_arch_helper as mah

OSSRUN = os.getenv('OSSRUN') == '1'

class TestLightningTask(unittest.TestCase):
    def _get_cfg(self, tmp_dir: str) -> CfgNode:
        runner = default_runner.Detectron2GoRunner()
        cfg = mah.create_detection_cfg(runner, tmp_dir)
        cfg.TEST.EVAL_PERIOD = cfg.SOLVER.MAX_ITER
        return cfg

    def _compare_state_dict(
        self, state1: Dict[str, Tensor], state2: Dict[str, Tensor]
    ) -> bool:
        if state1.keys() != state2.keys():
            return False

        for k in state1:
            if not torch.allclose(state1[k], state2[k]):
                return False
        return True

    @unittest.skipIf(OSSRUN, "not supported yet")
    def test_load_from_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            task = GeneralizedRCNNTask(self._get_cfg(tmp_dir))
            from stl.lightning.callbacks.model_checkpoint import ModelCheckpoint
            checkpoint_callback = ModelCheckpoint(
                directory=task.cfg.OUTPUT_DIR, has_user_data=False
            )
            params = {
                "max_steps": 1,
                "limit_train_batches": 1,
                "num_sanity_val_steps": 0,
                "checkpoint_callback": checkpoint_callback,
            }
            trainer = pl.Trainer(**params)
            with EventStorage() as storage:
                task.storage = storage
                trainer.fit(task)
                ckpt_path = os.path.join(tmp_dir, "test.ckpt")
                trainer.save_checkpoint(ckpt_path)
                self.assertTrue(os.path.exists(ckpt_path))

                # load model weights from checkpoint
                task2 = GeneralizedRCNNTask.load_from_checkpoint(ckpt_path)
                self.assertTrue(
                    self._compare_state_dict(
                        task.model.state_dict(), task2.model.state_dict()
                    )
                )

    def test_train_ema(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            cfg = self._get_cfg(tmp_dir)
            cfg.MODEL_EMA.ENABLED = True
            cfg.MODEL_EMA.DECAY = 0.7
            task = GeneralizedRCNNTask(cfg)
            init_state = deepcopy(task.model.state_dict())

            trainer = pl.Trainer(
                max_steps=1,
                limit_train_batches=1,
                num_sanity_val_steps=0,
            )
            with EventStorage() as storage:
                task.storage = storage
                trainer.fit(task)

            for k, v in task.model.state_dict().items():
                init_state[k].copy_(init_state[k] * 0.7 + 0.3 * v)

            self.assertTrue(
                self._compare_state_dict(init_state, task.ema_state.state_dict())
            )

    @unittest.skipIf(OSSRUN, "not supported yet")
    def test_load_ema_weights(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            cfg = self._get_cfg(tmp_dir)
            cfg.MODEL_EMA.ENABLED = True
            task = GeneralizedRCNNTask(cfg)
            from stl.lightning.callbacks.model_checkpoint import ModelCheckpoint
            checkpoint_callback = ModelCheckpoint(
                directory=task.cfg.OUTPUT_DIR, save_last=True
            )

            trainer = pl.Trainer(
                max_steps=1,
                limit_train_batches=1,
                num_sanity_val_steps=0,
                callbacks=[checkpoint_callback],
            )

            with EventStorage() as storage:
                task.storage = storage
                trainer.fit(task)

            # load EMA weights from checkpoint
            task2 = GeneralizedRCNNTask.load_from_checkpoint(os.path.join(tmp_dir, "last.ckpt"))
            self.assertTrue(self._compare_state_dict(task.ema_state.state_dict(), task2.ema_state.state_dict()))

            # apply EMA weights to model
            task2.ema_state.apply_to(task2.model)
            self.assertTrue(self._compare_state_dict(task.ema_state.state_dict(), task2.model.state_dict()))
