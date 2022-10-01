#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved


import os
import unittest
from copy import deepcopy
from tempfile import TemporaryDirectory
from typing import Dict

import pytorch_lightning as pl  # type: ignore
import torch
from d2go.config import CfgNode, temp_defrost
from d2go.quantization.modeling import set_backend_and_create_qconfig
from d2go.registry.builtin import META_ARCH_REGISTRY
from d2go.runner import create_runner
from d2go.runner.callbacks.quantization import QuantizationAwareTraining
from d2go.runner.lightning_task import GeneralizedRCNNTask
from d2go.utils.testing import meta_arch_helper as mah
from d2go.utils.testing.helper import tempdir
from detectron2.utils.events import EventStorage
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from torch import Tensor
from torch.ao.quantization.quantize_fx import convert_fx, prepare_qat_fx


class TestLightningTask(unittest.TestCase):
    def _get_cfg(self, tmp_dir: str) -> CfgNode:
        cfg = mah.create_detection_cfg(GeneralizedRCNNTask, tmp_dir)
        cfg.TEST.EVAL_PERIOD = cfg.SOLVER.MAX_ITER
        return cfg

    def _get_trainer(self, output_dir: str) -> pl.Trainer:
        checkpoint_callback = ModelCheckpoint(dirpath=output_dir, save_last=True)
        return pl.Trainer(
            max_steps=1,
            limit_train_batches=1,
            num_sanity_val_steps=0,
            callbacks=[checkpoint_callback],
            logger=False,
        )

    def _compare_state_dict(
        self, state1: Dict[str, Tensor], state2: Dict[str, Tensor]
    ) -> bool:
        if state1.keys() != state2.keys():
            return False

        for k in state1:
            if not torch.allclose(state1[k], state2[k]):
                return False
        return True

    @tempdir
    def test_load_from_checkpoint(self, tmp_dir) -> None:
        task = GeneralizedRCNNTask(self._get_cfg(tmp_dir))

        trainer = self._get_trainer(tmp_dir)
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

    @tempdir
    def test_train_ema(self, tmp_dir):
        cfg = self._get_cfg(tmp_dir)
        cfg.MODEL_EMA.ENABLED = True
        cfg.MODEL_EMA.DECAY = 0.7
        task = GeneralizedRCNNTask(cfg)
        init_state = deepcopy(task.model.state_dict())

        trainer = self._get_trainer(tmp_dir)
        with EventStorage() as storage:
            task.storage = storage
            trainer.fit(task)

        for k, v in task.model.state_dict().items():
            init_state[k].copy_(init_state[k] * 0.7 + 0.3 * v)

        self.assertTrue(
            self._compare_state_dict(init_state, task.ema_state.state_dict())
        )

    @tempdir
    def test_load_ema_weights(self, tmp_dir) -> None:
        cfg = self._get_cfg(tmp_dir)
        cfg.MODEL_EMA.ENABLED = True
        task = GeneralizedRCNNTask(cfg)
        trainer = self._get_trainer(tmp_dir)
        with EventStorage() as storage:
            task.storage = storage
            trainer.fit(task)

        # load EMA weights from checkpoint
        task2 = GeneralizedRCNNTask.load_from_checkpoint(
            os.path.join(tmp_dir, "last.ckpt")
        )
        self.assertTrue(
            self._compare_state_dict(
                task.ema_state.state_dict(), task2.ema_state.state_dict()
            )
        )

        # apply EMA weights to model
        task2.ema_state.apply_to(task2.model)
        self.assertTrue(
            self._compare_state_dict(
                task.ema_state.state_dict(), task2.model.state_dict()
            )
        )

    @tempdir
    def test_ema_eval_only_mode(self, tmp_dir: TemporaryDirectory) -> None:
        """Train one model for one iteration, then check if the
        second task is loaded correctly from config and applied to model.x"""
        cfg = self._get_cfg(tmp_dir)
        cfg.MODEL.MODELING_HOOKS = ["EMA"]
        cfg.MODEL_EMA.ENABLED = True

        task = GeneralizedRCNNTask(cfg)
        trainer = self._get_trainer(tmp_dir)
        with EventStorage() as storage:
            task.storage = storage
            trainer.fit(task)

        # load EMA weights from checkpoint
        cfg2 = self._get_cfg(tmp_dir)
        cfg2.MODEL.MODELING_HOOKS = ["EMA"]
        cfg2.MODEL_EMA.ENABLED = True
        cfg2.MODEL_EMA.USE_EMA_WEIGHTS_FOR_EVAL_ONLY = True
        cfg2.MODEL.WEIGHTS = os.path.join(tmp_dir, "last.ckpt")

        task2 = GeneralizedRCNNTask.from_config(cfg2)

        self.assertTrue(task2.ema_state, "EMA state is not loaded from checkpoint.")
        self.assertTrue(
            len(task2.ema_state.state_dict()) > 0, "EMA state should not be empty."
        )
        self.assertTrue(
            self._compare_state_dict(
                task.ema_state.state_dict(), task2.model.state_dict()
            ),
            "Task loaded from config should apply the ema_state to the model.",
        )

    def test_create_runner(self):
        task_cls = create_runner(
            f"{GeneralizedRCNNTask.__module__}.{GeneralizedRCNNTask.__name__}"
        )
        self.assertTrue(task_cls == GeneralizedRCNNTask)

    @tempdir
    def test_build_model(self, tmp_dir):
        cfg = self._get_cfg(tmp_dir)
        cfg.MODEL_EMA.ENABLED = True
        task = GeneralizedRCNNTask(cfg)
        trainer = self._get_trainer(tmp_dir)

        with EventStorage() as storage:
            task.storage = storage
            trainer.fit(task)

        # test building untrained model
        model = GeneralizedRCNNTask.build_model(cfg)
        self.assertTrue(model.training)

        # test loading regular weights
        with temp_defrost(cfg):
            cfg.MODEL.WEIGHTS = os.path.join(tmp_dir, "last.ckpt")
            model = GeneralizedRCNNTask.build_model(cfg, eval_only=True)
            self.assertFalse(model.training)
            self.assertTrue(
                self._compare_state_dict(model.state_dict(), task.model.state_dict())
            )

        # test loading EMA weights
        with temp_defrost(cfg):
            cfg.MODEL.WEIGHTS = os.path.join(tmp_dir, "last.ckpt")
            cfg.MODEL_EMA.USE_EMA_WEIGHTS_FOR_EVAL_ONLY = True
            model = GeneralizedRCNNTask.build_model(cfg, eval_only=True)
            self.assertFalse(model.training)
            self.assertTrue(
                self._compare_state_dict(
                    model.state_dict(), task.ema_state.state_dict()
                )
            )

    @tempdir
    def test_qat(self, tmp_dir):
        @META_ARCH_REGISTRY.register()
        class QuantizableDetMetaArchForTest(mah.DetMetaArchForTest):
            custom_config_dict = {"preserved_attributes": ["preserved_attr"]}

            def __init__(self, cfg):
                super().__init__(cfg)
                self.avgpool.preserved_attr = "foo"
                self.avgpool.not_preserved_attr = "bar"

            def custom_prepare_fx(self, cfg, is_qat, example_input=None):
                example_inputs = (torch.rand(1, 3, 3, 3),)
                self.avgpool = prepare_qat_fx(
                    self.avgpool,
                    {"": set_backend_and_create_qconfig(cfg, is_train=self.training)},
                    example_inputs,
                    self.custom_config_dict,
                )

                def convert_fx_callback(model):
                    model.avgpool = convert_fx(
                        model.avgpool, convert_custom_config=model.custom_config_dict
                    )
                    return model

                return self, convert_fx_callback

        cfg = self._get_cfg(tmp_dir)
        cfg.MODEL.META_ARCHITECTURE = "QuantizableDetMetaArchForTest"
        cfg.QUANTIZATION.QAT.ENABLED = True
        cfg.QUANTIZATION.EAGER_MODE = False
        task = GeneralizedRCNNTask(cfg)

        callbacks = [
            QuantizationAwareTraining.from_config(cfg),
            ModelCheckpoint(dirpath=task.cfg.OUTPUT_DIR, save_last=True),
        ]
        trainer = pl.Trainer(
            max_steps=1,
            limit_train_batches=1,
            num_sanity_val_steps=0,
            callbacks=callbacks,
            logger=False,
        )
        with EventStorage() as storage:
            task.storage = storage
            trainer.fit(task)
        prepared_avgpool = task._prepared.model.avgpool
        self.assertEqual(prepared_avgpool.preserved_attr, "foo")
        self.assertFalse(hasattr(prepared_avgpool, "not_preserved_attr"))

        with temp_defrost(cfg):
            cfg.MODEL.WEIGHTS = os.path.join(tmp_dir, "last.ckpt")
            model = GeneralizedRCNNTask.build_model(cfg, eval_only=True)
            self.assertTrue(isinstance(model.avgpool, torch.fx.GraphModule))

    @tempdir
    def test_meta_arch_training_step(self, tmp_dir):
        @META_ARCH_REGISTRY.register()
        class DetMetaArchForWithTrainingStep(mah.DetMetaArchForTest):
            def training_step(self, batch, batch_idx, opt, manual_backward):
                assert batch
                assert opt
                assert manual_backward
                # We step the optimizer for progress tracking to occur
                # This is reflected in the Trainer's global_step property
                # which is used to determine when to stop training
                # when specifying the loop bounds with Trainer(max_steps=N)
                opt.step()
                return {"total_loss": 0.4}

        cfg = self._get_cfg(tmp_dir)
        cfg.MODEL.META_ARCHITECTURE = "DetMetaArchForWithTrainingStep"

        task = GeneralizedRCNNTask(cfg)

        trainer = self._get_trainer(tmp_dir)
        with EventStorage() as storage:
            task.storage = storage
            trainer.fit(task)
