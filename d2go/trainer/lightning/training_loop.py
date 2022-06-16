#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import os
from typing import Dict

import pytorch_lightning as pl
from d2go.config import CfgNode, temp_defrost
from d2go.runner.lightning_task import GeneralizedRCNNTask
from d2go.utils.misc import dump_trained_model_configs
from detectron2.utils.events import EventStorage
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint


FINAL_MODEL_CKPT = f"model_final{ModelCheckpoint.FILE_EXTENSION}"


def _do_train(
    cfg: CfgNode, trainer: pl.Trainer, task: GeneralizedRCNNTask
) -> Dict[str, str]:
    """Runs the training loop with given trainer and task.

    Args:
        cfg: The normalized ConfigNode for this D2Go Task.
        trainer: PyTorch Lightning trainer.
        task: Lightning module instance.

    Returns:
        A map of model name to trained model config path.
    """
    with EventStorage() as storage:
        task.storage = storage
        trainer.fit(task)
        final_ckpt = os.path.join(cfg.OUTPUT_DIR, FINAL_MODEL_CKPT)
        trainer.save_checkpoint(final_ckpt)  # for validation monitor

        trained_cfg = cfg.clone()
        with temp_defrost(trained_cfg):
            trained_cfg.MODEL.WEIGHTS = final_ckpt
        model_configs = dump_trained_model_configs(
            cfg.OUTPUT_DIR, {"model_final": trained_cfg}
        )
    return model_configs


def _do_test(trainer: pl.Trainer, task: GeneralizedRCNNTask):
    """Runs the evaluation with a pre-trained model.

    Args:
        cfg: The normalized ConfigNode for this D2Go Task.
        trainer: PyTorch Lightning trainer.
        task: Lightning module instance.

    """
    with EventStorage() as storage:
        task.storage = storage
        trainer.test(task)
