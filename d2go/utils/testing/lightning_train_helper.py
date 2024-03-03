#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import pytorch_lightning as pl  # type: ignore
from detectron2.utils.events import EventStorage
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint


def get_lt_trainer(output_dir: str, cfg):
    checkpoint_callback = ModelCheckpoint(dirpath=output_dir, save_last=True)
    return pl.Trainer(
        max_epochs=10**8,
        max_steps=cfg.SOLVER.MAX_ITER,
        val_check_interval=(
            cfg.TEST.EVAL_PERIOD if cfg.TEST.EVAL_PERIOD > 0 else cfg.SOLVER.MAX_ITER
        ),
        callbacks=[checkpoint_callback],
        logger=False,
    )


def lt_train(task, trainer):
    with EventStorage() as storage:
        task.storage = storage
        trainer.fit(task)


def lt_test(task, trainer):
    with EventStorage() as storage:
        task.storage = storage
        trainer.test(task)
        return task.eval_res
