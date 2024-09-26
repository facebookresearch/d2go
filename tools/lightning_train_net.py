#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved


import logging
import os
from typing import Any, Dict, List, Type, Union

import mobile_cv.torch.utils_pytorch.comm as comm
import pytorch_lightning as pl  # type: ignore
from d2go.config import CfgNode
from d2go.runner.callbacks.quantization import QuantizationAwareTraining
from d2go.runner.lightning_task import DefaultTask
from d2go.setup import basic_argument_parser, prepare_for_launch, setup_after_launch
from d2go.trainer.api import TestNetOutput, TrainNetOutput
from d2go.trainer.helper import parse_precision_from_string
from d2go.trainer.lightning.training_loop import _do_test, _do_train
from detectron2.utils.file_io import PathManager
from pytorch_lightning.callbacks import Callback, LearningRateMonitor, TQDMProgressBar
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.strategies.ddp import DDPStrategy
from torch.distributed import get_rank


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("detectron2go.lightning.train_net")

FINAL_MODEL_CKPT = f"model_final{ModelCheckpoint.FILE_EXTENSION}"


def _get_trainer_callbacks(cfg: CfgNode) -> List[Callback]:
    """Gets the trainer callbacks based on the given D2Go Config.

    Args:
        cfg: The normalized ConfigNode for this D2Go Task.

    Returns:
        A list of configured Callbacks to be used by the Lightning Trainer.
    """
    callbacks: List[Callback] = [
        TQDMProgressBar(refresh_rate=10),  # Arbitrary refresh_rate.
        LearningRateMonitor(logging_interval="step"),
        ModelCheckpoint(
            dirpath=cfg.OUTPUT_DIR,
            save_last=True,
        ),
    ]
    if cfg.QUANTIZATION.QAT.ENABLED:
        callbacks.append(QuantizationAwareTraining.from_config(cfg))
    return callbacks


def _get_strategy(cfg: CfgNode) -> DDPStrategy:
    return DDPStrategy(find_unused_parameters=cfg.MODEL.DDP_FIND_UNUSED_PARAMETERS)


def _get_accelerator(use_cpu: bool) -> str:
    return "cpu" if use_cpu else "gpu"


def get_trainer_params(cfg: CfgNode) -> Dict[str, Any]:
    use_cpu = cfg.MODEL.DEVICE.lower() == "cpu"
    strategy = _get_strategy(cfg)
    accelerator = _get_accelerator(use_cpu)

    params = {
        "max_epochs": -1,
        "max_steps": cfg.SOLVER.MAX_ITER,
        "val_check_interval": (
            cfg.TEST.EVAL_PERIOD if cfg.TEST.EVAL_PERIOD > 0 else cfg.SOLVER.MAX_ITER
        ),
        "num_nodes": comm.get_num_nodes(),
        "devices": comm.get_local_size(),
        "strategy": strategy,
        "accelerator": accelerator,
        "callbacks": _get_trainer_callbacks(cfg),
        "logger": TensorBoardLogger(save_dir=cfg.OUTPUT_DIR),
        "num_sanity_val_steps": 0,
        "replace_sampler_ddp": False,
        "precision": (
            parse_precision_from_string(cfg.SOLVER.AMP.PRECISION, lightning=True)
            if cfg.SOLVER.AMP.ENABLED
            else 32
        ),
    }
    if cfg.SOLVER.CLIP_GRADIENTS.ENABLED:
        if (
            cfg.SOLVER.CLIP_GRADIENTS.CLIP_TYPE.lower() == "norm"
            and cfg.SOLVER.CLIP_GRADIENTS.NORM_TYPE != 2.0
        ):
            raise ValueError(
                "D2Go Lightning backend supports only L2-norm for norm-based gradient clipping!"
            )
        params["gradient_clip_val"] = cfg.SOLVER.CLIP_GRADIENTS.CLIP_VALUE
        params["gradient_clip_algorithm"] = cfg.SOLVER.CLIP_GRADIENTS.CLIP_TYPE

    # Allow specifying additional trainer parameters under `LIGHTNING_TRAINER` field.
    # Please note that:
    #   - the `LIGHTNING_TRAINER`` is not part of "base" config, users need to add this to their default config via `_DEFAULTS_` or `get_default_cfg`.
    #   - this is a temporal solution due to future refactor of config system.
    if hasattr(cfg, "LIGHTNING_TRAINER"):
        params.update(
            {
                "reload_dataloaders_every_n_epochs": cfg.LIGHTNING_TRAINER.RELOAD_DATALOADERS_EVERY_N_EPOCHS,
                "sync_batchnorm": cfg.LIGHTNING_TRAINER.SYNC_BATCHNORM,
                "benchmark": cfg.LIGHTNING_TRAINER.BENCHMARK,
            }
        )

    if (
        hasattr(cfg, "SOLVER.DETERMINISTIC")
        and hasattr(cfg.SOLVER, "DETERMINISTIC")
        and cfg.SOLVER.DETERMINISTIC
    ):
        params.update(
            {
                "sync_batchnorm": True,
                "deterministic": True,
                "replace_sampler_ddp": False,
            }
        )

    return params


def main(
    cfg: CfgNode,
    output_dir: str,
    runner_class: Union[str, Type[DefaultTask]],
    eval_only: bool = False,
) -> Union[TrainNetOutput, TestNetOutput]:
    """Main function for launching a training with lightning trainer
    Args:
        cfg: D2go config node
        num_machines: Number of nodes used for distributed training
        num_processes: Number of processes on each node.
        eval_only: True if run evaluation only.
    """
    task_cls: Type[DefaultTask] = setup_after_launch(cfg, output_dir, runner_class)

    task = task_cls.from_config(cfg, eval_only)
    trainer_params = get_trainer_params(cfg)

    last_checkpoint = os.path.join(cfg.OUTPUT_DIR, "last.ckpt")
    if PathManager.exists(last_checkpoint):
        # resume training from checkpoint
        trainer_params["resume_from_checkpoint"] = last_checkpoint
        logger.info(f"Resuming training from checkpoint: {last_checkpoint}.")

    trainer = pl.Trainer(**trainer_params)

    if eval_only:
        _do_test(trainer, task)
        return TestNetOutput(
            tensorboard_log_dir=trainer_params["logger"].log_dir,
            accuracy=task.eval_res,
            metrics=task.eval_res,
        )
    else:
        model_configs = _do_train(cfg, trainer, task)
        return TrainNetOutput(
            tensorboard_log_dir=trainer_params["logger"].log_dir,
            accuracy=task.eval_res,
            metrics=task.eval_res,
            model_configs=model_configs,
        )


def argument_parser():
    parser = basic_argument_parser(distributed=True, requires_output_dir=False)
    # Change default runner argument
    parser.set_defaults(runner="d2go.runner.lightning_task.GeneralizedRCNNTask")
    parser.add_argument(
        "--eval-only", action="store_true", help="perform evaluation only"
    )
    return parser


if __name__ == "__main__":
    args = argument_parser().parse_args()
    cfg, output_dir, runner_name = prepare_for_launch(args)

    ret = main(
        cfg,
        output_dir,
        runner_name,
        eval_only=args.eval_only,
    )
    if get_rank() == 0:
        print(ret)
