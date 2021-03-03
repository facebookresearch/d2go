#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved


import logging
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Type

import pytorch_lightning as pl  # type: ignore
from d2go.config import CfgNode, temp_defrost
from d2go.runner import get_class
from d2go.runner.lightning_task import GeneralizedRCNNTask
from d2go.setup import basic_argument_parser
from d2go.utils.misc import dump_trained_model_configs
from detectron2.utils.events import EventStorage
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from stl.lightning.callbacks.model_checkpoint import ModelCheckpoint
from stl.lightning.callbacks.quantization import QuantizationAwareTraining
from stl.lightning.io.filesystem import get_filesystem
from stl.lightning.loggers import ManifoldTensorBoardLogger
from stl.lightning.utilities.manifold import manifold_uri_to_bucket_and_path
from torch.distributed import get_rank


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("detectron2go.lightning.train_net")

FINAL_MODEL_CKPT = f"model_final{ModelCheckpoint.FILE_EXTENSION}"


@dataclass
class TrainOutput:
    output_dir: str
    accuracy: Optional[Dict[str, Any]] = None
    tensorboard_log_dir: Optional[str] = None
    model_configs: Optional[Dict[str, str]] = None


def get_tb_logger(output_dir: str) -> TensorBoardLogger:
    """Stores tensorboard outputs in output_dir."""
    if output_dir.startswith("manifold://"):
        bucket, path = manifold_uri_to_bucket_and_path(output_dir)
        return ManifoldTensorBoardLogger(manifold_bucket=bucket, manifold_path=path)
    return TensorBoardLogger(save_dir=output_dir)


def maybe_override_output_dir(cfg: CfgNode, output_dir: Optional[str]) -> None:
    if output_dir is not None and output_dir != cfg.OUTPUT_DIR:
        cfg.OUTPUT_DIR = output_dir
        logger.warning(
            f"Override cfg.OUTPUT_DIR ({cfg.OUTPUT_DIR}) to be the same as "
            f"output_dir {output_dir}"
        )


def _get_trainer_callbacks(cfg: CfgNode) -> List[Callback]:
    """Gets the trainer callbacks based on the given D2Go Config.

    Args:
        cfg: The normalized ConfigNode for this D2Go Task.

    Returns:
        A list of configured Callbacks to be used by the Lightning Traininer.
    """
    callbacks: List[Callback] = [
        LearningRateMonitor(logging_interval="step"),
        ModelCheckpoint(
            directory=cfg.OUTPUT_DIR,
            has_user_data=False,
            save_top_k=-1,
            every_n_epochs=-1,
            every_n_steps=cfg.SOLVER.CHECKPOINT_PERIOD,
            file_name_template="{step}",
            save_last=True,
        ),
    ]
    if cfg.QUANTIZATION.QAT.ENABLED:
        qat = cfg.QUANTIZATION.QAT
        callbacks.append(
            QuantizationAwareTraining(
                qconfig_dicts={
                    submodule: None for submodule in cfg.QUANTIZATION.MODULES
                }
                if cfg.QUANTIZATION.MODULES
                else None,
                start_step=qat.START_ITER,
                enable_observer=(qat.ENABLE_OBSERVER_ITER, qat.DISABLE_OBSERVER_ITER),
                freeze_bn_step=qat.FREEZE_BN_ITER,
            )
        )
    return callbacks


def main(
    cfg: CfgNode,
    output_dir: Optional[str] = None,
    task_cls: Type[GeneralizedRCNNTask] = GeneralizedRCNNTask,
    eval_only: bool = False,
    num_machines: int = 1,
    num_gpus: int = 0,
    num_processes: int = 1,
    accelerator: Optional[str] = "ddp",
) -> TrainOutput:
    """Main function for launching a training with lightning trainer
    Args:
        cfg: D2go config node
        num_machines: Number of nodes used for distributed training
        num_gpus: Number of GPUs to train on each node
        num_processes: Number of processes on each node.
            NOTE: Automatically set to the number of GPUs when using DDP.
            Set a value greater than 1 to mimic distributed training on CPUs.
        accelerator: Backend for distributed training. Only DDP
            and DPP_CPU are supported.
        eval_only: True if run evaluation only.
    """
    assert (
        num_processes == 1 or num_gpus == 0
    ), "Only set num_processes > 1 when training on CPUs"

    maybe_override_output_dir(cfg, output_dir)

    if cfg.MODEL.WEIGHTS:
        # only load model weights from checkpoint
        task = task_cls.load_from_checkpoint(cfg.MODEL.WEIGHTS, cfg=cfg)
        logger.info(f"Load model weights from checkpoint: {cfg.MODEL.WEIGHTS}.")
    else:
        task = task_cls(cfg)

    tb_logger = get_tb_logger(cfg.OUTPUT_DIR)
    trainer_params = {
        # training loop is bounded by max steps, use a large max_epochs to make
        # sure max_steps is met first
        "max_epochs": 10 ** 8,
        "max_steps": cfg.SOLVER.MAX_ITER,
        "val_check_interval": cfg.TEST.EVAL_PERIOD
        if cfg.TEST.EVAL_PERIOD > 0
        else cfg.SOLVER.MAX_ITER,
        "num_nodes": num_machines,
        "gpus": num_gpus,
        "num_processes": num_processes,
        "accelerator": accelerator,
        "callbacks": _get_trainer_callbacks(cfg),
        "logger": tb_logger,
        "num_sanity_val_steps": 0,
        "progress_bar_refresh_rate": 10,
    }

    last_checkpoint = os.path.join(cfg.OUTPUT_DIR, "last.ckpt")
    if get_filesystem(cfg.OUTPUT_DIR).exists(last_checkpoint):
        # resume training from checkpoint
        trainer_params["resume_from_checkpoint"] = last_checkpoint
        logger.info(f"Resuming training from checkpoint: {last_checkpoint}.")

    # pyre-fixme[16]: Module `pl` has no attribute `Trainer`.
    trainer = pl.Trainer(**trainer_params)
    # TODO: find a better place for event storage
    with EventStorage() as storage:
        task.storage = storage
        model_configs = None
        if eval_only:
            logger.info(
                f"start to evaluate with {num_machines} nodes and {num_gpus} GPUs"
            )
            trainer.test(task)
        else:
            logger.info(f"start to train with {num_machines} nodes and {num_gpus} GPUs")
            trainer.fit(task)
            final_ckpt = os.path.join(cfg.OUTPUT_DIR, FINAL_MODEL_CKPT)
            trainer.save_checkpoint(final_ckpt) # for validation monitor

            trained_cfg = cfg.clone()
            with temp_defrost(trained_cfg):
                trained_cfg.MODEL.WEIGHTS = final_ckpt
            model_configs = dump_trained_model_configs(cfg.OUTPUT_DIR, {"model_final": trained_cfg})

    tb_log_dir = (
        tb_logger.output_dir
        if isinstance(tb_logger, ManifoldTensorBoardLogger)
        else tb_logger.log_dir
    )
    return TrainOutput(
        output_dir=cfg.OUTPUT_DIR,
        tensorboard_log_dir=tb_log_dir,
        accuracy=task.eval_res,
        model_configs=model_configs
    )


def build_config(
    config_file: str,
    task_cls: Type[GeneralizedRCNNTask],
    opts: Optional[List[str]] = None,
) -> CfgNode:
    """Build config node from config file
    Args:
        config_file: Path to a D2go config file
        output_dir: When given, this will override the OUTPUT_DIR in the config
        opts: A list of config overrides. e.g. ["SOLVER.IMS_PER_BATCH", "2"]
    """
    cfg = task_cls.get_default_cfg()
    cfg.merge_from_file(config_file)

    if opts:
        cfg.merge_from_list(opts)
    return cfg


def argument_parser():
    parser = basic_argument_parser(distributed=True, requires_output_dir=False)
    parser.add_argument(
        "--num-gpus", type=int, default=0, help="number of GPUs per machine"
    )
    return parser


if __name__ == "__main__":
    args = argument_parser().parse_args()
    task_cls = get_class(args.runner) if args.runner else GeneralizedRCNNTask
    cfg = build_config(args.config_file, task_cls, args.opts)
    ret = main(
        cfg,
        args.output_dir,
        task_cls,
        eval_only=False,  # eval_only
        num_machines=args.num_machines,
        num_gpus=args.num_gpus,
        num_processes=args.num_processes,
        accelerator="ddp" if args.num_gpus > 0 else "ddp_cpu",
    )
    if get_rank() == 0:
        print(ret)
