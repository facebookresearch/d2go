#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved


import logging
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Type

import pytorch_lightning as pl  # type: ignore
from d2go.config import auto_scale_world_size, CfgNode, temp_defrost
from d2go.runner import create_runner
from d2go.runner.callbacks.quantization import QuantizationAwareTraining
from d2go.runner.lightning_task import GeneralizedRCNNTask
from d2go.setup import basic_argument_parser, setup_after_launch
from d2go.utils.misc import dump_trained_model_configs
from detectron2.utils.events import EventStorage
from detectron2.utils.file_io import PathManager
from pytorch_lightning.callbacks import Callback, LearningRateMonitor, TQDMProgressBar
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.strategies.ddp import DDPStrategy
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


def get_trainer_params(
    cfg: CfgNode, num_machines: int, num_processes: int
) -> Dict[str, Any]:
    use_cpu = cfg.MODEL.DEVICE.lower() == "cpu"
    strategy = _get_strategy(cfg)
    accelerator = _get_accelerator(use_cpu)

    return {
        "max_epochs": -1,
        "max_steps": cfg.SOLVER.MAX_ITER,
        "val_check_interval": cfg.TEST.EVAL_PERIOD
        if cfg.TEST.EVAL_PERIOD > 0
        else cfg.SOLVER.MAX_ITER,
        "num_nodes": num_machines,
        "devices": num_processes,
        "strategy": strategy,
        "accelerator": accelerator,
        "callbacks": _get_trainer_callbacks(cfg),
        "logger": TensorBoardLogger(save_dir=cfg.OUTPUT_DIR),
        "num_sanity_val_steps": 0,
        "replace_sampler_ddp": False,
    }


def do_train(
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


def do_test(trainer: pl.Trainer, task: GeneralizedRCNNTask):
    """Runs the evaluation with a pre-trained model.

    Args:
        cfg: The normalized ConfigNode for this D2Go Task.
        trainer: PyTorch Lightning trainer.
        task: Lightning module instance.

    """
    with EventStorage() as storage:
        task.storage = storage
        trainer.test(task)


def main(
    cfg: CfgNode,
    output_dir: str,
    task_cls: Type[GeneralizedRCNNTask] = GeneralizedRCNNTask,
    eval_only: bool = False,
    num_machines: int = 1,
    num_processes: int = 1,
) -> TrainOutput:
    """Main function for launching a training with lightning trainer
    Args:
        cfg: D2go config node
        num_machines: Number of nodes used for distributed training
        num_processes: Number of processes on each node.
        eval_only: True if run evaluation only.
    """
    # FIXME: make comm.get_world_size() work properly.
    setup_after_launch(cfg, output_dir, _scale_world_size=False)
    auto_scale_world_size(cfg, new_world_size=num_machines * num_processes)

    task = task_cls.from_config(cfg, eval_only)
    trainer_params = get_trainer_params(cfg, num_machines, num_processes)

    last_checkpoint = os.path.join(cfg.OUTPUT_DIR, "last.ckpt")
    if PathManager.exists(last_checkpoint):
        # resume training from checkpoint
        trainer_params["resume_from_checkpoint"] = last_checkpoint
        logger.info(f"Resuming training from checkpoint: {last_checkpoint}.")

    trainer = pl.Trainer(**trainer_params)
    model_configs = None
    if eval_only:
        do_test(trainer, task)
    else:
        model_configs = do_train(cfg, trainer, task)

    return TrainOutput(
        output_dir=cfg.OUTPUT_DIR,
        tensorboard_log_dir=trainer_params["logger"].log_dir,
        accuracy=task.eval_res,
        model_configs=model_configs,
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
    task_cls = create_runner(args.runner) if args.runner else GeneralizedRCNNTask
    cfg = build_config(args.config_file, task_cls, args.opts)
    ret = main(
        cfg,
        args.output_dir,
        task_cls,
        eval_only=False,  # eval_only
        num_machines=args.num_machines,
        num_processes=args.num_processes,
    )
    if get_rank() == 0:
        print(ret)
