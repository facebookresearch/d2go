#!/usr/bin/env python3
import os
from copy import deepcopy
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import pytorch_lightning as pl
import torch
from d2go.config import CfgNode
from d2go.runner.default_runner import (
    Detectron2GoRunner,
    GeneralizedRCNNRunner,
)
from d2go.setup import setup_after_launch
from detectron2.modeling import build_model
from detectron2.solver import (
    build_lr_scheduler as d2_build_lr_scheduler,
    build_optimizer as d2_build_optimizer,
)
from pytorch_lightning.utilities import rank_zero_info
from stl.lightning.utilities.ema_state import EMAState
from torch import nn


class ModelTag(str, Enum):
    DEFAULT = "default"
    EMA = "ema"


class DefaultTask(pl.LightningModule):
    def __init__(self, cfg: CfgNode):
        super().__init__()
        self.cfg = cfg
        self.model = build_model(cfg)
        self.storage = None
        # evaluators for validation datasets, split by model tag(default, ema),
        # in the order of DATASETS.TEST
        self.dataset_evaluators = {ModelTag.DEFAULT: []}
        self.save_hyperparameters()
        self.eval_res = None

        self.ema_state: Optional[EMAState] = None
        if cfg.MODEL_EMA.ENABLED:
            self.ema_state = EMAState(
                decay=cfg.MODEL_EMA.DECAY,
                device=cfg.MODEL_EMA.DEVICE or cfg.MODEL.DEVICE,
            )
            self.model_ema = deepcopy(self.model)
            self.dataset_evaluators[ModelTag.EMA] = []

    def setup(self, stage: str):
        setup_after_launch(self.cfg, self.cfg.OUTPUT_DIR, runner=None)

    @classmethod
    def get_default_cfg(cls):
        return Detectron2GoRunner.get_default_cfg()

    def training_step(self, batch, batch_idx):
        loss_dict = self.forward(batch)
        losses = sum(loss_dict.values())
        self.storage.step()

        self.log_dict(loss_dict, prog_bar=True)
        return losses

    def validation_step(self, batch, batch_idx: int, dataloader_idx: int = 0) -> None:
        if not isinstance(batch, List):
            batch = [batch]
        outputs = self.forward(batch)
        self.dataset_evaluators[ModelTag.DEFAULT][dataloader_idx].process(
            batch, outputs
        )

        if self.ema_state:
            ema_outputs = self.model_ema(batch)
            self.dataset_evaluators[ModelTag.EMA][dataloader_idx].process(
                batch, ema_outputs
            )

    def _log_dataset_evaluation_results(self) -> None:
        nested_res = {}
        for tag, evaluators in self.dataset_evaluators.items():
            nested_res[tag] = {}
            for idx, evaluator in enumerate(evaluators):
                dataset_name = self.cfg.DATASETS.TEST[idx]
                nested_res[tag][dataset_name] = evaluator.evaluate()

        self.eval_res = nested_res
        flattened = pl.loggers.LightningLoggerBase._flatten_dict(nested_res)
        self.log_dict(flattened)

    def validation_epoch_end(self, _outputs) -> None:
        self._log_dataset_evaluation_results()
        self._reset_dataset_evaluators()

    def configure_optimizers(
        self,
    ) -> Tuple[List[torch.optim.Optimizer], List]:
        optim = d2_build_optimizer(self.cfg, self.model)
        lr_scheduler = d2_build_lr_scheduler(self.cfg, optim)

        return [optim], [{"scheduler": lr_scheduler, "interval": "step"}]

    def train_dataloader(self):
        return Detectron2GoRunner.build_detection_train_loader(self.cfg)

    def _reset_dataset_evaluators(self):
        """reset validation dataset evaluator to be run in EVAL_PERIOD steps"""
        assert (
            not self.trainer.distributed_backend
            or self.trainer.distributed_backend.lower()
            in [
                "ddp",
                "ddp_cpu",
            ]
        ), (
            "Only DDP and DDP_CPU distributed backend are supported"
        )

        def _get_inference_dir_name(
            base_dir, inference_type, dataset_name, model_tag: ModelTag
        ):
            next_eval_iter = self.trainer.global_step + self.cfg.TEST.EVAL_PERIOD
            if self.trainer.global_step == 0:
                next_eval_iter -= 1
            return os.path.join(
                base_dir,
                inference_type,
                model_tag,
                str(next_eval_iter),
                dataset_name,
            )

        for tag, dataset_evaluators in self.dataset_evaluators.items():
            dataset_evaluators.clear()
            assert self.cfg.OUTPUT_DIR, "Expect output_dir to be specified in config"
            for dataset_name in self.cfg.DATASETS.TEST:
                # setup evaluator for each dataset
                output_folder = _get_inference_dir_name(
                    self.cfg.OUTPUT_DIR, "inference", dataset_name, tag
                )
                evaluator = Detectron2GoRunner.get_evaluator(
                    self.cfg, dataset_name, output_folder=output_folder
                )
                evaluator.reset()
                dataset_evaluators.append(evaluator)
                # TODO: add visualization evaluator

    def val_dataloader(self):
        # TODO: Support subsample n images
        assert len(self.cfg.DATASETS.TEST)

        dataloaders = []
        for dataset_name in self.cfg.DATASETS.TEST:
            dataloaders.append(
                Detectron2GoRunner.build_detection_test_loader(self.cfg, dataset_name)
            )

        self._reset_dataset_evaluators()
        return dataloaders

    def forward(self, input):
        return self.model(input)

    def on_pretrain_routine_end(self) -> None:
        if self.cfg.MODEL_EMA.ENABLED:
            if self.ema_state and self.ema_state.has_inited():
                # ema_state could have been loaded from checkpoint
                return
            self.ema_state = EMAState.from_model(
                self.model,
                decay=self.cfg.MODEL_EMA.DECAY,
                device=self.cfg.MODEL_EMA.DEVICE or self.cfg.MODEL.DEVICE,
            )

    def on_train_batch_end(self, outputs, batch, batch_idx, dataloader_idx) -> None:
        if self.ema_state:
            self.ema_state.update(self.model)

    def on_validation_epoch_start(self):
        if self.ema_state:
            self.ema_state.apply_to(self.model_ema)

    def on_save_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        if self.ema_state:
            checkpoint["model_ema"] = self.ema_state.state_dict()

    def on_load_checkpoint(self, checkpointed_state: Dict[str, Any]) -> None:
        if self.ema_state:
            if "model_ema" not in checkpointed_state:
                rank_zero_info(
                    "EMA is enabled but EMA state is not found in given checkpoint"
                )
            else:
                self.ema_state = EMAState()
                self.ema_state.load_state_dict(checkpointed_state["model_ema"])
                if not self.ema_state.device:
                    # EMA state device not given, move to module device
                    self.ema_state.to(self.device)


class GeneralizedRCNNTask(DefaultTask):
    @classmethod
    def get_default_cfg(cls):
        return GeneralizedRCNNRunner.get_default_cfg()
