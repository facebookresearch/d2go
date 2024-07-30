#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import logging
import os
from copy import deepcopy
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import pytorch_lightning as pl
import torch
from d2go.checkpoint.checkpoint_instrumentation import instrument_checkpoint
from d2go.config import CfgNode
from d2go.data.datasets import inject_coco_datasets, register_dynamic_datasets
from d2go.data.utils import update_cfg_if_using_adhoc_dataset
from d2go.modeling.api import build_meta_arch
from d2go.modeling.model_freezing_utils import set_requires_grad
from d2go.optimizer.build import build_optimizer_mapper
from d2go.runner.api import RunnerV2Mixin
from d2go.runner.callbacks.quantization import maybe_prepare_for_quantization, PREPARED
from d2go.runner.default_runner import (
    _get_tbx_writer,
    D2GoDataAPIMixIn,
    Detectron2GoRunner,
    GeneralizedRCNNRunner,
)
from d2go.utils.ema_state import EMAState
from d2go.utils.misc import get_tensorboard_log_dir
from detectron2.engine.train_loop import HookBase
from detectron2.solver import build_lr_scheduler as d2_build_lr_scheduler
from mobile_cv.common.misc.oss_utils import fb_overwritable
from pytorch_lightning.strategies import DDPStrategy, SingleDeviceStrategy
from pytorch_lightning.utilities import rank_zero_info, rank_zero_only
from pytorch_lightning.utilities.logger import _flatten_dict

_STATE_DICT_KEY = "state_dict"
_OLD_STATE_DICT_KEY = "model"
_OLD_EMA_KEY = "ema_state"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _is_lightning_checkpoint(checkpoint: Dict[str, Any]) -> bool:
    """Returns true if we believe this checkpoint to be a Lightning checkpoint."""
    return _STATE_DICT_KEY in checkpoint


def _is_d2go_checkpoint(checkpoint: Dict[str, Any]) -> bool:
    """Returns true if we believe this to be a D2Go checkpoint."""
    d2_go_keys = [_OLD_STATE_DICT_KEY, "iteration"]
    for key in d2_go_keys:
        if key not in checkpoint:
            return False
    return True


def _convert_to_lightning(d2_checkpoint: Dict[str, Any]) -> None:
    """Converst a D2Go Checkpoint to Lightning in-place by renaming keys."""
    prefix = "model"  # based on DefaultTask.model.
    old_keys = list(d2_checkpoint[_OLD_STATE_DICT_KEY])
    for key in old_keys:
        d2_checkpoint[_OLD_STATE_DICT_KEY][f"{prefix}.{key}"] = d2_checkpoint[
            _OLD_STATE_DICT_KEY
        ][key]
        del d2_checkpoint[_OLD_STATE_DICT_KEY][key]

    if "model.pixel_mean" in d2_checkpoint[_OLD_STATE_DICT_KEY]:
        del d2_checkpoint[_OLD_STATE_DICT_KEY]["model.pixel_mean"]

    if "model.pixel_std" in d2_checkpoint[_OLD_STATE_DICT_KEY]:
        del d2_checkpoint[_OLD_STATE_DICT_KEY]["model.pixel_std"]

    for old, new in zip(
        [_OLD_STATE_DICT_KEY, "iteration"], [_STATE_DICT_KEY, "global_step"]
    ):
        d2_checkpoint[new] = d2_checkpoint[old]
        del d2_checkpoint[old]

    for old, new in zip(
        ["optimizer", "scheduler"], ["optimizer_states", "lr_schedulers"]
    ):
        if old not in d2_checkpoint:
            continue
        d2_checkpoint[new] = [d2_checkpoint[old]]
        del d2_checkpoint[old]

    if _OLD_EMA_KEY in d2_checkpoint:
        d2_checkpoint["model_ema"] = d2_checkpoint[_OLD_EMA_KEY]
        del d2_checkpoint[_OLD_EMA_KEY]

    d2_checkpoint["epoch"] = 0


class ModelTag(str, Enum):
    DEFAULT = "default"
    EMA = "ema"


@fb_overwritable()
def get_gpu_profiler(cfg: CfgNode) -> Optional[HookBase]:
    return None


class DefaultTask(D2GoDataAPIMixIn, pl.LightningModule):
    def __init__(self, cfg: CfgNode):
        super().__init__()
        self.register(cfg)
        self.cfg = cfg
        self.model = self._build_model()
        self.storage = None
        # evaluators for validation datasets, split by model tag(default, ema),
        # in the order of DATASETS.TEST
        self.dataset_evaluators = {ModelTag.DEFAULT: []}
        self.save_hyperparameters()
        self.eval_res = None

        # Support custom training step in meta arch
        if hasattr(self.model, "training_step"):
            # activate manual optimization for custom training step
            self.automatic_optimization = False

        self.ema_state: Optional[EMAState] = None
        if cfg.MODEL_EMA.ENABLED:
            self.ema_state = EMAState(
                decay=cfg.MODEL_EMA.DECAY,
                device=cfg.MODEL_EMA.DEVICE or cfg.MODEL.DEVICE,
            )
            self.dataset_evaluators[ModelTag.EMA] = []
        self.gpu_profiler: Optional[HookBase] = get_gpu_profiler(cfg)

    def _build_model(self) -> torch.nn.Module:
        model = build_meta_arch(self.cfg)

        if self.cfg.MODEL.FROZEN_LAYER_REG_EXP:
            set_requires_grad(model, self.cfg.MODEL.FROZEN_LAYER_REG_EXP, value=False)

        return model

    @classmethod
    def from_config(cls, cfg: CfgNode, eval_only=False):
        """Builds Lightning module including model from config.
        To load weights from a pretrained checkpoint, please specify checkpoint
        path in `MODEL.WEIGHTS`.

        Args:
            cfg: D2go config node.
            eval_only: True if module should be in eval mode.
        """
        if eval_only and not cfg.MODEL.WEIGHTS:
            logger.warning("MODEL.WEIGHTS is missing for eval only mode.")

        if cfg.MODEL.WEIGHTS:
            # only load model weights from checkpoint
            logger.info(f"Load model weights from checkpoint: {cfg.MODEL.WEIGHTS}.")
            task = cls.load_from_checkpoint(cfg.MODEL.WEIGHTS, cfg=cfg, strict=False)
        else:
            task = cls(cfg)

        if cfg.MODEL_EMA.ENABLED and cfg.MODEL_EMA.USE_EMA_WEIGHTS_FOR_EVAL_ONLY:
            assert task.ema_state, "EMA state is not loaded from checkpoint."
            task.ema_state.apply_to(task.model)

        if eval_only:
            task.eval()
        return task

    def training_step(self, batch, batch_idx):
        if hasattr(self.model, "training_step"):
            return self._meta_arch_training_step(batch, batch_idx)

        return self._standard_training_step(batch, batch_idx)

    def _standard_training_step(self, batch, batch_idx):
        loss_dict = self.forward(batch)
        losses = sum(loss_dict.values())
        loss_dict["total_loss"] = losses
        self.storage.step()
        self.log_dict(loss_dict, prog_bar=True)
        return losses

    def _meta_arch_training_step(self, batch, batch_idx):
        opt = self.optimizers()
        loss_dict = self.model.training_step(
            batch, batch_idx, opt, self.manual_backward
        )
        sch = self.lr_schedulers()
        sch.step()
        self.storage.step()
        self.log_dict(loss_dict, prog_bar=True)
        return loss_dict

    def test_step(self, batch, batch_idx: int, dataloader_idx: int = 0) -> None:
        self._evaluation_step(batch, batch_idx, dataloader_idx)

    def validation_step(self, batch, batch_idx: int, dataloader_idx: int = 0) -> None:
        self._evaluation_step(batch, batch_idx, dataloader_idx)

    def _evaluation_step(self, batch, batch_idx: int, dataloader_idx: int) -> None:
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
            res = {}
            for idx, evaluator in enumerate(evaluators):
                dataset_name = self.cfg.DATASETS.TEST[idx]
                res[dataset_name] = evaluator.evaluate()
            nested_res[tag.lower()] = res

        self.eval_res = nested_res
        flattened = _flatten_dict(nested_res)

        if self.trainer.global_rank:
            assert (
                len(flattened) == 0
            ), "evaluation results should have been reduced on rank 0."
        self.log_dict(flattened, rank_zero_only=True)

    def test_epoch_end(self, _outputs) -> None:
        self._evaluation_epoch_end()

    def validation_epoch_end(self, _outputs) -> None:
        self._evaluation_epoch_end()

    def _evaluation_epoch_end(self) -> None:
        self._log_dataset_evaluation_results()
        self._reset_dataset_evaluators()

    def configure_optimizers(
        self,
    ) -> Tuple[List[torch.optim.Optimizer], List]:
        model = self.model
        if hasattr(self, PREPARED):
            # train the prepared model for FX quantization
            model = getattr(self, PREPARED)
        optim = build_optimizer_mapper(self.cfg, model)
        lr_scheduler = d2_build_lr_scheduler(self.cfg, optim)

        return [optim], [{"scheduler": lr_scheduler, "interval": "step"}]

    def train_dataloader(self):
        return self.build_detection_train_loader(self.cfg)

    def _reset_dataset_evaluators(self):
        """reset validation dataset evaluator to be run in EVAL_PERIOD steps"""
        assert isinstance(self.trainer.strategy, (SingleDeviceStrategy, DDPStrategy)), (
            "Only Single Device or DDP strategies are supported,"
            f" instead found: {self.trainer.strategy}"
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

        @rank_zero_only
        def _setup_visualization_evaluator(
            evaluator,
            dataset_name: str,
            model_tag: ModelTag,
        ) -> None:
            logger.info("Adding visualization evaluator ...")
            mapper = self.get_mapper(self.cfg, is_train=False)
            vis_eval_type = self.get_visualization_evaluator()
            # TODO: replace tbx_writter with Lightning's self.logger.experiment
            tbx_writter = _get_tbx_writer(get_tensorboard_log_dir(self.cfg.OUTPUT_DIR))
            if vis_eval_type is not None:
                evaluator._evaluators.append(
                    vis_eval_type(
                        self.cfg,
                        tbx_writter,
                        mapper,
                        dataset_name,
                        train_iter=self.trainer.global_step,
                        tag_postfix=model_tag,
                    )
                )

        for tag, dataset_evaluators in self.dataset_evaluators.items():
            dataset_evaluators.clear()
            assert self.cfg.OUTPUT_DIR, "Expect output_dir to be specified in config"
            for dataset_name in self.cfg.DATASETS.TEST:
                # setup evaluator for each dataset
                output_folder = _get_inference_dir_name(
                    self.cfg.OUTPUT_DIR, "inference", dataset_name, tag
                )
                evaluator = self.get_evaluator(
                    self.cfg, dataset_name, output_folder=output_folder
                )
                evaluator.reset()
                dataset_evaluators.append(evaluator)
                _setup_visualization_evaluator(evaluator, dataset_name, tag)

    def _evaluation_dataloader(self):
        # TODO: Support subsample n images
        assert len(self.cfg.DATASETS.TEST)

        dataloaders = []
        for dataset_name in self.cfg.DATASETS.TEST:
            dataloaders.append(self.build_detection_test_loader(self.cfg, dataset_name))

        self._reset_dataset_evaluators()
        return dataloaders

    def test_dataloader(self):
        return self._evaluation_dataloader()

    def val_dataloader(self):
        return self._evaluation_dataloader()

    def forward(self, input):
        return self.model(input)

    # ---------------------------------------------------------------------------
    # Runner methods
    # ---------------------------------------------------------------------------
    def register(self, cfg: CfgNode):
        inject_coco_datasets(cfg)
        register_dynamic_datasets(cfg)
        update_cfg_if_using_adhoc_dataset(cfg)

    @classmethod
    def build_model(cls, cfg: CfgNode, eval_only=False):
        """Builds D2go model instance from config.
        NOTE: For backward compatible with existing D2Go tools. Prefer
        `from_config` in other use cases.

        Args:
            cfg: D2go config node.
            eval_only: True if model should be in eval mode.
        """
        task = cls.from_config(cfg, eval_only)
        if hasattr(task, PREPARED):
            task = getattr(task, PREPARED)
        return task.model

    @classmethod
    def get_default_cfg(cls):
        return Detectron2GoRunner.get_default_cfg()

    @staticmethod
    def _initialize(cfg: CfgNode):
        pass

    @staticmethod
    def get_evaluator(cfg: CfgNode, dataset_name: str, output_folder: str):
        return Detectron2GoRunner.get_evaluator(
            cfg=cfg, dataset_name=dataset_name, output_folder=output_folder
        )

    @classmethod
    def cleanup(cls) -> None:
        pass

    # ---------------------------------------------------------------------------
    # Hooks
    # ---------------------------------------------------------------------------
    def on_fit_start(self) -> None:
        if self.cfg.MODEL_EMA.ENABLED:
            if self.ema_state and self.ema_state.has_inited():
                # ema_state could have been loaded from checkpoint
                # move to the current CUDA device if not on CPU
                self.ema_state.to(self.ema_state.device)
                return
            self.ema_state = EMAState.from_model(
                self.model,
                decay=self.cfg.MODEL_EMA.DECAY,
                device=self.cfg.MODEL_EMA.DEVICE or self.cfg.MODEL.DEVICE,
            )

    def on_train_batch_start(self, *_) -> None:
        if self.gpu_profiler is not None:
            self.gpu_profiler.before_step()

    def on_train_batch_end(self, *_) -> None:
        if self.ema_state:
            self.ema_state.update(self.model)
        if self.gpu_profiler is not None:
            # NOTE: keep this last in function to include all ops in this iteration of the trace
            self.gpu_profiler.after_step()

    def on_test_epoch_start(self):
        self._on_evaluation_epoch_start()

    def on_validation_epoch_start(self):
        self._on_evaluation_epoch_start()

    def _on_evaluation_epoch_start(self):
        if self.ema_state:
            self.model_ema = deepcopy(self.model)
            self.ema_state.apply_to(self.model_ema)

    def on_validation_epoch_end(self):
        if self.ema_state and hasattr(self, "model_ema"):
            del self.model_ema

    def on_test_epoch_end(self):
        if self.ema_state and hasattr(self, "model_ema"):
            del self.model_ema

    @instrument_checkpoint("save")
    def on_save_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        if self.ema_state:
            checkpoint["model_ema"] = self.ema_state.state_dict()

    @instrument_checkpoint("load")
    def on_load_checkpoint(self, checkpointed_state: Dict[str, Any]) -> None:
        """
        Called before model state is restored. Explicitly handles old model
        states so we can resume training from D2Go checkpoints transparently.

        Args:
            checkpointed_state: The raw checkpoint state as returned by torch.load
                or equivalent.
        """
        # If this is a non-Lightning checkpoint, we need to convert it.
        if not _is_lightning_checkpoint(checkpointed_state) and not _is_d2go_checkpoint(
            checkpointed_state
        ):
            raise ValueError(
                f"Invalid checkpoint state with keys: {checkpointed_state.keys()}"
            )
        if not _is_lightning_checkpoint(checkpointed_state):
            _convert_to_lightning(checkpointed_state)

        maybe_prepare_for_quantization(self, checkpointed_state)

        if self.ema_state:
            if "model_ema" not in checkpointed_state:
                rank_zero_info(
                    "EMA is enabled but EMA state is not found in given checkpoint"
                )
            else:
                self.ema_state = EMAState(
                    decay=self.cfg.MODEL_EMA.DECAY,
                    device=self.cfg.MODEL_EMA.DEVICE or self.cfg.MODEL.DEVICE,
                )
                self.ema_state.load_state_dict(checkpointed_state["model_ema"])
                rank_zero_info("Loaded EMA state from checkpoint.")


# TODO(T123654122): subclass of DefaultTask will be refactored
class GeneralizedRCNNTask(DefaultTask):
    @classmethod
    def get_default_cfg(cls):
        return GeneralizedRCNNRunner.get_default_cfg()


# TODO(T123654122): subclass of DefaultTask will be refactored
class GeneralizedRCNNTaskNoDefaultConfig(RunnerV2Mixin, DefaultTask):
    """
    Similar to `GeneralizedRCNNTask` but allowing specifying default config in yaml via `_defaults_`
    """

    pass
