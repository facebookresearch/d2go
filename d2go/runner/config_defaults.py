#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved


from d2go.config import CfgNode as CN
from d2go.data.build import (
    add_random_subset_training_sampler_default_configs,
    add_weighted_training_sampler_default_configs,
)
from d2go.data.config import add_d2go_data_default_configs
from d2go.modeling.backbone.fbnet_cfg import add_fbnet_v2_default_configs
from d2go.modeling.ema import add_model_ema_configs
from d2go.modeling.kmeans_anchors import add_kmeans_anchors_cfg
from d2go.modeling.meta_arch.fcos import add_fcos_configs
from d2go.modeling.model_freezing_utils import add_model_freezing_configs
from d2go.modeling.subclass import add_subclass_configs
from d2go.quantization.modeling import add_quantization_default_configs
from d2go.registry.builtin import CONFIG_UPDATER_REGISTRY
from d2go.trainer.activation_checkpointing import add_activation_checkpoint_configs
from d2go.trainer.fsdp import add_fsdp_configs
from d2go.utils.gpu_memory_profiler import (
    add_memory_profiler_configs,
    add_zoomer_default_config,
)
from d2go.utils.visualization import add_tensorboard_default_configs
from detectron2.config import get_cfg as get_d2_cfg
from mobile_cv.common.misc.oss_utils import fb_overwritable


def _add_abnormal_checker_configs(_C: CN) -> None:
    _C.ABNORMAL_CHECKER = CN()
    # check and log the iteration with bad losses if enabled
    _C.ABNORMAL_CHECKER.ENABLED = False


@fb_overwritable()
def _add_detectron2go_runner_default_fb_cfg(_C: CN) -> None:
    pass


@fb_overwritable()
def _add_base_runner_default_fb_cfg(_C: CN) -> None:
    pass


def add_distillation_configs(_C: CN) -> None:
    """Add default parameters to config

    The TEACHER.CONFIG field allows us to build a PyTorch model using an
    existing config.  We can build any model that is normally supported by
    D2Go (e.g., FBNet) because we just use the same config
    """
    _C.DISTILLATION = CN()
    _C.DISTILLATION.ALGORITHM = "LabelDistillation"
    _C.DISTILLATION.HELPER = "BaseDistillationHelper"
    _C.DISTILLATION.TEACHER = CN()
    _C.DISTILLATION.TEACHER.TORCHSCRIPT_FNAME = ""
    _C.DISTILLATION.TEACHER.DEVICE = ""
    _C.DISTILLATION.TEACHER.TYPE = "torchscript"
    _C.DISTILLATION.TEACHER.CONFIG_FNAME = ""
    _C.DISTILLATION.TEACHER.RUNNER_NAME = "d2go.runner.GeneralizedRCNNRunner"
    _C.DISTILLATION.TEACHER.OVERWRITE_OPTS = []


def _add_detectron2go_runner_default_cfg(_C: CN) -> None:
    # _C.MODEL.FBNET_V2...
    add_fbnet_v2_default_configs(_C)
    # _C.MODEL.FROZEN_LAYER_REG_EXP
    add_model_freezing_configs(_C)
    # _C.MODEL other models
    add_model_ema_configs(_C)
    # _C.D2GO_DATA...
    add_d2go_data_default_configs(_C)
    # _C.TENSORBOARD...
    add_tensorboard_default_configs(_C)
    # _C.MODEL.KMEANS...
    add_kmeans_anchors_cfg(_C)
    # _C.QUANTIZATION
    add_quantization_default_configs(_C)
    # _C.DATASETS.TRAIN_REPEAT_FACTOR
    add_weighted_training_sampler_default_configs(_C)
    # _C.DATALOADER.RANDOM_SUBSET_RATIO
    add_random_subset_training_sampler_default_configs(_C)
    # _C.ABNORMAL_CHECKER
    _add_abnormal_checker_configs(_C)
    # _C.MODEL.SUBCLASS
    add_subclass_configs(_C)
    # _C.MODEL.FCOS
    add_fcos_configs(_C)
    # _C.DISTILLATION
    add_distillation_configs(_C)
    # _C.FSDP
    add_fsdp_configs(_C)
    # _C.ACTIVATION_CHECKPOINT
    add_activation_checkpoint_configs(_C)

    # Set find_unused_parameters for DistributedDataParallel.
    _C.MODEL.DDP_FIND_UNUSED_PARAMETERS = False
    # Set FP16 gradient compression for DistributedDataParallel.
    _C.MODEL.DDP_FP16_GRAD_COMPRESS = False
    # Specify the gradients as views
    _C.MODEL.DDP_GRADIENT_AS_BUCKET_VIEW = False

    # Set default optimizer
    _C.SOLVER.OPTIMIZER = "sgd"
    _C.SOLVER.LR_MULTIPLIER_OVERWRITE = []
    _C.SOLVER.WEIGHT_DECAY_EMBED = 0.0
    _C.SOLVER.WEIGHT_DECAY_OVERWRITE = []
    assert not _C.SOLVER.AMP.ENABLED
    # AMP precision is used by both D2 and lightning backend. Can be "float16" or "bfloat16".
    _C.SOLVER.AMP.PRECISION = "float16"
    # log the grad scalar to the output
    _C.SOLVER.AMP.LOG_GRAD_SCALER = False

    # Betas are used in the AdamW optimizer
    _C.SOLVER.BETAS = (0.9, 0.999)
    _C.SOLVER.EPS = 1e-08
    _C.SOLVER.FUSED = False
    _C.SOLVER.DETERMINISTIC = False

    # RECOMPUTE_BOXES for LSJ Training
    _C.INPUT.RECOMPUTE_BOXES = False

    # Default world size in D2 is 0, which means scaling is not applied. For D2Go
    # auto scale is encouraged, setting it to 8
    assert _C.SOLVER.REFERENCE_WORLD_SIZE == 0
    _C.SOLVER.REFERENCE_WORLD_SIZE = 8
    # Besides scaling default D2 configs, also scale quantization configs
    _C.SOLVER.AUTO_SCALING_METHODS = [
        "default_scale_d2_configs",
        "default_scale_quantization_configs",
    ]

    # Modeling hooks
    # List of modeling hook names
    _C.MODEL.MODELING_HOOKS = []

    # Profiler
    _C.PROFILERS = ["default_flop_counter"]

    # GPU memory profiler
    add_memory_profiler_configs(_C)

    # Zoomer memory profiling
    add_zoomer_default_config(_C)

    # Checkpointing-specific config
    _C.LOAD_CKPT_TO_GPU = False

    # Add FB specific configs
    _add_detectron2go_runner_default_fb_cfg(_C)

    # Specify whether to perform NUMA binding
    _C.NUMA_BINDING = False

    # Specify whether to zero the gradients before forward
    _C.ZERO_GRAD_BEFORE_FORWARD = False

    # Whether to enforce rebuilding data loaders for datasets that have expiration
    _C.DATALOADER.ENFORE_EXPIRATION = False


def _add_rcnn_default_config(_C: CN) -> None:
    _C.EXPORT_CAFFE2 = CN()
    _C.EXPORT_CAFFE2.USE_HEATMAP_MAX_KEYPOINT = False

    # Options about how to export the model
    _C.RCNN_EXPORT = CN()
    # whether or not to include the postprocess (GeneralizedRCNN._postprocess) step
    # inside the exported model
    _C.RCNN_EXPORT.INCLUDE_POSTPROCESS = False

    _C.RCNN_PREPARE_FOR_EXPORT = "default_rcnn_prepare_for_export"
    _C.RCNN_PREPARE_FOR_QUANT = "default_rcnn_prepare_for_quant"
    _C.register_deprecated_key("RCNN_PREPARE_FOR_QUANT_CONVERT")


@CONFIG_UPDATER_REGISTRY.register("BaseRunner")
def get_base_runner_default_cfg(cfg: CN) -> CN:
    assert len(cfg) == 0, f"start from scratch, but previous cfg is non-empty: {cfg}"

    cfg = get_d2_cfg()
    # upgrade from D2's CfgNode to D2Go's CfgNode
    cfg = CN.cast_from_other_class(cfg)

    cfg.SOLVER.AUTO_SCALING_METHODS = ["default_scale_d2_configs"]

    # Frequency of metric gathering in trainer.
    cfg.GATHER_METRIC_PERIOD = 1
    # Frequency of metric printer, tensorboard writer, etc.
    cfg.WRITER_PERIOD = 20
    # Enable async writing metrics to tensorboard and logs to speed up training
    cfg.ASYNC_WRITE_METRICS = False

    # train_net specific arguments, define in runner but used in train_net
    # run evaluation after training is done
    cfg.TEST.FINAL_EVAL = True

    _add_base_runner_default_fb_cfg(cfg)

    return cfg


@CONFIG_UPDATER_REGISTRY.register("Detectron2GoRunner")
def get_detectron2go_runner_default_cfg(cfg: CN) -> CN:
    assert len(cfg) == 0, f"start from scratch, but previous cfg is non-empty: {cfg}"

    cfg = get_base_runner_default_cfg(cfg)
    _add_detectron2go_runner_default_cfg(cfg)

    return cfg


@CONFIG_UPDATER_REGISTRY.register("GeneralizedRCNNRunner")
def get_generalized_rcnn_runner_default_cfg(cfg: CN) -> CN:
    assert len(cfg) == 0, f"start from scratch, but previous cfg is non-empty: {cfg}"

    cfg = get_detectron2go_runner_default_cfg(cfg)
    _add_rcnn_default_config(cfg)

    return cfg


@fb_overwritable()
def preprocess_cfg(cfg: CN) -> CN:
    return cfg
