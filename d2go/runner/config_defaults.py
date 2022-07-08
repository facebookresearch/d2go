#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved


from d2go.config import CfgNode as CN
from d2go.data.build import (
    add_random_subset_training_sampler_default_configs,
    add_weighted_training_sampler_default_configs,
)
from d2go.data.config import add_d2go_data_default_configs
from d2go.modeling import kmeans_anchors, model_ema
from d2go.modeling.backbone.fbnet_cfg import add_fbnet_v2_default_configs
from d2go.modeling.distillation import add_distillation_configs
from d2go.modeling.meta_arch.fcos import add_fcos_configs
from d2go.modeling.model_freezing_utils import add_model_freezing_configs
from d2go.modeling.subclass import add_subclass_configs
from d2go.quantization.modeling import add_quantization_default_configs
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


def _add_detectron2go_runner_default_cfg(_C: CN) -> None:
    # _C.MODEL.FBNET_V2...
    add_fbnet_v2_default_configs(_C)
    # _C.MODEL.FROZEN_LAYER_REG_EXP
    add_model_freezing_configs(_C)
    # _C.MODEL other models
    model_ema.add_model_ema_configs(_C)
    # _C.D2GO_DATA...
    add_d2go_data_default_configs(_C)
    # _C.TENSORBOARD...
    add_tensorboard_default_configs(_C)
    # _C.MODEL.KMEANS...
    kmeans_anchors.add_kmeans_anchors_cfg(_C)
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

    # Set find_unused_parameters for DistributedDataParallel.
    _C.MODEL.DDP_FIND_UNUSED_PARAMETERS = False
    # Set FP16 gradient compression for DistributedDataParallel.
    _C.MODEL.DDP_FP16_GRAD_COMPRESS = False

    # Set default optimizer
    _C.SOLVER.OPTIMIZER = "sgd"
    _C.SOLVER.LR_MULTIPLIER_OVERWRITE = []
    _C.SOLVER.WEIGHT_DECAY_EMBED = 0.0

    # Betas are used in the AdamW optimizer
    _C.SOLVER.BETAS = (0.9, 0.999)

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

    # Add FB specific configs
    _add_detectron2go_runner_default_fb_cfg(_C)


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
    _C.RCNN_CUSTOM_CONVERT_FX = "default_rcnn_custom_convert_fx"
    _C.register_deprecated_key("RCNN_PREPARE_FOR_QUANT_CONVERT")


def get_base_runner_default_cfg(cfg: CN) -> CN:
    assert len(cfg) == 0, f"start from scratch, but previous cfg is non-empty: {cfg}"

    cfg = get_d2_cfg()
    # upgrade from D2's CfgNode to D2Go's CfgNode
    cfg = CN.cast_from_other_class(cfg)

    cfg.SOLVER.AUTO_SCALING_METHODS = ["default_scale_d2_configs"]

    return cfg


def get_detectron2go_runner_default_cfg(cfg: CN) -> CN:
    assert len(cfg) == 0, f"start from scratch, but previous cfg is non-empty: {cfg}"

    cfg = get_base_runner_default_cfg(cfg)
    _add_detectron2go_runner_default_cfg(cfg)

    return cfg


def get_generalized_rcnn_runner_default_cfg(cfg: CN) -> CN:
    assert len(cfg) == 0, f"start from scratch, but previous cfg is non-empty: {cfg}"

    cfg = get_detectron2go_runner_default_cfg(cfg)
    _add_rcnn_default_config(cfg)

    return cfg
