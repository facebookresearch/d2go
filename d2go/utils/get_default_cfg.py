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
from d2go.modeling.meta_arch.fcos import add_fcos_configs
from d2go.modeling.model_freezing_utils import add_model_freezing_configs
from d2go.modeling.subclass import add_subclass_configs
from d2go.quantization.modeling import add_quantization_default_configs
from mobile_cv.common.misc.oss_utils import fb_overwritable


@fb_overwritable()
def add_tensorboard_default_configs(_C):
    _C.TENSORBOARD = CN()
    # Output from dataloader will be written to tensorboard at this frequency
    _C.TENSORBOARD.TRAIN_LOADER_VIS_WRITE_PERIOD = 20
    # This controls max number of images over all batches, be considerate when
    # increasing this number because it takes disk space and slows down the training
    _C.TENSORBOARD.TRAIN_LOADER_VIS_MAX_IMAGES = 16
    # Max number of images per dataset to visualize in tensorboard during evaluation
    _C.TENSORBOARD.TEST_VIS_MAX_IMAGES = 16

    # TENSORBOARD.LOG_DIR will be determined solely by OUTPUT_DIR
    _C.register_deprecated_key("TENSORBOARD.LOG_DIR")


@fb_overwritable()
def add_abnormal_checker_configs(_C):
    _C.ABNORMAL_CHECKER = CN()
    # check and log the iteration with bad losses if enabled
    _C.ABNORMAL_CHECKER.ENABLED = False


@fb_overwritable()
def get_default_cfg(_C):
    # _C.MODEL.FBNET...
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
    add_abnormal_checker_configs(_C)
    # _C.MODEL.SUBCLASS
    add_subclass_configs(_C)
    # _C.MODEL.FCOS
    add_fcos_configs(_C)

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

    return _C
