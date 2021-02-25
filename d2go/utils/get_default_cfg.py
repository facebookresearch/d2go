from d2go.config import CfgNode as CN
from d2go.data.build import (
    add_weighted_training_sampler_default_configs,
)
from d2go.data.config import add_d2go_data_default_configs
from d2go.modeling.backbone.fbnet_cfg import (
    add_bifpn_default_configs,
    add_fbnet_v2_default_configs,
)
from d2go.modeling import kmeans_anchors, model_ema
from d2go.modeling.model_freezing_utils import add_model_freezing_configs
from d2go.modeling.quantization import add_quantization_default_configs
from d2go.modeling.subclass import add_subclass_configs


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


def add_abnormal_checker_configs(_C):
    _C.ABNORMAL_CHECKER = CN()
    # check and log the iteration with bad losses if enabled
    _C.ABNORMAL_CHECKER.ENABLED = False


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
    # _C.ABNORMAL_CHECKER
    add_abnormal_checker_configs(_C)
    # _C.MODEL.SUBCLASS
    add_subclass_configs(_C)

    # Set find_unused_parameters for DistributedDataParallel.
    _C.MODEL.DDP_FIND_UNUSED_PARAMETERS = False

    # Set default optimizer
    _C.SOLVER.OPTIMIZER = "sgd"
    _C.SOLVER.LR_MULTIPLIER_OVERWRITE = []

    # Default world size in D2 is 0, which means scaling is not applied. For D2Go
    # auto scale is encouraged, setting it to 8
    assert _C.SOLVER.REFERENCE_WORLD_SIZE == 0
    _C.SOLVER.REFERENCE_WORLD_SIZE = 8
    # Besides scaling default D2 configs, also scale quantization configs
    _C.SOLVER.AUTO_SCALING_METHODS = [
        "default_scale_d2_configs",
        "default_scale_quantization_configs",
    ]
    return _C
