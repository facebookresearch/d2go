#!/usr/bin/env python3

from .modeldef import (
    DEFAULT_STAGES,
    _repeat_last,
    FPN_UPSAMPLE_HEAD_STAGES,
    _BASIC_ARGS,
)

from .modeldef_retinanet_eff import (
    BASIC_ARGS,
    IRF_CFG,
)

from mobile_cv.arch.fbnet_v2.modeldef_registry import FBNetV2ModelArch
from mobile_cv.arch.fbnet_v2.modeldef_utils import e1, e6


"""
Baseline FPN: T. Lin et al., Feature Pyramid Networks for Object Detection
    Input0          Input1
      |               |
    1x1 Conv        1x1 Conv
      |               |
    Combiner   |--> Combiner   |-->
      |        |      |        |
    Skip       |    Skip       |
      |        |      |        |
      | --> Upsample  | --> Upsample
    3x3 Conv        3x3 Conv
      |               |
    Output0         Output1
"""
BASELINE_FPN_STAGES = [
    {
        "stages": [
            [("conv_k1", 128, 1, 1, {"relu_args": None})],
            [("noop", 128, 1, 1)],
            [("skip", 128, 1, 1)],
            [("conv_k3", 128, 1, 1, {"relu_args": None})],
            [("upsample", 128, 2, 1)],
        ] * 4,
        "stage_combiners": ["add"] * 4,
        "combiner_path": "low_res",
    },
]


EFFICIENTDET_D0_RETINA_FPN_STAGES = [
    {
        "stages": [
            [("conv_k1", 64, 1, 1, {"relu_args": None})],
            [("noop", 64, 1, 1)],
            [("skip", 64, 1, 1)],
            [("conv_k3", 64, 1, 1, {"relu_args": None})],
            [("upsample", 64, 2, 1)],
        ] * 5,
        "stage_combiners": ["add"] * 5,
        "combiner_path": "low_res",
    },
]


EFFICIENTDET_D0_RETINA_FPN_DW_STAGES = [
    {
        "stages": [
            [("conv_k1", 64, 1, 1, {"relu_args": None})],
            [("noop", 64, 1, 1)],
            [("skip", 64, 1, 1)],
            [(
                "dc_k3", 64, 1, 1, {
                    "relu_args": None,
                    "dw_skip_bnrelu": True,
                }
            )],
            [("upsample", 64, 2, 1)],
        ] * 5,
        "stage_combiners": ["add"] * 5,
        "combiner_path": "low_res",
    },
]


EFFICIENTDET_D0_RETINA_FPN_DW_3x_STAGES = [
    {
        "stages": [
            [("conv_k1", 64, 1, 1, {"relu_args": None})],
            [("noop", 64, 1, 1)],
            [("skip", 64, 1, 1)],
            [(
                "dc_k3", 64, 1, 1, {
                    "relu_args": None,
                    "dw_skip_bnrelu": True,
                }
            )],
            [("upsample", 64, 2, 1)],
        ] * 5,
        "stage_combiners": ["add"] * 5,
        "combiner_path": "low_res",
    },
    {
        "stages": [
            [("skip", 64, 1, 1)],
            [("noop", 64, 1, 1)],
            [("skip", 64, 1, 1)],
            [(
                "dc_k3", 64, 1, 1, {
                    "relu_args": None,
                    "dw_skip_bnrelu": True,
                }
            )],
            [("upsample", 64, 2, 1)],
        ] * 5,
        "stage_combiners": ["add"] * 5,
        "combiner_path": "low_res",
    },
    {
        "stages": [
            [("skip", 64, 1, 1)],
            [("noop", 64, 1, 1)],
            [("skip", 64, 1, 1)],
            [(
                "dc_k3", 64, 1, 1, {
                    "relu_args": None,
                    "dw_skip_bnrelu": True,
                }
            )],
            [("upsample", 64, 2, 1)],
        ] * 5,
        "stage_combiners": ["add"] * 5,
        "combiner_path": "low_res",
    },
]


"""
PANET: S. Liu et al., Path Aggregation Network for Instance Segmentation
    Input0          Input1
      |               |
    1x1 Conv        1x1 Conv
      |               |
    Combiner   |--> Combiner   |-->
      |        |      |        |
    Skip       |    Skip       |
      |        |      |        |
      | --> Upsample  | --> Upsample
    3x3 Conv        3x3 Conv
      |               |
      |               |
    Skip            Skip
      |               |
    Combiner <-|    Combiner <-|
      |        |      |        |
    3x3 Conv   |    3x3 Conv   |
      |        |      |        |
      |  3x3 Conv <-- |  3x3 Conv <--
    Skip            Skip
      |               |
    Output0         Output1
"""
BASELINE_PANET_STAGES = [
    {
        "stages": [
            [("conv_k1", 128, 1, 1, {"relu_args": None})],
            [("noop", 128, 1, 1)],
            [("skip", 128, 1, 1)],
            [("conv_k3", 128, 1, 1, {"relu_args": None})],
            [("upsample", 128, 2, 1)],
        ] * 4,
        "stage_combiners": ["add"] * 4,
        "combiner_path": "low_res",
    },
    {
        "stages": [
            [("skip", 128, 1, 1)],
            [("noop", 128, 1, 1)],
            [("conv_k3", 128, 1, 1, {"relu_args": None})],
            [("skip", 128, 1, 1)],
            [("conv_k3", 128, 2, 1, {"relu_args": None})],
        ] * 4,
        "stage_combiners": ["add"] * 4,
        "combiner_path": "high_res",
    },
]


"""
BiFPN: M. Tan et al., EfficientDet: Scalable and Efficient Object Detection
Implementation following
https://github.com/google/automl/blob/53286d6f2d4ab44999df9e968a7712c5117c4a6c/efficientdet/efficientdet_arch.py
    Input0          Input1
      |               |
    1x1 Conv        1x1 Conv
      |               |
    Combiner   |--> Combiner   |-->
      |        |      |        |
    Skip       |    Skip       |
      |        |      |        |
      | ---> Noop     | ---> Noop
    Skip            Skip
      |               |
      |               |
    Skip            Skip
      |               |
    Combiner   |--> Combiner   |-->
      |        |      |        |
    Skip       |    3x3 Conv   |
      |        |      |        |
      | --> Upsample  | --> Upsample
    Skip            Skip
      |               |    |
      |    |          |    |
      |    |          |    |
     Skip  Skip      Skip  Skip
      |    |          |    |
    Combiner <-|    Combiner <-|
      |        |      |        |
    3x3 Conv   |    3x3 Conv   |
      |        |      |        |
      |   MaxPool <-- |   MaxPool <--
    Skip            Skip
      |               |
    Output0         Output1
"""
BASELINE_BIFPN_STAGES = [
    {
        "stages": [
            [("conv_k1", 128, 1, 1, {"relu_args": None})],
            [("noop", 128, 1, 1)],
            [("skip", 128, 1, 1)],
            [("skip", 128, 1, 1)],
            [("noop", 128, 1, 1)],
        ] * 4,
        "combiner_path": "low_res",
    },
    {
        "stages": [
            [("skip", 128, 1, 1)],
            [("noop", 128, 1, 1)],
            [("conv_k3", 128, 1, 1, {"bn_args": "bn", "relu_args": "swish"})],
            [("skip", 128, 1, 1)],
            [("upsample", 128, 2, 1)],
        ] * 4,
        "combiner_path": "low_res",
    },
    {
        "stages": [
            [("skip", 128, 1, 1)],
            [("skip", 128, 1, 1)],
            [("conv_k3", 128, 1, 1, {"bn_args": "bn", "relu_args": "swish"})],
            [("skip", 128, 1, 1)],
            [("maxpool", 128, 2, 1, {"kernel_size": 3})],
        ] * 4,
        "combiner_path": "high_res",
    },
]


EFFICIENTDET_D0_RETINA_BIFPN_STAGES = [
    {
        "stages": [
            [("conv_k1", 64, 1, 1, {"relu_args": None})],
            [("noop", 64, 1, 1)],
            [("skip", 64, 1, 1)],
            [("skip", 64, 1, 1)],
            [("noop", 64, 1, 1)],
        # pyre-fixme[58]: `+` is not supported for operand types
        #  `List[typing.Union[typing.List[typing.Tuple[str, int, int, int]],
        #  typing.List[typing.Tuple[str, int, int, int, typing.Dict[str, None]]]]]` and
        #  `List[typing.List[typing.Tuple[str, int, int, int]]]`.
        ] * 3 + [
            [("skip", 64, 1, 1)],
            [("noop", 64, 1, 1)],
            [("skip", 64, 1, 1)],
            [("skip", 64, 1, 1)],
            [("noop", 64, 1, 1)],
        ] * 2,
        "combiner_path": "low_res",
    },
    {
        "stages": [
            [("skip", 64, 1, 1)],
            [("noop", 64, 1, 1)],
            # note different efficientdet implementation
            [("conv_k3", 64, 1, 1, {"bn_args": "bn", "relu_args": "swish"})],
            [("skip", 64, 1, 1)],
            [("upsample", 64, 2, 1)],
        ] * 5,
        "combiner_path": "low_res",
    },
    {
        "stages": [
            [("skip", 64, 1, 1)],
            [("skip", 64, 1, 1)],
            # note different efficientdet implementation
            [("conv_k3", 64, 1, 1, {"bn_args": "bn", "relu_args": "swish"})],
            [("skip", 64, 1, 1)],
            [("maxpool", 64, 2, 1, {"kernel_size": 3, "padding": 1})],
        ] * 5,
        "combiner_path": "high_res",
    },
]


EFFICIENTDET_D0_RETINA_BIFPN_DW_STAGES = [
    {
        "stages": [
            [("conv_k1", 64, 1, 1, {"relu_args": None})],
            [("noop", 64, 1, 1)],
            [("skip", 64, 1, 1)],
            [("skip", 64, 1, 1)],
            [("noop", 64, 1, 1)],
        # pyre-fixme[58]: `+` is not supported for operand types
        #  `List[typing.Union[typing.List[typing.Tuple[str, int, int, int]],
        #  typing.List[typing.Tuple[str, int, int, int, typing.Dict[str, None]]]]]` and
        #  `List[typing.List[typing.Tuple[str, int, int, int]]]`.
        ] * 3 + [
            [("skip", 64, 1, 1)],
            [("noop", 64, 1, 1)],
            [("skip", 64, 1, 1)],
            [("skip", 64, 1, 1)],
            [("noop", 64, 1, 1)],
        ] * 2,
        "combiner_path": "low_res",
    },
    {
        "stages": [
            [("skip", 64, 1, 1)],
            [("noop", 64, 1, 1)],
            # note different efficientdet implementation
            # [("conv_k3", 64, 1, 1, {"bn_args": "bn", "relu_args": "swish"})],
            [("dc_k3", 64, 1, 1, {"relu_args": "swish", "dw_skip_bnrelu": True, "act_first": True})],
            [("skip", 64, 1, 1)],
            [("upsample", 64, 2, 1)],
        ] * 5,
        "combiner_path": "low_res",
    },
    {
        "stages": [
            [("skip", 64, 1, 1)],
            [("skip", 64, 1, 1)],
            # note different efficientdet implementation
            # [("conv_k3", 64, 1, 1, {"bn_args": "bn", "relu_args": "swish"})],
            [("dc_k3", 64, 1, 1, {"relu_args": "swish", "dw_skip_bnrelu": True, "act_first": True})],
            [("skip", 64, 1, 1)],
            [("maxpool", 64, 2, 1, {"kernel_size": 3, "padding": 1})],
        ] * 5,
        "combiner_path": "high_res",
    },
]


EFFICIENTDET_D0_RETINA_BIFPN_DW_3x_STAGES = [
    {
        "stages": [
            [("conv_k1", 64, 1, 1, {"relu_args": None})],
            [("noop", 64, 1, 1)],
            [("skip", 64, 1, 1)],
            [("skip", 64, 1, 1)],
            [("noop", 64, 1, 1)],
        # pyre-fixme[58]: `+` is not supported for operand types
        #  `List[typing.Union[typing.List[typing.Tuple[str, int, int, int]],
        #  typing.List[typing.Tuple[str, int, int, int, typing.Dict[str, None]]]]]` and
        #  `List[typing.List[typing.Tuple[str, int, int, int]]]`.
        ] * 3 + [
            [("skip", 64, 1, 1)],
            [("noop", 64, 1, 1)],
            [("skip", 64, 1, 1)],
            [("skip", 64, 1, 1)],
            [("noop", 64, 1, 1)],
        ] * 2,
        "combiner_path": "low_res",
    },
    {
        "stages": [
            [("skip", 64, 1, 1)],
            [("noop", 64, 1, 1)],
            # note different efficientdet implementation
            # [("conv_k3", 64, 1, 1, {"bn_args": "bn", "relu_args": "swish"})],
            [("dc_k3", 64, 1, 1, {"relu_args": "swish", "dw_skip_bnrelu": True, "act_first": False})],
            [("skip", 64, 1, 1)],
            [("upsample", 64, 2, 1)],
        ] * 5,
        "combiner_path": "low_res",
    },
    {
        "stages": [
            [("skip", 64, 1, 1)],
            [("skip", 64, 1, 1)],
            # note different efficientdet implementation
            # [("conv_k3", 64, 1, 1, {"bn_args": "bn", "relu_args": "swish"})],
            [("dc_k3", 64, 1, 1, {"relu_args": "swish", "dw_skip_bnrelu": True, "act_first": False})],
            [("skip", 64, 1, 1)],
            [("maxpool", 64, 2, 1, {"kernel_size": 3, "padding": 1})],
        ] * 5,
        "combiner_path": "high_res",
    },
    {
        "stages": [
            [("skip", 64, 1, 1)],
            [("noop", 64, 1, 1)],
            # note different efficientdet implementation
            # [("conv_k3", 64, 1, 1, {"bn_args": "bn", "relu_args": "swish"})],
            [("dc_k3", 64, 1, 1, {"relu_args": "swish", "dw_skip_bnrelu": True, "act_first": False})],
            [("skip", 64, 1, 1)],
            [("upsample", 64, 2, 1)],
        ] * 5,
        "combiner_path": "low_res",
    },
    {
        "stages": [
            [("skip", 64, 1, 1)],
            [("skip", 64, 1, 1)],
            # note different efficientdet implementation
            # [("conv_k3", 64, 1, 1, {"bn_args": "bn", "relu_args": "swish"})],
            [("dc_k3", 64, 1, 1, {"relu_args": "swish", "dw_skip_bnrelu": True, "act_first": False})],
            [("skip", 64, 1, 1)],
            [("maxpool", 64, 2, 1, {"kernel_size": 3, "padding": 1})],
        ] * 5,
        "combiner_path": "high_res",
    },
    {
        "stages": [
            [("skip", 64, 1, 1)],
            [("noop", 64, 1, 1)],
            # note different efficientdet implementation
            # [("conv_k3", 64, 1, 1, {"bn_args": "bn", "relu_args": "swish"})],
            [("dc_k3", 64, 1, 1, {"relu_args": "swish", "dw_skip_bnrelu": True, "act_first": False})],
            [("skip", 64, 1, 1)],
            [("upsample", 64, 2, 1)],
        ] * 5,
        "combiner_path": "low_res",
    },
    {
        "stages": [
            [("skip", 64, 1, 1)],
            [("skip", 64, 1, 1)],
            # note different efficientdet implementation
            # [("conv_k3", 64, 1, 1, {"bn_args": "bn", "relu_args": "swish"})],
            [("dc_k3", 64, 1, 1, {"relu_args": "swish", "dw_skip_bnrelu": True, "act_first": False})],
            [("skip", 64, 1, 1)],
            [("maxpool", 64, 2, 1, {"kernel_size": 3, "padding": 1})],
        ] * 5,
        "combiner_path": "high_res",
    },
]


MODEL_ARCH_BIFPN = {
    "default_bifpn_fpn": {
        "trunk": DEFAULT_STAGES[0:5],  # FPN uses all 5 stages
        "bifpn": BASELINE_FPN_STAGES,
        "rpn": [[_repeat_last(DEFAULT_STAGES[3], n=1)]],
        "bbox": [DEFAULT_STAGES[4]],
        "mask": FPN_UPSAMPLE_HEAD_STAGES,
        "basic_args": _BASIC_ARGS,
    },
    "default_bifpn_panet": {
        "trunk": DEFAULT_STAGES[0:5],  # FPN uses all 5 stages
        "bifpn": BASELINE_PANET_STAGES,
        "rpn": [[_repeat_last(DEFAULT_STAGES[3], n=1)]],
        "bbox": [DEFAULT_STAGES[4]],
        "mask": FPN_UPSAMPLE_HEAD_STAGES,
        "basic_args": _BASIC_ARGS,
    },
    "default_bifpn_bifpn": {
        "trunk": DEFAULT_STAGES[0:5],  # FPN uses all 5 stages
        "bifpn": BASELINE_BIFPN_STAGES,
        "rpn": [[_repeat_last(DEFAULT_STAGES[3], n=1)]],
        "bbox": [DEFAULT_STAGES[4]],
        "mask": FPN_UPSAMPLE_HEAD_STAGES,
        "basic_args": _BASIC_ARGS,
    },
    "default_bifpn_fpn_1x": {
        "trunk": DEFAULT_STAGES[0:5],  # FPN uses all 5 stages
        "bifpn": EFFICIENTDET_D0_RETINA_FPN_STAGES,
        "rpn": [[_repeat_last(DEFAULT_STAGES[3], n=1)]],
        "bbox": [DEFAULT_STAGES[4]],
        "mask": FPN_UPSAMPLE_HEAD_STAGES,
        "basic_args": _BASIC_ARGS,
    },
    "default_bifpn_bifpn_1x": {
        "trunk": DEFAULT_STAGES[0:5],  # FPN uses all 5 stages
        "bifpn": EFFICIENTDET_D0_RETINA_BIFPN_STAGES,
        "rpn": [[_repeat_last(DEFAULT_STAGES[3], n=1)]],
        "bbox": [DEFAULT_STAGES[4]],
        "mask": FPN_UPSAMPLE_HEAD_STAGES,
        "basic_args": _BASIC_ARGS,
    },
    "default_bifpn_fpn_dw_3x": {
        "trunk": DEFAULT_STAGES[0:5],  # FPN uses all 5 stages
        "bifpn": EFFICIENTDET_D0_RETINA_FPN_DW_3x_STAGES,
        "rpn": [[_repeat_last(DEFAULT_STAGES[3], n=1)]],
        "bbox": [DEFAULT_STAGES[4]],
        "mask": FPN_UPSAMPLE_HEAD_STAGES,
        "basic_args": _BASIC_ARGS,
    },
    "default_bifpn_bifpn_dw_3x": {
        "trunk": DEFAULT_STAGES[0:5],  # FPN uses all 5 stages
        "bifpn": EFFICIENTDET_D0_RETINA_BIFPN_DW_3x_STAGES,
        "rpn": [[_repeat_last(DEFAULT_STAGES[3], n=1)]],
        "bbox": [DEFAULT_STAGES[4]],
        "mask": FPN_UPSAMPLE_HEAD_STAGES,
        "basic_args": _BASIC_ARGS,
    },
}
FBNetV2ModelArch.add_archs(MODEL_ARCH_BIFPN)


EFFICIENTNET_B0_BACKBONE = [
    # [c, s, n, ...]
    # stage 0
    [["conv_k3", 32, 2, 1], ["ir_k3_se", 16, 1, 1, e1, IRF_CFG]],
    # stage 1
    [["ir_k3_se", 24, 2, 2, e6, IRF_CFG]],
    # stage 2
    [["ir_k5_se", 40, 2, 2, e6, IRF_CFG]],
    # stage 3
    [
        ["ir_k3_se", 80, 2, 3, e6, IRF_CFG],
        ["ir_k5_se", 112, 1, 3, e6, IRF_CFG],
    ],
    # stage 4
    [
        ["ir_k5_se", 192, 2, 4, e6, IRF_CFG],
        ["ir_k3_se", 320, 1, 1, e6, IRF_CFG],
    ],
]

MODEL_ARCH_RETINANET_BIFPN = {
    "retinanet_fpn_1x": {
        "trunk": EFFICIENTNET_B0_BACKBONE,
        "bifpn": EFFICIENTDET_D0_RETINA_FPN_STAGES,
        "basic_args": BASIC_ARGS,
    },
    "retinanet_fpn_dw_1x": {
        "trunk": EFFICIENTNET_B0_BACKBONE,
        "bifpn": EFFICIENTDET_D0_RETINA_FPN_DW_STAGES,
        "basic_args": BASIC_ARGS,
    },
    "retinanet_fpn_dw_3x": {
        "trunk": EFFICIENTNET_B0_BACKBONE,
        "bifpn": EFFICIENTDET_D0_RETINA_FPN_DW_3x_STAGES,
        "basic_args": BASIC_ARGS,
    },
    "retinanet_bifpn_1x": {
        "trunk": EFFICIENTNET_B0_BACKBONE,
        "bifpn": EFFICIENTDET_D0_RETINA_BIFPN_STAGES,
        "basic_args": BASIC_ARGS,
    },
    "retinanet_bifpn_dw_1x": {
        "trunk": EFFICIENTNET_B0_BACKBONE,
        "bifpn": EFFICIENTDET_D0_RETINA_BIFPN_DW_STAGES,
        "basic_args": BASIC_ARGS,
    },
    "retinanet_bifpn_dw_3x": {
        "trunk": EFFICIENTNET_B0_BACKBONE,
        "bifpn": EFFICIENTDET_D0_RETINA_BIFPN_DW_3x_STAGES,
        "basic_args": BASIC_ARGS,
    },
}
FBNetV2ModelArch.add_archs(MODEL_ARCH_RETINANET_BIFPN)
