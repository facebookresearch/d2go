#!/usr/bin/env python3

import copy
from mobile_cv.arch.fbnet_v2.modeldef_utils import _ex, e1, e2, e1p, e3, e4, e6
from d2go.modeling.modeldef.fbnet_modeldef_registry import FBNetV2ModelArch


def _mutated_tuple(tp, pos, value):
    tp_list = list(tp)
    tp_list[pos] = value
    return tuple(tp_list)


def _repeat_last(stage, n=None):
    """
    Repeat the last "layer" of given stage, i.e. a (op_type, c, s, n_repeat, ...)
        tuple, reset n_repeat if specified otherwise kept the original value.
    """
    assert isinstance(stage, list)
    assert all(isinstance(x, tuple) for x in stage)
    last_layer = copy.deepcopy(stage[-1])
    if n is not None:
        last_layer = _mutated_tuple(last_layer, 3, n)
    return last_layer


_BASIC_ARGS = {
    # skil norm and activation for depthwise conv in IRF module, this make the
    # model easier to quantize.
    "dw_skip_bnrelu": True,
    # uncomment below (always_pw and bias) to match model definition of the
    # FBNetV1 builder.
    # "always_pw": True,
    # "bias": False,

    # temporarily disable zero_last_bn_gamma
    "zero_last_bn_gamma": False,
}


DEFAULT_STAGES = [
    # NOTE: each stage is a list of (op_type, out_channels, stride, n_repeat, ...)
    # resolution stage 0, equivalent to 224->112
    [("conv_k3", 32, 2, 1), ("ir_k3", 16, 1, 1, e1)],
    # resolution stage 1, equivalent to 112->56
    [("ir_k3", 24, 2, 2, e6)],
    # resolution stage 2, equivalent to 56->28
    [("ir_k3", 32, 2, 3, e6)],
    # resolution stage 3, equivalent to 28->14
    [("ir_k3", 64, 2, 4, e6), ("ir_k3", 96, 1, 3, e6)],
    # resolution stage 4, equivalent to 14->7
    [("ir_k3", 160, 2, 3, e6), ("ir_k3", 320, 1, 1, e6)],
    # final stage, equivalent to 7->1, ignored
]

IRF_CFG = {"less_se_channels": False}


FBNetV3_A_dsmask = [
    [
        ("conv_k3", 16, 2, 1),
        ("ir_k3", 16, 1, 1, {"expansion": 1}, IRF_CFG)
    ],
    [
        ("ir_k5", 32, 2, 1, {"expansion": 4}, IRF_CFG),
        ("ir_k5", 32, 1, 1, {"expansion": 2}, IRF_CFG),
    ],
    [
        ("ir_k5", 40, 2, 1, {"expansion": 4}, IRF_CFG),
        ("ir_k3", 40, 1, 3, {"expansion": 3}, IRF_CFG),
    ],
    [
        ("ir_k5", 72, 2, 1, {"expansion": 4}, IRF_CFG),
        ("ir_k3", 72, 1, 3, {"expansion": 3}, IRF_CFG),
        ("ir_k5", 112, 1, 1, {"expansion": 4}, IRF_CFG),
        ("ir_k5", 112, 1, 3, {"expansion": 4}, IRF_CFG),
    ],
    [
        ("ir_k5", 184, 2, 1, {"expansion": 4}, IRF_CFG),
        ("ir_k3", 184, 1, 4, {"expansion": 4}, IRF_CFG),
        ("ir_k5", 200, 1, 1, {"expansion": 6}, IRF_CFG),
    ],
]

FBNetV3_A_dsmask_tiny = [
    [
        ("conv_k3", 8, 2, 1),
        ("ir_k3", 8, 1, 1, {"expansion": 1}, IRF_CFG)
    ],
    [
        ("ir_k5", 16, 2, 1, {"expansion": 3}, IRF_CFG),
        ("ir_k5", 16, 1, 1, {"expansion": 2}, IRF_CFG),
    ],
    [
        ("ir_k5", 24, 2, 1, {"expansion": 4}, IRF_CFG),
        ("ir_k3", 24, 1, 2, {"expansion": 3}, IRF_CFG),
    ],
    [
        ("ir_k5", 40, 2, 1, {"expansion": 4}, IRF_CFG),
        ("ir_k3", 40, 1, 2, {"expansion": 3}, IRF_CFG),
        ("ir_k5", 64, 1, 1, {"expansion": 4}, IRF_CFG),
        ("ir_k5", 64, 1, 2, {"expansion": 3}, IRF_CFG),
    ],
    [
        ("ir_k5", 92, 2, 1, {"expansion": 4}, IRF_CFG),
        ("ir_k3", 92, 1, 2, {"expansion": 4}, IRF_CFG),
        ("ir_k5", 92, 1, 1, {"expansion": 6}, IRF_CFG),
    ],
]

FBNetV3_A = [
    # FBNetV3 arch without hs
    [
        ("conv_k3", 16, 2, 1),
        ("ir_k3", 16, 1, 2, {"expansion": 1}, IRF_CFG)
    ],
    [
        ("ir_k5", 24, 2, 1, {"expansion": 4}, IRF_CFG),
        ("ir_k5", 24, 1, 3, {"expansion": 3}, IRF_CFG),
    ],
    [
        ("ir_k5_se", 32, 2, 1, {"expansion": 4}, IRF_CFG),
        ("ir_k3_se", 32, 1, 3, {"expansion": 3}, IRF_CFG),
    ],
    [
        ("ir_k5", 64, 2, 1, {"expansion": 4}, IRF_CFG),
        ("ir_k3", 64, 1, 3, {"expansion": 3}, IRF_CFG),
        ("ir_k5_se", 112, 1, 1, {"expansion": 4}, IRF_CFG),
        ("ir_k5_se", 112, 1, 5, {"expansion": 3}, IRF_CFG),
    ],
    [
        ("ir_k5_se", 184, 2, 1, {"expansion": 4}, IRF_CFG),
        ("ir_k3_se", 184, 1, 4, {"expansion": 4}, IRF_CFG),
        ("ir_k5_se", 200, 1, 1, {"expansion": 6}, IRF_CFG),
    ],
]

FBNetV3_B = [
    [
        ("conv_k3", 16, 2, 1),
        ("ir_k3", 16, 1, 2 , {"expansion": 1}, IRF_CFG)
    ],
    [
        ("ir_k5", 24, 2, 1, {"expansion": 4}, IRF_CFG),
        ("ir_k5", 24, 1, 3, {"expansion": 2}, IRF_CFG),
    ],
    [
        ("ir_k5_se", 40, 2, 1, {"expansion": 5}, IRF_CFG),
        ("ir_k5_se", 40, 1, 4, {"expansion": 3}, IRF_CFG),
    ],
    [
        ("ir_k5", 72, 2, 1, {"expansion": 5}, IRF_CFG),
        ("ir_k3", 72, 1, 4, {"expansion": 3}, IRF_CFG),
        ("ir_k3_se", 120, 1, 1, {"expansion": 5}, IRF_CFG),
        ("ir_k5_se", 120, 1, 5, {"expansion": 3}, IRF_CFG),
    ],
    [
        ("ir_k3_se", 184, 2, 1, {"expansion": 6}, IRF_CFG),
        ("ir_k5_se", 184, 1, 5, {"expansion": 4}, IRF_CFG),
        ("ir_k5_se", 224, 1, 1, {"expansion": 6}, IRF_CFG),
    ],
]


FBNetV3_A_no_se = [
    # FBNetV3 without hs and SE (SE is not quantization friendly)
    [
        ("conv_k3", 16, 2, 1),
        ("ir_k3", 16, 1, 2, {"expansion": 1}, IRF_CFG)
    ],
    [
        ("ir_k5", 24, 2, 1, {"expansion": 4}, IRF_CFG),
        ("ir_k5", 24, 1, 3, {"expansion": 3}, IRF_CFG),
    ],
    [
        ("ir_k5", 32, 2, 1, {"expansion": 4}, IRF_CFG),
        ("ir_k3", 32, 1, 3, {"expansion": 3}, IRF_CFG),
    ],
    [
        ("ir_k5", 64, 2, 1, {"expansion": 4}, IRF_CFG),
        ("ir_k3", 64, 1, 3, {"expansion": 3}, IRF_CFG),
        ("ir_k5", 112, 1, 1, {"expansion": 4}, IRF_CFG),
        ("ir_k5", 112, 1, 5, {"expansion": 3}, IRF_CFG),
    ],
    [
        ("ir_k5", 184, 2, 1, {"expansion": 4}, IRF_CFG),
        ("ir_k3", 184, 1, 4, {"expansion": 4}, IRF_CFG),
        ("ir_k5", 200, 1, 1, {"expansion": 6}, IRF_CFG),
    ],
]

FBNetV3_B_no_se = [
    [
        ("conv_k3", 16, 2, 1),
        ("ir_k3", 16, 1, 2 , {"expansion": 1}, IRF_CFG)
    ],
    [
        ("ir_k5", 24, 2, 1, {"expansion": 4}, IRF_CFG),
        ("ir_k5", 24, 1, 3, {"expansion": 2}, IRF_CFG),
    ],
    [
        ("ir_k5", 40, 2, 1, {"expansion": 5}, IRF_CFG),
        ("ir_k5", 40, 1, 4, {"expansion": 3}, IRF_CFG),
    ],
    [
        ("ir_k5", 72, 2, 1, {"expansion": 5}, IRF_CFG),
        ("ir_k3", 72, 1, 4, {"expansion": 3}, IRF_CFG),
        ("ir_k3", 120, 1, 1, {"expansion": 5}, IRF_CFG),
        ("ir_k5", 120, 1, 5, {"expansion": 3}, IRF_CFG),
    ],
    [
        ("ir_k3", 184, 2, 1, {"expansion": 6}, IRF_CFG),
        ("ir_k5", 184, 1, 5, {"expansion": 4}, IRF_CFG),
        ("ir_k5", 224, 1, 1, {"expansion": 6}, IRF_CFG),
    ],
]


# FBNetV3_B model, a lighter version for real-time inference
FBNetV3_B_light_no_se = [
    [
        ("conv_k3", 16, 2, 1),
        ("ir_k3", 16, 1, 2 , {"expansion": 1}, IRF_CFG)
    ],
    [
        ("ir_k5", 24, 2, 1, {"expansion": 4}, IRF_CFG),
        ("ir_k5", 24, 1, 2, {"expansion": 2}, IRF_CFG),
    ],
    [
        ("ir_k5", 40, 2, 1, {"expansion": 5}, IRF_CFG),
        ("ir_k5", 40, 1, 3, {"expansion": 3}, IRF_CFG),
    ],
    [
        ("ir_k5", 72, 2, 1, {"expansion": 5}, IRF_CFG),
        ("ir_k3", 72, 1, 4, {"expansion": 3}, IRF_CFG),
        ("ir_k3", 120, 1, 1, {"expansion": 5}, IRF_CFG),
        ("ir_k5", 120, 1, 5, {"expansion": 3}, IRF_CFG),
    ],
    [
        ("ir_k3", 184, 2, 1, {"expansion": 6}, IRF_CFG),
        ("ir_k5", 184, 1, 5, {"expansion": 4}, IRF_CFG),
        ("ir_k5", 224, 1, 1, {"expansion": 6}, IRF_CFG),
    ],
]


XIRB16D_STAGES = [
    # NOTE: each stage is a list of (op_type, out_channels, stride, n_repeat, ...)
    # resolution stage 0, equivalent to 224->112
    [("conv_k3", 16, 2, 1), ("ir_k3", 16, 1, 1, e1)],
    # resolution stage 1, equivalent to 112->56
    [("ir_k3", 32, 2, 2, e6)],
    # resolution stage 2, equivalent to 56->28
    [("ir_k3", 48, 2, 3, e6)],
    # resolution stage 3, equivalent to 28->14
    [("ir_k3", 96, 2, 4, e6), ("ir_k3", 128, 1, 3, e6)],
    # resolution stage 4, equivalent to 14->7, ignored
    # final stage, equivalent to 7->1, ignored
]


LARGE_BOX_HEAD_STAGES = [
    [("ir_k3", 160, 2, 1, e4), ("ir_k3", 160, 1, 2, e6), ("ir_k3", 240, 1, 1, e6)],
]


SMALL_BOX_HEAD_STAGES = [
    [("ir_k3", 128, 2, 1, e4), ("ir_k3", 128, 1, 2, e6), ("ir_k3", 160, 1, 1, e6)],
]

TINY_BOX_HEAD_STAGES = [
    [("ir_k3", 64, 2, 1, e4), ("ir_k3", 64, 1, 2, e4), ("ir_k3", 80, 1, 1, e4)],
]

LARGE_UPSAMPLE_HEAD_STAGES = [
    [("ir_k3", 160, 1, 1, e4), ("ir_k3", 160, 1, 3, e6), ("ir_k3", 80, -2, 1, e3)],
]

LARGE_UPSAMPLE_HEAD_D21_STAGES = [
    [("ir_k3", 192, 1, 1, e4), ("ir_k3", 192, 1, 5, e3), ("ir_k3", 96, -2, 1, e3)],
]

SMALL_UPSAMPLE_HEAD_STAGES = [
    [("ir_k3", 128, 1, 1, e4), ("ir_k3", 128, 1, 3, e6), ("ir_k3", 64, -2, 1, e3)],
]


# NOTE: Compared with SMALL_UPSAMPLE_HEAD_STAGES, this does one more down-sample
# in the first "layer" and then up-sample twice
SMALL_DS_UPSAMPLE_HEAD_STAGES = [
    [("ir_k3", 128, 2, 1, e4), ("ir_k3", 128, 1, 2, e6), ("ir_k3", 128, -2, 1, e6), ("ir_k3", 64, -2, 1, e3)],  # noqa
]

TINY_DS_UPSAMPLE_HEAD_STAGES = [
    [("ir_k3", 64, 2, 1, e4), ("ir_k3", 64, 1, 2, e4), ("ir_k3", 64, -2, 1, e4), ("ir_k3", 40, -2, 1, e3)],  # noqa
]

MODEL_ARCH_BUILTIN = {
    "default": {
        "trunk": DEFAULT_STAGES[0:4],
        "rpn": [[_repeat_last(DEFAULT_STAGES[3])]],
        "bbox": LARGE_BOX_HEAD_STAGES,
        "mask": LARGE_UPSAMPLE_HEAD_STAGES,
        "kpts": LARGE_UPSAMPLE_HEAD_STAGES,
        "basic_args": _BASIC_ARGS,
    },
    "default_dsmask": {
        "trunk": DEFAULT_STAGES[0:4],
        "rpn": [[_repeat_last(DEFAULT_STAGES[3])]],
        "bbox": SMALL_BOX_HEAD_STAGES,
        "mask": SMALL_DS_UPSAMPLE_HEAD_STAGES,
        "kpts": SMALL_DS_UPSAMPLE_HEAD_STAGES,
        "basic_args": _BASIC_ARGS,
    },
    "FBNetV3_A_dsmask": {
        "trunk": FBNetV3_A_dsmask[0:4],
        "rpn": [[_repeat_last(FBNetV3_A_dsmask[3])]],
        "bbox": SMALL_BOX_HEAD_STAGES,
        "mask": SMALL_DS_UPSAMPLE_HEAD_STAGES,
        "kpts": SMALL_DS_UPSAMPLE_HEAD_STAGES,
        "basic_args": _BASIC_ARGS,
    },
    "FBNetV3_A_dsmask_tiny": {
        "trunk": FBNetV3_A_dsmask_tiny[0:4],
        "rpn": [[_repeat_last(FBNetV3_A_dsmask_tiny[3])]],
        "bbox": TINY_BOX_HEAD_STAGES,
        "mask": TINY_DS_UPSAMPLE_HEAD_STAGES,
        "kpts": TINY_DS_UPSAMPLE_HEAD_STAGES,
        "basic_args": _BASIC_ARGS,
    },
    "FBNetV3_B_light_large": {
        "trunk": FBNetV3_B_light_no_se[0:4],
        "rpn": [[_repeat_last(FBNetV3_B_light_no_se[3])]],
        "bbox": SMALL_BOX_HEAD_STAGES,
        "mask": SMALL_DS_UPSAMPLE_HEAD_STAGES,
        "kpts": LARGE_UPSAMPLE_HEAD_D21_STAGES,
        "basic_args": _BASIC_ARGS,
    },
    "xirb16d": {
        "trunk": XIRB16D_STAGES[0:4],
        "rpn": [[_repeat_last(XIRB16D_STAGES[3])]],
        "bbox": SMALL_BOX_HEAD_STAGES,
        "mask": SMALL_UPSAMPLE_HEAD_STAGES,
        "kpts": SMALL_UPSAMPLE_HEAD_STAGES,
        "basic_args": _BASIC_ARGS,
    },
    "xirb16d_dsmask": {
        "trunk": XIRB16D_STAGES[0:4],
        "rpn": [[_repeat_last(XIRB16D_STAGES[3])]],
        "bbox": SMALL_BOX_HEAD_STAGES,
        "mask": SMALL_DS_UPSAMPLE_HEAD_STAGES,
        "kpts": SMALL_DS_UPSAMPLE_HEAD_STAGES,
        "basic_args": _BASIC_ARGS,
    },
}

CHAM_V1_STAGES = [
    # NOTE: each stage is a list of (op_type, out_channels, stride, n_repeat, ...)
    # resolution stage 0, equivalent to 224->112
    [("conv_k3", 32, 2, 1), ("ir_k3", 24, 1, 1, _ex(1))],
    # resolution stage 1, equivalent to 112->56
    [("ir_k5", 48, 2, 2, _ex(4))],
    # resolution stage 2, equivalent to 56->28
    [("ir_k3", 64, 2, 5, _ex(7))],
    # resolution stage 3, equivalent to 28->14
    [("ir_k5", 56, 2, 7, _ex(12)), ("ir_k3", 88, 1, 5, _ex(8))],
    # bbox head stage, 2x down-sample (6x6 -> 3x3)
    [("ir_k3", 152, 2, 4, _ex(7)), ("ir_k3", 104, 1, 1, _ex(10))],
]


CHAM_V2_STAGES = [
    # NOTE: each stage is a list of (op_type, out_channels, stride, n_repeat, ...)
    # resolution stage 0, equivalent to 224->112
    [("conv_k3", 32, 2, 1), ("ir_k3", 24, 1, 1, _ex(1))],
    # resolution stage 1, equivalent to 112->56
    [("ir_k5", 32, 2, 4, _ex(8))],
    # resolution stage 2, equivalent to 56->28
    [("ir_k5", 48, 2, 6, _ex(5))],
    # resolution stage 3, equivalent to 28->14
    [("ir_k5", 56, 2, 3, _ex(9)), ("ir_k3", 56, 1, 6, _ex(6))],
    # bbox head stage, 2x down-sample (6x6 -> 3x3)
    [("ir_k3", 160, 2, 6, _ex(2)), ("ir_k3", 112, 1, 1, _ex(6))],
]


MODEL_ARCH_CHAM = {
    "cham_v1": {
        "trunk": CHAM_V1_STAGES[0:4],
        "rpn": [[_repeat_last(CHAM_V1_STAGES[3], n=1)]],
        "bbox": [CHAM_V1_STAGES[4]],
        "basic_args": _BASIC_ARGS,
    },
    "cham_v1a": {
        "trunk": CHAM_V1_STAGES[0:4],
        "rpn": [[_repeat_last(CHAM_V1_STAGES[3], n=3)]],
        "bbox": [CHAM_V1_STAGES[4]],
        "basic_args": _BASIC_ARGS,
    },
    "cham_v2": {
        "trunk": CHAM_V2_STAGES[0:4],
        "rpn": [[_repeat_last(CHAM_V2_STAGES[3], n=1)]],
        "bbox": [CHAM_V2_STAGES[4]],
        "basic_args": _BASIC_ARGS,
    },
    "cham_v2a": {
        "trunk": CHAM_V2_STAGES[0:4],
        "rpn": [[_repeat_last(CHAM_V2_STAGES[3], n=3)]],
        "bbox": [CHAM_V2_STAGES[4]],
        "basic_args": _BASIC_ARGS,
    },
}


# NOTE: a re-balanced version SMALL_UPSAMPLE_HEAD_STAGES
FPN_UPSAMPLE_HEAD_STAGES = [
    [("ir_k3", 96, 1, 1, e6), ("ir_k3", 160, 1, 3, e6), ("ir_k3", 80, -2, 1, e3)],
]


# NOTE: a copy of DEFAULT_STAGES, but does an extra 2x ds in first IRF layer
DEFAULT_FPN_S_STAGES = copy.deepcopy(DEFAULT_STAGES)
assert DEFAULT_FPN_S_STAGES[0][1][2] == 1
DEFAULT_FPN_S_STAGES[0][1] = _mutated_tuple(DEFAULT_FPN_S_STAGES[0][1], pos=2, value=2)


VOGUE_RPN_HEAD_STAGES = [
    [("ir_k3", 96, 1, 1, e6)],
]


VOGUE_DS_MASK_HEAD_STAGES = [
    [("ir_k3", 128, 2, 1, e4), ("ir_k3", 128, 1, 2, e6), ("ir_k3", 80, -2, 1, e3), ("ir_k3", 80, -2, 1, e3)],  # noqa
]


VOGUE_SMALL_MASK_HEAD_STAGES = [
    [("ir_k3", 64, 1, 1, e4), ("ir_k3", 64, 1, 3, e6), ("ir_k3", 80, -2, 1, e3)],
]


VOGUE_M2_STAGES = [
    # NOTE: each stage is a list of (op_type, out_channels, stride, n_repeat, ...)
    # resolution stage 0, equivalent to 224->112
    [("conv_k3", 32, 2, 1), ("ir_k3", 16, 2, 1, e1)],
    # resolution stage 1, equivalent to 112->56
    [("ir_k3", 24, 2, 2, e6)],
    # resolution stage 2, equivalent to 56->28
    [("ir_k3", 32, 2, 3, e6)],
    # resolution stage 3, equivalent to 28->14
    [("ir_k3", 32, 2, 3, e6)],
    # resolution stage 4, equivalent to 14->7
    [("ir_k3", 64, 2, 4, e6), ("ir_k3", 96, 1, 3, e6)],
]


# NOTE: 1.5x smaller than VOGUE_M2_STAGES
VOGUE_M1_STAGES = [
    # NOTE: each stage is a list of (op_type, out_channels, stride, n_repeat, ...)
    # resolution stage 0, equivalent to 224->112
    [("conv_k3", 32, 2, 1), ("ir_k3", 24, 2, 1, e1)],
    # resolution stage 1, equivalent to 112->56
    [("ir_k3", 32, 2, 2, e6)],
    # resolution stage 2, equivalent to 56->28
    [("ir_k3", 48, 2, 3, e6)],
    # resolution stage 3, equivalent to 28->14
    [("ir_k3", 48, 2, 3, e6)],
    # resolution stage 4, equivalent to 14->7
    [("ir_k3", 96, 2, 4, e6), ("ir_k3", 128, 1, 3, e6)],
]


MODEL_ARCH_VOGUE = {
    "default_fpn": {
        "trunk": DEFAULT_STAGES[0:5],  # FPN uses all 5 stages
        "rpn": [[_repeat_last(DEFAULT_STAGES[3], n=1)]],
        "bbox": [DEFAULT_STAGES[4]],
        "mask": FPN_UPSAMPLE_HEAD_STAGES,
        "basic_args": _BASIC_ARGS,
        "kpts": LARGE_UPSAMPLE_HEAD_STAGES,
    },
    "default_fpn_s": {
        "trunk": DEFAULT_FPN_S_STAGES[0:5],  # FPN uses all 5 stages
        "rpn": [[_repeat_last(DEFAULT_STAGES[3], n=1)]],
        "bbox": [DEFAULT_STAGES[4]],
        "mask": FPN_UPSAMPLE_HEAD_STAGES,
        "basic_args": _BASIC_ARGS,
    },
    "FBNetV3_A_fpn": {
        "trunk": FBNetV3_A[0:5],  # FPN uses all 5 stages
        "rpn": [[_repeat_last(FBNetV3_A[3], n=1)]],
        "bbox": [FBNetV3_A[4]],
        "mask": FPN_UPSAMPLE_HEAD_STAGES,
        "basic_args": _BASIC_ARGS,
    },
    "FBNetV3_B_fpn": {
        "trunk": FBNetV3_B[0:5],  # FPN uses all 5 stages
        "rpn": [[_repeat_last(FBNetV3_B[3], n=1)]],
        "bbox": [FBNetV3_B[4]],
        "mask": FPN_UPSAMPLE_HEAD_STAGES,
        "basic_args": _BASIC_ARGS,
    },
    "FBNetV3_A_fpn_no_se": {
        "trunk": FBNetV3_A_no_se[0:5],  # FPN uses all 5 stages
        "rpn": [[_repeat_last(FBNetV3_A_no_se[3], n=1)]],
        "bbox": [FBNetV3_A_no_se[4]],
        "mask": FPN_UPSAMPLE_HEAD_STAGES,
        "basic_args": _BASIC_ARGS,
    },
    "FBNetV3_B_fpn_no_se": {
        "trunk": FBNetV3_B_no_se[0:5],  # FPN uses all 5 stages
        "rpn": [[_repeat_last(FBNetV3_B_no_se[3], n=1)]],
        "bbox": [FBNetV3_B_no_se[4]],
        "mask": FPN_UPSAMPLE_HEAD_STAGES,
        "basic_args": _BASIC_ARGS,
    },
    # model optimized for vogue by wym & stzpz
    "vogue_fpn_s": {
        "trunk": VOGUE_M2_STAGES[0:5],  # FPN uses all 5 stages
        "rpn": VOGUE_RPN_HEAD_STAGES,
        "bbox": LARGE_BOX_HEAD_STAGES,
        "mask": VOGUE_DS_MASK_HEAD_STAGES,
        "basic_args": _BASIC_ARGS,
    },
    # others same as vogue_s, mask head upsample by 2 while shrink the channel size
    # by half
    "vogue_fpn_m2": {
        "trunk": VOGUE_M2_STAGES[0:5],  # FPN uses all 5 stages
        "rpn": VOGUE_RPN_HEAD_STAGES,
        "bbox": LARGE_BOX_HEAD_STAGES,
        "mask": VOGUE_SMALL_MASK_HEAD_STAGES,
        "basic_args": _BASIC_ARGS,
    },
    # others same as vogue_s, backbone channel size x1.5
    "vogue_fpn_m1": {
        "trunk": VOGUE_M1_STAGES[0:5],  # FPN uses all 5 stages
        "rpn": VOGUE_RPN_HEAD_STAGES,
        "bbox": LARGE_BOX_HEAD_STAGES,
        "mask": VOGUE_DS_MASK_HEAD_STAGES,
        "basic_args": _BASIC_ARGS,
    },
}


MODEL_ARCH_SOCIALEYE = {
    "cham_d_d32": {
        "basic_args": {
            "dw_skip_bnrelu": True,
            "bias": False,
            "zero_last_bn_gamma": False,
        },
        "trunk": [
            # TODO: arch is configured such that last output is trunk3
            #   should remove this requirement in regressor meta_arch
            # op, c, s, n, ...
            [("conv_k3", 32, 2, 1), ("ir_k3", 32, 1, 1, e1p)],
            [("ir_k3", 32, 2, 1, e6)],
            [("ir_k3", 32, 2, 2, e4)],
            [("ir_k3", 32, 2, 4, e6), ("ir_k3", 96, 1, 3, e6)],
        ],
    }
}

# A lightweight FBNetV3 architecture for real-time face landmark
FBNetV3_A_light = [
    [
        ("conv_k3", 16, 2, 1),
        ("ir_k3", 8, 1, 1, {"expansion": 1}, IRF_CFG)
    ],
    [
        ("ir_k5", 24, 2, 1, {"expansion": 4}, IRF_CFG),
        ("ir_k5", 24, 1, 1, {"expansion": 2}, IRF_CFG),
    ],
    [
        ("ir_k5", 40, 2, 1, {"expansion": 4}, IRF_CFG),
        ("ir_k3", 40, 1, 2, {"expansion": 3}, IRF_CFG),
    ],
    [
        ("ir_k5", 64, 2, 1, {"expansion": 4}, IRF_CFG),
        ("ir_k3", 64, 1, 3, {"expansion": 3}, IRF_CFG),
        ("ir_k3", 104, 1, 1, {"expansion": 3}, IRF_CFG),
        ("ir_k5", 104, 1, 2, {"expansion": 3}, IRF_CFG),
    ],
    [
        ("ir_k5", 184, 2, 1, {"expansion": 4}, IRF_CFG),
        ("ir_k3", 184, 1, 4, {"expansion": 4}, IRF_CFG),
        ("ir_k5", 200, 1, 1, {"expansion": 6}, IRF_CFG),
    ],
]

FBNetV3_A_tiny = [
    [
        ("conv_k3", 8, 2, 1),
        ("ir_k3", 8, 1, 1, {"expansion": 1}, IRF_CFG)
    ],
    [
        ("ir_k5", 16, 2, 1, {"expansion": 4}, IRF_CFG),
        ("ir_k5", 16, 1, 1, {"expansion": 2}, IRF_CFG),
    ],
    [
        ("ir_k5", 32, 2, 1, {"expansion": 4}, IRF_CFG),
        ("ir_k3", 32, 1, 2, {"expansion": 3}, IRF_CFG),
    ],
    [
        ("ir_k5", 48, 2, 1, {"expansion": 4}, IRF_CFG),
        ("ir_k3", 48, 1, 3, {"expansion": 3}, IRF_CFG),
        ("ir_k3", 80, 1, 1, {"expansion": 3}, IRF_CFG),
        ("ir_k5", 80, 1, 2, {"expansion": 3}, IRF_CFG),
    ],
    [
        ("ir_k5", 144, 2, 1, {"expansion": 4}, IRF_CFG),
        ("ir_k3", 144, 1, 4, {"expansion": 4}, IRF_CFG),
        ("ir_k5", 160, 1, 1, {"expansion": 6}, IRF_CFG),
    ],
]


FBNetV3_G_no_se = [
    [("conv_k3", 32, 2, 1), ("ir_k3", 24, 1, 3, {"expansion": 1}, IRF_CFG)],
    [
        ("ir_k5", 40, 2, 1, {"expansion": 4}, IRF_CFG),
        ("ir_k5", 40, 1, 4, {"expansion": 2}, IRF_CFG),
    ],
    [
        ("ir_k5", 56, 2, 1, {"expansion": 4}, IRF_CFG),
        ("ir_k5", 56, 1, 4, {"expansion": 3}, IRF_CFG),
    ],
    [
        ("ir_k5", 104, 2, 1, {"expansion": 5}, IRF_CFG),
        ("ir_k3", 104, 1, 4, {"expansion": 3}, IRF_CFG),
        ("ir_k3", 160, 1, 1, {"expansion": 5}, IRF_CFG),
        ("ir_k5", 160, 1, 8, {"expansion": 3}, IRF_CFG),
    ],
    [
        ("ir_k3", 264, 2, 1, {"expansion": 6}, IRF_CFG),
        ("ir_k5", 264, 1, 6, {"expansion": 5}, IRF_CFG),
        ("ir_k5", 288, 1, 2, {"expansion": 6}, IRF_CFG),
    ],
]


# For the face landmark keypoint model, we use a much deeper keypoint head due to the
# fact that the number of keypoints (100) is significantly more than others (e.g., COCO)

FACELM_SERVER_BOX_HEAD_STAGES = [
    [("ir_k3", 256, 2, 1, e4), ("ir_k3", 256, 1, 3, e6), ("ir_k3", 320, 1, 1, e6)],
]

FACELM_UPSAMPLE_W288_D27_HEAD_STAGES = [
    [("ir_k3", 288, 1, 1, e6), ("ir_k3", 288, 1, 7, e2), ("ir_k3", 288, -2, 1, e3)],  # noqa
]

FACELM_UPSAMPLE_W256_D27_HEAD_STAGES = [
    [("ir_k3", 256, 1, 1, e6), ("ir_k3", 256, 1, 7, e2), ("ir_k3", 256, -2, 1, e3)],  # noqa
]

FACELM_UPSAMPLE_W160_D18_HEAD_STAGES = [
    [("ir_k3", 160, 1, 1, e6), ("ir_k3", 160, 1, 4, e2), ("ir_k3", 160, -2, 1, e3)],  # noqa
]

FACELM_UPSAMPLE_W160_D15_HEAD_STAGES = [
    [("ir_k3", 160, 1, 1, e6), ("ir_k3", 160, 1, 3, e2), ("ir_k3", 160, -2, 1, e3)],  # noqa
]


FACELM_UPSAMPLE_W360_D30_HEAD_STAGES = [
    [("ir_k3", 360, 1, 1, e6), ("ir_k3", 360, 1, 8, e2), ("ir_k3", 360, -2, 1, e3)],  # noqa
]

FACELM_UPSAMPLE_W360_D36_HEAD_STAGES = [
    [("ir_k3", 360, 1, 1, e6), ("ir_k3", 360, 1, 10, e2), ("ir_k3", 360, -2, 1, e3)],  # noqa
]

FACELM_UPSAMPLE_W384_D42_HEAD_STAGES = [
    [("ir_k3", 384, 1, 1, e6), ("ir_k3", 384, 1, 12, e3), ("ir_k3", 384, -2, 1, e3)],  # noqa
]


MODEL_ARCH_FACELM = {
    # Naming: w - #channels in the keypoint head
    # d - #conv layers in the keypoint head
    "FBNetV3_A_tiny_facelm_w160_d15": {
        "trunk": FBNetV3_A_tiny[0:4],
        "rpn": [[_repeat_last(FBNetV3_A_tiny[3])]],
        "bbox": SMALL_BOX_HEAD_STAGES,
        "mask": SMALL_DS_UPSAMPLE_HEAD_STAGES,
        "kpts": FACELM_UPSAMPLE_W160_D15_HEAD_STAGES,
        "basic_args": _BASIC_ARGS,
    },
    "FBNetV3_A_tiny_facelm_w160_d18": {
        "trunk": FBNetV3_A_tiny[0:4],
        "rpn": [[_repeat_last(FBNetV3_A_tiny[3])]],
        "bbox": SMALL_BOX_HEAD_STAGES,
        "mask": SMALL_DS_UPSAMPLE_HEAD_STAGES,
        "kpts": FACELM_UPSAMPLE_W160_D18_HEAD_STAGES,
        "basic_args": _BASIC_ARGS,
    },
    "FBNetV3_A_light_facelm_w160_d15": {
        "trunk": FBNetV3_A_light[0:4],
        "rpn": [[_repeat_last(FBNetV3_A_light[3])]],
        "bbox": SMALL_BOX_HEAD_STAGES,
        "mask": SMALL_DS_UPSAMPLE_HEAD_STAGES,
        "kpts": FACELM_UPSAMPLE_W160_D15_HEAD_STAGES,
        "basic_args": _BASIC_ARGS,
    },
    "FBNetV3_A_light_facelm_w256_d27": {
        "trunk": FBNetV3_A_light[0:4],
        "rpn": [[_repeat_last(FBNetV3_A_light[3])]],
        "bbox": SMALL_BOX_HEAD_STAGES,
        "mask": SMALL_DS_UPSAMPLE_HEAD_STAGES,
        "kpts": FACELM_UPSAMPLE_W256_D27_HEAD_STAGES,
        "basic_args": _BASIC_ARGS,
    },
    "FBNetV3_A_light_facelm_w288_d27": {
        "trunk": FBNetV3_A_light[0:4],
        "rpn": [[_repeat_last(FBNetV3_A_light[3])]],
        "bbox": SMALL_BOX_HEAD_STAGES,
        "mask": SMALL_DS_UPSAMPLE_HEAD_STAGES,
        "kpts": FACELM_UPSAMPLE_W288_D27_HEAD_STAGES,
        "basic_args": _BASIC_ARGS,
    },
    "FBNetV3_A_facelm_small": {
        "trunk": FBNetV3_A_dsmask[0:4],
        "rpn": [[_repeat_last(FBNetV3_A_dsmask[3])]],
        "bbox": SMALL_BOX_HEAD_STAGES,
        "mask": SMALL_DS_UPSAMPLE_HEAD_STAGES,
        "kpts": FACELM_UPSAMPLE_W360_D30_HEAD_STAGES,
        "basic_args": _BASIC_ARGS,
    },
    "FBNetV3_A_facelm_large": {
        "trunk": FBNetV3_A_dsmask[0:4],
        "rpn": [[_repeat_last(FBNetV3_A_dsmask[3])]],
        "bbox": SMALL_BOX_HEAD_STAGES,
        "mask": SMALL_DS_UPSAMPLE_HEAD_STAGES,
        "kpts": FACELM_UPSAMPLE_W360_D36_HEAD_STAGES,
        "basic_args": _BASIC_ARGS,
    },
    "FBNetV3_G_facelm_server": {
        "trunk": FBNetV3_G_no_se[0:4],
        "rpn": [[_repeat_last(FBNetV3_G_no_se[3])]],
        "bbox": FACELM_SERVER_BOX_HEAD_STAGES,
        "mask": SMALL_DS_UPSAMPLE_HEAD_STAGES,
        "kpts": FACELM_UPSAMPLE_W384_D42_HEAD_STAGES,
        "basic_args": _BASIC_ARGS,
    },
}


FBNetV2ModelArch.add_archs(MODEL_ARCH_BUILTIN)
FBNetV2ModelArch.add_archs(MODEL_ARCH_CHAM)
FBNetV2ModelArch.add_archs(MODEL_ARCH_VOGUE)
FBNetV2ModelArch.add_archs(MODEL_ARCH_SOCIALEYE)
FBNetV2ModelArch.add_archs(MODEL_ARCH_FACELM)
