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
}

FBNetV2ModelArch.add_archs(MODEL_ARCH_BUILTIN)
