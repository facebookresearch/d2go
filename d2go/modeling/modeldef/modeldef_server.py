#!/usr/bin/env python3

import copy

from mobile_cv.arch.fbnet_v2.modeldef_registry import FBNetV2ModelArch
from mobile_cv.arch.fbnet_v2.modeldef_utils import _ex, e1, e2, e3, e4, e6


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
    assert all(isinstance(x, (tuple, list)) for x in stage)
    last_layer = copy.deepcopy(stage[-1])
    if n is not None:
        last_layer = _mutated_tuple(last_layer, 3, n)
    return last_layer


_BASIC_ARGS = {
    "dw_skip_bnrelu": True,
    "width_divisor": 8,
    # uncomment below (always_pw and bias) to match model definition of the
    # FBNetV1 builder.
    # "always_pw": True,
    # "bias": False,
    # temporarily disable zero_last_bn_gamma
    "zero_last_bn_gamma": False,
}


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
SMALL_BOX_HEAD_STAGES = {
    "type1": [
        [("ir_k3", 128, 2, 1, e4), ("ir_k3", 128, 1, 2, e6), ("ir_k3", 160, 1, 1, e6)],
    ],
    "type2": [
        [("ir_k3", 240, 2, 1, e2), ("ir_k3", 240, 1, 2, e2), ("ir_k3", 320, 1, 1, e2)],
    ],
    # for large classes
    "type3": [
        [
            ("ir_k3", 128, 2, 1, e4),
            ("ir_k3", 128, 1, 2, e6),
            ("ir_k3", 160, 1, 1, e6),
            ("ir_pool", 1600, 1, 1, e6),
        ],
    ],
    "type4": [
        [
            ("ir_k3", 128, 1, 1, e4),
            ("ir_k3", 128, 1, 2, e6),
            ("ir_k3", 160, 1, 1, e6),
            ("ir_pool", 1600, 1, 1, e6),
        ],
    ],
}


MODEL_ARCH_BUILTIN = {
    "xirb16d_a": {
        "trunk": XIRB16D_STAGES[0:4],
        "rpn": [[_repeat_last(XIRB16D_STAGES[3])]],
        "bbox": SMALL_BOX_HEAD_STAGES["type2"],
        "basic_args": _BASIC_ARGS,
    },
    "xirb16d_b": {
        "trunk": XIRB16D_STAGES[0:4],
        "rpn": [[_repeat_last(XIRB16D_STAGES[3])]],
        "bbox": LARGE_BOX_HEAD_STAGES,
        "basic_args": _BASIC_ARGS,
    },
    "xirb16dh3": {
        "trunk": XIRB16D_STAGES[0:4],
        "rpn": [[_repeat_last(XIRB16D_STAGES[3])]],
        "bbox": SMALL_BOX_HEAD_STAGES["type3"],
        "basic_args": _BASIC_ARGS,
    },
    "xirb16dh4": {
        "trunk": XIRB16D_STAGES[0:4],
        "rpn": [[_repeat_last(XIRB16D_STAGES[3])]],
        "bbox": SMALL_BOX_HEAD_STAGES["type4"],
        "basic_args": _BASIC_ARGS,
    },
}
FBNetV2ModelArch.add_archs(MODEL_ARCH_BUILTIN)


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

CHAM_V3_STAGES = [
    # NOTE: each stage is a list of (op_type, out_channels, stride, n_repeat, ...)
    # resolution stage 0, equivalent to 224->112
    [("conv_k3", 32, 2, 1), ("ir_k3", 24, 1, 1, _ex(1))],
    # resolution stage 1, equivalent to 112->56
    [("ir_k5", 32, 2, 4, _ex(8))],
    # resolution stage 2, equivalent to 56->28
    [("ir_k5", 48, 2, 6, _ex(5))],
    # resolution stage 3, equivalent to 28->14
    [("ir_k5", 56, 2, 3, _ex(9)), ("ir_k3", 96, 1, 4, _ex(6))],
    # bbox head stage, 2x down-sample (6x6 -> 3x3)
    [("ir_k3", 160, 2, 6, _ex(2)), ("ir_k3", 240, 1, 1, _ex(6))],
]

CHAM_RPN_STAGES = [
    [("ir_k3", 96, 1, 1, _ex(6))],
]


MODEL_ARCH_CHAM = {
    "cham2h3": {
        "trunk": CHAM_V2_STAGES[0:4],
        "rpn": CHAM_RPN_STAGES,
        "bbox": SMALL_BOX_HEAD_STAGES["type3"],
        "basic_args": _BASIC_ARGS,
    },
    "cham3h3": {
        "trunk": CHAM_V3_STAGES[0:4],
        "rpn": CHAM_RPN_STAGES,
        "bbox": SMALL_BOX_HEAD_STAGES["type3"],
        "basic_args": _BASIC_ARGS,
    },
}
FBNetV2ModelArch.add_archs(MODEL_ARCH_CHAM)


IRF_CFG = {"less_se_channels": False}
FBNET_MODELS = {
    "fbnet_c": [
        # [op, c, s, n, ...]
        # stage 0
        [("conv_k3", 16, 2, 1), ("ir_k3", 16, 1, 1, e1)],
        # stage 1
        [
            ("ir_k3", 24, 2, 1, e6),
            ("skip", 24, 1, 1, e1),
            ("ir_k3", 24, 1, 1, e1),
            ("ir_k3", 24, 1, 1, e1),
        ],
        # stage 2
        [
            ("ir_k5", 32, 2, 1, e6),
            ("ir_k5", 32, 1, 1, e3),
            # ("ir_k3_sep", 32, 1, 1, e6),
            ("ir_k5", 32, 1, 1, e6),
            ("ir_k3", 32, 1, 1, e6),
        ],
        # stage 3
        [
            ("ir_k5", 64, 2, 1, e6),
            ("ir_k5", 64, 1, 1, e3),
            ("ir_k5", 64, 1, 1, e6),
            ("ir_k5", 64, 1, 1, e6),
            ("ir_k5", 112, 1, 1, e6),
            #  ("ir_k3_sep", 112, 1, 1, e6),
            ("ir_k5", 112, 1, 1, e6),
            ("ir_k5", 112, 1, 1, e6),
            ("ir_k5", 112, 1, 1, e3),
        ],
        # stage 4
        [
            ("ir_k5", 184, 2, 1, e6),
            ("ir_k5", 184, 1, 1, e6),
            ("ir_k5", 184, 1, 1, e6),
            ("ir_k5", 184, 1, 1, e6),
            ("ir_k3", 352, 1, 1, e6),
            ("conv_k1", 1984, 1, 1),
        ],
    ],
    "fbnet_cse": [
        # [op, c, s, n, ...]
        # stage 0
        [("conv_k3_hs", 16, 2, 1), ("ir_k3", 16, 1, 1, e1)],
        # stage 1
        [
            ("ir_k3", 24, 2, 1, e6),
            ("skip", 24, 1, 1),
            ("ir_k3", 24, 1, 1, e1),
            ("ir_k3", 24, 1, 1, e1),
        ],
        # stage 2
        [
            ("ir_k5_sehsig", 32, 2, 1, e6),
            ("ir_k5_sehsig", 32, 1, 1, e3),
            ("ir_k5_sehsig", 32, 1, 1, e6),
            ("ir_k3_sehsig", 32, 1, 1, e6),
        ],
        # stage 3
        [
            ("ir_k5_hs", 64, 2, 1, e6),
            ("ir_k5_hs", 64, 1, 1, e3),
            ("ir_k5_hs", 64, 1, 1, e6),
            ("ir_k5_hs", 64, 1, 1, e6),
            ("ir_k5_hs", 112, 1, 1, e6),
            ("ir_k5_sehsig_hs", 112, 1, 1, e6),
            ("ir_k5_sehsig_hs", 112, 1, 1, e6),
            ("ir_k5_sehsig_hs", 112, 1, 1, e3),
        ],
        # stage 4
        [
            ("ir_k5_sehsig_hs", 184, 2, 1, e6),
            ("ir_k5_sehsig_hs", 184, 1, 1, e6),
            ("ir_k5_sehsig_hs", 184, 1, 1, e6),
            ("ir_k5_sehsig_hs", 184, 1, 1, e6),
            ("ir_pool_hs", 1984, 1, 1, e6),
        ],
    ],
    "dmasking_f5": [
        [("conv_k3_hs", 16, 2, 1), ("ir_k3_hs", 16, 1, 1, e1, IRF_CFG)],
        [
            ("ir_k5_hs", 24, 2, 1, _ex(5.4566), IRF_CFG),
            ("ir_k5_hs", 24, 1, 1, _ex(1.7912), IRF_CFG),
            ("ir_k3_hs", 24, 1, 1, _ex(1.7912), IRF_CFG),
            ("ir_k5_hs", 24, 1, 1, _ex(1.7912), IRF_CFG),
        ],
        [
            ("ir_k5_sehsig", 40, 2, 1, _ex(5.3501), IRF_CFG),
            ("ir_k5_sehsig", 32, 1, 1, _ex(3.5379), IRF_CFG),
            ("ir_k5_sehsig", 32, 1, 1, _ex(4.5379), IRF_CFG),
            ("ir_k5_sehsig", 32, 1, 1, _ex(4.5379), IRF_CFG),
        ],
        [
            ("ir_k5_hs", 64, 2, 1, _ex(5.7133), IRF_CFG),
            ("ir_k3_hs", 64, 1, 1, _ex(2.1212), IRF_CFG),
            ("skip", 64, 1, 1, _ex(3.1246), IRF_CFG),
            ("ir_k3_hs", 64, 1, 1, _ex(3.1246), IRF_CFG),
            ("ir_k3_hs", 112, 1, 1, _ex(5.0333), IRF_CFG),
            ("ir_k5_sehsig_hs", 112, 1, 1, _ex(2.5070), IRF_CFG),
            ("ir_k5_sehsig_hs", 112, 1, 1, _ex(1.7712), IRF_CFG),
            ("ir_k5_sehsig_hs", 112, 1, 1, _ex(2.7712), IRF_CFG),
            ("ir_k5_sehsig_hs", 112, 1, 1, _ex(3.7712), IRF_CFG),
            ("ir_k5_sehsig_hs", 112, 1, 1, _ex(3.7712), IRF_CFG),
        ],
        [
            ("ir_k3_sehsig_hs", 184, 2, 1, _ex(5.5685), IRF_CFG),
            ("ir_k5_sehsig_hs", 184, 1, 1, _ex(2.8400), IRF_CFG),
            ("ir_k5_sehsig_hs", 184, 1, 1, _ex(2.8400), IRF_CFG),
            ("ir_k5_sehsig_hs", 184, 1, 1, _ex(4.8754), IRF_CFG),
            ("ir_k5_sehsig_hs", 184, 1, 1, _ex(4.8754), IRF_CFG),
            ("skip", 224, 1, 1, _ex(6.5245), IRF_CFG),
            ("ir_pool_hs", 1984, 1, 1, e6),
        ],
    ],
}
MODEL_ARCH_FBNET = {
    "fbnet_c": {
        "trunk": FBNET_MODELS["fbnet_c"][0:4],
        "rpn": [[_repeat_last(FBNET_MODELS["fbnet_c"][3], 1)]],
        "bbox": [FBNET_MODELS["fbnet_c"][4]],
        "basic_args": _BASIC_ARGS,
    },
    "fbnet_c_h3": {
        "trunk": FBNET_MODELS["fbnet_c"][0:4],
        "rpn": [[_repeat_last(FBNET_MODELS["fbnet_c"][3], 1)]],
        "bbox": SMALL_BOX_HEAD_STAGES["type3"],
        "basic_args": _BASIC_ARGS,
    },
    "fbnet_cse": {
        "trunk": FBNET_MODELS["fbnet_cse"][0:4],
        "rpn": [[_repeat_last(FBNET_MODELS["fbnet_cse"][3], 1)]],
        "bbox": [FBNET_MODELS["fbnet_cse"][4]],
        "basic_args": _BASIC_ARGS,
    },
    "fbnet_cse_h3": {
        "trunk": FBNET_MODELS["fbnet_cse"][0:4],
        "rpn": [[_repeat_last(FBNET_MODELS["fbnet_cse"][3], 1)]],
        "bbox": SMALL_BOX_HEAD_STAGES["type3"],
        "basic_args": _BASIC_ARGS,
    },
    "dmasking_f5": {
        "trunk": FBNET_MODELS["dmasking_f5"][0:4],
        "rpn": [[_repeat_last(FBNET_MODELS["dmasking_f5"][3], 1)]],
        "bbox": [FBNET_MODELS["dmasking_f5"][4]],
        "basic_args": _BASIC_ARGS,
    },
    "dmasking_f5_h3": {
        "trunk": FBNET_MODELS["dmasking_f5"][0:4],
        "rpn": [[_repeat_last(FBNET_MODELS["dmasking_f5"][3], 1)]],
        "bbox": SMALL_BOX_HEAD_STAGES["type3"],
        "basic_args": _BASIC_ARGS,
    },
}
FBNetV2ModelArch.add_archs(MODEL_ARCH_FBNET)

FBNetV3_A = [
    # FBNetV3 arch without hs
    [("conv_k3", 16, 2, 1), ("ir_k3", 16, 1, 2, {"expansion": 1}, IRF_CFG)],
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
    [("conv_k3", 16, 2, 1), ("ir_k3", 16, 1, 2, {"expansion": 1}, IRF_CFG)],
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

FBNetV3_C = [
    [("conv_k3", 16, 2, 1), ("ir_k3", 16, 1, 2, {"expansion": 1}, IRF_CFG)],
    [
        ("ir_k5", 24, 2, 1, {"expansion": 5}, IRF_CFG),
        ("ir_k3", 24, 1, 4, {"expansion": 3}, IRF_CFG),
    ],
    [
        ("ir_k5_se", 48, 2, 1, {"expansion": 5}, IRF_CFG),
        ("ir_k5_se", 48, 1, 4, {"expansion": 2}, IRF_CFG),
    ],
    [
        ("ir_k5", 88, 2, 1, {"expansion": 4}, IRF_CFG),
        ("ir_k3", 88, 1, 4, {"expansion": 3}, IRF_CFG),
        ("ir_k3_se", 120, 1, 1, {"expansion": 4}, IRF_CFG),
        ("ir_k5_se", 120, 1, 5, {"expansion": 3}, IRF_CFG),
    ],
    [
        ("ir_k5_se", 216, 2, 1, {"expansion": 5}, IRF_CFG),
        ("ir_k5_se", 216, 1, 5, {"expansion": 5}, IRF_CFG),
        ("ir_k5_se", 216, 1, 1, {"expansion": 6}, IRF_CFG),
    ],
]

FBNetV3_D = [
    [("conv_k3", 24, 2, 1), ("ir_k3", 16, 1, 2, {"expansion": 1}, IRF_CFG)],
    [
        ("ir_k3", 24, 2, 1, {"expansion": 5}, IRF_CFG),
        ("ir_k3", 24, 1, 5, {"expansion": 2}, IRF_CFG),
    ],
    [
        ("ir_k5_se", 40, 2, 1, {"expansion": 4}, IRF_CFG),
        ("ir_k3_se", 40, 1, 4, {"expansion": 3}, IRF_CFG),
    ],
    [
        ("ir_k3", 72, 2, 1, {"expansion": 5}, IRF_CFG),
        ("ir_k3", 72, 1, 4, {"expansion": 3}, IRF_CFG),
        ("ir_k3_se", 128, 1, 1, {"expansion": 5}, IRF_CFG),
        ("ir_k5_se", 128, 1, 6, {"expansion": 3}, IRF_CFG),
    ],
    [
        ("ir_k3_se", 208, 2, 1, {"expansion": 6}, IRF_CFG),
        ("ir_k5_se", 208, 1, 5, {"expansion": 5}, IRF_CFG),
        ("ir_k5_se", 240, 1, 1, {"expansion": 6}, IRF_CFG),
    ],
]

FBNetV3_E = [
    [("conv_k3", 24, 2, 1), ("ir_k3", 16, 1, 3, {"expansion": 1}, IRF_CFG)],
    [
        ("ir_k5", 24, 2, 1, {"expansion": 4}, IRF_CFG),
        ("ir_k5", 24, 1, 4, {"expansion": 2}, IRF_CFG),
    ],
    [
        ("ir_k5_se", 48, 2, 1, {"expansion": 4}, IRF_CFG),
        ("ir_k5_se", 48, 1, 4, {"expansion": 3}, IRF_CFG),
    ],
    [
        ("ir_k5", 80, 2, 1, {"expansion": 5}, IRF_CFG),
        ("ir_k3", 80, 1, 4, {"expansion": 3}, IRF_CFG),
        ("ir_k3_se", 128, 1, 1, {"expansion": 5}, IRF_CFG),
        ("ir_k5_se", 128, 1, 7, {"expansion": 3}, IRF_CFG),
    ],
    [
        ("ir_k3_se", 216, 2, 1, {"expansion": 6}, IRF_CFG),
        ("ir_k5_se", 216, 1, 5, {"expansion": 5}, IRF_CFG),
        ("ir_k5_se", 240, 1, 1, {"expansion": 6}, IRF_CFG),
    ],
]

FBNetV3_F = [
    [("conv_k3", 24, 2, 1), ("ir_k3", 24, 1, 3, {"expansion": 1}, IRF_CFG)],
    [
        ("ir_k5", 32, 2, 1, {"expansion": 4}, IRF_CFG),
        ("ir_k5", 32, 1, 4, {"expansion": 2}, IRF_CFG),
    ],
    [
        ("ir_k5_se", 56, 2, 1, {"expansion": 4}, IRF_CFG),
        ("ir_k5_se", 56, 1, 4, {"expansion": 3}, IRF_CFG),
    ],
    [
        ("ir_k5", 88, 2, 1, {"expansion": 5}, IRF_CFG),
        ("ir_k3", 88, 1, 4, {"expansion": 3}, IRF_CFG),
        ("ir_k3_se", 144, 1, 1, {"expansion": 5}, IRF_CFG),
        ("ir_k5_se", 144, 1, 8, {"expansion": 3}, IRF_CFG),
    ],
    [
        ("ir_k3_se", 248, 2, 1, {"expansion": 6}, IRF_CFG),
        ("ir_k5_se", 248, 1, 6, {"expansion": 5}, IRF_CFG),
        ("ir_k5_se", 272, 1, 1, {"expansion": 6}, IRF_CFG),
    ],
]

FBNetV3_G = [
    [("conv_k3", 32, 2, 1), ("ir_k3", 24, 1, 3, {"expansion": 1}, IRF_CFG)],
    [
        ("ir_k5", 40, 2, 1, {"expansion": 4}, IRF_CFG),
        ("ir_k5", 40, 1, 4, {"expansion": 2}, IRF_CFG),
    ],
    [
        ("ir_k5_se", 56, 2, 1, {"expansion": 4}, IRF_CFG),
        ("ir_k5_se", 56, 1, 4, {"expansion": 3}, IRF_CFG),
    ],
    [
        ("ir_k5", 104, 2, 1, {"expansion": 5}, IRF_CFG),
        ("ir_k3", 104, 1, 4, {"expansion": 3}, IRF_CFG),
        ("ir_k3_se", 160, 1, 1, {"expansion": 5}, IRF_CFG),
        ("ir_k5_se", 160, 1, 8, {"expansion": 3}, IRF_CFG),
    ],
    [
        ("ir_k3_se", 264, 2, 1, {"expansion": 6}, IRF_CFG),
        ("ir_k5_se", 264, 1, 6, {"expansion": 5}, IRF_CFG),
        ("ir_k5_se", 288, 1, 2, {"expansion": 6}, IRF_CFG),
    ],
]

FBNetV3_H = [
    [("conv_k3", 48, 2, 1), ("ir_k3", 32, 1, 4, {"expansion": 1}, IRF_CFG)],
    [
        ("ir_k5", 64, 2, 1, {"expansion": 4}, IRF_CFG),
        ("ir_k5", 64, 1, 6, {"expansion": 2}, IRF_CFG),
    ],
    [
        ("ir_k5_se", 80, 2, 1, {"expansion": 4}, IRF_CFG),
        ("ir_k5_se", 80, 1, 6, {"expansion": 3}, IRF_CFG),
    ],
    [
        ("ir_k5", 160, 2, 1, {"expansion": 5}, IRF_CFG),
        ("ir_k3", 160, 1, 6, {"expansion": 3}, IRF_CFG),
        ("ir_k3_se", 240, 1, 1, {"expansion": 5}, IRF_CFG),
        ("ir_k5_se", 240, 1, 12, {"expansion": 3}, IRF_CFG),
    ],
    [
        ("ir_k3_se", 400, 2, 1, {"expansion": 6}, IRF_CFG),
        ("ir_k5_se", 400, 1, 8, {"expansion": 5}, IRF_CFG),
        ("ir_k5_se", 480, 1, 3, {"expansion": 6}, IRF_CFG),
    ],
]


# giant FBNetV3 models
FBNetV3_1s = [
    [("conv_k3", 32, 2, 1), ("ir_k3", 32, 1, 3, {"expansion": 1}, IRF_CFG)],
    [
        ("ir_k5", 56, 2, 1, {"expansion": 4}, IRF_CFG),
        ("ir_k5", 56, 1, 4, {"expansion": 2}, IRF_CFG),
    ],
    [
        ("ir_k5_se", 64, 2, 1, {"expansion": 4}, IRF_CFG),
        ("ir_k5_se", 64, 1, 4, {"expansion": 3}, IRF_CFG),
    ],
    [
        ("ir_k5", 128, 2, 1, {"expansion": 5}, IRF_CFG),
        ("ir_k3", 128, 1, 4, {"expansion": 3}, IRF_CFG),
        ("ir_k3_se", 256, 1, 1, {"expansion": 5}, IRF_CFG),
        ("ir_k5_se", 256, 1, 8, {"expansion": 3}, IRF_CFG),
    ],
    [
        ("ir_k3_se", 384, 2, 1, {"expansion": 6}, IRF_CFG),
        ("ir_k5_se", 384, 1, 6, {"expansion": 5}, IRF_CFG),
        ("ir_k5_se", 512, 1, 2, {"expansion": 6}, IRF_CFG),
    ],
]


FBNetV3_2s = [
    [("conv_k3", 40, 2, 1), ("ir_k3", 40, 1, 5, {"expansion": 1}, IRF_CFG)],
    [
        ("ir_k5", 72, 2, 1, {"expansion": 4}, IRF_CFG),
        ("ir_k5", 72, 1, 6, {"expansion": 2}, IRF_CFG),
    ],
    [
        ("ir_k5_se", 96, 2, 1, {"expansion": 4}, IRF_CFG),
        ("ir_k5_se", 96, 1, 6, {"expansion": 3}, IRF_CFG),
    ],
    [
        ("ir_k5", 160, 2, 1, {"expansion": 5}, IRF_CFG),
        ("ir_k3", 160, 1, 6, {"expansion": 3}, IRF_CFG),
        ("ir_k3_se", 320, 1, 1, {"expansion": 5}, IRF_CFG),
        ("ir_k5_se", 320, 1, 10, {"expansion": 3}, IRF_CFG),
    ],
    [
        ("ir_k3_se", 480, 2, 1, {"expansion": 6}, IRF_CFG),
        ("ir_k5_se", 480, 1, 7, {"expansion": 5}, IRF_CFG),
        ("ir_k5_se", 640, 1, 2, {"expansion": 6}, IRF_CFG),
    ],
]


FBNetV3_3s = [
    [("conv_k3", 48, 2, 1), ("ir_k3", 48, 1, 5, {"expansion": 1}, IRF_CFG)],
    [
        ("ir_k5", 80, 2, 1, {"expansion": 4}, IRF_CFG),
        ("ir_k5", 80, 1, 8, {"expansion": 2}, IRF_CFG),
    ],
    [
        ("ir_k5_se", 112, 2, 1, {"expansion": 4}, IRF_CFG),
        ("ir_k5_se", 112, 1, 8, {"expansion": 3}, IRF_CFG),
    ],
    [
        ("ir_k5", 168, 2, 1, {"expansion": 5}, IRF_CFG),
        ("ir_k3", 168, 1, 8, {"expansion": 3}, IRF_CFG),
        ("ir_k3_se", 360, 1, 1, {"expansion": 5}, IRF_CFG),
        ("ir_k5_se", 360, 1, 11, {"expansion": 3}, IRF_CFG),
    ],
    [
        ("ir_k3_se", 512, 2, 1, {"expansion": 6}, IRF_CFG),
        ("ir_k5_se", 512, 1, 8, {"expansion": 5}, IRF_CFG),
        ("ir_k5_se", 688, 1, 2, {"expansion": 6}, IRF_CFG),
    ],
]

FBNetV3_F_no_se = [
    [("conv_k3", 32, 2, 1), ("ir_k3", 24, 1, 3, {"expansion": 1}, IRF_CFG)],
    [
        ("ir_k5", 40, 2, 1, {"expansion": 4}, IRF_CFG),
        ("ir_k5", 40, 1, 4, {"expansion": 2}, IRF_CFG),
    ],
    [
        ("ir_k5", 64, 2, 1, {"expansion": 4}, IRF_CFG),
        ("ir_k5", 64, 1, 4, {"expansion": 3}, IRF_CFG),
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
        ("ir_k5", 288, 1, 1, {"expansion": 6}, IRF_CFG),
    ],
]

FBNetV3_H_no_se = [
    [("conv_k3", 48, 2, 1), ("ir_k3", 32, 1, 4, {"expansion": 1}, IRF_CFG)],
    [
        ("ir_k5", 64, 2, 1, {"expansion": 4}, IRF_CFG),
        ("ir_k5", 64, 1, 6, {"expansion": 2}, IRF_CFG),
    ],
    [
        ("ir_k5", 80, 2, 1, {"expansion": 4}, IRF_CFG),
        ("ir_k5", 80, 1, 6, {"expansion": 3}, IRF_CFG),
    ],
    [
        ("ir_k5", 160, 2, 1, {"expansion": 5}, IRF_CFG),
        ("ir_k3", 160, 1, 6, {"expansion": 3}, IRF_CFG),
        ("ir_k3", 240, 1, 1, {"expansion": 5}, IRF_CFG),
        ("ir_k5", 240, 1, 12, {"expansion": 3}, IRF_CFG),
    ],
    [
        ("ir_k3", 400, 2, 1, {"expansion": 6}, IRF_CFG),
        ("ir_k5", 400, 1, 8, {"expansion": 5}, IRF_CFG),
        ("ir_k5", 480, 1, 3, {"expansion": 6}, IRF_CFG),
    ],
]


# giant FBNetV3 models
FBNetV3_1s_no_se = [
    [("conv_k3", 32, 2, 1), ("ir_k3", 32, 1, 3, {"expansion": 1}, IRF_CFG)],
    [
        ("ir_k5", 56, 2, 1, {"expansion": 4}, IRF_CFG),
        ("ir_k5", 56, 1, 4, {"expansion": 2}, IRF_CFG),
    ],
    [
        ("ir_k5", 80, 2, 1, {"expansion": 4}, IRF_CFG),
        ("ir_k5", 80, 1, 4, {"expansion": 3}, IRF_CFG),
    ],
    [
        ("ir_k5", 128, 2, 1, {"expansion": 5}, IRF_CFG),
        ("ir_k3", 128, 1, 4, {"expansion": 3}, IRF_CFG),
        ("ir_k3", 272, 1, 1, {"expansion": 5}, IRF_CFG),
        ("ir_k5", 272, 1, 8, {"expansion": 3}, IRF_CFG),
    ],
    [
        ("ir_k3", 400, 2, 1, {"expansion": 6}, IRF_CFG),
        ("ir_k5", 400, 1, 6, {"expansion": 5}, IRF_CFG),
        ("ir_k5", 528, 1, 2, {"expansion": 6}, IRF_CFG),
    ],
]

FBNetV3_2s_no_se = [
    [("conv_k3", 40, 2, 1), ("ir_k3", 40, 1, 5, {"expansion": 1}, IRF_CFG)],
    [
        ("ir_k5", 72, 2, 1, {"expansion": 4}, IRF_CFG),
        ("ir_k5", 72, 1, 6, {"expansion": 2}, IRF_CFG),
    ],
    [
        ("ir_k5", 96, 2, 1, {"expansion": 4}, IRF_CFG),
        ("ir_k5", 96, 1, 6, {"expansion": 3}, IRF_CFG),
    ],
    [
        ("ir_k5", 160, 2, 1, {"expansion": 5}, IRF_CFG),
        ("ir_k3", 160, 1, 6, {"expansion": 3}, IRF_CFG),
        ("ir_k3", 320, 1, 1, {"expansion": 5}, IRF_CFG),
        ("ir_k5", 320, 1, 10, {"expansion": 3}, IRF_CFG),
    ],
    [
        ("ir_k3", 480, 2, 1, {"expansion": 6}, IRF_CFG),
        ("ir_k5", 480, 1, 7, {"expansion": 5}, IRF_CFG),
        ("ir_k5", 640, 1, 2, {"expansion": 6}, IRF_CFG),
    ],
]


FBNetV3_3s_no_se = [
    [("conv_k3", 48, 2, 1), ("ir_k3", 48, 1, 5, {"expansion": 1}, IRF_CFG)],
    [
        ("ir_k5", 80, 2, 1, {"expansion": 4}, IRF_CFG),
        ("ir_k5", 80, 1, 8, {"expansion": 2}, IRF_CFG),
    ],
    [
        ("ir_k5", 112, 2, 1, {"expansion": 4}, IRF_CFG),
        ("ir_k5", 112, 1, 8, {"expansion": 3}, IRF_CFG),
    ],
    [
        ("ir_k5", 168, 2, 1, {"expansion": 5}, IRF_CFG),
        ("ir_k3", 168, 1, 8, {"expansion": 3}, IRF_CFG),
        ("ir_k3", 360, 1, 1, {"expansion": 5}, IRF_CFG),
        ("ir_k5", 360, 1, 11, {"expansion": 3}, IRF_CFG),
    ],
    [
        ("ir_k3", 512, 2, 1, {"expansion": 6}, IRF_CFG),
        ("ir_k5", 512, 1, 8, {"expansion": 5}, IRF_CFG),
        ("ir_k5", 688, 1, 2, {"expansion": 6}, IRF_CFG),
    ],
]

MODEL_ARCH_FBNETV3 = {
    "FBNetV3_A": {
        "trunk": FBNetV3_A[0:4],
        "rpn": [[_repeat_last(FBNetV3_A[3])]],
        "bbox": [FBNetV3_A[4]],
        "basic_args": _BASIC_ARGS,
    },
    "FBNetV3_B": {
        "trunk": FBNetV3_B[0:4],
        "rpn": [[_repeat_last(FBNetV3_B[3])]],
        "bbox": [FBNetV3_B[4]],
        "basic_args": _BASIC_ARGS,
    },
    "FBNetV3_C": {
        "trunk": FBNetV3_C[0:4],
        "rpn": [[_repeat_last(FBNetV3_C[3])]],
        "bbox": [FBNetV3_C[4]],
        "basic_args": _BASIC_ARGS,
    },
    "FBNetV3_D": {
        "trunk": FBNetV3_D[0:4],
        "rpn": [[_repeat_last(FBNetV3_D[3])]],
        "bbox": [FBNetV3_D[4]],
        "basic_args": _BASIC_ARGS,
    },
    "FBNetV3_E": {
        "trunk": FBNetV3_E[0:4],
        "rpn": [[_repeat_last(FBNetV3_E[3])]],
        "bbox": [FBNetV3_E[4]],
        "basic_args": _BASIC_ARGS,
    },
    "FBNetV3_F": {
        "trunk": FBNetV3_F[0:4],
        "rpn": [[_repeat_last(FBNetV3_F[3])]],
        "bbox": [FBNetV3_F[4]],
        "basic_args": _BASIC_ARGS,
    },
    "FBNetV3_G": {
        "trunk": FBNetV3_G[0:4],
        "rpn": [[_repeat_last(FBNetV3_G[3])]],
        "bbox": [FBNetV3_G[4]],
        "basic_args": _BASIC_ARGS,
    },
    "FBNetV3_H": {
        "trunk": FBNetV3_H[0:4],
        "rpn": [[_repeat_last(FBNetV3_H[3])]],
        "bbox": [FBNetV3_H[4]],
        "basic_args": _BASIC_ARGS,
    },
    "FBNetV3_1s": {
        "trunk": FBNetV3_1s[0:4],
        "rpn": [[_repeat_last(FBNetV3_1s[3])]],
        "bbox": [FBNetV3_1s[4]],
        "basic_args": _BASIC_ARGS,
    },
    "FBNetV3_2s": {
        "trunk": FBNetV3_2s[0:4],
        "rpn": [[_repeat_last(FBNetV3_2s[3])]],
        "bbox": [FBNetV3_2s[4]],
        "basic_args": _BASIC_ARGS,
    },
    "FBNetV3_3s": {
        "trunk": FBNetV3_3s[0:4],
        "rpn": [[_repeat_last(FBNetV3_3s[3])]],
        "bbox": [FBNetV3_3s[4]],
        "basic_args": _BASIC_ARGS,
    },
    "FBNetV3_F_no_se": {
        "trunk": FBNetV3_F_no_se[0:4],
        "rpn": [[_repeat_last(FBNetV3_F_no_se[3])]],
        "bbox": [FBNetV3_F_no_se[4]],
        "basic_args": _BASIC_ARGS,
    },
    "FBNetV3_H_no_se": {
        "trunk": FBNetV3_H_no_se[0:4],
        "rpn": [[_repeat_last(FBNetV3_H_no_se[3])]],
        "bbox": [FBNetV3_H_no_se[4]],
        "basic_args": _BASIC_ARGS,
    },
    "FBNetV3_1s_no_se": {
        "trunk": FBNetV3_1s_no_se[0:4],
        "rpn": [[_repeat_last(FBNetV3_1s_no_se[3])]],
        "bbox": [FBNetV3_1s_no_se[4]],
        "basic_args": _BASIC_ARGS,
    },
    "FBNetV3_2s_no_se": {
        "trunk": FBNetV3_2s_no_se[0:4],
        "rpn": [[_repeat_last(FBNetV3_2s_no_se[3])]],
        "bbox": [FBNetV3_2s_no_se[4]],
        "basic_args": _BASIC_ARGS,
    },
    "FBNetV3_3s_no_se": {
        "trunk": FBNetV3_3s_no_se[0:4],
        "rpn": [[_repeat_last(FBNetV3_3s_no_se[3])]],
        "bbox": [FBNetV3_3s_no_se[4]],
        "basic_args": _BASIC_ARGS,
    },
}
FBNetV2ModelArch.add_archs(MODEL_ARCH_FBNETV3)
