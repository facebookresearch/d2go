#!/usr/bin/env python3
from mobile_cv.arch.fbnet_v2.modeldef_utils import _ex, e1, e1p, e3, e4, e6  # noqa
from .modeldef import FBNetV2ModelArch

_BASIC_ARGS = {
    "dw_skip_bnrelu": False,
    "width_divisor": 8,
    # temporarily disable zero_last_bn_gamma
    # "zero_last_bn_gamma": False,
}

IRF_CFG = {
    "less_se_channels": True,
    "zero_last_bn_gamma": True,
}

V1_ARGS = {
    "dw_skip_bnrelu": True,
    "width_divisor": 8,
    # uncomment below (always_pw and bias) to match model definition of the
    # FBNetV1 builder.
    "always_pw": True,
    "bias": False,

    # temporarily disable zero_last_bn_gamma
    "zero_last_bn_gamma": False,
}

FD_ARCH = {
    "default_v1_from_f150822062": {
        "trunk": [
            # [c, s, n, t]
            [("conv_k3", 32, 2, 1), ("ir_k3", 16, 1, 1, e1)],
            # resolution stage 1, equivalent to 112->56
            [("ir_k3", 24, 2, 2, e6)],
            # resolution stage 2, equivalent to 56->28
            [("ir_k3", 32, 2, 3, e6)],
            # resolution stage 3, equivalent to 28->14
            [("ir_k3", 64, 2, 4, e6), ("ir_k3", 96, 1, 3, e6)],
        ],
        "bbox": [[("ir_k3", 160, 2, 1, e4), ("ir_k3", 160, 1, 2, e6), ("ir_k3", 240, 1, 1, e6)]],
        "rpn": [[("ir_k3", 96, 1, 3, e6)]],
        "basic_args": V1_ARGS,
    },
    "cham_e_fd_v2_from_v1": {
        "trunk": [
            # [c, s, n, t]
            [("conv_k3", 8, 2, 1), ("ir_k3", 8, 1, 1, e1)],
            [("ir_k3", 24, 2, 1, e4)],
            [("ir_k3", 32, 2, 3, e4)],
            [("ir_k3", 64, 2, 1, e6), ("ir_k3", 64, 1, 3, e6)],
        ],
        "bbox": [[("ir_k3", 128, 1, 1, e4)]],
        "rpn": [[("ir_k3", 96, 1, 1, e6)]],
        "basic_args": V1_ARGS,
    },
    "v1_to_v2_f150822062": {
        "trunk": [
            # [c, s, n, t]
            [("conv_k3", 32, 2, 1), ("ir_k3", 16, 1, 1, e1)],
            [("ir_k3", 24, 2, 2, e6)],
            [("ir_k3", 32, 2, 3, e6)],
            [("ir_k3", 64, 2, 4, e6), ("ir_k3", 96, 1, 3, e6)],
        ],
        "bbox": [
            [("ir_k3", 160, 2, 1, e4), ("ir_k3", 160, 1, 2, e6), ("ir_k3", 240, 1, 1, e6)],
        ],
        "rpn": [[("ir_k3", 96, 1, 3, e6)]],
        "basic_args": V1_ARGS,
    },
    "v1_to_v2_f143877782": {
        "trunk": [
            # [c, s, n, t]
            [("conv_k3", 8, 2, 1), ("ir_k3", 8, 1, 1, e1)],
            [("ir_k3", 16, 2, 1, e4)],
            [("ir_k3", 16, 2, 3, e4)],
            [("ir_k3", 32, 2, 2, e6)],
        ],
        "bbox": [
            [
                ("ir_k3", 48, 2, 1, e6),
                ("ir_k3", 64, 1, 1, e4),
            ],
        ],
        "rpn": [[("ir_k3", 48, 1, 1, e6)]],
        "basic_args": V1_ARGS,
    },
    "chame_e_fd_v3_from_f144058202": {
        "trunk": [
            # [c, s, n, t]
            [("conv_k3", 8, 2, 1), ("ir_k3", 8, 1, 1, e1)],
            [("ir_k3", 16, 2, 1, e4)],
            [("ir_k3", 16, 2, 1, e4)],
            [("ir_k3", 32, 2, 4, e6)],
        ],
        "bbox": [[("ir_k3", 64, 1, 1, e4)]],
        "rpn": [[("ir_k3", 48, 1, 1, e6)]],
        "kpts": [[
                    ("ir_k3", 16, 2, 1, e4),
                    ("ir_k3", 16, 1, 2, e6),
                    ("ir_k3", 16, -2, 1, e6),
                    ("ir_k3", 8, -2, 1, e3),
                ]],
        "basic_args": V1_ARGS,
    },
    "chame_e_fd_v3_from_f143877782": {
        "trunk": [
            # [c, s, n, t]
            [("conv_k3", 8, 2, 1), ("ir_k3", 8, 1, 1, e1)],
            [("ir_k3", 16, 2, 1, e4)],
            [("ir_k3", 16, 2, 3, e4)],
            [("ir_k3", 32, 2, 2, e6)],
        ],
        "bbox": [
            [("ir_k3", 48, 2, 1, e6)],
            [("ir_k3", 64, 1, 1, e4)],
        ],
        "rpn": [[("ir_k3", 48, 1, 1, e6)]],
        "basic_args": V1_ARGS,
    },
    "cham_e_fd_v4": {
        "trunk": [
            # [c, s, n, t]
            [("conv_k3", 4, 2, 1), ("ir_k3", 2, 1, 1, e1)],
            [("ir_k3", 2, 2, 1, e4)],
            [("ir_k3", 2, 2, 1, e4)],
            [("ir_k3", 4, 2, 1, e6)],
        ],
        "bbox": [
            [("ir_k3", 8, 1, 1, e4)],
        ],
        "rpn": [[("ir_k3", 6, 1, 1, e6)]],
        "basic_args": {
            "dw_skip_bnrelu": True,
            "width_divisor": 2,
            # temporarily disable zero_last_bn_gamma
            "zero_last_bn_gamma": False,
        },
    },
    "retina_fbnetv3_G":{
        "basic_args": _BASIC_ARGS,
        "trunk": [
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
        ],
    },
}
FBNetV2ModelArch.add_archs(FD_ARCH)
