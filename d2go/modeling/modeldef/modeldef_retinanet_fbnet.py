#!/usr/bin/env python3

from mobile_cv.arch.fbnet_v2.modeldef_registry import FBNetV2ModelArch
from mobile_cv.arch.fbnet_v2.modeldef_utils import e1, e3, e6

BASIC_ARGS = {
    "width_divisor": 8,
}

# IRF_CFG = {
#     "less_se_channels": True,
#     "zero_last_bn_gamma": True,
# }


MODEL_ARCH_RETINA_EFF = {
    "retina_default": {
        "basic_args": BASIC_ARGS,
        "trunk": [
            # [op, c, s, n, ...]
            # stage 0
            [("conv_k3", 32, 2, 1), ("ir_k3", 16, 1, 1, e1)],
            # stage 2
            [("ir_k3", 24, 2, 2, e6)],
            # stage 3
            [("ir_k3", 32, 2, 3, e6)],
            # stage 4
            [("ir_k3", 64, 2, 4, e6), ("ir_k3", 96, 1, 3, e6)],
            # stage 5
            [("ir_k3", 160, 2, 3, e6), ("ir_k3", 320, 1, 1, e6)],
        ],
    },
    "retina_fbnet_a": {
        "basic_args": BASIC_ARGS,
        "trunk": [
            # [op, c, s, n, ...]
            # stage 0
            [["conv_k3", 16, 2, 1]],
            # stage 2
            [
                ["ir_k3", 24, 2, 1, e3],
                ["ir_k3", 24, 1, 1, e1],
                ["skip", 24, 1, 1],
                ["skip", 24, 1, 1],
            ],
            # stage 3
            [
                ["ir_k5", 32, 2, 1, e6],
                ["ir_k3", 32, 1, 1, e3],
                ["ir_k5", 32, 1, 1, e1],
                ["ir_k3", 32, 1, 1, e3],
            ],
            # stage 4
            [
                ["ir_k5", 64, 2, 1, e6],
                ["ir_k5", 64, 1, 1, e3],
                ["ir_k5_g2", 64, 1, 1, e1],
                ["ir_k5", 64, 1, 1, e6],
                ["ir_k3", 112, 1, 1, e6],
                ["ir_k5_g2", 112, 1, 1, e1],
                ["ir_k5", 112, 1, 1, e3],
                ["ir_k3_g2", 112, 1, 1, e1],
            ],
            # stage 5
            [
                ["ir_k5", 184, 2, 1, e6],
                ["ir_k5", 184, 1, 1, e6],
                ["ir_k5", 184, 1, 1, e3],
                ["ir_k5", 184, 1, 1, e6],
                ["ir_k5", 352, 1, 1, e6],
            ],
        ],
    },
    "retina_fbnet_b": {
        "basic_args": BASIC_ARGS,
        "trunk": [
            # [op, c, s, n, ...]
            # stage 0
            [["conv_k3", 16, 2, 1], ["ir_k3", 16, 1, 1, e1]],
            # stage 2
            [
                ["ir_k3", 24, 2, 1, e6],
                ["ir_k5", 24, 1, 1, e1],
                ["ir_k3", 24, 1, 1, e1],
                ["ir_k3", 24, 1, 1, e1],
            ],
            # stage 3
            [
                ["ir_k5", 32, 2, 1, e6],
                ["ir_k5", 32, 1, 1, e3],
                ["ir_k3", 32, 1, 1, e6],
                ["ir_k5", 32, 1, 1, e6],
            ],
            # stage 4
            [
                ["ir_k5", 64, 2, 1, e6],
                ["ir_k5", 64, 1, 1, e6],
                ["ir_k5", 64, 1, 1, e3],
                ["ir_k5", 112, 1, 1, e6],
                ["ir_k3", 112, 1, 1, e1],
                ["ir_k5", 112, 1, 1, e1],
                ["ir_k5", 112, 1, 1, e3],
            ],
            # stage 5
            [
                ["ir_k5", 184, 2, 1, e6],
                ["ir_k5", 184, 1, 1, e1],
                ["ir_k5", 184, 1, 1, e6],
                ["ir_k5", 184, 1, 1, e6],
                ["ir_k3", 352, 1, 1, e6],
            ],
        ],
    },
    "retina_fbnet_c": {
        "basic_args": BASIC_ARGS,
        "trunk": [
            # [op, c, s, n, ...]
            # stage 0
            [["conv_k3", 16, 2, 1], ["ir_k3", 16, 1, 1, e1]],
            # stage 2
            [
                ["ir_k3", 24, 2, 1, e6],
                ["ir_k3", 24, 1, 1, e1],
                ["ir_k3", 24, 1, 1, e1],
            ],
            # stage 3
            [
                ["ir_k5", 32, 2, 1, e6],
                ["ir_k5", 32, 1, 1, e3],
                ["ir_k5", 32, 1, 1, e6],
                ["ir_k3", 32, 1, 1, e6],
            ],
            # stage 4
            [
                ["ir_k5", 64, 2, 1, e6],
                ["ir_k5", 64, 1, 1, e3],
                ["ir_k5", 64, 1, 1, e6],
                ["ir_k5", 64, 1, 1, e6],
                ["ir_k5", 112, 1, 1, e6],
                ["ir_k5", 112, 1, 1, e6],
                ["ir_k5", 112, 1, 1, e6],
                ["ir_k5", 112, 1, 1, e3],
            ],
            # stage 5
            [
                ["ir_k5", 184, 2, 1, e6],
                ["ir_k5", 184, 1, 1, e6],
                ["ir_k5", 184, 1, 1, e6],
                ["ir_k5", 184, 1, 1, e6],
                ["ir_k3", 352, 1, 1, e6],
            ],
            # stage 6
            # [("conv_k1", 1984, 1, 1)],
        ],
    },
}
FBNetV2ModelArch.add_archs(MODEL_ARCH_RETINA_EFF)
