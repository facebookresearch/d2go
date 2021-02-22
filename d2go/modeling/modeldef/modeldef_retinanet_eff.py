#!/usr/bin/env python3

from mobile_cv.arch.fbnet_v2.modeldef_registry import FBNetV2ModelArch
from mobile_cv.arch.fbnet_v2.modeldef_utils import e1, e6

BASIC_ARGS = {
    "relu_args": "swish",
    "width_divisor": 8,
}

IRF_CFG = {
    "less_se_channels": True,
    "zero_last_bn_gamma": True,
}


MODEL_ARCH_RETINA_EFF = {
    "retina_eff_d0": {
        "trunk": [
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
        ],
        "basic_args": BASIC_ARGS,
    },
    "retina_eff_d3": {
        "basic_args": BASIC_ARGS,
        "trunk": [
            # [c, s, n, ...]
            # stage 0
            [["conv_k3", 40, 2, 1], ["ir_k3_se", 24, 1, 2, e1, IRF_CFG]],
            # stage 1
            [["ir_k3_se", 32, 2, 3, e6, IRF_CFG]],
            # stage 2
            [["ir_k5_se", 48, 2, 3, e6, IRF_CFG]],
            # stage 3
            [
                ["ir_k3_se", 96, 2, 5, e6, IRF_CFG],
                ["ir_k5_se", 136, 1, 5, e6, IRF_CFG],
            ],
            # stage 4
            [
                ["ir_k5_se", 232, 2, 6, e6, IRF_CFG],
                ["ir_k3_se", 384, 1, 2, e6, IRF_CFG],
            ],
        ],
    },
}
FBNetV2ModelArch.add_archs(MODEL_ARCH_RETINA_EFF)
