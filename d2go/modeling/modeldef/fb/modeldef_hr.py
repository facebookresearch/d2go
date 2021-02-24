#!/usr/bin/env python3

from d2go.modeling.modeldef.fbnet_modeldef_registry import FBNetV2ModelArch
from mobile_cv.arch.fbnet_v2.modeldef_utils import e1p, e6, e12


BASIC_ARGS = {
    "dw_skip_bnrelu": True,
    "bias": False,
    "zero_last_bn_gamma": False,
    "width_divisor": 8,
}
CONV_ONLY = {"bn_args": None, "relu_args": None, "weight_init": None}


MODEL_ARCH_SEG = {
    # arch_def explanation in `build_model` https://fburl.com/diff/0f7g763r
    "test_arch": {
        "basic_args": BASIC_ARGS,
        "stages": [
            # [op, c, s, n, ...]
            # original res
            [("conv_k3", 16, 1, 1)],
            [("skip", 16, 1, 1)],
            [("conv_k3", 16, 1, 1)],
            # downsampled x2
            [("skip", 16, 1, 1)],
            [("skip", 16, 1, 1)],
            [("conv_k3", 16, 1, 1)],
        ],
    },
    "tfv3.1": {
        "basic_args": BASIC_ARGS,
        "stages": [
            # [op, c, s, n, ...]
            # original res
            [("conv_k3", 128, 1, 1)],
            [("ir_k3", 64, 1, 3, e1p)],
            [("conv_k3", 1, 1, 1, CONV_ONLY)],
            # downsampled (x2)
            [("ir_k3", 64, 2, 3, e1p)],
            [("ir_k3", 64, 1, 3, e1p)],
            [("ir_k3", 64, -2, 3, e1p)],
            # downsampled (x4)
            [("ir_k3", 64, 2, 3, e6)],
            [("ir_k3", 64, 1, 3, e6)],
            [("ir_k3", 64, -2, 3, e6)],
            # downsampled (x8)
            [("ir_k3", 64, 2, 3, e6)],
            [("ir_k3", 64, 1, 3, e6)],
            [("ir_k3", 64, -2, 3, e6)],
            # downsampled (x16)
            [("ir_k3", 64, 2, 3, e12)],
            [("ir_k3", 64, 1, 3, e12)],
            [("ir_k3", 64, -2, 3, e12)],
            # downsampled (x32)
            [("ir_k3", 64, 2, 3, e12)],
            [("ir_k3", 64, 1, 3, e12)],
            [("ir_k3", 64, -2, 3, e12)],
        ],
    },
}

FBNetV2ModelArch.add_archs(MODEL_ARCH_SEG)


MODEL_ARCH_ARGOS = {
    "as_f181421878": {
        "basic_args": BASIC_ARGS,
        "stages": [
            # [op, c, s, n, ...]
            # original res
            [["conv_k3", 56, 1, 1]],
            [["ir_k3", 40, 1, 3, {"expansion": 1}]],
            [["conv_k3", 1, 1, 1, CONV_ONLY]],
            # downsampled (x2)
            [["ir_k3", 80, 2, 3, {"expansion": 1}]],
            [["ir_k3", 8, 1, 3, {"expansion": 1}]],
            [["ir_k3", 40, -2, 3, {"expansion": 1}]],
            # downsampled (x4)
            [["ir_k3", 128, 2, 3, {"expansion": 6}]],
            [["ir_k3", 80, 1, 3, {"expansion": 6}]],
            [["ir_k3", 8, -2, 3, {"expansion": 6}]],
            # downsampled (x8)
            [["ir_k3", 72, 2, 4, {"expansion": 5}]],
            [["ir_k3", 64, 1, 4, {"expansion": 5}]],
            [["ir_k3", 80, -2, 4, {"expansion": 5}]],
            # downsampled (x16)
            [["ir_k3", 88, 2, 2, {"expansion": 11}]],
            [["ir_k3", 48, 1, 2, {"expansion": 11}]],
            [["ir_k3", 64, -2, 2, {"expansion": 11}]],
            # downsampled (x32)
            [["ir_k3", 80, 2, 5, {"expansion": 6}]],
            [["ir_k3", 24, 1, 5, {"expansion": 6}]],
            [["ir_k3", 48, -2, 5, {"expansion": 6}]],
        ],
        "nviews": 2,
        "correlation": {"k": 0, "d": 0, "s1": 1, "s2": 1},
    },
    "as_f181421878_8last1x1": {
        "basic_args": BASIC_ARGS,
        "stages": [
            # [op, c, s, n, ...]
            # original res
            [["conv_k3", 56, 1, 1]],
            [["ir_k3", 40, 1, 3, {"expansion": 1}]],
            [
                ("conv_k1", 8, 1, 1, {"bn_args": None, "relu_args": None}),
                ("conv_k3", 1, 1, 1, CONV_ONLY),
            ],
            # downsampled (x2)
            [["ir_k3", 80, 2, 3, {"expansion": 1}]],
            [["ir_k3", 8, 1, 3, {"expansion": 1}]],
            [["ir_k3", 40, -2, 3, {"expansion": 1}]],
            # downsampled (x4)
            [["ir_k3", 128, 2, 3, {"expansion": 6}]],
            [["ir_k3", 80, 1, 3, {"expansion": 6}]],
            [["ir_k3", 8, -2, 3, {"expansion": 6}]],
            # downsampled (x8)
            [["ir_k3", 72, 2, 4, {"expansion": 5}]],
            [["ir_k3", 64, 1, 4, {"expansion": 5}]],
            [["ir_k3", 80, -2, 4, {"expansion": 5}]],
            # downsampled (x16)
            [["ir_k3", 88, 2, 2, {"expansion": 11}]],
            [["ir_k3", 48, 1, 2, {"expansion": 11}]],
            [["ir_k3", 64, -2, 2, {"expansion": 11}]],
            # downsampled (x32)
            [["ir_k3", 80, 2, 5, {"expansion": 6}]],
            [["ir_k3", 24, 1, 5, {"expansion": 6}]],
            [["ir_k3", 48, -2, 5, {"expansion": 6}]],
        ],
        "nviews": 2,
        "correlation": {
            "corr_type": "unfold",
            "k": 0,
            "d": 10,
            "s1": 1,
            "s2": 1,
        }
    },
    "as_f181421878_8last1x1_corr45": {
        "basic_args": BASIC_ARGS,
        "stages": [
            # [op, c, s, n, ...]
            # original res
            [["conv_k3", 56, 1, 1]],
            [["ir_k3", 40, 1, 3, {"expansion": 1}]],
            [
                ("conv_k1", 8, 1, 1, {"bn_args": None, "relu_args": None}),
                ("conv_k3", 1, 1, 1, CONV_ONLY),
            ],
            # downsampled (x2)
            [["ir_k3", 80, 2, 3, {"expansion": 1}]],
            [["ir_k3", 8, 1, 3, {"expansion": 1}]],
            [["ir_k3", 40, -2, 3, {"expansion": 1}]],
            # downsampled (x4)
            [["ir_k3", 128, 2, 3, {"expansion": 6}]],
            [["ir_k3", 80, 1, 3, {"expansion": 6}]],
            [["ir_k3", 8, -2, 3, {"expansion": 6}]],
            # downsampled (x8)
            [["ir_k3", 72, 2, 4, {"expansion": 5}]],
            [["ir_k3", 64, 1, 4, {"expansion": 5}]],
            [["ir_k3", 80, -2, 4, {"expansion": 5}]],
            # downsampled (x16)
            [["ir_k3", 88, 2, 2, {"expansion": 11}]],
            [["ir_k3", 48, 1, 2, {"expansion": 11}]],
            [["ir_k3", 64, -2, 2, {"expansion": 11}]],
            # downsampled (x32)
            [["ir_k3", 80, 2, 5, {"expansion": 6}]],
            [["ir_k3", 24, 1, 5, {"expansion": 6}]],
            [["ir_k3", 48, -2, 5, {"expansion": 6}]],
        ],
        "nviews": 2,
        "correlation": {
            "corr_type": "unfold",
            "k": 0,
            "d": 10,
            "s1": 1,
            "s2": 1,
            "selected_layers": [4, 5],
        }
    },
    "as_f181421878_8last1x1_corr345_deepproj": {
        "basic_args": BASIC_ARGS,
        "stages": [
            # [op, c, s, n, ...]
            # original res
            [["conv_k3", 56, 1, 1]],
            [["ir_k3", 40, 1, 3, {"expansion": 1}]],
            [
                ("conv_k1", 8, 1, 1, {"bn_args": None, "relu_args": None}),
                ("conv_k3", 1, 1, 1, CONV_ONLY),
            ],
            # downsampled (x2)
            [["ir_k3", 80, 2, 3, {"expansion": 1}]],
            [["ir_k3", 8, 1, 3, {"expansion": 1}]],
            [["ir_k3", 40, -2, 3, {"expansion": 1}]],
            # downsampled (x4)
            [["ir_k3", 128, 2, 3, {"expansion": 6}]],
            [["ir_k3", 80, 1, 3, {"expansion": 6}]],
            [["ir_k3", 8, -2, 3, {"expansion": 6}]],
            # downsampled (x8)
            [["ir_k3", 72, 2, 4, {"expansion": 5}]],
            [["ir_k3", 64, 1, 4, {"expansion": 5}]],
            [["ir_k3", 80, -2, 4, {"expansion": 5}]],
            # downsampled (x16)
            [["ir_k3", 88, 2, 2, {"expansion": 11}]],
            [["ir_k3", 48, 1, 2, {"expansion": 11}]],
            [["ir_k3", 64, -2, 2, {"expansion": 11}]],
            # downsampled (x32)
            [["ir_k3", 80, 2, 5, {"expansion": 6}]],
            [["ir_k3", 24, 1, 5, {"expansion": 6}]],
            [["ir_k3", 48, -2, 5, {"expansion": 6}]],
        ],
        "nviews": 2,
        "correlation": {
            "corr_type": "unfold",
            "k": 0,
            "d": 10,
            "s1": 1,
            "s2": 1,
            "selected_layers": [3, 4, 5],
            "num_conv": 1,
            "conv_depth": 128,
        }
    },
    "as_f181421878_8last1x1_corr45_deepproj": {
        "basic_args": BASIC_ARGS,
        "stages": [
            # [op, c, s, n, ...]
            # original res
            [["conv_k3", 56, 1, 1]],
            [["ir_k3", 40, 1, 3, {"expansion": 1}]],
            [
                ("conv_k1", 8, 1, 1, {"bn_args": None, "relu_args": None}),
                ("conv_k3", 1, 1, 1, CONV_ONLY),
            ],
            # downsampled (x2)
            [["ir_k3", 80, 2, 3, {"expansion": 1}]],
            [["ir_k3", 8, 1, 3, {"expansion": 1}]],
            [["ir_k3", 40, -2, 3, {"expansion": 1}]],
            # downsampled (x4)
            [["ir_k3", 128, 2, 3, {"expansion": 6}]],
            [["ir_k3", 80, 1, 3, {"expansion": 6}]],
            [["ir_k3", 8, -2, 3, {"expansion": 6}]],
            # downsampled (x8)
            [["ir_k3", 72, 2, 4, {"expansion": 5}]],
            [["ir_k3", 64, 1, 4, {"expansion": 5}]],
            [["ir_k3", 80, -2, 4, {"expansion": 5}]],
            # downsampled (x16)
            [["ir_k3", 88, 2, 2, {"expansion": 11}]],
            [["ir_k3", 48, 1, 2, {"expansion": 11}]],
            [["ir_k3", 64, -2, 2, {"expansion": 11}]],
            # downsampled (x32)
            [["ir_k3", 80, 2, 5, {"expansion": 6}]],
            [["ir_k3", 24, 1, 5, {"expansion": 6}]],
            [["ir_k3", 48, -2, 5, {"expansion": 6}]],
        ],
        "nviews": 2,
        "correlation": {
            "corr_type": "unfold",
            "k": 0,
            "d": 10,
            "s1": 1,
            "s2": 1,
            "selected_layers": [4, 5],
            "num_conv": 1,
            "conv_depth": 128,
        }
    },
    "as_f181421878_8last1x1_deepproj": {
        "basic_args": BASIC_ARGS,
        "stages": [
            # [op, c, s, n, ...]
            # original res
            [["conv_k3", 56, 1, 1]],
            [["ir_k3", 40, 1, 3, {"expansion": 1}]],
            [("conv_k1", 8, 1, 1, {"bn_args": None, "relu_args": None}),
             ("conv_k3", 1, 1, 1, CONV_ONLY)],
            # downsampled (x2)
            [["ir_k3", 80, 2, 3, {"expansion": 1}]],
            [["ir_k3", 8, 1, 3, {"expansion": 1}]],
            [["ir_k3", 40, -2, 3, {"expansion": 1}]],
            # downsampled (x4)
            [["ir_k3", 128, 2, 3, {"expansion": 6}]],
            [["ir_k3", 80, 1, 3, {"expansion": 6}]],
            [["ir_k3", 8, -2, 3, {"expansion": 6}]],
            # downsampled (x8)
            [["ir_k3", 72, 2, 4, {"expansion": 5}]],
            [["ir_k3", 64, 1, 4, {"expansion": 5}]],
            [["ir_k3", 80, -2, 4, {"expansion": 5}]],
            # downsampled (x16)
            [["ir_k3", 88, 2, 2, {"expansion": 11}]],
            [["ir_k3", 48, 1, 2, {"expansion": 11}]],
            [["ir_k3", 64, -2, 2, {"expansion": 11}]],
            # downsampled (x32)
            [["ir_k3", 80, 2, 5, {"expansion": 6}]],
            [["ir_k3", 24, 1, 5, {"expansion": 6}]],
            [["ir_k3", 48, -2, 5, {"expansion": 6}]],
        ],
        "nviews": 2,
        "correlation": {
            "corr_type": "matmul",
            "k": 0,
            "d": 10,
            "s1": 1,
            "s2": 1,
            "num_conv": 3,
            "conv_depth": 128,
        }
    },
}
FBNetV2ModelArch.add_archs(MODEL_ARCH_ARGOS)
