#!/usr/bin/env python3

import copy

from d2go.modeling.modeldef.fbnet_modeldef_registry import FBNetV2ModelArch
from mobile_cv.arch.fbnet_v2.modeldef_utils import e1, e2, e3, e4, e5, e6, _ex


BASIC_ARGS = {
    "dw_skip_bnrelu": True,
    "bias": False,
    "zero_last_bn_gamma": False,
    "width_divisor": 8,
}
BASIC_ARGS1 = {
    "dw_skip_bnrelu": True,
    "bias": False,
    "zero_last_bn_gamma": True,
    "width_divisor": 8,
}
CONV_ONLY = {"bn_args": None, "relu_args": None, "weight_init": None}
NO_BN = {"bn_args": None}
PS_UNET_MAXPOOL = {"kernel_size": 2, "padding": 0}
CONV_PAD_REFLECT = {"padding_mode": "reflect"}
DW_PAD_REFLECT = {"dw_args": {**CONV_PAD_REFLECT}}
SE_RELU = {"se_args": {"name": "se_hsig", "relu_args": "relu"}}
UPSAMPLE_BILINEAR = {"upsample_args": {"name": "default", "mode": "bilinear"}}


MODEL_ARCH_PERSON_SEGMENTATION = {
    # arch_def explanation in `build_model` https://fburl.com/diff/0f7g763r
    "fbunet0": {
        "basic_args": BASIC_ARGS,
        "stages": [
            # [op, c, s, n, ...]
            # original res
            [("ir_k3", 8, 1, 1, e1)],
            [("skip", 8, 1, 1)],
            [("conv_k3", 1, 1, 1, CONV_ONLY)],
            # downsampled (x2)
            [("ir_k3", 16, 2, 1, e1)],
            [("skip", 16, 1, 1)],
            [("ir_k3", 8, -2, 1, e1)],
            # downsampled (x4)
            [("ir_k3", 24, 2, 1, e2)],
            [("skip", 24, 1, 1)],
            [("ir_k3", 16, -2, 1, e2)],
            # downsampled (x8)
            [("ir_k3", 32, 2, 1, e3)],
            [("skip", 32, 1, 1)],
            [("ir_k3", 24, -2, 1, e3)],
            # downsampled (x16)
            [("ir_k3", 40, 2, 1, e6)],
            [("skip", 40, 1, 1)],
            [("ir_k3", 32, -2, 1, e6)],
            # downsampled (x32)
            [("ir_k3", 48, 2, 1, e6)],
            [("skip", 48, 1, 1)],
            [("ir_k3", 40, -2, 1, e6)],
        ],
    },
    "xirp1": {
        # nparams: 0.316376, nflops 168.823296
        "basic_args": BASIC_ARGS,
        "stages": [
            # [op, c, s, n, ...]
            # original res
            [("conv_k3", 8, 1, 1)],
            [("ir_k3", 8, 1, 1, e1)],
            [("conv_k3", 1, 1, 1, CONV_ONLY)],
            # downsampled (x2)
            [("ir_k3", 16, 2, 1, e3)],
            [("ir_k3", 16, 1, 1, e1)],
            [("ir_k3", 8, -2, 1, e1)],
            # downsampled (x4)
            [("ir_k3", 24, 2, 2, e3)],
            [("ir_k3", 24, 1, 1, e1)],
            [("ir_k3", 16, -2, 1, e2)],
            # downsampled (x8)
            [("ir_k3", 32, 2, 3, e4)],
            [("ir_k3", 32, 1, 1, e1)],
            [("ir_k3", 24, -2, 2, e4)],
            # downsampled (x16)
            [("ir_k3", 40, 2, 3, e6)],
            [("ir_k3", 40, 1, 1, e1)],
            [("ir_k3", 32, -2, 3, e6)],
            # downsampled (x32)
            [("ir_k3", 48, 2, 3, e6)],
            [("ir_k3", 48, 1, 1, e1)],
            [("ir_k3", 40, -2, 3, e6)],
        ],
    },
    "xirp1a": {
        # nparams: 0.316376, nflops 168.823296
        "basic_args": BASIC_ARGS,
        "stages": [
            # [op, c, s, n, ...]
            # original res
            [("conv_k3", 8, 1, 1)],
            [("ir_k3", 8, 1, 1, e1)],
            [("conv_k3", 1, 1, 1, CONV_ONLY)],
            # downsampled (x2)
            [("ir_k3", 16, 2, 1, e3)],
            [("ir_k3", 16, 1, 1, e1)],
            [("ir_k3", 8, -2, 1, e1)],
            # downsampled (x4)
            [("ir_k3", 24, 2, 2, e3)],
            [("ir_k3", 24, 1, 1, e1)],
            [("ir_k3", 16, -2, 1, e2)],
            # downsampled (x8)
            [("ir_k3", 32, 2, 3, e4)],
            [("ir_k3", 32, 1, 1, e1)],
            [("ir_k3", 24, -2, 2, e4)],
            # downsampled (x16)
            [("ir_k3", 40, 2, 3, e6)],
            [("ir_k3", 40, 1, 1, e1)],
            [("ir_k3", 32, -2, 3, e6)],
            # downsampled (x16)
            [("ir_k3", 48, 1, 3, e6)],
            [("ir_k3", 48, 1, 1, e1)],
            [("ir_k3", 40, 1, 3, e6)],
        ],
    },
    "xirp2": {
        # nparams: 0.240152, nflops 133.917696
        "basic_args": BASIC_ARGS,
        "stages": [
            # [op, c, s, n, ...]
            # original res
            [("conv_k3", 8, 1, 1)],
            [("ir_k3", 8, 1, 1, e1)],
            [("conv_k3", 1, 1, 1, CONV_ONLY)],
            # downsampled (x2)
            [("ir_k3", 16, 2, 1, e3)],
            [("ir_k3", 16, 1, 1, e1)],
            [("ir_k3", 8, -2, 1, e1)],
            # downsampled (x4)
            [("ir_k3", 24, 2, 2, e3)],
            [("ir_k3", 24, 1, 1, e1)],
            [("ir_k3", 16, -2, 1, e2)],
            # downsampled (x8)
            [("ir_k3", 32, 2, 3, e4)],
            [("ir_k3", 32, 1, 1, e1)],
            [("ir_k3", 24, -2, 1, e4)],
            # downsampled (x16)
            [("ir_k3", 40, 2, 3, e6)],
            [("ir_k3", 40, 1, 1, e1)],
            [("ir_k3", 32, -2, 1, e6)],
            # downsampled (x32)
            [("ir_k3", 48, 2, 3, e6)],
            [("ir_k3", 48, 1, 1, e1)],
            [("ir_k3", 40, -2, 1, e6)],
        ],
    },
    "xirp2a": {
        # nparams: 0.240152, nflops 133.917696
        "basic_args": BASIC_ARGS,
        "stages": [
            # [op, c, s, n, ...]
            # original res
            [("conv_k3", 8, 1, 1)],
            [("ir_k3", 8, 1, 1, e1)],
            [("conv_k3", 1, 1, 1, CONV_ONLY)],
            # downsampled (x2)
            [("ir_k3", 16, 2, 1, e3)],
            [("ir_k3", 16, 1, 1, e1)],
            [("ir_k3", 8, -2, 1, e1)],
            # downsampled (x4)
            [("ir_k3", 24, 2, 2, e3)],
            [("ir_k3", 24, 1, 1, e1)],
            [("ir_k3", 16, -2, 1, e2)],
            # downsampled (x8)
            [("ir_k3", 32, 2, 3, e4)],
            [("ir_k3", 32, 1, 1, e1)],
            [("ir_k3", 24, -2, 1, e4)],
            # downsampled (x16)
            [("ir_k3", 40, 2, 3, e6)],
            [("ir_k3", 40, 1, 1, e1)],
            [("ir_k3", 32, -2, 1, e6)],
            # downsampled (x16)
            [("ir_k3", 48, 1, 3, e6)],
            [("ir_k3", 48, 1, 1, e1)],
            [("ir_k3", 40, 1, 1, e6)],
        ],
    },
    "xirp3": {
        # nparams: 0.23864, nflops 129.20544
        "basic_args": BASIC_ARGS,
        "stages": [
            # [op, c, s, n, ...]
            # original res
            [("conv_k3", 8, 1, 1)],
            [("conv_k1", 8, 1, 1)],
            [("conv_k3", 1, 1, 1, CONV_ONLY)],
            # downsampled (x2)
            [("ir_k3", 16, 2, 1, e3)],
            [("conv_k1", 16, 1, 1)],
            [("ir_k3", 8, -2, 1, e1)],
            # downsampled (x4)
            [("ir_k3", 24, 2, 2, e3)],
            [("conv_k1", 24, 1, 1)],
            [("ir_k3", 16, -2, 1, e2)],
            # downsampled (x8)
            [("ir_k3", 32, 2, 3, e4)],
            [("conv_k1", 32, 1, 1)],
            [("ir_k3", 24, -2, 1, e4)],
            # downsampled (x16)
            [("ir_k3", 40, 2, 3, e6)],
            [("conv_k1", 40, 1, 1)],
            [("ir_k3", 32, -2, 1, e6)],
            # downsampled (x32)
            [("ir_k3", 48, 2, 3, e6)],
            [("conv_k1", 48, 1, 1)],
            [("ir_k3", 40, -2, 1, e6)],
        ],
    },
    "xirp3a": {
        # nparams: 0.23864, nflops 129.20544
        "basic_args": BASIC_ARGS,
        "stages": [
            # [op, c, s, n, ...]
            # original res
            [("conv_k3", 8, 1, 1)],
            [("conv_k1", 8, 1, 1)],
            [("conv_k3", 1, 1, 1, CONV_ONLY)],
            # downsampled (x2)
            [("ir_k3", 16, 2, 1, e3)],
            [("conv_k1", 16, 1, 1)],
            [("ir_k3", 8, -2, 1, e1)],
            # downsampled (x4)
            [("ir_k3", 24, 2, 2, e3)],
            [("conv_k1", 24, 1, 1)],
            [("ir_k3", 16, -2, 1, e2)],
            # downsampled (x8)
            [("ir_k3", 32, 2, 3, e4)],
            [("conv_k1", 32, 1, 1)],
            [("ir_k3", 24, -2, 1, e4)],
            # downsampled (x16)
            [("ir_k3", 40, 2, 3, e6)],
            [("conv_k1", 40, 1, 1)],
            [("ir_k3", 32, -2, 1, e6)],
            # downsampled (x16)
            [("ir_k3", 48, 1, 3, e6)],
            [("conv_k1", 48, 1, 1)],
            [("ir_k3", 40, 1, 1, e6)],
        ],
    },
    "xirp4": {
        # nparams: 0.227912, nflops 110.667456
        "basic_args": BASIC_ARGS,
        "stages": [
            # [op, c, s, n, ...]
            # original res
            [("conv_k3", 8, 1, 1)],
            [("conv_k1", 8, 1, 1)],
            [("conv_k3", 1, 1, 1, CONV_ONLY)],
            # downsampled (x2)
            [("ir_k3", 16, 2, 1, e2)],
            [("conv_k1", 16, 1, 1)],
            [("ir_k3", 8, -2, 1, e1)],
            # downsampled (x4)
            [("ir_k3", 24, 2, 2, e2)],
            [("conv_k1", 24, 1, 1)],
            [("ir_k3", 16, -2, 1, e2)],
            # downsampled (x8)
            [("ir_k3", 32, 2, 3, e3)],
            [("conv_k1", 32, 1, 1)],
            [("ir_k3", 24, -2, 1, e3)],
            # downsampled (x16)
            [("ir_k3", 40, 2, 3, e6)],
            [("conv_k1", 40, 1, 1)],
            [("ir_k3", 32, -2, 1, e6)],
            # downsampled (x32)
            [("ir_k3", 48, 2, 3, e6)],
            [("conv_k1", 48, 1, 1)],
            [("ir_k3", 40, -2, 1, e6)],
        ],
    },
    "xirp4a": {
        # nparams: 0.227912, nflops 110.667456
        "basic_args": BASIC_ARGS,
        "stages": [
            # [op, c, s, n, ...]
            # original res
            [("conv_k3", 8, 1, 1)],
            [("conv_k1", 8, 1, 1)],
            [("conv_k3", 1, 1, 1, CONV_ONLY)],
            # downsampled (x2)
            [("ir_k3", 16, 2, 1, e2)],
            [("conv_k1", 16, 1, 1)],
            [("ir_k3", 8, -2, 1, e1)],
            # downsampled (x4)
            [("ir_k3", 24, 2, 2, e2)],
            [("conv_k1", 24, 1, 1)],
            [("ir_k3", 16, -2, 1, e2)],
            # downsampled (x8)
            [("ir_k3", 32, 2, 3, e3)],
            [("conv_k1", 32, 1, 1)],
            [("ir_k3", 24, -2, 1, e3)],
            # downsampled (x16)
            [("ir_k3", 40, 2, 3, e6)],
            [("conv_k1", 40, 1, 1)],
            [("ir_k3", 32, -2, 1, e6)],
            # downsampled (x16)
            [("ir_k3", 48, 1, 3, e6)],
            [("conv_k1", 48, 1, 1)],
            [("ir_k3", 40, 1, 1, e6)],
        ],
    },
    "xirp5": {
        # 192: nparams: 0.349936, nflops 130.568256
        # 96: nparams: 0.349936, nflops 32.642064
        "basic_args": BASIC_ARGS,
        "stages": [
            # [op, c, s, n, ...]
            # original res
            [("conv_k3", 8, 1, 1)],
            [("conv_k1", 8, 1, 1)],
            [("conv_k3", 1, 1, 1, CONV_ONLY)],
            # downsampled (x2)
            [("ir_k3", 16, 2, 1, e1)],
            [("conv_k1", 16, 1, 1)],
            [("ir_k3", 8, -2, 1, e1)],
            # downsampled (x4)
            [("ir_k3", 24, 2, 2, e2)],
            [("conv_k1", 24, 1, 1)],
            [("ir_k3", 16, -2, 1, e2)],
            # downsampled (x8)
            [("ir_k3", 32, 2, 3, e3)],
            [("conv_k1", 32, 1, 1)],
            [("ir_k3", 24, -2, 1, e3)],
            # downsampled (x16)
            [("ir_k3", 40, 2, 4, e6)],
            [("conv_k1", 40, 1, 1)],
            [("ir_k3", 32, -2, 3, e6)],
            # downsampled (x32)
            [("ir_k3", 48, 2, 4, e6)],
            [("conv_k1", 48, 1, 1)],
            [("ir_k3", 40, -2, 3, e6)],
        ],
    },
    "xirp5a": {
        # 96: nparams: 0.349936, nflops 35.89632
        "basic_args": BASIC_ARGS,
        "stages": [
            # [op, c, s, n, ...]
            # original res
            [("conv_k3", 8, 1, 1)],
            [("conv_k1", 8, 1, 1)],
            [("conv_k3", 1, 1, 1, CONV_ONLY)],
            # downsampled (x2)
            [("ir_k3", 16, 2, 1, e1)],
            [("conv_k1", 16, 1, 1)],
            [("ir_k3", 8, -2, 1, e1)],
            # downsampled (x4)
            [("ir_k3", 24, 2, 2, e2)],
            [("conv_k1", 24, 1, 1)],
            [("ir_k3", 16, -2, 1, e2)],
            # downsampled (x8)
            [("ir_k3", 32, 2, 3, e3)],
            [("conv_k1", 32, 1, 1)],
            [("ir_k3", 24, -2, 1, e3)],
            # downsampled (x16)
            [("ir_k3", 40, 2, 4, e6)],
            [("conv_k1", 40, 1, 1)],
            [("ir_k3", 32, -2, 3, e6)],
            # downsampled (x16)
            [("ir_k3", 48, 1, 4, e6)],
            [("conv_k1", 48, 1, 1)],
            [("ir_k3", 40, 1, 3, e6)],
        ],
    },
    "xirp5b": {
        # 96: nparams: 0.33064, nflops 33.687936
        "basic_args": BASIC_ARGS,
        "stages": [
            # [op, c, s, n, ...]
            # original res
            [("conv_k3", 8, 1, 1)],
            [("conv_k1", 8, 1, 1)],
            [("conv_k3", 1, 1, 1, CONV_ONLY)],
            # downsampled (x2)
            [("ir_k3", 16, 2, 1, e1)],
            [("conv_k1", 16, 1, 1)],
            [("ir_k3", 8, -2, 1, e1)],
            # downsampled (x4)
            [("ir_k3", 24, 2, 2, e2)],
            [("conv_k1", 24, 1, 1)],
            [("ir_k3", 16, -2, 1, e2)],
            # downsampled (x8)
            [("ir_k3", 32, 2, 3, e3)],
            [("conv_k1", 32, 1, 1)],
            [("ir_k3", 24, -2, 1, e3)],
            # downsampled (x16)
            [("ir_k3", 40, 2, 7, e6)],
            [("conv_k1", 40, 1, 1)],
            [("ir_k3", 32, -2, 2, e6)],
            # downsampled (x16)
            [("ir_k3", 48, 1, 1, e6)],
            [("conv_k1", 48, 1, 1)],
            [("ir_k3", 40, 1, 4, e6)],
        ],
    },
    "xirp5c": {
        # 96: nparams: 0.33064, nflops 33.687936
        "basic_args": BASIC_ARGS,
        "stages": [
            # [op, c, s, n, ...]
            # original res
            [("conv_k3", 8, 1, 1)],
            [("conv_k1", 8, 1, 1)],
            [("conv_k3", 1, 1, 1, CONV_ONLY)],
            # downsampled (x2)
            [("ir_k3", 16, 2, 1, e1)],
            [("conv_k1", 16, 1, 1)],
            [("ir_k3", 8, -2, 1, e1)],
            # downsampled (x4)
            [("ir_k3", 24, 2, 2, e2)],
            [("conv_k1", 24, 1, 1)],
            [("ir_k3", 16, -2, 1, e2)],
            # downsampled (x8)
            [("ir_k3", 32, 2, 3, e3)],
            [("conv_k1", 32, 1, 1)],
            [("ir_k3", 24, -2, 1, e3)],
            # downsampled (x16)
            [("ir_k3", 40, 2, 6, e6)],
            [("conv_k1", 40, 1, 1)],
            [("ir_k3", 32, -2, 1, e6)],
            # downsampled (x16)
            [("ir_k3", 48, 1, 1, e6)],
            [("conv_k1", 48, 1, 1)],
            [("ir_k3", 40, 1, 2, e6)],
        ],
    },
    "xirp5d": {
        # 96: nparams: 0.319888, nflops 31.99968
        "basic_args": BASIC_ARGS,
        "stages": [
            # [op, c, s, n, ...]
            # original res
            [("conv_k3", 8, 1, 1)],
            [("conv_k1", 8, 1, 1)],
            [("conv_k3", 1, 1, 1, CONV_ONLY)],
            # downsampled (x2)
            [("ir_k3", 16, 2, 1, e1)],
            [("conv_k1", 16, 1, 1)],
            [("ir_k3", 8, -2, 1, e1)],
            # downsampled (x4)
            [("ir_k3", 24, 2, 2, e2)],
            [("conv_k1", 24, 1, 1)],
            [("ir_k3", 16, -2, 1, e2)],
            # downsampled (x8)
            [("ir_k3", 32, 2, 3, e3)],
            [("conv_k1", 32, 1, 1)],
            [("ir_k3", 24, -2, 1, e3)],
            # downsampled (x16)
            [("ir_k3", 40, 2, 3, e6), ("ir_k3", 48, 1, 3, e6)],
            [("conv_k1", 48, 1, 1)],
            [("ir_k3", 32, -2, 1, e6)],
            # downsampled (x16)
            [("ir_k3", 64, 1, 1, e6)],
            [("conv_k1", 64, 1, 1)],
            [("ir_k3", 48, 1, 2, e6)],
        ],
    },
    "xirp6": {
        # 96: nparams: 0.349936, nflops 35.89632
        "basic_args": BASIC_ARGS,
        "stages": [
            # [op, c, s, n, ...]
            # original res
            [("conv_k3", 8, 1, 1)],
            [("skip", 8, 1, 1)],
            [("conv_k3", 1, 1, 1, CONV_ONLY)],
            # downsampled (x2)
            [("ir_k3", 16, 2, 1, e1)],
            [("skip", 16, 1, 1)],
            [("ir_k3", 8, -2, 1, e1)],
            # downsampled (x4)
            [("ir_k3", 24, 2, 2, e2)],
            [("skip", 24, 1, 1)],
            [("ir_k3", 16, -2, 1, e2)],
            # downsampled (x8)
            [("ir_k3", 32, 2, 3, e3)],
            [("skip", 32, 1, 1)],
            [("ir_k3", 24, -2, 1, e3)],
            # downsampled (x16)
            [("ir_k3", 40, 2, 4, e6)],
            [("skip", 40, 1, 1)],
            [("ir_k3", 32, -2, 3, e6)],
            # downsampled (x16)
            [("ir_k3", 48, 1, 4, e6)],
            [("skip", 48, 1, 1)],
            [("ir_k3", 40, 1, 3, e6)],
        ],
    },
    "xirp6a": {
        # 96: nparams: 0.324816, nflops 31.888512
        "basic_args": BASIC_ARGS,
        "stages": [
            # [op, c, s, n, ...]
            # original res
            [("conv_k3", 8, 1, 1)],
            [("skip", 8, 1, 1)],
            [("conv_k3", 1, 1, 1, CONV_ONLY)],
            # downsampled (x2)
            [("ir_k3", 16, 2, 1, e1)],
            [("skip", 16, 1, 1)],
            [("ir_k3", 8, -2, 1, e1)],
            # downsampled (x4)
            [("ir_k3", 24, 2, 2, e2)],
            [("skip", 24, 1, 1)],
            [("ir_k3", 16, -2, 1, e2)],
            # downsampled (x8)
            [("ir_k3", 32, 2, 3, e3)],
            [("skip", 32, 1, 1)],
            [("ir_k3", 24, -2, 1, e3)],
            # downsampled (x16)
            [("ir_k3", 40, 2, 7, e6)],
            [("skip", 40, 1, 1)],
            [("ir_k3", 32, -2, 2, e6)],
            # downsampled (x16)
            [("ir_k3", 48, 1, 1, e6)],
            [("skip", 48, 1, 1)],
            [("ir_k3", 40, 1, 4, e6)],
        ],
    },
    "xirp6b": {
        # 96: nparams: 0.311568, nflops 30.1104
        "basic_args": BASIC_ARGS,
        "stages": [
            # [op, c, s, n, ...]
            # original res
            [("conv_k3", 8, 1, 1)],
            [("skip", 8, 1, 1)],
            [("conv_k3", 1, 1, 1, CONV_ONLY)],
            # downsampled (x2)
            [("ir_k3", 16, 2, 1, e1)],
            [("skip", 16, 1, 1)],
            [("ir_k3", 8, -2, 1, e1)],
            # downsampled (x4)
            [("ir_k3", 24, 2, 2, e2)],
            [("skip", 24, 1, 1)],
            [("ir_k3", 16, -2, 1, e2)],
            # downsampled (x8)
            [("ir_k3", 32, 2, 3, e3)],
            [("skip", 32, 1, 1)],
            [("ir_k3", 24, -2, 1, e3)],
            # downsampled (x16)
            [("ir_k3", 40, 2, 3, e6), ("ir_k3", 48, 1, 3, e6)],
            [("skip", 48, 1, 1)],
            [("ir_k3", 32, -2, 1, e6)],
            # downsampled (x16)
            [("ir_k3", 64, 1, 1, e6)],
            [("skip", 64, 1, 1)],
            [("ir_k3", 48, 1, 2, e6)],
        ],
    },
    "unet_s2": {
        # 96: nparams: 1.133136, nflops 57.42576
        "basic_args": BASIC_ARGS,
        "stages": [
            # [op, c, s, n, ...]
            # original res
            [("conv_k3", 12, 1, 1)],
            [("skip", 12, 1, 1)],
            [("conv_k3", 1, 1, 1, CONV_ONLY)],
            # downsampled (x2)
            [("conv_k3", 24, 2, 1)],
            [("skip", 24, 1, 1)],
            [("conv_k3", 12, -2, 1)],
            # downsampled (x4)
            [("conv_k3", 48, 2, 1)],
            [("skip", 48, 1, 1)],
            [("conv_k3", 24, -2, 1)],
            # downsampled (x8)
            [("conv_k3", 96, 2, 1)],
            [("skip", 96, 1, 1)],
            [("conv_k3", 48, -2, 1)],
            # downsampled (x16)
            [("conv_k3", 180, 2, 1)],
            [("skip", 180, 1, 1)],
            [("conv_k3", 96, -2, 1)],
            # downsampled (x32)
            [("conv_k3", 220, 2, 1)],
            [("skip", 220, 1, 1)],
            [("conv_k3", 180, -2, 1)],
        ],
    },
    "ps_unet": {
        # 192: nparams: 1.133136, nflops 229.70304
        "basic_args": BASIC_ARGS,
        "stages": [
            # [op, c, s, n, ...]
            # original res
            [("conv_k3", 12, 1, 1)],
            [("skip", 12, 1, 1)],
            [("conv_k3", 1, 1, 1, CONV_ONLY)],
            # downsampled (x2)
            [("maxpool", 12, 2, 1, PS_UNET_MAXPOOL), ("conv_k3", 24, 1, 1)],
            [("skip", 24, 1, 1)],
            [("conv_k3", 12, -2, 1)],
            # downsampled (x4)
            [("maxpool", 24, 2, 1, PS_UNET_MAXPOOL), ("conv_k3", 48, 1, 1)],
            [("skip", 48, 1, 1)],
            [("conv_k3", 24, -2, 1)],
            # downsampled (x8)
            [("maxpool", 48, 2, 1, PS_UNET_MAXPOOL), ("conv_k3", 96, 1, 1)],
            [("skip", 96, 1, 1)],
            [("conv_k3", 48, -2, 1)],
            # downsampled (x16)
            [("maxpool", 96, 2, 1, PS_UNET_MAXPOOL), ("conv_k3", 180, 1, 1)],
            [("skip", 180, 1, 1)],
            [("conv_k3", 96, -2, 1)],
            # downsampled (x32)
            [("maxpool", 180, 2, 1, PS_UNET_MAXPOOL), ("conv_k3", 220, 1, 1)],
            [("skip", 220, 1, 1)],
            [("conv_k3", 180, -2, 1)],
        ],
    },
    "xiru1": {
        # 96: nparams: 0.13912, nflops 25.760952
        "basic_args": BASIC_ARGS,
        "stages": [
            # [op, c, s, n, ...]
            # original res
            [("conv_k3", 16, 1, 1)],
            [("skip", 16, 1, 1)],
            [("conv_k3", 1, 1, 1, CONV_ONLY)],
            # downsampled (x2)
            [("ir_k3", 24, 2, 1, e1)],
            [("skip", 24, 1, 1)],
            [("ir_k3", 16, -2, 1, e1)],
            # downsampled (x4)
            [("ir_k3", 48, 2, 1, e1)],
            [("skip", 48, 1, 1)],
            [("ir_k3", 24, -2, 1, e1)],
            # downsampled (x8)
            [("ir_k3", 96, 2, 1, e1)],
            [("skip", 96, 1, 1)],
            [("ir_k3", 48, -2, 1, e1)],
            # downsampled (x16)
            [("ir_k3", 184, 2, 1, e1)],
            [("skip", 184, 1, 1)],
            [("ir_k3", 96, -2, 1, e1)],
            # downsampled (x32)
            [("ir_k3", 224, 2, 1, e1)],
            [("skip", 224, 1, 1)],
            [("ir_k3", 184, -2, 1, e1)],
        ],
    },
    "xiru2": {
        # 96: nparams: 0.152352, nflops 29.321784
        "basic_args": BASIC_ARGS,
        "stages": [
            # [op, c, s, n, ...]
            # original res
            [("conv_k3", 16, 1, 1)],
            [("skip", 16, 1, 1)],
            [("conv_k3", 1, 1, 1, CONV_ONLY)],
            # downsampled (x2)
            [("ir_k3", 32, 2, 1, e1)],
            [("skip", 32, 1, 1)],
            [("ir_k3", 16, -2, 1, e1)],
            # downsampled (x4)
            [("ir_k3", 48, 2, 1, e1)],
            [("skip", 48, 1, 1)],
            [("ir_k3", 32, -2, 1, e1)],
            # downsampled (x8)
            [("ir_k3", 96, 2, 1, e1)],
            [("skip", 96, 1, 1)],
            [("ir_k3", 48, -2, 1, e1)],
            # downsampled (x16)
            [("ir_k3", 184, 2, 1, e1)],
            [("skip", 184, 1, 1)],
            [("ir_k3", 96, -2, 1, e1)],
            # downsampled (x32)
            [("ir_k3", 256, 2, 1, e1)],
            [("skip", 256, 1, 1)],
            [("ir_k3", 184, -2, 1, e1)],
        ],
    },
    "xiru3": {
        # 96: nparams: 0.298896, nflops 50.754816
        "basic_args": BASIC_ARGS,
        "stages": [
            # [op, c, s, n, ...]
            # original res
            [("conv_k3", 16, 1, 1)],
            [("skip", 16, 1, 1)],
            [("conv_k3", 1, 1, 1, CONV_ONLY)],
            # downsampled (x2)
            [("ir_k3", 24, 2, 1, e1)],
            [("skip", 24, 1, 1)],
            [("ir_k3", 16, -2, 1, e1)],
            # downsampled (x4)
            [("ir_k3", 32, 2, 1, e3)],
            [("skip", 32, 1, 1)],
            [("ir_k3", 24, -2, 1, e3)],
            # downsampled (x8)
            [("ir_k3", 48, 2, 1, e6)],
            [("skip", 48, 1, 1)],
            [("ir_k3", 32, -2, 1, e6)],
            # downsampled (x16)
            [("ir_k3", 64, 2, 1, e6)],
            [("skip", 64, 1, 1)],
            [("ir_k3", 48, -2, 1, e6)],
            # downsampled (x32)
            [("ir_k3", 96, 2, 1, e6)],
            [("skip", 96, 1, 1)],
            [("ir_k3", 64, -2, 1, e6)],
        ],
    },
    "xiru4": {
        # 96: nparams: 0.197464, nflops 31.635072
        "basic_args": BASIC_ARGS,
        "stages": [
            # [op, c, s, n, ...]
            # original res
            [("conv_k3", 16, 1, 1)],
            [("skip", 16, 1, 1)],
            [("conv_k3", 1, 1, 1, CONV_ONLY)],
            # downsampled (x2)
            [("ir_k3", 24, 2, 1, e1)],
            [("skip", 24, 1, 1)],
            [("ir_k3", 16, -2, 1, e1)],
            # downsampled (x4)
            [("ir_k3", 32, 2, 1, e2)],
            [("skip", 32, 1, 1)],
            [("ir_k3", 24, -2, 1, e1)],
            # downsampled (x8)
            [("ir_k3", 48, 2, 1, e4)],
            [("skip", 48, 1, 1)],
            [("ir_k3", 32, -2, 1, e2)],
            # downsampled (x16)
            [("ir_k3", 64, 2, 1, e6)],
            [("skip", 64, 1, 1)],
            [("ir_k3", 48, -2, 1, e3)],
            # downsampled (x32)
            [("ir_k3", 96, 2, 1, e6)],
            [("skip", 96, 1, 1)],
            [("ir_k3", 64, -2, 1, e3)],
        ],
    },
    "xiru5": {
        # 96: nparams: 0.197464, nflops 31.635072
        "basic_args": BASIC_ARGS,
        "stages": [
            # [op, c, s, n, ...]
            # original res
            [("conv_k3", 16, 1, 1)],
            [("skip", 16, 1, 1)],
            [("conv_k3", 1, 1, 1, CONV_ONLY)],
            # downsampled (x2)
            [("ir_k3", 24, 2, 1, e1)],
            [("skip", 24, 1, 1)],
            [("ir_k3", 16, -2, 1, e1)],
            # downsampled (x4)
            [("ir_k3", 32, 2, 1, e4)],
            [("skip", 32, 1, 1)],
            [("ir_k3", 24, -2, 1, e1)],
            # downsampled (x8)
            [("ir_k3", 48, 2, 1, e6)],
            [("skip", 48, 1, 1)],
            [("ir_k3", 32, -2, 1, e1)],
            # downsampled (x16)
            [("ir_k3", 64, 2, 1, e6)],
            [("skip", 64, 1, 1)],
            [("ir_k3", 48, -2, 1, e1)],
            # downsampled (x32)
            [("ir_k3", 96, 2, 1, e6)],
            [("skip", 96, 1, 1)],
            [("ir_k3", 64, -2, 1, e1)],
        ],
    },
    "xiru6": {
        # 96: nparams: 0.197464, nflops 31.635072
        "basic_args": BASIC_ARGS,
        "stages": [
            # [op, c, s, n, ...]
            # original res
            [("conv_k3", 16, 1, 1)],
            [("skip", 16, 1, 1)],
            [("conv_k3", 1, 1, 1, CONV_ONLY)],
            # downsampled (x2)
            [("ir_k3", 24, 2, 1, e1)],
            [("skip", 24, 1, 1)],
            [("ir_k3", 16, -2, 1, e1)],
            # downsampled (x4)
            [("ir_k3", 32, 2, 1, e4)],
            [("skip", 32, 1, 1)],
            [("ir_k3", 24, -2, 1, e1)],
            # downsampled (x8)
            [("ir_k3", 48, 2, 1, e6)],
            [("skip", 48, 1, 1)],
            [("ir_k3", 32, -2, 1, e1)],
            # downsampled (x16)
            [("ir_k3", 64, 2, 1, e6)],
            [("skip", 64, 1, 1)],
            [("ir_k3", 48, -2, 1, e1)],
            # downsampled (x16)
            [("ir_k3", 96, 1, 1, e6)],
            [("skip", 96, 1, 1)],
            [("ir_k3", 64, 1, 1, e1)],
        ],
    },
    "xirp7": {
        # 96: nparams: 0.232816, nflops 32.939712
        "basic_args": BASIC_ARGS,
        "stages": [
            # [op, c, s, n, ...]
            # original res
            [("conv_k3", 8, 1, 1)],
            [("skip", 8, 1, 1)],
            [("conv_k3", 1, 1, 1, CONV_ONLY)],
            # downsampled (x2)
            [("ir_k3", 16, 2, 1, e3)],
            [("skip", 16, 1, 1)],
            [("ir_k3", 8, -2, 1, e1)],
            # downsampled (x4)
            [("ir_k3", 24, 2, 2, e3)],
            [("skip", 24, 1, 1)],
            [("ir_k3", 16, -2, 1, e2)],
            # downsampled (x8)
            [("ir_k3", 32, 2, 3, e4)],
            [("skip", 32, 1, 1)],
            [("ir_k3", 24, -2, 1, e4)],
            # downsampled (x16)
            [("ir_k3", 40, 2, 3, e6)],
            [("skip", 40, 1, 1)],
            [("ir_k3", 32, -2, 1, e6)],
            # downsampled (x16)
            [("ir_k3", 48, 1, 3, e6)],
            [("skip", 48, 1, 1)],
            [("ir_k3", 40, 1, 1, e6)],
        ],
    },
    "xirp7a": {
        # nparams: 0.257776, nflops 33.838272
        "basic_args": BASIC_ARGS,
        "stages": [
            # [op, c, s, n, ...]
            # original res
            [("conv_k3", 8, 1, 1)],
            [("skip", 8, 1, 1)],
            [("conv_k3", 1, 1, 1, CONV_ONLY)],
            # downsampled (x2)
            [("ir_k3", 16, 2, 1, e3)],
            [("skip", 16, 1, 1)],
            [("ir_k3", 8, -2, 1, e1)],
            # downsampled (x4)
            [("ir_k3", 24, 2, 2, e3)],
            [("skip", 24, 1, 1)],
            [("ir_k3", 16, -2, 1, e2)],
            # downsampled (x8)
            [("ir_k3", 32, 2, 3, e4)],
            [("skip", 32, 1, 1)],
            [("ir_k3", 24, -2, 1, e4)],
            # downsampled (x16)
            [("ir_k3", 40, 2, 6, e6)],
            [("skip", 40, 1, 1)],
            [("ir_k3", 32, -2, 1, e6)],
            # downsampled (x16)
            [("ir_k3", 48, 1, 1, e6)],
            [("skip", 48, 1, 1)],
            [("ir_k3", 40, 1, 2, e6)],
        ],
    },
    "xirp7b": {
        # 96: nparams: 0.216232, nflops 34.014528
        "basic_args": BASIC_ARGS,
        "stages": [
            # [op, c, s, n, ...]
            # original res
            [("conv_k3", 8, 1, 1)],
            [("skip", 8, 1, 1)],
            [("conv_k3", 1, 1, 1, CONV_ONLY)],
            # downsampled (x2)
            [("ir_k3", 16, 2, 1, e3)],
            [("skip", 16, 1, 1)],
            [("ir_k3", 8, -2, 1, e1)],
            # downsampled (x4)
            [("ir_k3", 24, 2, 2, e3)],
            [("skip", 24, 1, 1)],
            [("ir_k3", 16, -2, 1, e3)],
            # downsampled (x8)
            [("ir_k3", 32, 2, 3, e4)],
            [("skip", 32, 1, 1)],
            [("ir_k3", 24, -2, 1, e4)],
            # downsampled (x16)
            [("ir_k3", 40, 2, 5, e6)],
            [("skip", 40, 1, 1)],
            [("ir_k3", 32, -2, 1, e6)],
            # downsampled (x16)
            [("ir_k3", 48, 1, 1, e6)],
            [("skip", 48, 1, 1)],
            [("ir_k3", 40, 1, 1, e6)],
        ],
    },
    "xirp7c": {
        # 96: nparams: 0.216232, nflops 34.014528
        "basic_args": BASIC_ARGS,
        "stages": [
            # [op, c, s, n, ...]
            # original res
            [("conv_k3", 8, 1, 1)],
            [("skip", 8, 1, 1)],
            [("conv_k3", 1, 1, 1, CONV_ONLY)],
            # downsampled (x2)
            [("ir_k3", 16, 2, 1, e3)],
            [("skip", 16, 1, 1)],
            [("ir_k3", 8, -2, 1, e1)],
            # downsampled (x4)
            [("ir_k3", 24, 2, 2, e3)],
            [("skip", 24, 1, 1)],
            [("ir_k3", 16, -2, 1, e3)],
            # downsampled (x8)
            [("ir_k3", 32, 2, 3, e4)],
            [("skip", 32, 1, 1)],
            [("ir_k3", 24, -2, 1, e4)],
            # downsampled (x16)
            [("ir_k3", 40, 2, 5, e6), ("ir_k3", 48, 1, 1, e6)],
            [("skip", 48, 1, 1)],
            [("ir_k3", 40, 1, 1, e6), ("ir_k3", 32, -2, 1, e6)],
        ],
    },
    "xirp7d": {
        # 96: nparams: 0.165016, nflops 32.170752
        "basic_args": BASIC_ARGS,
        "stages": [
            # [op, c, s, n, ...]
            # original res
            [("conv_k3", 8, 1, 1)],
            [("skip", 8, 1, 1)],
            [("conv_k3", 1, 1, 1, CONV_ONLY)],
            # downsampled (x2)
            [("ir_k3", 16, 2, 1, e3)],
            [("skip", 16, 1, 1)],
            [("ir_k3", 8, -2, 1, e1)],
            # downsampled (x4)
            [("ir_k3", 24, 2, 2, e3)],
            [("skip", 24, 1, 1)],
            [("ir_k3", 16, -2, 1, e3)],
            # downsampled (x8)
            [("ir_k3", 32, 2, 3, e4)],
            [("skip", 32, 1, 1)],
            [("ir_k3", 24, -2, 1, e4)],
            # downsampled (x16)
            [("ir_k3", 40, 2, 5, e6)],
            [("skip", 40, 1, 1)],
            [("ir_k3", 32, -2, 1, e6)],
        ],
    },
    "xirp7e": {
        # 96: nparams: 0.179032, nflops 34.189056
        "basic_args": BASIC_ARGS,
        "stages": [
            # [op, c, s, n, ...]
            # original res
            [("conv_k3", 8, 1, 1)],
            [("skip", 8, 1, 1)],
            [("conv_k3", 1, 1, 1, CONV_ONLY)],
            # downsampled (x2)
            [("ir_k3", 16, 2, 1, e3)],
            [("skip", 16, 1, 1)],
            [("ir_k3", 8, -2, 1, e1)],
            # downsampled (x4)
            [("ir_k3", 24, 2, 2, e3)],
            [("skip", 24, 1, 1)],
            [("ir_k3", 16, -2, 1, e3)],
            # downsampled (x8)
            [("ir_k3", 32, 2, 3, e4)],
            [("skip", 32, 1, 1)],
            [("ir_k3", 24, -2, 1, e4)],
            # downsampled (x16)
            [("ir_k3", 40, 2, 5, e6)],
            [("skip", 40, 1, 1)],
            [("ir_k3", 32, -2, 2, e6)],
        ],
    },
    "xirp7f": {
        # 96: nparams: 0.136312, nflops 32.651136
        "basic_args": BASIC_ARGS,
        "stages": [
            # [op, c, s, n, ...]
            # original res
            [("conv_k3", 8, 1, 1)],
            [("skip", 8, 1, 1)],
            [("conv_k3", 1, 1, 1, CONV_ONLY)],
            # downsampled (x2)
            [("ir_k3", 16, 2, 1, e3)],
            [("skip", 16, 1, 1)],
            [("ir_k3", 8, -2, 1, e1)],
            # downsampled (x4)
            [("ir_k3", 24, 2, 2, e3)],
            [("skip", 24, 1, 1)],
            [("ir_k3", 16, -2, 1, e3)],
            # downsampled (x8)
            [("ir_k3", 32, 2, 3, e4)],
            [("skip", 32, 1, 1)],
            [("ir_k3", 24, -2, 1, e4)],
            # downsampled (x16)
            [("ir_k3", 40, 2, 3, e6)],
            [("skip", 40, 1, 1)],
            [("ir_k3", 32, -2, 2, e6)],
        ],
    },
    "xirp7g": {
        # 96: nparams: 0.12308, nflops 34.319232
        "basic_args": BASIC_ARGS,
        "stages": [
            # [op, c, s, n, ...]
            # original res
            [("conv_k3", 8, 1, 1)],
            [("skip", 8, 1, 1)],
            [("conv_k3", 1, 1, 1, CONV_ONLY)],
            # downsampled (x2)
            [("ir_k3", 16, 2, 1, e3)],
            [("skip", 16, 1, 1)],
            [("ir_k3", 8, -2, 1, e2)],
            # downsampled (x4)
            [("ir_k3", 24, 2, 2, e3)],
            [("skip", 24, 1, 1)],
            [("ir_k3", 16, -2, 1, e3)],
            # downsampled (x8)
            [("ir_k3", 32, 2, 3, e4)],
            [("skip", 32, 1, 1)],
            [("ir_k3", 24, -2, 1, e4)],
            # downsampled (x16)
            [("ir_k3", 40, 2, 3, e6)],
            [("skip", 40, 1, 1)],
            [("ir_k3", 32, -2, 1, e6)],
        ],
    },
    "xirp7h": {
        # 96: nparams: 0.123016, nflops 33.729408
        "basic_args": BASIC_ARGS,
        "stages": [
            # [op, c, s, n, ...]
            # original res
            [("conv_k3", 8, 1, 1)],
            [("skip", 8, 1, 1)],
            [("conv_k1", 1, 1, 1, CONV_ONLY)],
            # downsampled (x2)
            [("ir_k3", 16, 2, 1, e3)],
            [("skip", 16, 1, 1)],
            [("ir_k3", 8, -2, 1, e2)],
            # downsampled (x4)
            [("ir_k3", 24, 2, 2, e3)],
            [("skip", 24, 1, 1)],
            [("ir_k3", 16, -2, 1, e3)],
            # downsampled (x8)
            [("ir_k3", 32, 2, 3, e4)],
            [("skip", 32, 1, 1)],
            [("ir_k3", 24, -2, 1, e4)],
            # downsampled (x16)
            [("ir_k3", 40, 2, 3, e6)],
            [("skip", 40, 1, 1)],
            [("ir_k3", 32, -2, 1, e6)],
        ],
    },
    "xirp7i": {
        # nparams: 0.257712, nflops 33.248448
        "basic_args": BASIC_ARGS,
        "stages": [
            # [op, c, s, n, ...]
            # original res
            [("conv_k3", 8, 1, 1)],
            [("skip", 8, 1, 1)],
            [("conv_k1", 1, 1, 1, CONV_ONLY)],
            # downsampled (x2)
            [("ir_k3", 16, 2, 1, e3)],
            [("skip", 16, 1, 1)],
            [("ir_k3", 8, -2, 1, e1)],
            # downsampled (x4)
            [("ir_k3", 24, 2, 2, e3)],
            [("skip", 24, 1, 1)],
            [("ir_k3", 16, -2, 1, e2)],
            # downsampled (x8)
            [("ir_k3", 32, 2, 3, e4)],
            [("skip", 32, 1, 1)],
            [("ir_k3", 24, -2, 1, e4)],
            # downsampled (x16)
            [("ir_k3", 40, 2, 6, e6)],
            [("skip", 40, 1, 1)],
            [("ir_k3", 32, -2, 1, e6)],
            # downsampled (x16)
            [("ir_k3", 48, 1, 1, e6)],
            [("skip", 48, 1, 1)],
            [("ir_k3", 40, 1, 2, e6)],
        ],
    },
    "xirp7i_rf": {
        # nparams: 0.257712, nflops 33.248448
        "basic_args": BASIC_ARGS,
        "stages": [
            # [op, c, s, n, ...]
            # original res
            [("conv_k3", 8, 1, 1, CONV_PAD_REFLECT)],
            [("skip", 8, 1, 1)],
            [("conv_k1", 1, 1, 1, CONV_ONLY)],
            # downsampled (x2)
            [("ir_k3", 16, 2, 1, e3, DW_PAD_REFLECT)],
            [("skip", 16, 1, 1)],
            [("ir_k3", 8, -2, 1, e1, DW_PAD_REFLECT)],
            # downsampled (x4)
            [("ir_k3", 24, 2, 2, e3, DW_PAD_REFLECT)],
            [("skip", 24, 1, 1)],
            [("ir_k3", 16, -2, 1, e2, DW_PAD_REFLECT)],
            # downsampled (x8)
            [("ir_k3", 32, 2, 3, e4, DW_PAD_REFLECT)],
            [("skip", 32, 1, 1)],
            [("ir_k3", 24, -2, 1, e4, DW_PAD_REFLECT)],
            # downsampled (x16)
            [("ir_k3", 40, 2, 6, e6, DW_PAD_REFLECT)],
            [("skip", 40, 1, 1)],
            [("ir_k3", 32, -2, 1, e6, DW_PAD_REFLECT)],
            # downsampled (x16)
            [("ir_k3", 48, 1, 1, e6, DW_PAD_REFLECT)],
            [("skip", 48, 1, 1)],
            [("ir_k3", 40, 1, 2, e6, DW_PAD_REFLECT)],
        ],
    },
    "xirp7i_add_input": {
        # nparams: 0.261288, nflops 33.377184
        "basic_args": BASIC_ARGS,
        "stages": [
            # [op, c, s, n, ...]
            # original res
            [("conv_k3", 8, 1, 1)],
            [("skip", 8, 1, 1)],
            [("conv_k1", 1, 1, 1, CONV_ONLY)],
            # downsampled (x2)
            [("ir_k3", 16, 2, 1, e3)],
            [("skip", 16, 1, 1)],
            [("ir_k3", 8, -2, 1, e1)],
            # downsampled (x4)
            [("ir_k3", 24, 2, 2, e3)],
            [("skip", 24, 1, 1)],
            [("ir_k3", 16, -2, 1, e2)],
            # downsampled (x8)
            [("ir_k3", 32, 2, 3, e4)],
            [("skip", 32, 1, 1)],
            [("ir_k3", 24, -2, 1, e4)],
            # downsampled (x16)
            [("ir_k3", 40, 2, 6, e6)],
            [("skip", 40, 1, 1)],
            [("ir_k3", 32, -2, 1, e6)],
            # downsampled (x16)
            [("ir_k3", 48, 1, 1, e6)],
            [("skip", 48, 1, 1)],
            [("ir_k3", 40, 1, 2, e6)],
        ],
        "additional_inputs": {
            "17": 4,  # stage 5 has additional input
        },
    },
    "xirp8": {
        # 96: nparams: 0.177856, nflops 32.47488
        "basic_args": BASIC_ARGS,
        "stages": [
            # [op, c, s, n, ...]
            # original res
            [("conv_k3", 8, 1, 1)],
            [("skip", 8, 1, 1)],
            [("conv_k3", 1, 1, 1, CONV_ONLY)],
            # downsampled (x2)
            [("ir_k3", 16, 2, 1, e3)],
            [("skip", 16, 1, 1)],
            [("ir_k3", 8, -2, 1, e1)],
            # downsampled (x4)
            [("ir_k3", 24, 2, 2, e3)],
            [("skip", 24, 1, 1)],
            [("ir_k3", 16, -2, 1, e2)],
            # downsampled (x8)
            [("ir_k3", 32, 2, 3, e4)],
            [("skip", 32, 1, 1)],
            [("ir_k3", 24, -2, 1, e4)],
            # downsampled (x16)
            [("ir_k3", 40, 2, 5, e6)],
            [("skip", 40, 1, 1)],
            [("ir_k3", 32, -2, 2, e6)],
        ],
    },
    "xirp8a": {
        # 96: nparams: 0.177792, nflops 31.885056
        "basic_args": BASIC_ARGS,
        "stages": [
            # [op, c, s, n, ...]
            # original res
            [("conv_k3", 8, 1, 1)],
            [("skip", 8, 1, 1)],
            [("conv_k1", 1, 1, 1, CONV_ONLY)],
            # downsampled (x2)
            [("ir_k3", 16, 2, 1, e3)],
            [("skip", 16, 1, 1)],
            [("ir_k3", 8, -2, 1, e1)],
            # downsampled (x4)
            [("ir_k3", 24, 2, 2, e3)],
            [("skip", 24, 1, 1)],
            [("ir_k3", 16, -2, 1, e2)],
            # downsampled (x8)
            [("ir_k3", 32, 2, 3, e4)],
            [("skip", 32, 1, 1)],
            [("ir_k3", 24, -2, 1, e4)],
            # downsampled (x16)
            [("ir_k3", 40, 2, 5, e6)],
            [("skip", 40, 1, 1)],
            [("ir_k3", 32, -2, 2, e6)],
        ],
    },
    "xirp8b": {
        # 96: nparams: 0.155384, nflops 30.538368
        "basic_args": BASIC_ARGS,
        "stages": [
            # [op, c, s, n, ...]
            # original res
            [("conv_k3", 8, 1, 1)],
            [("skip", 8, 1, 1)],
            [("conv_k1", 1, 1, 1, CONV_ONLY)],
            # downsampled (x2)
            [("ir_k3", 16, 2, 1, e3)],
            [("skip", 16, 1, 1)],
            [("ir_k3", 8, -2, 1, e1)],
            # downsampled (x4)
            [("ir_k3", 24, 2, 2, e3)],
            [("skip", 24, 1, 1)],
            [("ir_k3", 16, -2, 1, e2)],
            # downsampled (x8)
            [("ir_k3", 32, 2, 3, e4)],
            [("skip", 32, 1, 1)],
            [("ir_k3", 24, -2, 1, e4)],
            # downsampled (x16)
            [("ir_k3", 40, 2, 5, e5)],
            [("skip", 40, 1, 1)],
            [("ir_k3", 32, -2, 2, e5)],
        ],
    },
    "xirp8c": {
        # 96: nparams: 0.147728, nflops 29.152512
        "basic_args": BASIC_ARGS,
        "stages": [
            # [op, c, s, n, ...]
            # original res
            [("conv_k3", 8, 1, 1)],
            [("skip", 8, 1, 1)],
            [("conv_k1", 1, 1, 1, CONV_ONLY)],
            # downsampled (x2)
            [("ir_k3", 16, 2, 1, e3)],
            [("skip", 16, 1, 1)],
            [("ir_k3", 8, -2, 1, e1)],
            # downsampled (x4)
            [("ir_k3", 24, 2, 2, e3)],
            [("skip", 24, 1, 1)],
            [("ir_k3", 16, -2, 1, e2)],
            # downsampled (x8)
            [("ir_k3", 32, 2, 3, e4)],
            [("skip", 32, 1, 1)],
            [("ir_k3", 24, -2, 1, e3)],
            # downsampled (x16)
            [("ir_k3", 40, 2, 5, e5)],
            [("skip", 40, 1, 1)],
            [("ir_k3", 32, -2, 2, e4)],
        ],
    },
    "xirp8d": {
        # 96: nparams: 0.165264, nflops 28.742688
        "basic_args": BASIC_ARGS,
        "stages": [
            # [op, c, s, n, ...]
            # original res
            [("conv_k3", 8, 1, 1)],
            [("skip", 8, 1, 1)],
            [("conv_k1", 1, 1, 1, CONV_ONLY)],
            # downsampled (x2)
            [("ir_k3", 16, 2, 1, e2)],
            [("skip", 16, 1, 1)],
            [("ir_k3", 8, -2, 1, e1)],
            # downsampled (x4)
            [("ir_k3", 24, 2, 2, e3)],
            [("skip", 24, 1, 1)],
            [("ir_k3", 16, -2, 1, e2)],
            # downsampled (x8)
            [("ir_k3", 32, 2, 3, e4)],
            [("skip", 32, 1, 1)],
            [("ir_k3", 24, -2, 1, e3)],
            # downsampled (x16)
            [("ir_k3", 40, 2, 6, e5)],
            [("skip", 40, 1, 1)],
            [("ir_k3", 32, -2, 2, e4)],
        ],
    },
    "xirp8e": {
        # 96: nparams: 0.12032, nflops 26.115552
        "basic_args": BASIC_ARGS,
        "stages": [
            # [op, c, s, n, ...]
            # original res
            [("conv_k3", 8, 1, 1)],
            [("skip", 8, 1, 1)],
            [("conv_k1", 1, 1, 1, CONV_ONLY)],
            # downsampled (x2)
            [("ir_k3", 16, 2, 1, e2)],
            [("skip", 16, 1, 1)],
            [("ir_k3", 8, -2, 1, e1)],
            # downsampled (x4)
            [("ir_k3", 24, 2, 2, e3)],
            [("skip", 24, 1, 1)],
            [("ir_k3", 16, -2, 1, e2)],
            # downsampled (x8)
            [("ir_k3", 32, 2, 3, e4)],
            [("skip", 32, 1, 1)],
            [("ir_k3", 24, -2, 1, e3)],
            # downsampled (x16)
            [("ir_k3", 40, 2, 4, e5)],
            [("skip", 40, 1, 1)],
            [("ir_k3", 32, -2, 1, e4)],
        ],
    },
    "xirp8f": {
        # 96: nparams: 0.120155, nflops 24.594912
        "basic_args": BASIC_ARGS,
        "stages": [
            # [op, c, s, n, ...]
            # original res
            [("ir_k3", 8, 1, 1, e1)],
            [("skip", 8, 1, 1)],
            [("conv_k1", 1, 1, 1, CONV_ONLY)],
            # downsampled (x2)
            [("ir_k3", 16, 2, 1, e2)],
            [("skip", 16, 1, 1)],
            [("ir_k3", 8, -2, 1, e1)],
            # downsampled (x4)
            [("ir_k3", 24, 2, 2, e3)],
            [("skip", 24, 1, 1)],
            [("ir_k3", 16, -2, 1, e2)],
            # downsampled (x8)
            [("ir_k3", 32, 2, 3, e4)],
            [("skip", 32, 1, 1)],
            [("ir_k3", 24, -2, 1, e3)],
            # downsampled (x16)
            [("ir_k3", 40, 2, 4, e5)],
            [("skip", 40, 1, 1)],
            [("ir_k3", 32, -2, 1, e4)],
        ],
    },
    "xirp8g": {
        # 96: nparams: 0.120264, nflops 25.599456
        "basic_args": BASIC_ARGS,
        "stages": [
            # [op, c, s, n, ...]
            # original res
            [("ir_k3", 8, 1, 1, e2)],
            [("skip", 8, 1, 1)],
            [("conv_k1", 1, 1, 1, CONV_ONLY)],
            # downsampled (x2)
            [("ir_k3", 16, 2, 1, e2)],
            [("skip", 16, 1, 1)],
            [("ir_k3", 8, -2, 1, e1)],
            # downsampled (x4)
            [("ir_k3", 24, 2, 2, e3)],
            [("skip", 24, 1, 1)],
            [("ir_k3", 16, -2, 1, e2)],
            # downsampled (x8)
            [("ir_k3", 32, 2, 3, e4)],
            [("skip", 32, 1, 1)],
            [("ir_k3", 24, -2, 1, e3)],
            # downsampled (x16)
            [("ir_k3", 40, 2, 4, e5)],
            [("skip", 40, 1, 1)],
            [("ir_k3", 32, -2, 1, e4)],
        ],
    },
    "xirp8h": {
        # 96: nparams: 0.084304, nflops 23.498784
        "basic_args": BASIC_ARGS,
        "stages": [
            # [op, c, s, n, ...]
            # original res
            [("conv_k3", 8, 1, 1)],
            [("skip", 8, 1, 1)],
            [("conv_k1", 1, 1, 1, CONV_ONLY)],
            # downsampled (x2)
            [("ir_k3", 16, 2, 1, e2)],
            [("skip", 16, 1, 1)],
            [("ir_k3", 8, -2, 1, e1)],
            # downsampled (x4)
            [("ir_k3", 24, 2, 2, e3)],
            [("skip", 24, 1, 1)],
            [("ir_k3", 16, -2, 1, e2)],
            # downsampled (x8)
            [("ir_k3", 32, 2, 3, e3)],
            [("skip", 32, 1, 1)],
            [("ir_k3", 24, -2, 1, e3)],
            # downsampled (x16)
            [("ir_k3", 40, 2, 4, e3)],
            [("skip", 40, 1, 1)],
            [("ir_k3", 32, -2, 1, e3)],
        ],
    },
    "xirp8i": {
        # 96: nparams: 0.084304, nflops 23.498784
        "basic_args": BASIC_ARGS,
        "stages": [
            # [op, c, s, n, ...]
            # original res
            [("ir_k3", 8, 1, 1, e2)],
            [("skip", 8, 1, 1)],
            [("conv_k1", 1, 1, 1, CONV_ONLY)],
            # downsampled (x2)
            [("ir_k3", 16, 2, 1, e2)],
            [("skip", 16, 1, 1)],
            [("ir_k3", 8, -2, 1, e1)],
            # downsampled (x4)
            [("ir_k3", 24, 2, 2, e3)],
            [("skip", 24, 1, 1)],
            [("ir_k3", 16, -2, 1, e2)],
            # downsampled (x8)
            [("ir_k3", 32, 2, 3, e3)],
            [("skip", 32, 1, 1)],
            [("ir_k3", 24, -2, 1, e3)],
            # downsampled (x16)
            [("ir_k3", 40, 2, 4, e3)],
            [("skip", 40, 1, 1)],
            [("ir_k3", 32, -2, 1, e3)],
        ],
    },
    "xirp12a": {
        # nparams: 0.232824, nflops 33.01344
        "basic_args": BASIC_ARGS,
        "stages": [
            # [op, c, s, n, ...]
            # original res
            [("conv_k3", 8, 1, 1)],
            [("skip", 8, 1, 1)],
            [("conv_k1", 1, 1, 1, CONV_ONLY)],
            # downsampled (x2)
            [("ir_k3", 16, 2, 1, e3)],
            [("skip", 16, 1, 1)],
            [("ir_k3", 8, -2, 1, e1)],
            # downsampled (x4)
            [("ir_k3", 24, 2, 2, e3)],
            [("skip", 24, 1, 1)],
            [("ir_k3", 16, -2, 1, e2)],
            # downsampled (x8)
            [("ir_k3", 32, 2, 3, e4)],
            [("skip", 32, 1, 1)],
            [("ir_k3", 24, -2, 1, e4)],
            # downsampled (x16)
            [("ir_k3", 40, 2, 2, e6)],
            [("skip", 40, 1, 1)],
            [("ir_k3", 32, -2, 1, e6)],
            # downsampled (x16)
            [("ir_k3", 48, 1, 3, e6)],
            [("skip", 48, 1, 1)],
            [("ir_k3", 40, 1, 2, e6)],
        ],
    },
    "xirp12b": {
        # nparams: 0.233176, nflops 32.592384
        "basic_args": BASIC_ARGS,
        "stages": [
            # [op, c, s, n, ...]
            # original res
            [("conv_k3", 8, 1, 1)],
            [("skip", 8, 1, 1)],
            [("conv_k1", 1, 1, 1, CONV_ONLY)],
            # downsampled (x2)
            [("ir_k3", 16, 2, 1, e3)],
            [("skip", 16, 1, 1)],
            [("ir_k3", 8, -2, 1, e1)],
            # downsampled (x4)
            [("ir_k3", 24, 2, 2, e3)],
            [("skip", 24, 1, 1)],
            [("ir_k3", 16, -2, 1, e2)],
            # downsampled (x8)
            [("ir_k3", 32, 2, 3, e4)],
            [("skip", 32, 1, 1)],
            [("ir_k3", 24, -2, 1, e4)],
            # downsampled (x16)
            [("ir_k3", 48, 2, 4, e4)],
            [("skip", 48, 1, 1)],
            [("ir_k3", 32, -2, 1, e4)],
            # downsampled (x16)
            [("ir_k3", 48, 1, 3, e4)],
            [("skip", 48, 1, 1)],
            [("ir_k3", 48, 1, 2, e4)],
        ],
    },
    "xirp12c": {
        # nparams: 0.234424, nflops 43.61184
        "basic_args": BASIC_ARGS,
        "stages": [
            # [op, c, s, n, ...]
            # original res
            [("conv_k3", 16, 1, 1)],
            [("skip", 16, 1, 1)],
            [("conv_k1", 1, 1, 1, CONV_ONLY)],
            # downsampled (x2)
            [("ir_k3", 16, 2, 1, e3)],
            [("skip", 16, 1, 1)],
            [("ir_k3", 16, -2, 1, e1)],
            # downsampled (x4)
            [("ir_k3", 24, 2, 2, e3)],
            [("skip", 24, 1, 1)],
            [("ir_k3", 16, -2, 1, e2)],
            # downsampled (x8)
            [("ir_k3", 32, 2, 3, e4)],
            [("skip", 32, 1, 1)],
            [("ir_k3", 24, -2, 1, e4)],
            # downsampled (x16)
            [("ir_k3", 40, 2, 2, e6)],
            [("skip", 40, 1, 1)],
            [("ir_k3", 32, -2, 1, e6)],
            # downsampled (x16)
            [("ir_k3", 48, 1, 3, e6)],
            [("skip", 48, 1, 1)],
            [("ir_k3", 40, 1, 2, e6)],
        ],
    },
    "xirp12d": {
        # nparams: 0.171288, nflops 40.76352
        "basic_args": BASIC_ARGS,
        "stages": [
            # [op, c, s, n, ...]
            # original res
            [("conv_k3", 16, 1, 1)],
            [("skip", 16, 1, 1)],
            [("conv_k1", 1, 1, 1, CONV_ONLY)],
            # downsampled (x2)
            [("ir_k3", 16, 2, 1, e3)],
            [("skip", 16, 1, 1)],
            [("ir_k3", 16, -2, 1, e1)],
            # downsampled (x4)
            [("ir_k3", 24, 2, 2, e3)],
            [("skip", 24, 1, 1)],
            [("ir_k3", 16, -2, 1, e2)],
            # downsampled (x8)
            [("ir_k3", 32, 2, 3, e4)],
            [("skip", 32, 1, 1)],
            [("ir_k3", 24, -2, 1, e4)],
            # downsampled (x16)
            [("ir_k3", 40, 2, 2, e4)],
            [("skip", 40, 1, 1)],
            [("ir_k3", 32, -2, 1, e4)],
            # downsampled (x16)
            [("ir_k3", 48, 1, 3, e4)],
            [("skip", 48, 1, 1)],
            [("ir_k3", 40, 1, 2, e4)],
        ],
    },
    "xirp12e": {
        # nparams: 0.170632, nflops 37.482624
        "basic_args": BASIC_ARGS,
        "stages": [
            # [op, c, s, n, ...]
            # original res
            [("conv_k3", 16, 1, 1)],
            [("skip", 16, 1, 1)],
            [("conv_k1", 1, 1, 1, CONV_ONLY)],
            # downsampled (x2)
            [("ir_k3", 16, 2, 1, e2)],
            [("skip", 16, 1, 1)],
            [("ir_k3", 16, -2, 1, e1)],
            # downsampled (x4)
            [("ir_k3", 24, 2, 2, e3)],
            [("skip", 24, 1, 1)],
            [("ir_k3", 16, -2, 1, e2)],
            # downsampled (x8)
            [("ir_k3", 32, 2, 3, e4)],
            [("skip", 32, 1, 1)],
            [("ir_k3", 24, -2, 1, e4)],
            # downsampled (x16)
            [("ir_k3", 40, 2, 2, e4)],
            [("skip", 40, 1, 1)],
            [("ir_k3", 32, -2, 1, e4)],
            # downsampled (x16)
            [("ir_k3", 48, 1, 3, e4)],
            [("skip", 48, 1, 1)],
            [("ir_k3", 40, 1, 2, e4)],
        ],
    },
    "xirp12f": {
        # nparams: 0.160168, nflops 33.898752
        "basic_args": BASIC_ARGS,
        "stages": [
            # [op, c, s, n, ...]
            # original res
            [("conv_k3", 16, 1, 1)],
            [("skip", 16, 1, 1)],
            [("conv_k1", 1, 1, 1, CONV_ONLY)],
            # downsampled (x2)
            [("ir_k3", 16, 2, 1, e2)],
            [("skip", 16, 1, 1)],
            [("ir_k3", 16, -2, 1, e1)],
            # downsampled (x4)
            [("ir_k3", 24, 2, 2, e2)],
            [("skip", 24, 1, 1)],
            [("ir_k3", 16, -2, 1, e2)],
            # downsampled (x8)
            [("ir_k3", 32, 2, 3, e3)],
            [("skip", 32, 1, 1)],
            [("ir_k3", 24, -2, 1, e3)],
            # downsampled (x16)
            [("ir_k3", 40, 2, 2, e4)],
            [("skip", 40, 1, 1)],
            [("ir_k3", 32, -2, 1, e4)],
            # downsampled (x16)
            [("ir_k3", 48, 1, 3, e4)],
            [("skip", 48, 1, 1)],
            [("ir_k3", 40, 1, 2, e4)],
        ],
    },
    "xirp13a": {
        # nparams: 0.187928, nflops 41.36256
        "basic_args": BASIC_ARGS,
        "stages": [
            # [op, c, s, n, ...]
            # original res
            [("conv_k3", 16, 1, 1)],
            [("skip", 16, 1, 1)],
            [("conv_k1", 1, 1, 1, CONV_ONLY)],
            # downsampled (x2)
            [("ir_k3", 16, 2, 1, e3)],
            [("skip", 16, 1, 1)],
            [("ir_k3", 16, -2, 1, e1)],
            # downsampled (x4)
            [("ir_k3", 24, 2, 2, e3)],
            [("skip", 24, 1, 1)],
            [("ir_k3", 16, -2, 1, e2)],
            # downsampled (x8)
            [("ir_k3", 32, 2, 3, e4)],
            [("skip", 32, 1, 1)],
            [("ir_k3", 24, -2, 1, e4)],
            # downsampled (x16)
            [("ir_k3", 40, 2, 6, e4)],
            [("skip", 40, 1, 1)],
            [("ir_k3", 32, -2, 1, e4)],
            # downsampled (x16)
            [("ir_k3", 48, 1, 1, e4)],
            [("skip", 48, 1, 1)],
            [("ir_k3", 40, 1, 2, e4)],
        ],
    },
    "xirp13b": {
        # nparams: 0.187584, nflops 38.192256
        "basic_args": BASIC_ARGS,
        "stages": [
            # [op, c, s, n, ...]
            # original res
            [("ir_k3", 16, 1, 1, e1)],
            [("skip", 16, 1, 1)],
            [("conv_k1", 1, 1, 1, CONV_ONLY)],
            # downsampled (x2)
            [("ir_k3", 16, 2, 1, e3)],
            [("skip", 16, 1, 1)],
            [("ir_k3", 16, -2, 1, e1)],
            # downsampled (x4)
            [("ir_k3", 24, 2, 2, e3)],
            [("skip", 24, 1, 1)],
            [("ir_k3", 16, -2, 1, e2)],
            # downsampled (x8)
            [("ir_k3", 32, 2, 3, e4)],
            [("skip", 32, 1, 1)],
            [("ir_k3", 24, -2, 1, e4)],
            # downsampled (x16)
            [("ir_k3", 40, 2, 6, e4)],
            [("skip", 40, 1, 1)],
            [("ir_k3", 32, -2, 1, e4)],
            # downsampled (x16)
            [("ir_k3", 48, 1, 1, e4)],
            [("skip", 48, 1, 1)],
            [("ir_k3", 40, 1, 2, e4)],
        ],
    },
    "xirp13c": {
        # nparams: 0.144864, nflops 36.654336
        "basic_args": BASIC_ARGS,
        "stages": [
            # [op, c, s, n, ...]
            # original res
            [("ir_k3", 16, 1, 1, e1)],
            [("skip", 16, 1, 1)],
            [("conv_k1", 1, 1, 1, CONV_ONLY)],
            # downsampled (x2)
            [("ir_k3", 16, 2, 1, e3)],
            [("skip", 16, 1, 1)],
            [("ir_k3", 16, -2, 1, e1)],
            # downsampled (x4)
            [("ir_k3", 24, 2, 2, e3)],
            [("skip", 24, 1, 1)],
            [("ir_k3", 16, -2, 1, e2)],
            # downsampled (x8)
            [("ir_k3", 32, 2, 3, e4)],
            [("skip", 32, 1, 1)],
            [("ir_k3", 24, -2, 1, e4)],
            # downsampled (x16)
            [("ir_k3", 40, 2, 3, e4)],
            [("skip", 40, 1, 1)],
            [("ir_k3", 32, -2, 1, e4)],
            # downsampled (x16)
            [("ir_k3", 48, 1, 1, e4)],
            [("skip", 48, 1, 1)],
            [("ir_k3", 40, 1, 2, e4)],
        ],
    },
    "xirp13d": {
        # nparams: 0.188704, nflops 33.090048
        "basic_args": BASIC_ARGS,
        "stages": [
            # [op, c, s, n, ...]
            # original res
            [("ir_k3", 16, 1, 1, e1)],
            [("skip", 16, 1, 1)],
            [("ir_k3", 1, 1, 1, e1)],
            # downsampled (x2)
            [("ir_k3", 16, 2, 1, e2)],
            [("skip", 16, 1, 1)],
            [("ir_k3", 16, -2, 1, e1)],
            # downsampled (x4)
            [("ir_k3", 24, 2, 2, e2)],
            [("skip", 24, 1, 1)],
            [("ir_k3", 16, -2, 1, e2)],
            # downsampled (x8)
            [("ir_k3", 32, 2, 3, e3)],
            [("skip", 32, 1, 1)],
            [("ir_k3", 24, -2, 1, e3)],
            # downsampled (x16)
            [("ir_k3", 40, 2, 6, e4)],
            [("skip", 40, 1, 1)],
            [("ir_k3", 32, -2, 1, e4)],
            # downsampled (x16)
            [("ir_k3", 48, 1, 1, e5)],
            [("skip", 48, 1, 1)],
            [("ir_k3", 40, 1, 2, e5)],
        ],
    },
    "xirp14a": {
        # nparams: 0.271872, nflops 15.057792
        "basic_args": BASIC_ARGS,
        "stages": [
            # [op, c, s, n, ...]
            # original res
            [("conv_k3", 8, 2, 1)],
            [("skip", 8, 1, 1)],
            [("ir_k3", 1, -2, 1, e1, {"skip_pwl_bn": True})],
            # downsampled (x2)
            [("ir_k3", 16, 2, 1, e3)],
            [("skip", 16, 1, 1)],
            [("ir_k3", 8, -2, 1, e1)],
            # downsampled (x4)
            [("ir_k3", 24, 2, 2, e3)],
            [("skip", 24, 1, 1)],
            [("ir_k3", 16, -2, 1, e2)],
            # downsampled (x8)
            [("ir_k3", 32, 2, 3, e4)],
            [("skip", 32, 1, 1)],
            [("ir_k3", 24, -2, 1, e4)],
            # downsampled (x16)
            [("ir_k3", 40, 1, 6, e6)],
            [("skip", 40, 1, 1)],
            [("ir_k3", 32, 1, 2, e6)],
            # downsampled (x16)
            [("ir_k3", 48, 1, 1, e6)],
            [("skip", 48, 1, 1)],
            [("ir_k3", 40, 1, 2, e6)],
        ],
        "stage_combiners": [
            # original res
            "choose_right",
            # downsampled (x2)
            "add",
            # downsampled (x4)
            "add",
            # downsampled (x8)
            "add",
            # downsampled (x16)
            "add",
        ],
    },
    "xirp14b": {
        # nparams: 0.395872, nflops 25.505856
        "basic_args": BASIC_ARGS,
        "stages": [
            # [op, c, s, n, ...]
            # original res
            [("conv_k3", 16, 2, 1)],
            [("skip", 16, 1, 1)],
            [("ir_k3", 1, -2, 1, e1, {"skip_pwl_bn": True})],
            # downsampled (x2)
            [("ir_k3", 24, 2, 1, e3)],
            [("skip", 24, 1, 1)],
            [("ir_k3", 16, -2, 1, e1)],
            # downsampled (x4)
            [("ir_k3", 32, 2, 2, e3)],
            [("skip", 32, 1, 1)],
            [("ir_k3", 24, -2, 1, e2)],
            # downsampled (x8)
            [("ir_k3", 40, 2, 3, e4)],
            [("skip", 40, 1, 1)],
            [("ir_k3", 32, -2, 1, e4)],
            # downsampled (x16)
            [("ir_k3", 48, 1, 6, e6)],
            [("skip", 48, 1, 1)],
            [("ir_k3", 40, 1, 2, e6)],
            # downsampled (x16)
            [("ir_k3", 56, 1, 1, e6)],
            [("skip", 56, 1, 1)],
            [("ir_k3", 48, 1, 2, e6)],
        ],
        "stage_combiners": [
            # original res
            "choose_right",
            # downsampled (x2)
            "add",
            # downsampled (x4)
            "add",
            # downsampled (x8)
            "add",
            # downsampled (x16)
            "add",
        ],
    },
    "xirp14b_rf": {
        # nparams: 0.395872, nflops 25.505856
        "basic_args": BASIC_ARGS,
        "stages": [
            # [op, c, s, n, ...]
            # original res
            [("conv_k3", 16, 2, 1, CONV_PAD_REFLECT)],
            [("skip", 16, 1, 1)],
            [("ir_k3", 1, -2, 1, e1, {"skip_pwl_bn": True}, DW_PAD_REFLECT)],
            # downsampled (x2)
            [("ir_k3", 24, 2, 1, e3, DW_PAD_REFLECT)],
            [("skip", 24, 1, 1)],
            [("ir_k3", 16, -2, 1, e1, DW_PAD_REFLECT)],
            # downsampled (x4)
            [("ir_k3", 32, 2, 2, e3, DW_PAD_REFLECT)],
            [("skip", 32, 1, 1)],
            [("ir_k3", 24, -2, 1, e2, DW_PAD_REFLECT)],
            # downsampled (x8)
            [("ir_k3", 40, 2, 3, e4, DW_PAD_REFLECT)],
            [("skip", 40, 1, 1)],
            [("ir_k3", 32, -2, 1, e4, DW_PAD_REFLECT)],
            # downsampled (x16)
            [("ir_k3", 48, 1, 6, e6, DW_PAD_REFLECT)],
            [("skip", 48, 1, 1)],
            [("ir_k3", 40, 1, 2, e6, DW_PAD_REFLECT)],
            # downsampled (x16)
            [("ir_k3", 56, 1, 1, e6, DW_PAD_REFLECT)],
            [("skip", 56, 1, 1)],
            [("ir_k3", 48, 1, 2, e6, DW_PAD_REFLECT)],
        ],
        "stage_combiners": [
            # original res
            "choose_right",
            # downsampled (x2)
            "add",
            # downsampled (x4)
            "add",
            # downsampled (x8)
            "add",
            # downsampled (x16)
            "add",
        ],
    },
    "xirp14c": {
        # nparams: 0.399704, nflops 28.30752
        "basic_args": BASIC_ARGS,
        "stages": [
            # [op, c, s, n, ...]
            # original res
            [("conv_k3", 16, 2, 1)],
            [("skip", 16, 1, 1)],
            [("ir_k3", 1, -2, 1, e1, {"skip_pwl_bn": True})],
            # downsampled (x2)
            [("ir_k3", 24, 2, 1, e3)],
            [("skip", 24, 1, 1)],
            [("ir_k3", 16, -2, 1, e2)],
            # downsampled (x4)
            [("ir_k3", 32, 2, 2, e3)],
            [("skip", 32, 1, 1)],
            [("ir_k3", 24, -2, 1, e3)],
            # downsampled (x8)
            [("ir_k3", 40, 2, 3, e4)],
            [("skip", 40, 1, 1)],
            [("ir_k3", 32, -2, 1, e4)],
            # downsampled (x16)
            [("ir_k3", 48, 1, 6, e6)],
            [("skip", 48, 1, 1)],
            [("ir_k3", 40, 1, 2, e6)],
            # downsampled (x16)
            [("ir_k3", 56, 1, 1, e6)],
            [("skip", 56, 1, 1)],
            [("ir_k3", 48, 1, 2, e6)],
        ],
        "stage_combiners": [
            # original res
            "choose_right",
            # downsampled (x2)
            "add",
            # downsampled (x4)
            "add",
            # downsampled (x8)
            "add",
            # downsampled (x16)
            "add",
        ],
    },
    "xirp14d": {
        # nparams: 0.529736, nflops 33.130368
        "basic_args": BASIC_ARGS,
        "stages": [
            # [op, c, s, n, ...]
            # original res
            [("conv_k3", 16, 2, 1)],
            [("skip", 16, 1, 1)],
            [("ir_k3", 1, -2, 1, e1, {"skip_pwl_bn": True})],
            # downsampled (x2)
            [("ir_k3", 24, 2, 1, e3)],
            [("skip", 24, 1, 1)],
            [("ir_k3", 16, -2, 1, e2)],
            # downsampled (x4)
            [("ir_k3", 32, 2, 2, e3)],
            [("skip", 32, 1, 1)],
            [("ir_k3", 24, -2, 1, e3)],
            # downsampled (x8)
            [("ir_k3", 48, 2, 3, e4)],
            [("skip", 48, 1, 1)],
            [("ir_k3", 32, -2, 1, e4)],
            # downsampled (x16)
            [("ir_k3", 56, 1, 6, e6)],
            [("skip", 56, 1, 1)],
            [("ir_k3", 48, 1, 2, e6)],
            # downsampled (x16)
            [("ir_k3", 64, 1, 1, e6)],
            [("skip", 64, 1, 1)],
            [("ir_k3", 56, 1, 2, e6)],
        ],
        "stage_combiners": [
            # original res
            "choose_right",
            # downsampled (x2)
            "add",
            # downsampled (x4)
            "add",
            # downsampled (x8)
            "add",
            # downsampled (x16)
            "add",
        ],
    },
    "xirp14d_rf": {
        # nparams: 0.529736, nflops 33.130368
        "basic_args": BASIC_ARGS,
        "stages": [
            # [op, c, s, n, ...]
            # original res
            [("conv_k3", 16, 2, 1, CONV_PAD_REFLECT)],
            [("skip", 16, 1, 1)],
            [("ir_k3", 1, -2, 1, e1, {"skip_pwl_bn": True}, DW_PAD_REFLECT)],
            # downsampled (x2)
            [("ir_k3", 24, 2, 1, e3, DW_PAD_REFLECT)],
            [("skip", 24, 1, 1)],
            [("ir_k3", 16, -2, 1, e2, DW_PAD_REFLECT)],
            # downsampled (x4)
            [("ir_k3", 32, 2, 2, e3, DW_PAD_REFLECT)],
            [("skip", 32, 1, 1)],
            [("ir_k3", 24, -2, 1, e3, DW_PAD_REFLECT)],
            # downsampled (x8)
            [("ir_k3", 48, 2, 3, e4, DW_PAD_REFLECT)],
            [("skip", 48, 1, 1)],
            [("ir_k3", 32, -2, 1, e4, DW_PAD_REFLECT)],
            # downsampled (x16)
            [("ir_k3", 56, 1, 6, e6, DW_PAD_REFLECT)],
            [("skip", 56, 1, 1)],
            [("ir_k3", 48, 1, 2, e6, DW_PAD_REFLECT)],
            # downsampled (x16)
            [("ir_k3", 64, 1, 1, e6, DW_PAD_REFLECT)],
            [("skip", 64, 1, 1)],
            [("ir_k3", 56, 1, 2, e6, DW_PAD_REFLECT)],
        ],
        "stage_combiners": [
            # original res
            "choose_right",
            # downsampled (x2)
            "add",
            # downsampled (x4)
            "add",
            # downsampled (x8)
            "add",
            # downsampled (x16)
            "add",
        ],
    },
    "xirp14e": {
        # nparams: 0.594344, nflops 35.456256
        "basic_args": BASIC_ARGS,
        "stages": [
            # [op, c, s, n, ...]
            # original res
            [("conv_k3", 16, 2, 1)],
            [("skip", 16, 1, 1)],
            [("ir_k3", 1, -2, 1, e1, {"skip_pwl_bn": True})],
            # downsampled (x2)
            [("ir_k3", 24, 2, 1, e3)],
            [("skip", 24, 1, 1)],
            [("ir_k3", 16, -2, 1, e2)],
            # downsampled (x4)
            [("ir_k3", 32, 2, 2, e3)],
            [("skip", 32, 1, 1)],
            [("ir_k3", 24, -2, 1, e3)],
            # downsampled (x8)
            [("ir_k3", 48, 2, 3, e4)],
            [("skip", 48, 1, 1)],
            [("ir_k3", 32, -2, 1, e4)],
            # downsampled (x16)
            [("ir_k3", 64, 1, 6, e6)],
            [("skip", 64, 1, 1)],
            [("ir_k3", 48, 1, 1, e6)],
            # downsampled (x16)
            [("ir_k3", 64, 1, 1, e6)],
            [("skip", 64, 1, 1)],
            [("ir_k3", 64, 1, 2, e6)],
        ],
        "stage_combiners": [
            # original res
            "choose_right",
            # downsampled (x2)
            "add",
            # downsampled (x4)
            "add",
            # downsampled (x8)
            "add",
            # downsampled (x16)
            "add",
        ],
    },
    "xirp15a": {
        # nparams: 0.198912, nflops 34.537536
        "basic_args": BASIC_ARGS,
        "stages": [
            # [op, c, s, n, ...]
            # original res
            [("conv_k3", 8, 1, 1)],
            [("skip", 8, 1, 1)],
            [("conv_k1", 1, 1, 1, CONV_ONLY)],
            # downsampled (x2)
            [("ir_k5", 16, 2, 1, e1), ("ir_k5", 16, 1, 1, e2)],
            [("skip", 16, 1, 1)],
            [("ir_k3", 8, -2, 1, e1)],
            # downsampled (x4)
            [("ir_k5", 24, 2, 1, e3), ("ir_k3", 24, 1, 2, e2)],
            [("skip", 24, 1, 1)],
            [("ir_k3", 16, -2, 1, e2)],
            # downsampled (x8)
            [
                ("ir_k5", 32, 2, 1, e3),
                # ("ir_k3", 32, 1, 3, e2),
                # ("ir_k5", 40, 1, 1, e3),
                ("ir_k5", 40, 1, 3, e2)
            ],
            [("skip", 40, 1, 1)],
            [("ir_k3", 24, -2, 1, e3)],
            # downsampled (x16)
            [("ir_k5", 48, 2, 1, e4), ("ir_k3", 48, 1, 3, e3)],
            [("skip", 48, 1, 1)],
            [("ir_k3", 40, -2, 1, e3)],
            # downsampled (x16)
            [("ir_k5", 56, 1, 1, e4)],
            [("skip", 56, 1, 1)],
            [("ir_k3", 48, 1, 2, e4)],
        ],
    },
    "xirp15b": {
        # nparams: 0.173576, nflops 34.660512
        "basic_args": BASIC_ARGS,
        "stages": [
            # [op, c, s, n, ...]
            # original res
            [("conv_k3", 8, 1, 1)],
            [("skip", 8, 1, 1)],
            [("conv_k1", 1, 1, 1, CONV_ONLY)],
            # downsampled (x2)
            [("ir_k5", 16, 2, 1, e1), ("ir_k5", 16, 1, 1, e2)],
            [("skip", 16, 1, 1)],
            [("ir_k3", 8, -2, 1, e1)],
            # downsampled (x4)
            [("ir_k5", 24, 2, 1, e3), ("ir_k3", 24, 1, 2, e2)],
            [("skip", 24, 1, 1)],
            [("ir_k3", 16, -2, 1, e2)],
            # downsampled (x8)
            [("ir_k5", 32, 2, 1, e3), ("ir_k5", 32, 1, 5, e3)],
            [("skip", 32, 1, 1)],
            [("ir_k3", 24, -2, 1, e3)],
            # downsampled (x16)
            [("ir_k5", 40, 2, 1, e4), ("ir_k3", 40, 1, 3, e3)],
            [("skip", 40, 1, 1)],
            [("ir_k3", 32, -2, 1, e3)],
            # downsampled (x16)
            [("ir_k5", 48, 1, 1, e4)],
            [("skip", 48, 1, 1)],
            [("ir_k3", 40, 1, 2, e4)],
        ],
    },
    "xirp15c": {
        # nparams: 0.397368, nflops 44.654592
        "basic_args": BASIC_ARGS,
        "stages": [
            # [op, c, s, n, ...]
            # original res (x2)
            [("conv_k3", 16, 2, 1), ("ir_k3", 16, 1, 1, e1)],
            [("skip", 16, 1, 1)],
            [("ir_k3", 1, -2, 1, e1, {"skip_pwl_bn": True})],
            # downsampled (x4)
            [("ir_k3", 24, 2, 1, e3), ("ir_k5", 24, 1, 1, e2)],
            [("skip", 24, 1, 1)],
            [("ir_k5", 16, -2, 1, e1)],
            # downsampled (x8)
            [("ir_k5", 32, 2, 1, e3), ("ir_k3", 32, 1, 2, e2)],
            [("skip", 32, 1, 1)],
            [("ir_k3", 24, -2, 1, e2)],
            # downsampled (x16)
            [
                ("ir_k5", 40, 2, 1, e4), ("ir_k3", 40, 1, 3, e3),
                ("ir_k5", 48, 1, 1, e4), ("ir_k3", 48, 1, 5, e3),
            ],
            [("skip", 48, 1, 1)],
            [("ir_k5", 32, -2, 1, e3)],
            # downsampled (x32)
            [("ir_k5", 56, 2, 1, e4), ("ir_k3", 56, 1, 4, e4), ("ir_k5", 64, 1, 1, e4)],
            [("skip", 64, 1, 1)],
            [("ir_k3", 48, -2, 2, e4)],
        ],
        "stage_combiners": [
            # original res
            "choose_right",
            # downsampled (x2)
            "add",
            # downsampled (x4)
            "add",
            # downsampled (x8)
            "add",
        ],
    },
    "xirp15d": {
        # nparams: 0.199968, nflops 33.641472
        "basic_args": BASIC_ARGS,
        "stages": [
            # [op, c, s, n, ...]
            # original res (x2)
            [("conv_k3", 16, 2, 1), ("ir_k3", 16, 1, 1, e1)],
            [("skip", 16, 1, 1)],
            [("ir_k3", 1, -2, 1, e1, {"skip_pwl_bn": True})],
            # downsampled (x4)
            [("ir_k3", 24, 2, 1, e2), ("ir_k5", 24, 1, 1, e2)],
            [("skip", 24, 1, 1)],
            [("ir_k5", 16, -2, 1, e1)],
            # downsampled (x8)
            [("ir_k5", 32, 2, 1, e3), ("ir_k3", 32, 1, 2, e2)],
            [("skip", 32, 1, 1)],
            [("ir_k3", 24, -2, 1, e2)],
            # downsampled (x16)
            [
                ("ir_k5", 40, 2, 1, e4), ("ir_k3", 40, 1, 3, e3),
                # ("ir_k5", 40, 1, 1, e4), ("ir_k3", 40, 1, 5, e3),
            ],
            [("skip", 40, 1, 1)],
            [("ir_k5", 32, -2, 1, e3)],
            # downsampled (x32)
            [("ir_k5", 48, 2, 1, e4), ("ir_k3", 48, 1, 4, e4)],
            [("skip", 48, 1, 1)],
            [("ir_k3", 40, -2, 1, e4)],
        ],
        "stage_combiners": [
            # original res
            "choose_right",
            # downsampled (x2)
            "add",
            # downsampled (x4)
            "add",
            # downsampled (x8)
            "add",
        ],
    },
    "xirp15e": {
        # nparams: 0.199968, nflops 38.5152, res=128
        "basic_args": BASIC_ARGS,
        "stages": [
            # [op, c, s, n, ...]
            # original res (x2)
            [("conv_k3", 16, 2, 1), ("ir_k3", 16, 1, 1, e1)],
            [("skip", 16, 1, 1)],
            [("ir_k3", 1, -2, 1, e1, {"skip_pwl_bn": True})],
            # downsampled (x4)
            [("ir_k3", 24, 2, 1, e2), ("ir_k5", 24, 1, 1, e2)],
            [("skip", 24, 1, 1)],
            [("ir_k5", 16, -2, 1, e1)],
            # downsampled (x8)
            [("ir_k5", 32, 2, 1, e3), ("ir_k3", 32, 1, 2, e2)],
            [("skip", 32, 1, 1)],
            [("ir_k3", 24, -2, 1, e2)],
            # downsampled (x16)
            [
                ("ir_k5", 40, 2, 1, e4), ("ir_k3", 40, 1, 3, e3),
                # ("ir_k5", 40, 1, 1, e4), ("ir_k3", 40, 1, 5, e3),
            ],
            [("skip", 40, 1, 1)],
            [("ir_k5", 32, -2, 1, e3)],
            # downsampled (x32)
            [("ir_k5", 48, 1, 1, e4), ("ir_k3", 48, 1, 4, e4)],
            [("skip", 48, 1, 1)],
            [("ir_k3", 40, 1, 1, e4)],
        ],
        "stage_combiners": [
            # original res
            "choose_right",
            # downsampled (x2)
            "add",
            # downsampled (x4)
            "add",
            # downsampled (x8)
            "add",
        ],
    },
    "xirp15f": {
        # nparams: 0.082624, nflops 31.005184, res=128
        "basic_args": BASIC_ARGS,
        "stages": [
            # [op, c, s, n, ...]
            # original res (x2)
            [("conv_k3", 16, 2, 1), ("ir_k3", 16, 1, 1, e1)],
            [("skip", 16, 1, 1)],
            [("ir_k3", 1, -2, 1, e1, {"skip_pwl_bn": True})],
            # downsampled (x4)
            [("ir_k3", 24, 2, 1, e2), ("ir_k5", 24, 1, 1, e2)],
            [("skip", 24, 1, 1)],
            [("ir_k5", 16, -2, 1, e1)],
            # downsampled (x8)
            [("ir_k5", 32, 2, 1, e3), ("ir_k3", 32, 1, 2, e2)],
            [("skip", 32, 1, 1)],
            [("ir_k3", 24, -2, 1, e2)],
            # downsampled (x16)
            [
                ("ir_k5", 40, 2, 1, e4), ("ir_k3", 40, 1, 3, e3),
                # ("ir_k5", 40, 1, 1, e4), ("ir_k3", 40, 1, 5, e3),
            ],
            [("skip", 40, 1, 1)],
            [("ir_k5", 32, -2, 1, e3)],
        ],
        "stage_combiners": [
            # original res
            "choose_right",
            # downsampled (x2)
            "add",
            # downsampled (x4)
            "add",
        ],
    },
    "xirp15g": {
        # nparams: 0.179672, nflops 24.252416, res=128
        "basic_args": BASIC_ARGS,
        "stages": [
            # [op, c, s, n, ...]
            # original res (x2)
            [("conv_k3", 8, 2, 1), ("ir_k3", 8, 1, 1, e1)],
            [("skip", 8, 1, 1)],
            [("ir_k3", 1, -2, 1, e1, {"skip_pwl_bn": True})],
            # downsampled (x4)
            [("ir_k3", 16, 2, 1, e2), ("ir_k5", 16, 1, 1, e2)],
            [("skip", 16, 1, 1)],
            [("ir_k5", 8, -2, 1, e1)],
            # downsampled (x8)
            [("ir_k5", 24, 2, 1, e3), ("ir_k3", 24, 1, 2, e2)],
            [("skip", 24, 1, 1)],
            [("ir_k3", 16, -2, 1, e2)],
            # downsampled (x16)
            [
                ("ir_k5", 32, 2, 1, e4), ("ir_k3", 32, 1, 3, e3),
                ("ir_k5", 32, 1, 1, e4), ("ir_k3", 32, 1, 5, e3),
            ],
            [("skip", 32, 1, 1)],
            [("ir_k5", 24, -2, 1, e3)],
            # downsampled (x32)
            [("ir_k5", 40, 1, 1, e4), ("ir_k3", 40, 1, 4, e4)],
            [("skip", 40, 1, 1)],
            [("ir_k3", 32, 1, 1, e4)],
        ],
        "stage_combiners": [
            # original res
            "choose_right",
            # downsampled (x2)
            "add",
            # downsampled (x4)
            "add",
            # downsampled (x8)
            "add",
        ],
    },
    "xirp15h": {
        # nparams: 0.179672, nflops 24.252416, res=128
        "basic_args": BASIC_ARGS,
        "stages": [
            # [op, c, s, n, ...]
            # original res (x2)
            [("conv_k3", 8, 2, 1), ("ir_k3", 8, 1, 1, e1)],
            [("skip", 8, 1, 1)],
            [("ir_k3", 1, -2, 1, e1, {"skip_pwl_bn": True})],
            # downsampled (x4)
            [("ir_k3", 16, 2, 1, e2), ("ir_k5", 16, 1, 1, e2)],
            [("skip", 16, 1, 1)],
            [("ir_k5", 8, -2, 1, e1)],
            # downsampled (x8)
            [("ir_k5", 24, 2, 1, e3), ("ir_k3", 24, 1, 2, e2)],
            [("skip", 24, 1, 1)],
            [("ir_k3", 16, -2, 1, e2)],
            # downsampled (x16)
            [
                ("ir_k5", 32, 2, 1, e4), ("ir_k3", 32, 1, 3, e3),
                ("ir_k5", 40, 1, 1, e4), ("ir_k3", 40, 1, 5, e3),
            ],
            [("skip", 32, 1, 1)],
            [("ir_k5", 24, -2, 1, e3)],
        ],
        "stage_combiners": [
            # original res
            "choose_right",
            # downsampled (x2)
            "add",
            # downsampled (x4)
            "add",
        ],
    },
    "xirp15i": {
        # nparams: 0.129352, nflops 33.098752, res=128
        "basic_args": BASIC_ARGS,
        "stages": [
            # [op, c, s, n, ...]
            # original res (x2)
            [("conv_k3", 16, 2, 1), ("ir_k3", 16, 1, 1, e1)],
            [("skip", 16, 1, 1)],
            [("ir_k3", 1, -2, 1, e1, {"skip_pwl_bn": True})],
            # downsampled (x4)
            [("ir_k3", 24, 2, 1, e2), ("ir_k5", 24, 1, 1, e2)],
            [("skip", 24, 1, 1)],
            [("ir_k5", 16, -2, 1, e1)],
            # downsampled (x8)
            [("ir_k5", 32, 2, 1, e3), ("ir_k3", 32, 1, 1, e2)],
            [("skip", 32, 1, 1)],
            [("ir_k3", 24, -2, 1, e2)],
            # downsampled (x16)
            [
                ("ir_k5", 40, 2, 1, e4), ("ir_k3", 40, 1, 1, e3),
                ("ir_k5", 40, 1, 1, e4), ("ir_k3", 40, 1, 1, e3),
            ],
            [("skip", 40, 1, 1)],
            [("ir_k5", 32, -2, 1, e3)],
            # downsampled (x32)
            [("ir_k5", 40, 1, 1, e4), ("ir_k3", 40, 1, 1, e4)],
            [("skip", 40, 1, 1)],
            [("ir_k3", 40, 1, 1, e4)],
        ],
        "stage_combiners": [
            # original res
            "choose_right",
            # downsampled (x2)
            "add",
            # downsampled (x4)
            "add",
            # downsampled (x8)
            "add",
        ],
    },
    "xirp15j": {
        # nparams: 0.056592, nflops 28.442112, res=128
        "basic_args": BASIC_ARGS,
        "stages": [
            # [op, c, s, n, ...]
            # original res (x2)
            [("conv_k3", 16, 2, 1), ("ir_k3", 16, 1, 1, e1)],
            [("skip", 16, 1, 1)],
            [("ir_k3", 1, -2, 1, e1, {"skip_pwl_bn": True})],
            # downsampled (x4)
            [("ir_k3", 24, 2, 1, e2), ("ir_k5", 24, 1, 1, e2)],
            [("skip", 24, 1, 1)],
            [("ir_k5", 16, -2, 1, e1)],
            # downsampled (x8)
            [("ir_k5", 32, 2, 1, e3), ("ir_k3", 32, 1, 1, e2)],
            [("skip", 32, 1, 1)],
            [("ir_k3", 24, -2, 1, e2)],
            # downsampled (x16)
            [
                ("ir_k5", 40, 2, 1, e4), ("ir_k3", 40, 1, 1, e3),
                # ("ir_k5", 40, 1, 1, e4), ("ir_k3", 40, 1, 1, e3),
            ],
            [("skip", 40, 1, 1)],
            [("ir_k5", 32, -2, 1, e3)],
        ],
        "stage_combiners": [
            # original res
            "choose_right",
            # downsampled (x2)
            "add",
            # downsampled (x4)
            "add",
        ],
    },
    "xirp16": {
        # nparams: 0.217328, nflops 25.73328, res=96x160
        # nparams: 0.217328, nflops 61.759872, res=144x256
        "basic_args": BASIC_ARGS1,
        "stages": [
            # [op, c, s, n, ...]
            # original res (x2)
            [("conv_k3", 16, 2, 1)],
            [("skip", 16, 1, 1)],
            [("conv_k3", 1, -2, 1, CONV_ONLY, UPSAMPLE_BILINEAR)],
            # [("ir_k3", 1, -2, 1, e1, {"skip_pwl_bn": True})],
            # downsampled (x4)
            [("ir_k3", 16, 2, 1, e1)],
            [("skip", 16, 1, 1)],
            [("ir_k3", 16, -2, 1, e1)],
            # downsampled (x8)
            [("ir_k3", 24, 2, 1, _ex(4.5)), ("ir_k3", 24, 1, 1, _ex(3.67))],
            [("skip", 24, 1, 1)],
            [("ir_k3", 24, 1, 1, _ex(3.67)), ("ir_k3", 16, -2, 1, _ex(4.5))],
            # downsampled (x16)
            [
                ("ir_k5", 40, 2, 1, e4),
                ("ir_k5", 40, 1, 2, e6),
                ("ir_k5", 48, 1, 2, e3),
            ],
            [
                ("skip", 48, 1, 1)
            ],
            [
                ("ir_k5", 48, 1, 2, e3),
                ("ir_k5", 40, 1, 2, e6),
                ("ir_k5", 24, -2, 1, e4),
            ],
        ],
        "stage_combiners": [
            # downsampled (x2)
            "add",
            # downsampled (x4)
            "add",
            # downsampled (x8)
            "add",
        ],
    },
    "xirp16a": {
        # nparams: 0.41112, nflops 26.141632, res=96x160
        # nparams: 0.41112, nflops 62.468608, res=144x256
        "basic_args": BASIC_ARGS1,
        "stages": [
            # [op, c, s, n, ...]
            # original res (x2)
            [("conv_k3", 16, 2, 1)],
            [("skip", 16, 1, 1)],
            [("conv_k3", 1, -2, 1, CONV_ONLY, UPSAMPLE_BILINEAR)],
            # [("ir_k3", 1, -2, 1, e1, {"skip_pwl_bn": True})],
            # downsampled (x4)
            [("ir_k3", 16, 2, 1, e1, SE_RELU)],
            [("skip", 16, 1, 1)],
            [("ir_k3", 16, -2, 1, e1, SE_RELU)],
            # downsampled (x8)
            [("ir_k3", 24, 2, 1, _ex(4.5)), ("ir_k3", 24, 1, 1, _ex(3.67))],
            [("skip", 24, 1, 1)],
            [("ir_k3", 24, 1, 1, _ex(3.67)), ("ir_k3", 16, -2, 1, _ex(4.5))],
            # downsampled (x16)
            [
                ("ir_k5", 40, 2, 1, e4, SE_RELU),
                ("ir_k5", 40, 1, 2, e6, SE_RELU),
                ("ir_k5", 48, 1, 2, e3, SE_RELU),
            ],
            [
                ("skip", 48, 1, 1)
            ],
            [
                ("ir_k5", 48, 1, 2, e3, SE_RELU),
                ("ir_k5", 40, 1, 2, e6, SE_RELU),
                ("ir_k5", 24, -2, 1, e4, SE_RELU),
            ],
        ],
        "stage_combiners": [
            # downsampled (x2)
            "add",
            # downsampled (x4)
            "add",
            # downsampled (x8)
            "add",
        ],
    },
    "xirp16b": {
        # nparams: 0.41112, nflops 26.141632, res=96x160
        # nparams: 0.41112, nflops 62.468608, res=144x256
        "basic_args": BASIC_ARGS1,
        "stages": [
            # [op, c, s, n, ...]
            # original res (x2)
            [("conv_k3_hs", 16, 2, 1)],
            [("skip", 16, 1, 1)],
            [("conv_k3", 1, -2, 1, CONV_ONLY, UPSAMPLE_BILINEAR)],
            # [("ir_k3", 1, -2, 1, e1, {"skip_pwl_bn": True})],
            # downsampled (x4)
            [("ir_k3", 16, 2, 1, e1, SE_RELU)],
            [("skip", 16, 1, 1)],
            [("ir_k3", 16, -2, 1, e1, SE_RELU)],
            # downsampled (x8)
            [("ir_k3", 24, 2, 1, _ex(4.5)), ("ir_k3", 24, 1, 1, _ex(3.67))],
            [("skip", 24, 1, 1)],
            [("ir_k3", 24, 1, 1, _ex(3.67)), ("ir_k3", 16, -2, 1, _ex(4.5))],
            # downsampled (x16)
            [
                ("ir_k5_hs", 40, 2, 1, e4, SE_RELU),
                ("ir_k5_hs", 40, 1, 2, e6, SE_RELU),
                ("ir_k5_hs", 48, 1, 2, e3, SE_RELU),
            ],
            [
                ("skip", 48, 1, 1)
            ],
            [
                ("ir_k5_hs", 48, 1, 2, e3, SE_RELU),
                ("ir_k5_hs", 40, 1, 2, e6, SE_RELU),
                ("ir_k5_hs", 24, -2, 1, e4, SE_RELU),
            ],
        ],
        "stage_combiners": [
            # downsampled (x2)
            "add",
            # downsampled (x4)
            "add",
            # downsampled (x8)
            "add",
        ],
    },
    "xirp16c": {
        # nparams: 0.228784, nflops 22.8696, res=96x160
        # nparams: 0.228784, nflops 54.88704, res=160x256
        "basic_args": BASIC_ARGS1,
        "stages": [
            # [op, c, s, n, ...]
            # original res (x2)
            [("conv_k3", 16, 2, 1)],
            [("skip", 16, 1, 1)],
            [("conv_k3", 1, -2, 1, CONV_ONLY, UPSAMPLE_BILINEAR)],
            # [("ir_k3", 1, -2, 1, e1, {"skip_pwl_bn": True})],
            # downsampled (x4)
            [("ir_k3", 16, 2, 1, e1)],
            [("skip", 16, 1, 1)],
            [("conv_k3", 16, -2, 1)],
            # downsampled (x8)
            [("ir_k3", 24, 2, 1, _ex(4.5)), ("ir_k3", 24, 1, 1, _ex(3.67))],
            [("skip", 24, 1, 1)],
            [("ir_k3", 16, -2, 1, e1)],
            # downsampled (x16)
            [
                ("ir_k5", 40, 2, 1, e4),
                ("ir_k5", 40, 1, 2, e6),
                ("ir_k5", 48, 1, 2, e3),
            ],
            [
                ("ir_k5", 48, 1, 2, e3),
                ("ir_k5", 40, 1, 2, e6),
                ("ir_k5", 40, 1, 1, e4),
            ],
            [
                ("ir_k3", 24, 1, 1, _ex(3.67)),
                ("ir_k3", 24, -2, 1, _ex(4.5))
            ],
        ],
        "stage_combiners": [
            # downsampled (x2)
            "add",
            # downsampled (x4)
            "add",
            # downsampled (x8)
            "add",
        ],
    },
    "xirp16d": {
        # nparams: 0.42232, nflops 23.187456, res=96x160
        # nparams: 0.42232, nflops 55.378944, res=160x256
        "basic_args": BASIC_ARGS1,
        "stages": [
            # [op, c, s, n, ...]
            # original res (x2)
            [("conv_k3", 16, 2, 1)],
            [("skip", 16, 1, 1)],
            [("conv_k3", 1, -2, 1, CONV_ONLY, UPSAMPLE_BILINEAR)],
            # [("ir_k3", 1, -2, 1, e1, {"skip_pwl_bn": True})],
            # downsampled (x4)
            [("ir_k3", 16, 2, 1, e1, SE_RELU)],
            [("skip", 16, 1, 1)],
            [("conv_k3", 16, -2, 1)],
            # downsampled (x8)
            [("ir_k3", 24, 2, 1, _ex(4.5)), ("ir_k3", 24, 1, 1, _ex(3.67))],
            [("skip", 24, 1, 1)],
            [("ir_k3", 16, -2, 1, e1)],
            # downsampled (x16)
            [
                ("ir_k5", 40, 2, 1, e4, SE_RELU),
                ("ir_k5", 40, 1, 2, e6, SE_RELU),
                ("ir_k5", 48, 1, 2, e3, SE_RELU),
            ],
            [
                ("ir_k5", 48, 1, 2, e3, SE_RELU),
                ("ir_k5", 40, 1, 2, e6, SE_RELU),
                ("ir_k5", 40, 1, 1, e4, SE_RELU),
            ],
            [
                ("ir_k3", 24, 1, 1, _ex(3.67)),
                ("ir_k3", 24, -2, 1, _ex(4.5))
            ],
        ],
        "stage_combiners": [
            # downsampled (x2)
            "add",
            # downsampled (x4)
            "add",
            # downsampled (x8)
            "add",
        ],
    },
    "xirp16e": {
        # nparams: 0.42232, nflops 23.187456, res=96x160
        # nparams: 0.42232, nflops 55.378944, res=160x256
        "basic_args": BASIC_ARGS1,
        "stages": [
            # [op, c, s, n, ...]
            # original res (x2)
            [("conv_k3_hs", 16, 2, 1)],
            [("skip", 16, 1, 1)],
            [("conv_k3", 1, -2, 1, CONV_ONLY, UPSAMPLE_BILINEAR)],
            # [("ir_k3", 1, -2, 1, e1, {"skip_pwl_bn": True})],
            # downsampled (x4)
            [("ir_k3", 16, 2, 1, e1, SE_RELU)],
            [("skip", 16, 1, 1)],
            [("conv_k3_hs", 16, -2, 1)],
            # downsampled (x8)
            [("ir_k3", 24, 2, 1, _ex(4.5)), ("ir_k3", 24, 1, 1, _ex(3.67))],
            [("skip", 24, 1, 1)],
            [("ir_k3", 16, -2, 1, e1)],
            # downsampled (x16)
            [
                ("ir_k5_hs", 40, 2, 1, e4, SE_RELU),
                ("ir_k5_hs", 40, 1, 2, e6, SE_RELU),
                ("ir_k5_hs", 48, 1, 2, e3, SE_RELU),
            ],
            [
                ("ir_k5_hs", 48, 1, 2, e3, SE_RELU),
                ("ir_k5_hs", 40, 1, 2, e6, SE_RELU),
                ("ir_k5_hs", 40, 1, 1, e4, SE_RELU),
            ],
            [
                ("ir_k3", 24, 1, 1, _ex(3.67)),
                ("ir_k3", 24, -2, 1, _ex(4.5))
            ],
        ],
        "stage_combiners": [
            # downsampled (x2)
            "add",
            # downsampled (x4)
            "add",
            # downsampled (x8)
            "add",
        ],
    },
    "xirp17a": {
        # nparams: 0.395872, nflops 85.01952, res=128x240
        "basic_args": BASIC_ARGS1,
        "stages": [
            # [op, c, s, n, ...]
            # downsampled (x2)
            [("conv_k3", 16, 2, 1)],
            [("skip", 16, 1, 1)],
            [("ir_k3", 1, -2, 1, e1, {"skip_pwl_bn": True})],
            # downsampled (x4)
            [("ir_k3", 24, 2, 1, e3)],
            [("skip", 24, 1, 1)],
            [("ir_k3", 16, -2, 1, e1)],
            # downsampled (x8)
            [("ir_k3", 32, 2, 2, e3)],
            [("skip", 32, 1, 1)],
            [("ir_k3", 24, -2, 1, e2)],
            # downsampled (x16)
            [("ir_k3", 40, 2, 3, e4), ("ir_k3", 48, 1, 6, e6), ("ir_k3", 56, 1, 1, e6)],
            [("skip", 56, 1, 1)],
            [("ir_k3", 48, 1, 2, e6), ("ir_k3", 40, 1, 2, e6), ("ir_k3", 32, -2, 1, e4)],
        ],
        "stage_combiners": [
            # original res
            "add",
            # downsampled (x2)
            "add",
            # downsampled (x4)
            "add",
            # downsampled (x8)
        ],
    },
    "xirp17b": {
        # nparams: 0.399704, nflops 28.30752
        "basic_args": BASIC_ARGS1,
        "stages": [
            # [op, c, s, n, ...]
            # original res
            [("conv_k3", 16, 2, 1)],
            [("skip", 16, 1, 1)],
            [("ir_k3", 1, -2, 1, e1, {"skip_pwl_bn": True})],
            # downsampled (x2)
            [("ir_k3", 24, 2, 1, e3)],
            [("skip", 24, 1, 1)],
            [("ir_k3", 16, -2, 1, e2)],
            # downsampled (x4)
            [("ir_k3", 32, 2, 2, e3)],
            [("skip", 32, 1, 1)],
            [("ir_k3", 24, -2, 1, e3)],
            # downsampled (x8)
            [("ir_k3", 40, 2, 3, e4), ("ir_k3", 48, 1, 6, e6), ("ir_k3", 56, 1, 1, e6)],
            [("skip", 56, 1, 1)],
            [("ir_k3", 48, 1, 2, e6), ("ir_k3", 40, 1, 2, e6), ("ir_k3", 32, -2, 1, e4)],
        ],
        "stage_combiners": [
            # original res
            "add",
            # downsampled (x2)
            "add",
            # downsampled (x4)
            "add",
        ],
    },
    "xirp17c": {
        # nparams: 0.594344, nflops 35.456256
        "basic_args": BASIC_ARGS1,
        "stages": [
            # [op, c, s, n, ...]
            # original res
            [("conv_k3", 16, 2, 1)],
            [("skip", 16, 1, 1)],
            [("ir_k3", 1, -2, 1, e1, {"skip_pwl_bn": True})],
            # downsampled (x2)
            [("ir_k3", 24, 2, 1, e3)],
            [("skip", 24, 1, 1)],
            [("ir_k3", 16, -2, 1, e2)],
            # downsampled (x4)
            [("ir_k3", 32, 2, 2, e3)],
            [("skip", 32, 1, 1)],
            [("ir_k3", 24, -2, 1, e3)],
            # downsampled (x8)
            [("ir_k3", 48, 2, 3, e4), ("ir_k3", 64, 1, 6, e6), ("ir_k3", 64, 1, 1, e6)],
            [("skip", 64, 1, 1)],
            [("ir_k3", 64, 1, 2, e6), ("ir_k3", 48, 1, 1, e6), ("ir_k3", 32, -2, 1, e4)],
        ],
        "stage_combiners": [
            # original res
            "add",
            # downsampled (x2)
            "add",
            # downsampled (x4)
            "add",
        ],
    },
    "xirp17d": {
        # nparams: 0.29648, nflops 77.40096, res=128x240
        "basic_args": BASIC_ARGS1,
        "stages": [
            # [op, c, s, n, ...]
            # downsampled (x2)
            [("conv_k3", 16, 2, 1)],
            [("skip", 16, 1, 1)],
            [("ir_k3", 1, -2, 1, e1, {"skip_pwl_bn": True})],
            # downsampled (x4)
            [("ir_k3", 24, 2, 1, e3)],
            [("skip", 24, 1, 1)],
            [("ir_k3", 16, -2, 1, e1)],
            # downsampled (x8)
            [("ir_k3", 32, 2, 2, e3)],
            [("skip", 32, 1, 1)],
            [("ir_k3", 24, -2, 1, e2)],
            # downsampled (x16)
            [("ir_k3", 40, 2, 3, e4), ("ir_k3", 48, 1, 6, e6), ("ir_k3", 56, 1, 1, e6)],
            [("skip", 56, 1, 1)],
            [("ir_k3", 32, -2, 2, e4)],
        ],
        "stage_combiners": [
            # original res
            "add",
            # downsampled (x2)
            "add",
            # downsampled (x4)
            "add",
            # downsampled (x8)
        ],
    },
    "xirp17e": {
        # nparams: 0.300312, nflops 86.73984, res=128x240
        "basic_args": BASIC_ARGS1,
        "stages": [
            # [op, c, s, n, ...]
            # original res
            [("conv_k3", 16, 2, 1)],
            [("skip", 16, 1, 1)],
            [("ir_k3", 1, -2, 1, e1, {"skip_pwl_bn": True})],
            # downsampled (x2)
            [("ir_k3", 24, 2, 1, e3)],
            [("skip", 24, 1, 1)],
            [("ir_k3", 16, -2, 1, e2)],
            # downsampled (x4)
            [("ir_k3", 32, 2, 2, e3)],
            [("skip", 32, 1, 1)],
            [("ir_k3", 24, -2, 1, e3)],
            # downsampled (x8)
            [("ir_k3", 40, 2, 3, e4), ("ir_k3", 48, 1, 6, e6), ("ir_k3", 56, 1, 1, e6)],
            [("skip", 56, 1, 1)],
            [("ir_k3", 32, -2, 2, e4)],
        ],
        "stage_combiners": [
            # original res
            "add",
            # downsampled (x2)
            "add",
            # downsampled (x4)
            "add",
        ],
    },
    "xirp17f": {
        # nparams: 0.4618, nflops 106.59072, res=128x240
        "basic_args": BASIC_ARGS1,
        "stages": [
            # [op, c, s, n, ...]
            # original res
            [("conv_k3", 16, 2, 1)],
            [("skip", 16, 1, 1)],
            [("ir_k3", 1, -2, 1, e1, {"skip_pwl_bn": True})],
            # downsampled (x2)
            [("ir_k3", 24, 2, 1, e3)],
            [("skip", 24, 1, 1)],
            [("ir_k3", 16, -2, 1, e2)],
            # downsampled (x4)
            [("ir_k3", 32, 2, 2, e3)],
            [("skip", 32, 1, 1)],
            [("ir_k3", 24, -2, 1, e3)],
            # downsampled (x8)
            [("ir_k3", 48, 2, 3, e4), ("ir_k3", 64, 1, 6, e6), ("ir_k3", 64, 1, 1, e6)],
            [("skip", 64, 1, 1)],
            [("ir_k3", 32, -2, 2, e4)],
        ],
        "stage_combiners": [
            # original res
            "add",
            # downsampled (x2)
            "add",
            # downsampled (x4)
            "add",
        ],
    },
    "xirp18a": {
        # nparams: 0.199968, nflops 38.5152, res=128
        # nparams: 0.199968, nflops 72.216, res=128x240
        "basic_args": BASIC_ARGS1,
        "stages": [
            # [op, c, s, n, ...]
            # original res (x2)
            [("conv_k3", 16, 2, 1), ("ir_k3", 16, 1, 1, e1)],
            [("skip", 16, 1, 1)],
            [("ir_k3", 1, -2, 1, e1, {"skip_pwl_bn": True})],
            # downsampled (x4)
            [("ir_k3", 24, 2, 1, e2), ("ir_k5", 24, 1, 1, e2)],
            [("skip", 24, 1, 1)],
            [("ir_k5", 16, -2, 1, e1)],
            # downsampled (x8)
            [("ir_k5", 32, 2, 1, e3), ("ir_k3", 32, 1, 2, e2)],
            [("skip", 32, 1, 1)],
            [("ir_k3", 24, -2, 1, e2)],
            # downsampled (x16)
            [
                ("ir_k5", 40, 2, 1, e4), ("ir_k3", 40, 1, 3, e3),
                # ("ir_k5", 40, 1, 1, e4), ("ir_k3", 40, 1, 5, e3),
            ],
            [("skip", 40, 1, 1)],
            [("ir_k5", 32, -2, 1, e3)],
            # downsampled (x32)
            [("ir_k5", 48, 1, 1, e4), ("ir_k3", 48, 1, 4, e4)],
            [("skip", 48, 1, 1)],
            [("ir_k3", 40, 1, 1, e4)],
        ],
        "stage_combiners": [
            # original res
            "add",
            # downsampled (x2)
            "add",
            # downsampled (x4)
            "add",
            # downsampled (x8)
            "add",
        ],
    },
    "xirp18b": {
        # nparams: 0.082624, nflops 31.005184, res=128
        # nparams: 0.082624, nflops 58.13472, res=128x240
        "basic_args": BASIC_ARGS1,
        "stages": [
            # [op, c, s, n, ...]
            # original res (x2)
            [("conv_k3", 16, 2, 1), ("ir_k3", 16, 1, 1, e1)],
            [("skip", 16, 1, 1)],
            [("ir_k3", 1, -2, 1, e1, {"skip_pwl_bn": True})],
            # downsampled (x4)
            [("ir_k3", 24, 2, 1, e2), ("ir_k5", 24, 1, 1, e2)],
            [("skip", 24, 1, 1)],
            [("ir_k5", 16, -2, 1, e1)],
            # downsampled (x8)
            [("ir_k5", 32, 2, 1, e3), ("ir_k3", 32, 1, 2, e2)],
            [("skip", 32, 1, 1)],
            [("ir_k3", 24, -2, 1, e2)],
            # downsampled (x16)
            [
                ("ir_k5", 40, 2, 1, e4), ("ir_k3", 40, 1, 3, e3),
                # ("ir_k5", 40, 1, 1, e4), ("ir_k3", 40, 1, 5, e3),
            ],
            [("skip", 40, 1, 1)],
            [("ir_k5", 32, -2, 1, e3)],
        ],
        "stage_combiners": [
            # original res
            "add",
            # downsampled (x2)
            "add",
            # downsampled (x4)
            "add",
        ],
    },
    "xirp18c": {
        # nparams: 0.152824, nflops 66.55872, res=128x240
        "basic_args": BASIC_ARGS1,
        "stages": [
            # [op, c, s, n, ...]
            # original res (x2)
            [("conv_k3", 16, 2, 1), ("ir_k3", 16, 1, 1, e1)],
            [("skip", 16, 1, 1)],
            [("ir_k3", 1, -2, 1, e1, {"skip_pwl_bn": True})],
            # downsampled (x4)
            [("ir_k3", 24, 2, 1, e2), ("ir_k5", 24, 1, 1, e2)],
            [("skip", 24, 1, 1)],
            [("ir_k5", 16, -2, 1, e1)],
            # downsampled (x8)
            [("ir_k5", 32, 2, 1, e3), ("ir_k3", 32, 1, 2, e2)],
            [("skip", 32, 1, 1)],
            [("ir_k3", 24, -2, 1, e2)],
            # downsampled (x16)
            [
                ("ir_k5", 40, 2, 1, e4), ("ir_k3", 40, 1, 3, e3),
                ("ir_k5", 40, 1, 1, e4), ("ir_k3", 40, 1, 5, e3),
            ],
            [("skip", 40, 1, 1)],
            [("ir_k5", 32, -2, 1, e3)],
        ],
        "stage_combiners": [
            # original res
            "add",
            # downsampled (x2)
            "add",
            # downsampled (x4)
            "add",
        ],
    },
    "xirp18d": {
        # nparams: 0.32956, nflops 76.11744, res=128x240
        "basic_args": BASIC_ARGS1,
        "stages": [
            # [op, c, s, n, ...]
            # original res (x2)
            [("conv_k3", 8, 2, 1), ("ir_k5", 8, 1, 1, e1)],
            [("skip", 8, 1, 1)],
            [("ir_k3", 1, -2, 1, e1, {"skip_pwl_bn": True})],
            # downsampled (x4)
            [("ir_k5", 16, 2, 1, e4), ("ir_k5", 16, 1, 1, e3)],
            [("skip", 16, 1, 1)],
            [("ir_k5", 8, -2, 1, e1)],
            # downsampled (x8)
            [("ir_k5", 24, 2, 1, e4), ("ir_k5", 24, 1, 1, e3)],
            [("skip", 24, 1, 1)],
            [("ir_k3", 16, -2, 1, e2)],
            # downsampled (x16)
            [
                ("ir_k5", 48, 2, 1, e4), ("ir_k3", 40, 1, 1, e3),
                ("ir_k5", 64, 1, 1, e3), ("ir_k3", 64, 1, 1, e2),
                ("ir_k5", 128, 1, 1, e3), ("ir_k3", 160, 1, 1, e3),
            ],
            [("skip", 160, 1, 1)],
            [("ir_k5", 24, -2, 1, e3)],
        ],
        "stage_combiners": [
            # original res
            "add",
            # downsampled (x2)
            "add",
            # downsampled (x4)
            "add",
        ],
    },
    "xirp18e": {
        # nparams: 0.095224, nflops 42.9168, res=128x240
        "basic_args": BASIC_ARGS1,
        "stages": [
            # [op, c, s, n, ...]
            # original res (x2)
            [("conv_k3", 8, 2, 1), ("ir_k5", 8, 1, 1, e1)],
            [("skip", 8, 1, 1)],
            [("ir_k3", 1, -2, 1, e1, {"skip_pwl_bn": True})],
            # downsampled (x4)
            [("ir_k5", 16, 2, 1, e4), ("ir_k5", 16, 1, 1, e3)],
            [("skip", 16, 1, 1)],
            [("ir_k5", 8, -2, 1, e1)],
            # downsampled (x8)
            [("ir_k5", 24, 2, 1, e4), ("ir_k5", 24, 1, 1, e3)],
            [("skip", 24, 1, 1)],
            [("ir_k3", 16, -2, 1, e2)],
            # downsampled (x16)
            [
                ("ir_k5", 48, 2, 1, e4), ("ir_k3", 40, 1, 1, e3),
                ("ir_k5", 64, 1, 1, e3), ("ir_k3", 64, 1, 1, e2),
                # ("ir_k5", 128, 1, 1, e3), ("ir_k3", 160, 1, 1, e3),
            ],
            [("skip", 64, 1, 1)],
            [("ir_k5", 24, -2, 1, e3)],
        ],
        "stage_combiners": [
            # original res
            "add",
            # downsampled (x2)
            "add",
            # downsampled (x4)
            "add",
        ],
    },
    "xirp19a": {
        # nparams: 0.273296, nflops 127.81536, res=128x240
        "basic_args": BASIC_ARGS1,
        "stages": [
            # [op, c, s, n, ...]
            # original res (x2)
            [("conv_k3", 16, 2, 1), ("ir_k3", 16, 1, 2, e1)],
            [("skip", 16, 1, 1)],
            [("ir_k3", 1, -2, 1, e1, {"skip_pwl_bn": True})],
            # downsampled (x4)
            [("ir_k5", 24, 2, 1, e4), ("ir_k5", 24, 1, 3, e2)],
            [("skip", 24, 1, 1)],
            [("ir_k5", 16, -2, 1, e1)],
            # downsampled (x8)
            [("ir_k5", 40, 2, 1, e5), ("ir_k5", 40, 1, 4, e3)],
            [("skip", 40, 1, 1)],
            [("ir_k3", 24, -2, 1, e2)],
            # downsampled (x16)
            [
                ("ir_k5", 72, 2, 1, e5), ("ir_k3", 72, 1, 4, e3),
                # ("ir_k3", 120, 1, 1, e5), ("ir_k5", 120, 1, 5, e3),
            ],
            [("skip", 72, 1, 1)],
            [("ir_k5", 40, -2, 1, e3)],
            # # downsampled (x32)
            # [("ir_k3", 184, 2, 1, e6), ("ir_k5", 184, 1, 5, e4), ("ir_k5", 224, 1, 1, e6)],
            # [("skip", 224, 1, 1)],
            # [("ir_k3", 120, 1, 1, e4)],
        ],
        "stage_combiners": [
            # original res
            "add",
            # downsampled (x2)
            "add",
            # downsampled (x4)
            "add",
            # # downsampled (x8)
            # "add",
        ],
    },
    "xirpl1": {
        # nparams: 8.345952, nflops 2123.283456
        "basic_args": BASIC_ARGS,
        "stages": [
            # [op, c, s, n, ...]
            # original res
            [("conv_k3", 32, 1, 1, e1)],
            [("skip", 32, 1, 1)],
            [("conv_k3", 1, 1, 1, e1, CONV_ONLY)],
            # downsampled (x2)
            [("ir_k3", 64, 2, 3, e6)],
            [("skip", 64, 1, 1)],
            [("ir_k3", 32, -2, 3, e6)],
            # downsampled (x4)
            [("ir_k3", 96, 2, 3, e6)],
            [("skip", 96, 1, 1)],
            [("ir_k3", 64, -2, 3, e6)],
            # downsampled (x8)
            [("ir_k3", 128, 2, 4, e6)],
            [("skip", 128, 1, 1)],
            [("ir_k3", 96, -2, 4, e6)],
            # downsampled (x16)
            [("ir_k3", 160, 2, 5, e6)],
            [("skip", 160, 1, 1)],
            [("ir_k3", 128, -2, 5, e6)],
            # downsampled (x16)
            [("ir_k3", 160, 1, 6, e6)],
            [("skip", 160, 1, 1)],
            [("ir_k3", 160, 1, 6, e6)],
        ],
    },
    "xirpl1a": {
        # nparams: 8.345952, nflops 2123.283456
        "basic_args": BASIC_ARGS,
        "stages": [
            # [op, c, s, n, ...]
            # original res
            [("conv_k3", 32, 1, 1, e1)],
            [("skip", 32, 1, 1)],
            [("conv_k3", 1, 1, 1, e1, CONV_ONLY)],
            # downsampled (x2)
            [("ir_k3", 64, 2, 3, e6)],
            [("skip", 64, 1, 1)],
            [("ir_k3", 32, -2, 3, e6)],
            # downsampled (x4)
            [("ir_k3", 96, 2, 3, e6)],
            [("skip", 96, 1, 1)],
            [("ir_k3", 64, -2, 3, e6)],
            # downsampled (x8)
            [("ir_k3", 128, 2, 4, e6)],
            [("skip", 128, 1, 1)],
            [("ir_k3", 96, -2, 4, e6)],
            # downsampled (x16)
            [("ir_k3", 160, 2, 5, e6)],
            [("skip", 160, 1, 1)],
            [("ir_k3", 128, -2, 5, e6)],
            # downsampled (x16)
            [("ir_k3", 224, 2, 6, e6)],
            [("skip", 224, 1, 1)],
            [("ir_k3", 160, -2, 6, e6)],
        ],
    },
    "xirpl2": {
        # nparams: 20.49264, nflops 2151.714816
        "basic_args": BASIC_ARGS,
        "stages": [
            # [op, c, s, n, ...]
            # original res
            [("conv_k3", 32, 1, 1, e1)],
            [("skip", 32, 1, 1)],
            [("conv_k3", 1, 1, 1, e1, CONV_ONLY)],
            # downsampled (x2)
            [("ir_k3_se", 64, 2, 3, e6)],
            [("skip", 64, 1, 1)],
            [("ir_k3_se", 32, -2, 3, e6)],
            # downsampled (x4)
            [("ir_k3_se", 96, 2, 3, e6)],
            [("skip", 96, 1, 1)],
            [("ir_k3_se", 64, -2, 3, e6)],
            # downsampled (x8)
            [("ir_k3_se", 128, 2, 4, e6)],
            [("skip", 128, 1, 1)],
            [("ir_k3_se", 96, -2, 4, e6)],
            # downsampled (x16)
            [("ir_k3_se", 160, 2, 5, e6)],
            [("skip", 160, 1, 1)],
            [("ir_k3_se", 128, -2, 5, e6)],
            # downsampled (x16)
            [("ir_k3_se", 160, 1, 6, e6)],
            [("skip", 160, 1, 1)],
            [("ir_k3_se", 160, 1, 6, e6)],
        ],
    },
    "xirpl2a": {
        # nparams: 20.49264, nflops 2151.714816
        "basic_args": BASIC_ARGS,
        "stages": [
            # [op, c, s, n, ...]
            # original res
            [("conv_k3", 32, 1, 1, e1)],
            [("skip", 32, 1, 1)],
            [("conv_k3", 1, 1, 1, e1, CONV_ONLY)],
            # downsampled (x2)
            [("ir_k3_se", 64, 2, 3, e6)],
            [("skip", 64, 1, 1)],
            [("ir_k3_se", 32, -2, 3, e6)],
            # downsampled (x4)
            [("ir_k3_se", 96, 2, 3, e6)],
            [("skip", 96, 1, 1)],
            [("ir_k3_se", 64, -2, 3, e6)],
            # downsampled (x8)
            [("ir_k3_se", 128, 2, 4, e6)],
            [("skip", 128, 1, 1)],
            [("ir_k3_se", 96, -2, 4, e6)],
            # downsampled (x16)
            [("ir_k3_se", 160, 2, 5, e6)],
            [("skip", 160, 1, 1)],
            [("ir_k3_se", 128, -2, 5, e6)],
            # downsampled (x16)
            [("ir_k3_se", 224, 2, 6, e6)],
            [("skip", 224, 1, 1)],
            [("ir_k3_se", 160, -2, 6, e6)],
        ],
    },
    "xirpl3": {
        # nparams: 11.845888, nflops 1781.40288, res=128x240
        "basic_args": BASIC_ARGS1,
        "stages": [
            # [op, c, s, n, ...]
            # original res (x2)
            [("conv_k3", 32, 2, 1), ("ir_k3", 24, 1, 3, e1)],
            [("skip", 24, 1, 1)],
            [("ir_k3", 1, -2, 1, e3, {"skip_pwl_bn": True})],
            # downsampled (x4)
            [("ir_k5", 40, 2, 1, e4), ("ir_k5", 40, 1, 4, e2)],
            [("skip", 40, 1, 1)],
            [("ir_k5", 24, -2, 4, e1)],
            # downsampled (x8)
            [("ir_k5", 56, 2, 1, e4), ("ir_k5", 56, 1, 4, e3)],
            [("skip", 56, 1, 1)],
            [("ir_k3", 40, -2, 5, e2)],
            # downsampled (x16)
            [
                ("ir_k5", 104, 2, 1, e5), ("ir_k3", 104, 1, 4, e3),
                ("ir_k3", 160, 1, 1, e5), ("ir_k5", 160, 1, 8, e3),
            ],
            [("skip", 160, 1, 1)],
            [("ir_k5", 56, -2, 5, e3)],
            # downsampled (x32)
            [("ir_k3", 264, 1, 1, e6), ("ir_k5", 264, 1, 6, e5), ("ir_k5", 288, 1, 2, e6)],
            [("skip", 288, 1, 1)],
            [("ir_k3", 160, 1, 1, e6), ("ir_k3", 160, 1, 10, e4)],
        ],
        "stage_combiners": [
            # original res
            "add",
            # downsampled (x2)
            "add",
            # downsampled (x4)
            "add",
            # downsampled (x8)
            "add",
        ],
    },
    "xirpl4": {
        # nparams: 11.845888, nflops 1524.16384, res=160x256
        "basic_args": BASIC_ARGS1,
        "stages": [
            # [op, c, s, n, ...]
            # original res (x2)
            [("conv_k3", 32, 2, 1), ("ir_k3", 24, 1, 3, e1)],
            [("skip", 24, 1, 1)],
            [("ir_k3", 1, -2, 1, e3, {"skip_pwl_bn": True})],
            # downsampled (x4)
            [("ir_k5", 40, 2, 1, e4), ("ir_k5", 40, 1, 4, e2)],
            [("skip", 40, 1, 1)],
            [("ir_k5", 24, -2, 4, e1)],
            # downsampled (x8)
            [("ir_k5", 56, 2, 1, e4), ("ir_k5", 56, 1, 4, e3)],
            [("skip", 56, 1, 1)],
            [("ir_k3", 40, -2, 5, e2)],
            # downsampled (x16)
            [
                ("ir_k5", 104, 2, 1, e5), ("ir_k3", 104, 1, 4, e3),
                ("ir_k3", 160, 1, 1, e5), ("ir_k5", 160, 1, 8, e3),
            ],
            [("skip", 160, 1, 1)],
            [("ir_k5", 56, -2, 5, e3)],
            # downsampled (x32)
            [("ir_k3", 264, 2, 1, e6), ("ir_k5", 264, 1, 6, e5), ("ir_k5", 288, 1, 2, e6)],
            [("skip", 288, 1, 1)],
            [("ir_k3", 160, -2, 1, e6), ("ir_k3", 160, 1, 10, e4)],
        ],
        "stage_combiners": [
            # original res
            "add",
            # downsampled (x2)
            "add",
            # downsampled (x4)
            "add",
            # downsampled (x8)
            "add",
        ],
    },
}

FBNetV2ModelArch.add_archs(MODEL_ARCH_PERSON_SEGMENTATION)


def _change_out_channels(arch_def, out_channels):
    arch_def = copy.deepcopy(arch_def)
    c_idx = 1  # eg. ('conv_k3', c, s, n, ...)
    output_stage = arch_def["stages"][2]
    stage_list = list(output_stage[-1])
    stage_list[c_idx] = out_channels
    output_stage[-1] = tuple(stage_list)
    assert arch_def["stages"][2][-1][1] == out_channels
    return arch_def


MODEL_ARCH_MCS = {
    "ps_unet_mcs3": _change_out_channels(MODEL_ARCH_PERSON_SEGMENTATION["ps_unet"], 3),
    "xirp7i_mcs3": _change_out_channels(MODEL_ARCH_PERSON_SEGMENTATION["xirp7i"], 3),
    "xirp14b_mcs3": _change_out_channels(MODEL_ARCH_PERSON_SEGMENTATION["xirp14b"], 3),
    "xirp14b_mcs7": _change_out_channels(MODEL_ARCH_PERSON_SEGMENTATION["xirp14b"], 7),
}


FBNetV2ModelArch.add_archs(MODEL_ARCH_MCS)


XIRP7I_STAGES = [
    # [op, c, s, n, ...]
    # original res
    [("conv_k3", 8, 1, 1)],
    [("skip", 8, 1, 1)],
    [("conv_k1", 1, 1, 1, CONV_ONLY)],
    # downsampled (x2)
    [("ir_k3", 16, 2, 1, e3)],
    [("skip", 16, 1, 1)],
    [("ir_k3", 8, -2, 1, e1)],
    # downsampled (x4)
    [("ir_k3", 24, 2, 2, e3)],
    [("skip", 24, 1, 1)],
    [("ir_k3", 16, -2, 1, e2)],
    # downsampled (x8)
    [("ir_k3", 32, 2, 3, e4)],
    [("skip", 32, 1, 1)],
    [("ir_k3", 24, -2, 1, e4)],
    # downsampled (x16)
    [("ir_k3", 40, 2, 6, e6)],
    [("skip", 40, 1, 1)],
    [("ir_k3", 32, -2, 1, e6)],
    # downsampled (x16)
    [("ir_k3", 48, 1, 1, e6)],
    [("skip", 48, 1, 1)],
    [("ir_k3", 40, 1, 2, e6)],
]


MODEL_ARCH_PERSON_SEGMENTATION_CONNECT = {
    "xirp9": {
        # nparams: 0.257712, nflops 33.248448
        "basic_args": BASIC_ARGS,
        "stages": XIRP7I_STAGES,
        "stage_combiners": [
            # original res
            "choose_right",
            # downsampled (x2)
            "choose_right",
            # downsampled (x4)
            "choose_right",
            # downsampled (x8)
            "choose_right",
            # downsampled (x16)
            "choose_right",
        ],
    },
    "xirp9a": {
        # nparams: 0.257712, nflops 33.248448
        "basic_args": BASIC_ARGS,
        "stages": XIRP7I_STAGES,
        "stage_combiners": [
            # original res
            "add",
            # downsampled (x2)
            "choose_right",
            # downsampled (x4)
            "add",
            # downsampled (x8)
            "choose_right",
            # downsampled (x16)
            "choose_right",
        ],
    },
    "xirp9b": {
        # nparams: 0.257712, nflops 33.248448
        "basic_args": BASIC_ARGS,
        "stages": XIRP7I_STAGES,
        "stage_combiners": [
            # original res
            "choose_right",
            # downsampled (x2)
            "add",
            # downsampled (x4)
            "choose_right",
            # downsampled (x8)
            "add",
            # downsampled (x16)
            "choose_right",
        ],
    },
    "xirp9c": {
        # nparams: 0.257712, nflops 33.248448
        "basic_args": BASIC_ARGS,
        "stages": XIRP7I_STAGES,
        "stage_combiners": [
            # original res
            "choose_right",
            # downsampled (x2)
            "choose_right",
            # downsampled (x4)
            "add",
            # downsampled (x8)
            "choose_right",
            # downsampled (x16)
            "choose_right",
        ],
    },
    "xirp9d": {
        # nparams: 0.257792, nflops 33.985728
        "basic_args": BASIC_ARGS,
        "stages": [
            # [op, c, s, n, ...]
            # original res
            [("conv_k3", 8, 1, 1)],
            [("skip", 8, 1, 1)],
            [("conv_k1", 1, 1, 1, CONV_ONLY, {"in_channels": 16})],
            # downsampled (x2)
            [("ir_k3", 16, 2, 1, e3)],
            [("skip", 16, 1, 1)],
            [("ir_k3", 8, -2, 1, e1)],
            # downsampled (x4)
            [("ir_k3", 24, 2, 2, e3)],
            [("skip", 24, 1, 1)],
            [("ir_k3", 16, -2, 1, e2)],
            # downsampled (x8)
            [("ir_k3", 32, 2, 3, e4)],
            [("skip", 32, 1, 1)],
            [("ir_k3", 24, -2, 1, e4)],
            # downsampled (x16)
            [("ir_k3", 40, 2, 6, e6)],
            [("skip", 40, 1, 1)],
            [("ir_k3", 32, -2, 1, e6)],
            # downsampled (x16)
            [("ir_k3", 48, 1, 1, e6)],
            [("skip", 48, 1, 1)],
            [("ir_k3", 40, 1, 2, e6)],
        ],
        "stage_combiners": [
            # original res
            "concat",
            # downsampled (x2)
            "add",
            # downsampled (x4)
            "add",
            # downsampled (x8)
            "add",
            # downsampled (x16)
            "add",
        ],
    },
    "xirp9e": {
        # nparams: 0.258064, nflops 36.49248
        "basic_args": BASIC_ARGS,
        "stages": [
            # [op, c, s, n, ...]
            # original res
            [("conv_k3", 8, 1, 1)],
            [("skip", 8, 1, 1)],
            [("conv_k1", 1, 1, 1, CONV_ONLY, {"in_channels": 16})],
            # downsampled (x2)
            [("ir_k3", 16, 2, 1, e3)],
            [("skip", 16, 1, 1)],
            [("ir_k3", 8, -2, 1, e1, {"in_channels": 32})],
            # downsampled (x4)
            [("ir_k3", 24, 2, 2, e3)],
            [("skip", 24, 1, 1)],
            [("ir_k3", 16, -2, 1, e2)],
            # downsampled (x8)
            [("ir_k3", 32, 2, 3, e4)],
            [("skip", 32, 1, 1)],
            [("ir_k3", 24, -2, 1, e4)],
            # downsampled (x16)
            [("ir_k3", 40, 2, 6, e6)],
            [("skip", 40, 1, 1)],
            [("ir_k3", 32, -2, 1, e6)],
            # downsampled (x16)
            [("ir_k3", 48, 1, 1, e6)],
            [("skip", 48, 1, 1)],
            [("ir_k3", 40, 1, 2, e6)],
        ],
        "stage_combiners": [
            # original res
            "concat",
            # downsampled (x2)
            "concat",
            # downsampled (x4)
            "add",
            # downsampled (x8)
            "add",
            # downsampled (x16)
            "add",
        ],
    },
    "xirp10": {
        # nparams: 0.257712, nflops 33.248448
        "basic_args": BASIC_ARGS,
        "stages": [
            # [op, c, s, n, ...]
            # original res
            [("conv_k3", 8, 1, 1)],
            [("conv_k1", 1, 1, 1)],
            [("conv_k1", 1, 1, 1, CONV_ONLY)],
            # downsampled (x2)
            [("ir_k3", 16, 2, 1, e3)],
            [("conv_k1", 16, 1, 1)],
            [("ir_k3", 8, -2, 1, e1)],
            # downsampled (x4)
            [("ir_k3", 24, 2, 2, e3)],
            [("conv_k1", 24, 1, 1)],
            [("ir_k3", 16, -2, 1, e2)],
            # downsampled (x8)
            [("ir_k3", 32, 2, 3, e4)],
            [("conv_k1", 32, 1, 1)],
            [("ir_k3", 24, -2, 1, e4)],
            # downsampled (x16)
            [("ir_k3", 40, 2, 6, e6)],
            [("conv_k1", 40, 1, 1)],
            [("ir_k3", 32, -2, 1, e6)],
            # downsampled (x16)
            [("ir_k3", 48, 1, 1, e6)],
            [("skip", 48, 1, 1)],
            [("ir_k3", 40, 1, 2, e6)],
        ],
        "stage_combiners": [
            # original res
            "mul",
            # downsampled (x2)
            "add",
            # downsampled (x4)
            "add",
            # downsampled (x8)
            "add",
            # downsampled (x16)
            "add",
        ],
    },
    "xirp11a": {
        # nparams: 0.159696, nflops 25.343712
        "basic_args": BASIC_ARGS,
        "stages": [
            # [op, c, s, n, ...]
            # original res
            [("conv_k3", 8, 1, 1)],
            [("skip", 8, 1, 1)],
            [("conv_k1", 1, 1, 1, CONV_ONLY)],
            # downsampled (x2)
            [("gb_k3_r2", 16, 2, 1, e3)],
            [("skip", 16, 1, 1)],
            [("gb_k3_r2", 8, -2, 1, e1)],
            # downsampled (x4)
            [("gb_k3_r2", 24, 2, 2, e3)],
            [("skip", 24, 1, 1)],
            [("gb_k3_r2", 16, -2, 1, e2)],
            # downsampled (x8)
            [("gb_k3_r2", 32, 2, 3, e4)],
            [("skip", 32, 1, 1)],
            [("gb_k3_r2", 24, -2, 1, e4)],
            # downsampled (x16)
            [("gb_k3_r2", 40, 2, 6, e6)],
            [("skip", 40, 1, 1)],
            [("gb_k3_r2", 32, -2, 1, e6)],
            # downsampled (x16)
            [("gb_k3_r2", 48, 1, 1, e6)],
            [("skip", 48, 1, 1)],
            [("gb_k3_r2", 40, 1, 2, e6)],
        ],
    },
}

FBNetV2ModelArch.add_archs(MODEL_ARCH_PERSON_SEGMENTATION_CONNECT)
