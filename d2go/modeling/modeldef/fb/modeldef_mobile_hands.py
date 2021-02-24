#!/usr/bin/env python3

from d2go.modeling.modeldef.fbnet_modeldef_registry import FBNetV2ModelArch
from mobile_cv.arch.fbnet_v2.modeldef_utils import e1, e4, e6


BASIC_ARGS = {
    # "relu_args": "swish",
    "width_divisor": 8,
    "bn_args": "bn",
}

PS_UNET_MAXPOOL = {"kernel_size": 2, "padding": 0}
CONV_PAD_REFLECT = {"padding_mode": "reflect"}
DW_PAD_REFLECT = {"dw_args": {**CONV_PAD_REFLECT}}


_XAPL9_TRUNK = [
    [["conv_k3", 8, 2, 1], ["ir_k3", 8, 1, 1, e1]],
    [["ir_k3", 10, 2, 1, e4]],
    [["ir_k3", 24, 2, 2, e6]],
    [["ir_k3", 40, 2, 2, e6], ["ir_k3", 64, 1, 2, e6]]
]

MODEL_ARCH_MOBILE_HAND_TRACKING = {
    # Caffe2 "xapl9" legacy arch def, augmented version with no downsampling in kpts and bbox.
    # Latency around 40ms across devices
    "xapl9_kps1_bbs1": {
        "trunk": _XAPL9_TRUNK,
        "kpts": [
            [["ir_k3", 112, 1, 3, e6], ["ir_k3", 160, 1, 1, e4]]
        ],
        "bbox": [
            [["ir_k3", 112, 1, 3, e6], ["ir_k3", 160, 1, 1, e4]]
        ],
        "rpn": [
            [["ir_k3", 64, 1, 1, e6]]
        ],
        "basic_args": BASIC_ARGS,
    },
    # Caffe2 "xapl9" legacy arch def, original version with (kpts + bbox) downsampled x2
    # Latency around 20ms across devices
    "xapl9_kps2_bbs2": {
        "trunk": _XAPL9_TRUNK,
        "kpts": [
            [["ir_k3", 112, 2, 3, e6], ["ir_k3", 160, 1, 1, e4]]
        ],
        "bbox": [
            [["ir_k3", 112, 2, 3, e6], ["ir_k3", 160, 1, 1, e4]]
        ],
        "rpn": [
            [["ir_k3", 64, 1, 1, e6]]
        ],
        "basic_args": BASIC_ARGS,
    },
    # Reduce keypoints head to minimum, keep for API back-compat
    "xapl9_kpminimal_bbs2": {
        "trunk": _XAPL9_TRUNK,
        "kpts": [
            [["conv_k3", 1, 20, 1]]
        ],
        "bbox": [
            [["ir_k3", 112, 2, 3, e6], ["ir_k3", 160, 1, 1, e4]]
        ],
        "rpn": [
            [["ir_k3", 64, 1, 1, e6]]
        ],
        "basic_args": BASIC_ARGS,
    },
    # Reduce keypoints head to minimum, keep for API back-compat. Bbox higher-res
    "xapl9_kpminimal_bbs1": {
        "trunk": _XAPL9_TRUNK,
        "kpts": [
            [["ir_k3", 1, 20, 1]]
        ],
        "bbox": [
            [["ir_k3", 112, 1, 3, e6], ["ir_k3", 160, 1, 1, e4]]
        ],
        "rpn": [
            [["ir_k3", 64, 1, 1, e6]]
        ],
        "basic_args": BASIC_ARGS,
    },
    "xapl9_nokp_bbs2": {
        "trunk": _XAPL9_TRUNK,
        "bbox": [
            [["ir_k3", 112, 2, 3, e6], ["ir_k3", 160, 1, 1, e4]]
        ],
        "rpn": [
            [["ir_k3", 64, 1, 1, e6]]
        ],
        "basic_args": BASIC_ARGS,
    },
    "xapl9_nokp_bbs1": {
        "trunk": _XAPL9_TRUNK,
        "bbox": [
            [["ir_k3", 112, 1, 3, e6], ["ir_k3", 160, 1, 1, e4]]
        ],
        "rpn": [
            [["ir_k3", 64, 1, 1, e6]]
        ],
        "basic_args": BASIC_ARGS,
    }
}
FBNetV2ModelArch.add_archs(MODEL_ARCH_MOBILE_HAND_TRACKING)
