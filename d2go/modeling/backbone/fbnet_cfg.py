#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved


from d2go.config import CfgNode as CN


def add_fbnet_default_configs(_C):
    """FBNet options and default values"""
    _C.MODEL.FBNET = CN()
    _C.MODEL.FBNET.ARCH = "default"
    # custom arch
    _C.MODEL.FBNET.ARCH_DEF = ""
    _C.MODEL.FBNET.BN_TYPE = "bn"
    _C.MODEL.FBNET.NUM_GROUPS = 32  # for gn usage only
    _C.MODEL.FBNET.SCALE_FACTOR = 1.0
    # the output channels will be divisible by WIDTH_DIVISOR
    _C.MODEL.FBNET.WIDTH_DIVISOR = 1
    _C.MODEL.FBNET.DW_CONV_SKIP_BN = True
    _C.MODEL.FBNET.DW_CONV_SKIP_RELU = True

    # > 0 scale, == 0 skip, < 0 same dimension
    _C.MODEL.FBNET.DET_HEAD_LAST_SCALE = 1.0
    _C.MODEL.FBNET.DET_HEAD_BLOCKS = []
    # overwrite the stride for the head, 0 to use original value
    _C.MODEL.FBNET.DET_HEAD_STRIDE = 0

    # > 0 scale, == 0 skip, < 0 same dimension
    _C.MODEL.FBNET.KPTS_HEAD_LAST_SCALE = 0.0
    _C.MODEL.FBNET.KPTS_HEAD_BLOCKS = []
    # overwrite the stride for the head, 0 to use original value
    _C.MODEL.FBNET.KPTS_HEAD_STRIDE = 0

    # > 0 scale, == 0 skip, < 0 same dimension
    _C.MODEL.FBNET.MASK_HEAD_LAST_SCALE = 0.0
    _C.MODEL.FBNET.MASK_HEAD_BLOCKS = []
    # overwrite the stride for the head, 0 to use original value
    _C.MODEL.FBNET.MASK_HEAD_STRIDE = 0

    # 0 to use all blocks defined in arch_def
    _C.MODEL.FBNET.RPN_HEAD_BLOCKS = 0
    _C.MODEL.FBNET.RPN_BN_TYPE = ""

    # number of channels input to trunk
    _C.MODEL.FBNET.STEM_IN_CHANNELS = 3


def add_fbnet_v2_default_configs(_C):
    _C.MODEL.FBNET_V2 = CN()

    _C.MODEL.FBNET_V2.ARCH = "default"
    _C.MODEL.FBNET_V2.ARCH_DEF = []
    # number of channels input to trunk
    _C.MODEL.FBNET_V2.STEM_IN_CHANNELS = 3
    _C.MODEL.FBNET_V2.SCALE_FACTOR = 1.0
    # the output channels will be divisible by WIDTH_DIVISOR
    _C.MODEL.FBNET_V2.WIDTH_DIVISOR = 1

    # normalization configs
    # name of norm such as "bn", "sync_bn", "gn"
    _C.MODEL.FBNET_V2.NORM = "bn"
    # for advanced use case that requries extra arguments, passing a list of
    # dict such as [{"num_groups": 8}, {"momentum": 0.1}] (merged in given order).
    # Note that string written it in .yaml will be evaluated by yacs, thus this
    # node will become normal python object.
    # https://github.com/rbgirshick/yacs/blob/master/yacs/config.py#L410
    _C.MODEL.FBNET_V2.NORM_ARGS = []

    _C.MODEL.VT_FPN = CN()

    _C.MODEL.VT_FPN.IN_FEATURES = ["res2", "res3", "res4", "res5"]
    _C.MODEL.VT_FPN.OUT_CHANNELS = 256
    _C.MODEL.VT_FPN.LAYERS = 3
    _C.MODEL.VT_FPN.TOKEN_LS = [16, 16, 8, 8]
    _C.MODEL.VT_FPN.TOKEN_C = 1024
    _C.MODEL.VT_FPN.HEADS = 16
    _C.MODEL.VT_FPN.MIN_GROUP_PLANES = 64
    _C.MODEL.VT_FPN.NORM = "BN"
    _C.MODEL.VT_FPN.POS_HWS = []
    _C.MODEL.VT_FPN.POS_N_DOWNSAMPLE = []


def add_bifpn_default_configs(_C):
    _C.MODEL.BIFPN = CN()

    _C.MODEL.BIFPN.DEPTH_MULTIPLIER = 1
    _C.MODEL.BIFPN.SCALE_FACTOR = 1
    _C.MODEL.BIFPN.WIDTH_DIVISOR = 8
    _C.MODEL.BIFPN.NORM = "bn"
    _C.MODEL.BIFPN.NORM_ARGS = []
    _C.MODEL.BIFPN.TOP_BLOCK_BEFORE_FPN = False
