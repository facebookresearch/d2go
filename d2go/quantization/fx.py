#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved


from typing import Tuple

import torch
from mobile_cv.common.misc.oss_utils import fb_overwritable


TORCH_VERSION: Tuple[int, ...] = tuple(int(x) for x in torch.__version__.split(".")[:2])
if TORCH_VERSION > (1, 10):
    from torch.ao.quantization.quantize_fx import convert_fx, prepare_fx, prepare_qat_fx
else:
    from torch.quantization.quantize_fx import convert_fx, prepare_fx, prepare_qat_fx


@fb_overwritable()
def get_prepare_fx_fn(cfg, is_qat):
    return prepare_qat_fx if is_qat else prepare_fx


@fb_overwritable()
def get_convert_fx_fn(cfg, example_inputs):
    return convert_fx
