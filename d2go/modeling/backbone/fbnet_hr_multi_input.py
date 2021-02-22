#!/usr/bin/env python3

from d2go.modeling.backbone.fbnet_v2 import _get_builder_norm_args, _parse_arch_def
from detectron2.layers import ShapeSpec
from mobile_cv.arch.fbnet_v2 import fbnet_builder as mbuilder
from mobile_cv.arch.fbnet_v2.fbnet_hr_add_input import FBNetHRMultiInputBuilder


def build_fbnet_hr_multi_input(cfg, name, in_channels, *args, **kwargs):
    """
    Similar to build_fbnet but support to add input at the decoder stage
    The additional inputs are defined in arch_def['additional_inputs'], which is a dict
    The key of the dict denotes where to add additional input and the value is the number
        of channels to add.
    For example, arch_def['additional_inputs'] = {5: 1} denotes an additional one-channel
        input is concatenated channel-wise to the input of 5-th operator of stage 2 (decoder
        stage)
    """
    fbnet_hr_builder = FBNetHRMultiInputBuilder(
        mbuilder.FBNetBuilder(
            width_ratio=cfg.MODEL.FBNET_V2.SCALE_FACTOR,
            width_divisor=cfg.MODEL.FBNET_V2.WIDTH_DIVISOR,
            bn_args=_get_builder_norm_args(cfg),
        )
    )
    arch_def = _parse_arch_def(cfg)
    model = fbnet_hr_builder.build_model(
        arch_def, in_channels, *args, **kwargs
    )
    size_divisibility = fbnet_hr_builder.get_size_divisibility()
    shape_spec_per_stage = [ShapeSpec(channels=arch_def["stages"][2][-1][1], stride=1)]
    return arch_def, model, shape_spec_per_stage, size_divisibility
