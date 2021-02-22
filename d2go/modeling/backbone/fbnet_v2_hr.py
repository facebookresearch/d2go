#!/usr/bin/env python3

import logging

from detectron2.layers import ShapeSpec
from detectron2.modeling import BACKBONE_REGISTRY, Backbone
from mobile_cv.arch.fbnet_v2 import fbnet_builder as mbuilder
from mobile_cv.arch.fbnet_v2.fbnet_hr import FBNetHRBuilder
from mobile_cv.arch.utils.quantize_utils import QuantizableModule

from .fbnet_v2 import _get_builder_norm_args, _parse_arch_def

logger = logging.getLogger(__name__)


def build_fbnet_hr(cfg, name, in_channels, *args, **kwargs):
    """
    Similar to build_fbnet
    """
    fbnet_hr_builder = FBNetHRBuilder(
        mbuilder.FBNetBuilder(
            width_ratio=cfg.MODEL.FBNET_V2.SCALE_FACTOR,
            width_divisor=cfg.MODEL.FBNET_V2.WIDTH_DIVISOR,
            bn_args=_get_builder_norm_args(cfg),
        )
    )
    arch_def = _parse_arch_def(cfg)
    model = fbnet_hr_builder.build_model(arch_def, in_channels, *args, **kwargs)
    size_divisibility = fbnet_hr_builder.get_size_divisibility()

    # NOTE: Although FBNetHR has many blocks, but treat it as a single stage.
    # TODO: channels and stride for final output can't be obtained from builder.
    shape_spec_per_stage = [ShapeSpec(channels=arch_def["stages"][2][-1][1], stride=1)]
    return model, shape_spec_per_stage, size_divisibility


@BACKBONE_REGISTRY.register()
class FBNetV2HRBackbone(QuantizableModule, Backbone):
    def __init__(self, cfg, input_shape):
        super(FBNetV2HRBackbone, self).__init__(cfg.QUANTIZATION.EAGER_MODE)
        self.body, shape_specs, size_divisibility = build_fbnet_hr(
            cfg, name=None, in_channels=input_shape.channels
        )
        self._out_features = ["sem_seg_logits"]
        self._out_feature_strides = {"sem_seg_logits": shape_specs[-1].stride}
        self._out_feature_channels = {"sem_seg_logits": shape_specs[-1].channels}
        self._size_divisibility = size_divisibility

    @property
    def size_divisibility(self):
        return self._size_divisibility

    def forward(self, x):
        sem_seg_logits = self.body(x)
        return {"sem_seg_logits": sem_seg_logits}
