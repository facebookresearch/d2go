from typing import Dict

import numpy as np
import torch
import torch.nn.functional as F
from detectron2.modeling import build_backbone
from detectron2.utils.registry import Registry
from detr.models.backbone import Joiner
from detr.models.position_encoding import PositionEmbeddingSine
from detr.util.misc import NestedTensor
from torch import nn

DETR_MODEL_REGISTRY = Registry("DETR_MODEL")


def build_detr_backbone(cfg):
    if "resnet" in cfg.MODEL.BACKBONE.NAME.lower():
        d2_backbone = ResNetMaskedBackbone(cfg)
    elif "fbnet" in cfg.MODEL.BACKBONE.NAME.lower():
        d2_backbone = FBNetMaskedBackbone(cfg)
    elif cfg.MODEL.BACKBONE.SIMPLE:
        d2_backbone = SimpleSingleStageBackbone(cfg)
    else:
        raise NotImplementedError

    N_steps = cfg.MODEL.DETR.HIDDEN_DIM // 2
    centered_position_encoding = cfg.MODEL.DETR.CENTERED_POSITION_ENCODIND

    backbone = Joiner(
        d2_backbone,
        PositionEmbeddingSine(
            N_steps, normalize=True, centered=centered_position_encoding
        ),
    )
    backbone.num_channels = d2_backbone.num_channels
    return backbone


def build_detr_model(cfg):
    name = cfg.MODEL.DETR.NAME
    return DETR_MODEL_REGISTRY.get(name)(cfg)


class ResNetMaskedBackbone(nn.Module):
    """This is a thin wrapper around D2's backbone to provide padding masking"""

    def __init__(self, cfg):
        super().__init__()
        self.backbone = build_backbone(cfg)
        backbone_shape = self.backbone.output_shape()
        if cfg.MODEL.DETR.NUM_FEATURE_LEVELS > 1:
            self.strides = [8, 16, 32]
        else:
            self.strides = [32]

        if cfg.MODEL.RESNETS.RES5_DILATION == 2:
            # fix dilation from d2
            self.backbone.stages[-1][0].conv2.dilation = (1, 1)
            self.backbone.stages[-1][0].conv2.padding = (1, 1)
            self.strides[-1] = self.strides[-1] // 2

        self.feature_strides = [backbone_shape[f].stride for f in backbone_shape.keys()]
        self.num_channels = [backbone_shape[k].channels for k in backbone_shape.keys()]

    def forward(self, tensor_list: NestedTensor):
        xs = self.backbone(tensor_list.tensors)
        out: Dict[str, NestedTensor] = {}
        for name, x in xs.items():
            m = tensor_list.mask
            assert m is not None
            mask = F.interpolate(m[None].float(), size=x.shape[-2:]).to(torch.bool)[0]
            out[name] = NestedTensor(x, mask)
        return out


class FBNetMaskedBackbone(ResNetMaskedBackbone):
    """This is a thin wrapper around D2's backbone to provide padding masking"""

    def __init__(self, cfg):
        nn.Module.__init__(self)
        self.backbone = build_backbone(cfg)
        self.out_features = cfg.MODEL.FBNET_V2.OUT_FEATURES
        self.feature_strides = list(self.backbone._out_feature_strides.values())
        self.num_channels = [
            self.backbone._out_feature_channels[k] for k in self.out_features
        ]
        self.strides = [
            self.backbone._out_feature_strides[k] for k in self.out_features
        ]


class SimpleSingleStageBackbone(ResNetMaskedBackbone):
    """This is a simple wrapper for single stage backbone,
    please set the required configs:
    cfg.MODEL.BACKBONE.SIMPLE == True,
    cfg.MODEL.BACKBONE.STRIDE, cfg.MODEL.BACKBONE.CHANNEL
    """

    def __init__(self, cfg):
        nn.Module.__init__(self)
        self.backbone = build_backbone(cfg)
        self.out_features = ["out"]
        assert cfg.MODEL.BACKBONE.SIMPLE is True
        self.feature_strides = [cfg.MODEL.BACKBONE.STRIDE]
        self.num_channels = [cfg.MODEL.BACKBONE.CHANNEL]
        self.strides = [cfg.MODEL.BACKBONE.STRIDE]
