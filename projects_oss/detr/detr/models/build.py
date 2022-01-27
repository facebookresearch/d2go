import numpy as np
import torch
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

    def forward(self, images):
        features = self.backbone(images.tensor)
        # one tensor per feature level. Each tensor has shape (B, maxH, maxW)
        masks = self.mask_out_padding(
            [features_per_level.shape for features_per_level in features.values()],
            images.image_sizes,
            images.tensor.device,
        )
        assert len(features) == len(masks)
        for i, k in enumerate(features.keys()):
            features[k] = NestedTensor(features[k], masks[i])
        return features

    def mask_out_padding(self, feature_shapes, image_sizes, device):
        masks = []
        assert len(feature_shapes) == len(self.feature_strides)
        for idx, shape in enumerate(feature_shapes):
            N, _, H, W = shape
            masks_per_feature_level = torch.ones(
                (N, H, W), dtype=torch.bool, device=device
            )
            for img_idx, (h, w) in enumerate(image_sizes):
                masks_per_feature_level[
                    img_idx,
                    : int(np.ceil(float(h) / self.feature_strides[idx])),
                    : int(np.ceil(float(w) / self.feature_strides[idx])),
                ] = 0
            masks.append(masks_per_feature_level)
        return masks


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

    def forward(self, images):
        features = self.backbone(images.tensor)
        masks = self.mask_out_padding(
            [features_per_level.shape for features_per_level in features.values()],
            images.image_sizes,
            images.tensor.device,
        )
        assert len(features) == len(masks)
        ret_features = {}
        for i, k in enumerate(features.keys()):
            if k in self.out_features:
                ret_features[k] = NestedTensor(features[k], masks[i])
        return ret_features


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

    def forward(self, images):
        y = self.backbone(images.tensor)
        masks = self.mask_out_padding(
            [y.shape],
            images.image_sizes,
            images.tensor.device,
        )
        assert len(masks) == 1
        ret_features = {}
        ret_features[self.out_features[0]] = NestedTensor(y, masks[0])
        return ret_features
