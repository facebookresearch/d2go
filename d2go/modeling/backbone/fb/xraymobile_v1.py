#!/usr/bin/env python3

from detectron2.layers import Conv2d, ConvTranspose2d, ShapeSpec, cat, interpolate
from detectron2.modeling import BACKBONE_REGISTRY, Backbone
from detectron2.modeling.roi_heads import box_head, keypoint_head
from d2go.config import CfgNode
from torch import nn


def add_xraymobile_v1_default_configs(_C):
    _C.MODEL.XRAYMOBILE_V1 = CfgNode()
    _C.MODEL.XRAYMOBILE_V1.SCALE_CHANNELS = 1


def conv_block(in_channels, out_channels, kernel_size, stride=1, padding=None):
    """ Conv-BN-ReLU block """
    if padding is None:
        padding = kernel_size // 2
    conv = Conv2d(
        in_channels,
        out_channels,
        kernel_size,
        stride=stride,
        padding=padding,
        bias=False,
    )
    bn = nn.BatchNorm2d(out_channels)
    relu = nn.ReLU()
    return nn.Sequential(conv, bn, relu)


def init_parameters(module):
    if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
        nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
        if module.bias is not None:
            nn.init.constant_(module.bias, 0)
    elif isinstance(module, nn.BatchNorm2d):
        nn.init.constant_(module.weight, 1)
        if module.bias is not None:
            nn.init.constant_(module.bias, 0)


class Inception(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.channels = channels
        self.conv1x1 = conv_block(channels[0], channels[1], 1)
        self.conv3x3 = nn.Sequential(
            conv_block(channels[0], channels[2], 1),
            conv_block(channels[2], channels[3], 3),
        )
        self.conv5x5 = nn.Sequential(
            conv_block(channels[0], channels[4], 1),
            conv_block(channels[4], channels[5], 5),
        )
        self.apply(init_parameters)
        self.out_channels = channels[1] + channels[3] + channels[5]

    def forward(self, x):
        return cat([self.conv1x1(x), self.conv3x3(x), self.conv5x5(x)], dim=1)


@BACKBONE_REGISTRY.register()
class XRayMobileV1(Backbone):
    channels = {
        "stem": [32, 32, 32, 72],
        "inception_3a": [72, 48, 32, 64, 24, 48],
        "inception_4a": [160, 64, 48, 96, 32, 64],
        "inception_4b": [224, 64, 48, 96, 32, 64],
    }

    def __init__(self, cfg, input_shape: ShapeSpec):
        super().__init__()
        self.cfg = cfg
        self.input_shape = input_shape

        s = cfg.MODEL.XRAYMOBILE_V1.SCALE_CHANNELS
        self.stem = self.build_stem(input_shape.channels, s * self.channels["stem"])
        self.inception_3a = Inception(s * self.channels["inception_3a"])
        self.inception_4a = Inception(s * self.channels["inception_4a"])
        self.inception_4b = Inception(s * self.channels["inception_4b"])
        self.pool = nn.MaxPool2d(3, padding=1, stride=2)
        self.apply(init_parameters)

        self._out_features = ["inception_4b"]
        self._out_feature_channels = {"inception_4b": self.inception_4b.out_channels}
        self._out_feature_strides = {"inception_4b": 16}

        self.out_channels = self.inception_4b.out_channels

    def forward(self, x):
        x = self.stem(x)
        x = self.inception_3a(x)
        x = self.pool(x)
        x = self.inception_4a(x)
        x = self.inception_4b(x)
        return {"inception_4b": x}

    def build_stem(self, in_channels, channels):
        return nn.Sequential(
            conv_block(in_channels, channels[0], 3, stride=2),
            conv_block(channels[0], channels[1], 3, stride=2),
            conv_block(channels[1], channels[2], 1, stride=1),
            conv_block(channels[2], channels[3], 3, stride=2),
        )


@box_head.ROI_BOX_HEAD_REGISTRY.register()
class XRayMobileV1RoIBoxHead(nn.Module):
    channels = [224, 64, 48, 96, 32, 64]

    def __init__(self, cfg, input_shape):
        super().__init__()
        self.cfg = cfg
        s = cfg.MODEL.XRAYMOBILE_V1.SCALE_CHANNELS
        self.roi_box_conv = Inception(s * self.channels)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.output_shape = ShapeSpec(channels=self.roi_box_conv.out_channels)

    def forward(self, x):
        x = self.roi_box_conv(x)
        x = self.avgpool(x)
        return x


@keypoint_head.ROI_KEYPOINT_HEAD_REGISTRY.register()
class XRayMobileV1RoIKeypointHead(keypoint_head.BaseKeypointRCNNHead):
    channels = [224, 64, 48, 96, 32, 64]

    def __init__(self, cfg, input_shape):
        super().__init__(cfg, input_shape)
        self.cfg = cfg
        s = cfg.MODEL.XRAYMOBILE_V1.SCALE_CHANNELS
        self.roi_keypoint_head = Inception(s * self.channels)
        self.kps_deconv = ConvTranspose2d(input_shape.channels, 256, 2, stride=2)
        self.kps_score_lowres = ConvTranspose2d(256, self.num_keypoints, 2, stride=2)
        self.kps_conv = Conv2d(
            self.num_keypoints, self.num_keypoints, 3, stride=1, padding=1
        )

        self.apply(init_parameters)

    def layers(self, x):
        x = self.roi_keypoint_head(x)
        x = self.kps_deconv(x)
        x = self.kps_score_lowres(x)
        x = interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        x = self.kps_conv(x)
        return x
