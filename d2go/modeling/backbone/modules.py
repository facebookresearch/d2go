#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

from typing import List

import torch
import torch.nn as nn
from detectron2 import layers
from detectron2.utils.tracing import is_fx_tracing
from mobile_cv.arch.fbnet_v2.irf_block import IRFBlock


class RPNHeadConvRegressor(nn.Module):
    """
    A simple RPN Head for classification and bbox regression
    """

    def __init__(self, in_channels, num_anchors, box_dim=4):
        """
        Arguments:
            in_channels (int): number of channels of the input feature
            num_anchors (int): number of anchors to be predicted
            box_dim (int): dimension of bbox
        """
        super(RPNHeadConvRegressor, self).__init__()
        self.cls_logits = nn.Conv2d(in_channels, num_anchors, kernel_size=1, stride=1)
        self.bbox_pred = nn.Conv2d(
            in_channels, num_anchors * box_dim, kernel_size=1, stride=1
        )

        for l in [self.cls_logits, self.bbox_pred]:
            torch.nn.init.normal_(l.weight, std=0.01)
            torch.nn.init.constant_(l.bias, 0)

    def forward(self, x: List[torch.Tensor]):
        if not is_fx_tracing():
            torch._assert(isinstance(x, (list, tuple)), "Unexpected data type")
        logits = [self.cls_logits(y) for y in x]
        bbox_reg = [self.bbox_pred(y) for y in x]

        return logits, bbox_reg


class MaskRCNNConv1x1Predictor(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(MaskRCNNConv1x1Predictor, self).__init__()
        num_classes = out_channels
        num_inputs = in_channels

        self.mask_fcn_logits = nn.Conv2d(num_inputs, num_classes, 1, 1, 0)

        for name, param in self.named_parameters():
            if "bias" in name:
                nn.init.constant_(param, 0)
            elif "weight" in name:
                # Caffe2 implementation uses MSRAFill, which in fact
                # corresponds to kaiming_normal_ in PyTorch
                nn.init.kaiming_normal_(param, mode="fan_out", nonlinearity="relu")

    def forward(self, x):
        return self.mask_fcn_logits(x)


class KeypointRCNNPredictor(nn.Module):
    def __init__(self, in_channels, num_keypoints):
        super(KeypointRCNNPredictor, self).__init__()
        input_features = in_channels
        deconv_kernel = 4
        self.kps_score_lowres = nn.ConvTranspose2d(
            input_features,
            num_keypoints,
            deconv_kernel,
            stride=2,
            padding=deconv_kernel // 2 - 1,
        )
        nn.init.kaiming_normal_(
            self.kps_score_lowres.weight, mode="fan_out", nonlinearity="relu"
        )
        nn.init.constant_(self.kps_score_lowres.bias, 0)
        self.up_scale = 2
        self.out_channels = num_keypoints

    def forward(self, x):
        x = self.kps_score_lowres(x)
        x = layers.interpolate(
            x, scale_factor=self.up_scale, mode="bilinear", align_corners=False
        )
        return x


class KeypointRCNNPredictorNoUpscale(nn.Module):
    def __init__(self, in_channels, num_keypoints):
        super(KeypointRCNNPredictorNoUpscale, self).__init__()
        input_features = in_channels
        deconv_kernel = 4
        self.kps_score_lowres = nn.ConvTranspose2d(
            input_features,
            num_keypoints,
            deconv_kernel,
            stride=2,
            padding=deconv_kernel // 2 - 1,
        )
        nn.init.kaiming_normal_(
            self.kps_score_lowres.weight, mode="fan_out", nonlinearity="relu"
        )
        nn.init.constant_(self.kps_score_lowres.bias, 0)
        self.out_channels = num_keypoints

    def forward(self, x):
        x = self.kps_score_lowres(x)
        return x


class KeypointRCNNIRFPredictorNoUpscale(nn.Module):
    def __init__(self, cfg, in_channels, num_keypoints):
        super(KeypointRCNNIRFPredictorNoUpscale, self).__init__()
        input_features = in_channels

        self.kps_score_lowres = IRFBlock(
            input_features,
            num_keypoints,
            stride=-2,
            expansion=3,
            bn_args="none",
            dw_skip_bnrelu=True,
        )
        self.out_channels = num_keypoints

    def forward(self, x):
        x = self.kps_score_lowres(x)
        return x


class KeypointRCNNConvUpsamplePredictorNoUpscale(nn.Module):
    def __init__(self, cfg, in_channels, num_keypoints):
        super(KeypointRCNNConvUpsamplePredictorNoUpscale, self).__init__()
        input_features = in_channels

        self.kps_score_lowres = nn.Conv2d(
            input_features,
            num_keypoints,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        self.out_channels = num_keypoints

    def forward(self, x):
        x = layers.interpolate(x, scale_factor=(2, 2), mode="nearest")
        x = self.kps_score_lowres(x)
        return x
