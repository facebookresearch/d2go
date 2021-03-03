#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved


import d2go.data.transforms.crop as tfm_crop
import d2go.data.transforms.tensor as tfm_tensor
import detectron2.data.transforms as transforms
import torch
from detectron2.data.transforms.augmentation import AugmentationList
from torch import nn


class ImagePooler(nn.Module):
    """Get a subset of image
    Returns the transforms that could be used to inverse the image/boxes/keypoints
    as well.
    Only available for inference. The code is not tracable/scriptable.
    """

    def __init__(
        self,
        resize_type="resize_shortest",
        resize_short=None,
        resize_max=None,
        box_scale_factor=1.0,
    ):
        super().__init__()

        assert resize_type in ["resize_shortest", "resize", "None", None]

        resizer = None
        if resize_type == "resize_shortest":
            resizer = transforms.ResizeShortestEdge(resize_short, resize_max)
        elif resize_type == "resize":
            resizer = transforms.Resize(resize_short)

        self.aug = [
            tfm_tensor.Tensor2Array(),
            tfm_crop.CropBoxAug(box_scale_factor=box_scale_factor),
            *([resizer] if resizer else []),
            tfm_tensor.Array2Tensor(),
        ]

    def forward(self, x: torch.Tensor, box: torch.Tensor):
        """box: 1 x 4 tensor in XYXY format"""
        assert not self.training
        assert isinstance(x, torch.Tensor)
        assert isinstance(box, torch.Tensor)
        # box: 1 x 4 in xyxy format
        inputs = tfm_tensor.AugInput(image=x.cpu(), boxes=box.cpu())
        transforms = AugmentationList(self.aug)(inputs)
        return (
            inputs.image.to(x.device),
            torch.Tensor(inputs.boxes).to(box.device),
            transforms,
        )
