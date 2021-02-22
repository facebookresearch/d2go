#!/usr/bin/env python3

# misc.py
# modules that are used in different places but are not a specific type (e.g., backbone)

import torch
import torch.nn as nn


class SplitAndConcat(nn.Module):
    """Split the data from split_dim and concatenate in concat_dim.

    @param split_dim from which axis the data will be chunk
    @param concat_dim to which axis the data will be concatenated
    @param chunk size of the data to be chunk/concatenated

    copied: oculus/face/social_eye/lib/model/resnet_backbone.py
    """

    def __init__(self, split_dim: int = 1, concat_dim: int = 0, chunk: int = 2):
        super(SplitAndConcat, self).__init__()
        self.split_dim = split_dim
        self.concat_dim = concat_dim
        self.chunk = chunk

    def forward(self, x):
        x = torch.chunk(x, self.chunk, dim=self.split_dim)
        x = torch.cat(x, dim=self.concat_dim)
        return x

    def extra_repr(self):
        return (
            f"split_dim={self.split_dim}, concat_dim={self.concat_dim}, "
            f"chunk={self.chunk}"
        )
