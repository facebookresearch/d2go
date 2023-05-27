#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved


# misc.py
# modules that are used in different places but are not a specific type (e.g., backbone)

from typing import Any, Callable, Optional

import torch
import torch.nn as nn


class SplitAndConcat(nn.Module):
    """Split the data from split_dim and concatenate in concat_dim.

    @param split_dim from which axis the data will be chunk
    @param concat_dim to which axis the data will be concatenated
    @param chunk size of the data to be chunk/concatenated
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


class AddCoordChannels(nn.Module):
    """Appends coordinate location values to the channel dimension.

    @param with_r include radial distance from centroid as additional channel (default: False)
    """

    def __init__(self, with_r: bool = False) -> None:
        super().__init__()
        self.with_r = with_r

    def forward(self, input_tensor):
        batch_size_shape, channel_in_shape, dim_y, dim_x = input_tensor.shape
        device = input_tensor.device

        xx_ones = torch.ones([1, 1, 1, dim_x], dtype=torch.int32)
        yy_ones = torch.ones([1, 1, 1, dim_y], dtype=torch.int32)

        xx_range = torch.arange(dim_y, dtype=torch.int32)
        yy_range = torch.arange(dim_x, dtype=torch.int32)
        xx_range = xx_range[None, None, :, None]
        yy_range = yy_range[None, None, :, None]

        xx_channel = torch.matmul(xx_range, xx_ones)
        yy_channel = torch.matmul(yy_range, yy_ones)

        # transpose y
        yy_channel = yy_channel.permute(0, 1, 3, 2)

        xx_channel = xx_channel.float() / (dim_y - 1)
        yy_channel = yy_channel.float() / (dim_x - 1)

        xx_channel = xx_channel * 2 - 1
        yy_channel = yy_channel * 2 - 1

        xx_channel = xx_channel.repeat(batch_size_shape, 1, 1, 1)
        yy_channel = yy_channel.repeat(batch_size_shape, 1, 1, 1)

        out = torch.cat(
            [input_tensor, xx_channel.to(device), yy_channel.to(device)], dim=1
        )

        if self.with_r:
            rr = torch.sqrt(
                torch.pow(xx_channel - 0.5, 2) + torch.pow(yy_channel - 0.5, 2)
            )
            out = torch.cat([out, rr], dim=1)

        return out


def inplace_delegate(
    self,
    api_name: str,
    sub_module_name: str,
    setter_fn: Optional[Callable],
    *args,
    **kwargs,
) -> Any:
    """Helper function to delegate API calls to its submodule"""

    sub_module = getattr(self, sub_module_name)
    api_name = f"delegate_{api_name}"
    if hasattr(sub_module, api_name):
        func = getattr(sub_module, api_name)
        orig_ret = func(*args, **kwargs)
        if setter_fn is None:
            # Assume the return of `func` will replace the submodule
            setattr(self, sub_module_name, orig_ret)
            return self
        else:
            return setter_fn(self, sub_module_name, orig_ret)
    else:
        raise RuntimeError(
            f"It seems the {sub_module_name} doesn't implement {api_name},"
            " quantization might fail."
        )
