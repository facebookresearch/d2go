#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------------------------------
# Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------------------------------
# Modified from https://github.com/chengdazhi/Deformable-Convolution-V2-PyTorch/tree/pytorch_1.0.0
# ------------------------------------------------------------------------------------------------

import io
import unittest
from functools import wraps

import torch
from detr.functions.ms_deform_attn_func import (
    ms_deform_attn_core_pytorch,
    MSDeformAttnFunction,
)
from torch.autograd import gradcheck

USE_CUDA = torch.cuda.device_count() > 0


N, M, D = 1, 2, 2
Lq, L, P = 2, 2, 2
if USE_CUDA:
    shapes = torch.as_tensor([(6, 4), (3, 2)], dtype=torch.long).cuda()
    level_start_index = torch.cat(
        (shapes.new_zeros((1,)), shapes.prod(1).cumsum(0)[:-1])
    )
    S = sum([(H * W).item() for H, W in shapes])

torch.manual_seed(3)


class Tester(unittest.TestCase):
    @unittest.skipIf(not USE_CUDA, "CI does not have gpu")
    @torch.no_grad()
    def test_forward_equal_with_pytorch_double(self):
        value = torch.rand(N, S, M, D).cuda() * 0.01
        sampling_locations = torch.rand(N, Lq, M, L, P, 2).cuda()
        attention_weights = torch.rand(N, Lq, M, L, P).cuda() + 1e-5
        attention_weights /= attention_weights.sum(-1, keepdim=True).sum(
            -2, keepdim=True
        )
        im2col_step = 2
        output_pytorch = (
            ms_deform_attn_core_pytorch(
                value.double(),
                shapes,
                sampling_locations.double(),
                attention_weights.double(),
            )
            .detach()
            .cpu()
        )
        output_cuda = (
            MSDeformAttnFunction.apply(
                value.double(),
                shapes,
                level_start_index,
                sampling_locations.double(),
                attention_weights.double(),
                im2col_step,
            )
            .detach()
            .cpu()
        )
        fwdok = torch.allclose(output_cuda, output_pytorch)
        max_abs_err = (output_cuda - output_pytorch).abs().max()
        max_rel_err = (
            (output_cuda - output_pytorch).abs() / output_pytorch.abs()
        ).max()

        print(
            f"* {fwdok} test_forward_equal_with_pytorch_double: max_abs_err {max_abs_err:.2e} max_rel_err {max_rel_err:.2e}"
        )

    @unittest.skipIf(not USE_CUDA, "CI does not have gpu")
    @torch.no_grad()
    def test_forward_equal_with_pytorch_float(self):
        value = torch.rand(N, S, M, D).cuda() * 0.01
        sampling_locations = torch.rand(N, Lq, M, L, P, 2).cuda()
        attention_weights = torch.rand(N, Lq, M, L, P).cuda() + 1e-5
        attention_weights /= attention_weights.sum(-1, keepdim=True).sum(
            -2, keepdim=True
        )
        im2col_step = 2
        output_pytorch = (
            ms_deform_attn_core_pytorch(
                value, shapes, sampling_locations, attention_weights
            )
            .detach()
            .cpu()
        )
        output_cuda = (
            MSDeformAttnFunction.apply(
                value,
                shapes,
                level_start_index,
                sampling_locations,
                attention_weights,
                im2col_step,
            )
            .detach()
            .cpu()
        )
        fwdok = torch.allclose(output_cuda, output_pytorch, rtol=1e-2, atol=1e-3)
        max_abs_err = (output_cuda - output_pytorch).abs().max()
        max_rel_err = (
            (output_cuda - output_pytorch).abs() / output_pytorch.abs()
        ).max()

        print(
            f"* {fwdok} test_forward_equal_with_pytorch_float: max_abs_err {max_abs_err:.2e} max_rel_err {max_rel_err:.2e}"
        )

    @unittest.skipIf(not USE_CUDA, "CI does not have gpu")
    def test_gradient_numerical(
        self, channels=4, grad_value=True, grad_sampling_loc=True, grad_attn_weight=True
    ):

        value = torch.rand(N, S, M, channels).cuda() * 0.01
        sampling_locations = torch.rand(N, Lq, M, L, P, 2).cuda()
        attention_weights = torch.rand(N, Lq, M, L, P).cuda() + 1e-5
        attention_weights /= attention_weights.sum(-1, keepdim=True).sum(
            -2, keepdim=True
        )
        im2col_step = 2
        func = MSDeformAttnFunction.apply

        value.requires_grad = grad_value
        sampling_locations.requires_grad = grad_sampling_loc
        attention_weights.requires_grad = grad_attn_weight

        gradok = gradcheck(
            func,
            (
                value.double(),
                shapes,
                level_start_index,
                sampling_locations.double(),
                attention_weights.double(),
                im2col_step,
            ),
        )

        print(f"* {gradok} test_gradient_numerical(D={channels})")


if __name__ == "__main__":
    unittest.main()
