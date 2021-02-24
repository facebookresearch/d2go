#!/usr/bin/env python3

import torch.nn as nn
import torch

from typing import NamedTuple, List, Sequence, Tuple

import numpy as np


class DataLoss(object):
    def __init__(self, disable_scale_invariance: bool) -> None:
        self.disable_scale_invariance = disable_scale_invariance

    def __call__(
        self,
        predictions: torch.FloatTensor,
        mask: torch.BoolTensor,
        flow: torch.FloatTensor,
    ) -> torch.FloatTensor:
        diffs = predictions - flow
        # pyre-fixme
        diffs[~(mask.expand_as(diffs))] = 0
        N = diffs.size(0)
        diffs = diffs.reshape(N, -1)
        # pyre-fixme
        mask = mask.reshape(N, -1).float()

        s1 = torch.sum(
            torch.pow(diffs, 2) / torch.clamp(mask.sum(dim=1, keepdim=True), min=1), 1
        )

        if self.disable_scale_invariance:
            losses = s1
        else:
            s2 = torch.pow(
                torch.sum(diffs / torch.clamp(mask.sum(dim=1, keepdim=True), min=1), 1),
                2,
            )
            losses = s1 - s2

        return (losses.sum() / max(1, N)).float()


class GradientLoss(object):
    def __call__(
        self,
        prediction: torch.FloatTensor,
        mask: torch.BoolTensor,
        gt: torch.FloatTensor,
    ) -> torch.FloatTensor:
        assert torch.isfinite(prediction).all()

        diff = prediction - gt

        N = prediction.size(0)

        v_gradient = torch.abs(diff[..., :-1, :] - diff[..., 1:, :])
        v_mask = mask[..., :-1, :] & mask[..., 1:, :]
        v_gradient[~(v_mask.expand_as(v_gradient))] = 0
        v_gradient = v_gradient.reshape(N, -1)
        v_mask = v_mask.reshape(N, -1).float()
        v_loss = v_gradient.sum(dim=1) / torch.clamp(v_mask.sum(dim=1), min=1)

        h_gradient = torch.abs(diff[..., :-1] - diff[..., 1:])
        h_mask = mask[..., :-1] & mask[..., 1:]
        h_gradient[~(h_mask.expand_as(h_gradient))] = 0
        h_gradient = h_gradient.reshape(N, -1)
        h_mask = h_mask.reshape(N, -1).float()
        h_loss = h_gradient.sum(dim=1) / torch.clamp(h_mask.sum(dim=1), min=1)

        return (v_loss + h_loss).sum() / max(1, N)


class JointLossOutput(NamedTuple):
    data_loss: torch.FloatTensor
    gradient_losses: List[torch.FloatTensor]


class JointLoss(object):
    def __init__(
        self,
        midas_shift_scale: bool,
        num_grad_scales: int,
        weight_data: float,
        weight_grad: float,
        disable_scale_invariance: bool,
    ) -> None:
        super().__init__()
        self.midas_shift_scale = midas_shift_scale
        self.num_grad_scales = num_grad_scales
        self.weight_data = weight_data
        self.weight_grad = weight_grad

        self.data_loss = DataLoss(disable_scale_invariance)
        self.gradient_loss = GradientLoss()

    def _midas_shift_scale(
        self,
        prediction: torch.FloatTensor,
        flows: torch.FloatTensor,
        masks: torch.BoolTensor,
    ) -> Tuple[torch.FloatTensor, Sequence[torch.FloatTensor]]:
        if prediction.size(0) == 0:
            # pyre-fixme[7]: Expected `Tuple[torch.FloatTensor,
            #  Sequence[torch.FloatTensor]]` but got `Tuple[torch.FloatTensor,
            #  torch.FloatTensor]`.
            return prediction, flows

        prediction = prediction.clone()
        flows = flows.clone()

        def apply_shift_scale(x: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
            # pyre-fixme[16]: `Tensor` has no attribute `__iter__`.
            values = [x_slice[m.squeeze(0)] for x_slice in x]
            # compute a shift for x and y disparities
            medians = [value.median() for value in values]
            # only use the x disparities to generate the scale
            values = values[0] - medians[0]
            values = values.abs()
            scale = values.mean()
            scale = torch.clamp(scale, min=1e-3)
            x_slices = [
                (x_slice - median) / scale
                # pyre-fixme
                for x_slice, median in zip(x, medians)
            ]
            x = torch.stack(x_slices, 0)
            return x

        for i in range(flows.size(0)):
            m = masks[i]
            if m.sum() < 2:
                prediction[i][m] = flows[i][m]
                continue

            prediction[i] = apply_shift_scale(prediction[i], m)
            with torch.no_grad():
                flows[i] = apply_shift_scale(flows[i], m)

        # pyre-fixme[7]: Expected `Tuple[torch.FloatTensor,
        #  Sequence[torch.FloatTensor]]` but got `Tuple[torch.FloatTensor,
        #  torch.FloatTensor]`.
        return prediction, flows

    def _decimate_mask(self, mask: np.ndarray) -> np.ndarray:
        mask_a = mask[..., ::2, ::2].clone().detach()
        mask_b = mask[..., 1::2, ::2]
        mask_c = mask[..., ::2, 1::2]
        mask_d = mask[..., 1::2, 1::2]

        mask_a[..., : mask_b.shape[-2], : mask_b.shape[-1]] &= mask_b
        mask_a[..., : mask_c.shape[-2], : mask_c.shape[-1]] &= mask_c
        mask_a[..., : mask_d.shape[-2], : mask_d.shape[-1]] &= mask_d

        return mask_a

    def __call__(
        self,
        predictions: torch.FloatTensor,
        masks: torch.BoolTensor,
        flows: torch.FloatTensor,
    ) -> JointLossOutput:
        assert torch.isfinite(predictions).all()

        if self.midas_shift_scale:
            # pyre-fixme[9]: flows has type `FloatTensor`; used as
            #  `Sequence[torch.FloatTensor]`.
            predictions, flows = self._midas_shift_scale(predictions, flows, masks)

        # data term is applied only to original scale
        data_loss = self.data_loss(predictions, masks, flows)

        smallest_predictions = predictions

        # gradient term is applied to all scales
        gradient_losses = []
        for _ in range(self.num_grad_scales):
            gradient_loss = self.gradient_loss(smallest_predictions, masks, flows)
            gradient_losses.append(gradient_loss)

            smallest_predictions = smallest_predictions[..., ::2, ::2]

            # pyre-fixme[9]: masks has type `BoolTensor`; used as `ndarray`.
            # pyre-fixme[6]: Expected `ndarray` for 1st param but got `BoolTensor`.
            masks = self._decimate_mask(masks)
            flows = flows[..., ::2, ::2]

            N, _, H, W = flows.shape
            assert (N, 1, H, W) == masks.shape

        return JointLossOutput(
            # pyre-fixme
            data_loss=self.weight_data * data_loss,
            gradient_losses=[
                self.weight_grad * gradient_loss for gradient_loss in gradient_losses
            ],
        )
