#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import copy
import json
import random
from typing import List, Optional, Tuple

import cv2
import numpy as np
import torchvision.transforms as T
from d2go.data.transforms.build import TRANSFORM_OP_REGISTRY
from detectron2.config import CfgNode

from detectron2.data.transforms.augmentation import TransformGen
from fvcore.transforms.transform import NoOpTransform, Transform


class AffineTransform(Transform):
    def __init__(
        self,
        M: np.ndarray,
        img_w: int,
        img_h: int,
        flags: Optional[int] = None,
        border_mode: Optional[int] = None,
        is_inversed_M: bool = False,
    ):
        """
        Args:
           will transform img according to affine transform M
        """
        super().__init__()
        self._set_attributes(locals())
        self.warp_kwargs = {}
        if flags is not None:
            self.warp_kwargs["flags"] = flags
        if border_mode is not None:
            self.warp_kwargs["borderMode"] = border_mode

    def _warp_array(self, input_data: np.array, interp_flag: Optional[int] = None):
        warp_kwargs = copy.deepcopy(self.warp_kwargs)

        if interp_flag is not None:
            flags = warp_kwargs.get("flags", 0)
            # remove previous interp and add the new one
            flags = (flags - (flags & cv2.INTER_MAX)) + interp_flag
            warp_kwargs["flags"] = flags

        M = self.M
        if self.is_inversed_M:
            M = M[:2]
        img = cv2.warpAffine(
            input_data,
            M,
            (int(self.img_w), (self.img_h)),
            **warp_kwargs,
        )
        return img

    def apply_image(self, img: np.ndarray) -> np.ndarray:
        return self._warp_array(img)

    def apply_coords(self, coords: np.ndarray) -> np.ndarray:
        # Add row of ones to enable matrix multiplication
        coords = coords.T
        ones = np.ones((1, coords.shape[1]))
        coords = np.vstack((coords, ones))
        M = self.M
        if self.is_inversed_M:
            M = np.linalg.inv(M)
        coords = (M @ coords)[:2, :].T
        return coords

    def apply_segmentation(self, img: np.ndarray) -> np.ndarray:
        return self._warp_array(img, interp_flag=cv2.INTER_NEAREST)


class RandomPivotScaling(TransformGen):
    """
    Uniformly pick a random pivot point inside image frame, scaling the image
    around the pivot point using the scale factor sampled from a list of
    given scales. The pivot point's location is unchanged after the transform.

    Arguments:
        scales: List[float]: each element can be any positive float number,
            when larger than 1.0 objects become larger after transform
            and vice versa.
    """

    def __init__(self, scales: List[int]):
        super().__init__()
        self._init(locals())
        self.scales = scales

    def get_transform(self, img: np.ndarray) -> Transform:
        img_h, img_w, _ = img.shape
        img_h = float(img_h)
        img_w = float(img_w)
        pivot_y = self._rand_range(0.0, img_h)
        pivot_x = self._rand_range(0.0, img_w)

        def _interp(p1, p2, alpha):
            dx = p2[0] - p1[0]
            dy = p2[1] - p1[1]
            p_x = p1[0] + alpha * dx
            p_y = p1[1] + alpha * dy
            return (p_x, p_y)

        scale = np.random.choice(self.scales)
        lt = (0.0, 0.0)
        rb = (img_w, img_h)
        pivot = (pivot_x, pivot_y)
        pts1 = np.float32([lt, pivot, rb])
        pts2 = np.float32(
            [_interp(pivot, lt, scale), pivot, _interp(pivot, rb, scale)],
        )

        M = cv2.getAffineTransform(pts1, pts2)
        return AffineTransform(M, img_w, img_h)


class RandomAffine(TransformGen):
    """
    Apply random affine trasform to the image given
    probabilities and ranges in each dimension.
    """

    def __init__(
        self,
        prob: float = 0.5,
        angle_range: Tuple[float, float] = (-90, 90),
        translation_range: Tuple[float, float] = (0, 0),
        scale_range: Tuple[float, float] = (1.0, 1.0),
        shear_range: Tuple[float, float] = (0, 0),
        fit_in_frame: bool = True,
        keep_aspect_ratio: bool = False,
    ):
        """
        Args:
            prob (float): probability of applying transform.
            angle_range (tuple of integers): min/max rotation angle in degrees
                between -180 and 180.
            translation_range (tuple of integers): min/max translation
                (post re-centered rotation).
            scale_range (tuple of floats): min/max scale (post re-centered rotation).
            shear_range (tuple of intgers): min/max shear angle value in degrees
                between -180 to 180.
            fit_in_frame: warped image is scaled into the output frame
            keep_aspect_ratio: aspect ratio is kept instead of creating a squared image
                with dimension of max dimension
        """
        super().__init__()
        # Turn all locals into member variables.
        self._init(locals())

    def _compute_scale_adjustment(
        self,
        im_w: float,
        im_h: float,
        out_w: float,
        out_h: float,
        center: Tuple[float, float],
        angle: float,
        shear: Tuple[float, float],
    ) -> float:
        M_inv = T.functional._get_inverse_affine_matrix(
            center, angle, [0.0, 0.0], 1.0, shear
        )
        M_inv.extend([0.0, 0.0, 1.0])
        M_inv = np.array(M_inv).reshape((3, 3))
        M = np.linalg.inv(M_inv)

        # Center in output patch
        img_corners = np.array(
            [
                [0, 0, im_w - 1, im_w - 1],
                [0, im_h - 1, 0, im_h - 1],
                [1, 1, 1, 1],
            ]
        )
        new_corners = M @ img_corners
        x_range = np.ceil(np.amax(new_corners[0]) - np.amin(new_corners[0]))
        y_range = np.ceil(np.amax(new_corners[1]) - np.amin(new_corners[1]))

        # Apply translation and scale after centering in output patch
        scale_adjustment = min(out_w / x_range, out_h / y_range)
        return scale_adjustment

    def get_transform(self, img: np.ndarray) -> Transform:
        do = self._rand_range() < self.prob
        if not do:
            return NoOpTransform()

        im_h, im_w = img.shape[:2]
        center = [im_w / 2, im_h / 2]
        angle = random.uniform(self.angle_range[0], self.angle_range[1])
        translation = [
            random.uniform(self.translation_range[0], self.translation_range[1]),
            random.uniform(self.translation_range[0], self.translation_range[1]),
        ]
        scale = random.uniform(self.scale_range[0], self.scale_range[1])
        shear = [
            random.uniform(self.shear_range[0], self.shear_range[1]),
            random.uniform(self.shear_range[0], self.shear_range[1]),
        ]

        # Determine output image size
        max_size = max(im_w, im_h)
        out_w, out_h = (im_w, im_h) if self.keep_aspect_ratio else (max_size, max_size)

        # Apply translation adjustment
        translation_adjustment = [(out_w - im_w) / 2, (out_h - im_h) / 2]
        translation[0] += translation_adjustment[0]
        translation[1] += translation_adjustment[1]

        # Apply scale adjustment
        if self.fit_in_frame:
            scale_adjustment = self._compute_scale_adjustment(
                im_w, im_h, out_w, out_h, center, angle, shear
            )
            scale *= scale_adjustment

        # Compute the affine transform
        M_inv = T.functional._get_inverse_affine_matrix(
            center, angle, translation, scale, shear
        )
        M_inv = np.array(M_inv).reshape((2, 3))
        M_inv = np.vstack([M_inv, [0.0, 0.0, 1.0]])

        return AffineTransform(
            M_inv,
            out_w,
            out_h,
            flags=cv2.WARP_INVERSE_MAP + cv2.INTER_LINEAR,
            border_mode=cv2.BORDER_REPLICATE,
            is_inversed_M=True,
        )


# example repr: "RandomPivotScalingOp::[1.0, 0.75, 0.5]"
@TRANSFORM_OP_REGISTRY.register()
def RandomPivotScalingOp(cfg: CfgNode, arg_str: str, is_train: bool) -> List[Transform]:
    assert is_train
    scales = json.loads(arg_str)
    assert isinstance(scales, list)
    assert all(isinstance(scale, (float, int)) for scale in scales)
    return [RandomPivotScaling(scales=scales)]


@TRANSFORM_OP_REGISTRY.register()
def RandomAffineOp(cfg: CfgNode, arg_str: str, is_train: bool) -> List[Transform]:
    assert is_train
    kwargs = json.loads(arg_str) if arg_str is not None else {}
    assert isinstance(kwargs, dict)
    return [RandomAffine(**kwargs)]
