#!/usr/bin/env python3

import random
import cv2
import json
import numpy as np

from .build import TRANSFORM_OP_REGISTRY
from detectron2.data.transforms import Transform, TransformGen, NoOpTransform
import torchvision.transforms as T


class AffineTransform(Transform):
    def __init__(self, M, img_w, img_h, flags=None, border_mode=None, is_inversed_M=False):
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

    def apply_image(self, img):
        M = self.M
        if self.is_inversed_M:
            M = M[:2]
        img = cv2.warpAffine(
            img,
            M,
            (int(self.img_w), (self.img_h)),
            **self.warp_kwargs,
        )
        return img

    def apply_coords(self, coords):
        # Add row of ones to enable matrix multiplication
        coords = coords.T
        ones = np.ones((1, coords.shape[1]))
        coords = np.vstack((coords, ones))
        M = self.M
        if self.is_inversed_M:
            M = np.linalg.inv(M)
        coords = (M @ coords)[:2, :].T
        return coords


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
    def __init__(self, scales):
        super().__init__()
        self._init(locals())
        self.scales = scales

    def get_transform(self, img):
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
        pts2 = np.float32([
            _interp(pivot, lt, scale),
            pivot,
            _interp(pivot, rb, scale)],
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
        prob=0.5,
        angle_range=(-90, 90),
        translation_range=(0, 0),
        scale_range=(1.0, 1.0),
        shear_range=(0, 0),
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
        """
        super().__init__()
        # Turn all locals into member variables.
        self._init(locals())

    def get_transform(self, img):
        im_h, im_w = img.shape[:2]
        max_size = max(im_w, im_h)
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

        dummy_translation = [0.0, 0.0]
        dummy_scale = 1.0
        M_inv = T.functional._get_inverse_affine_matrix(
            center, angle, dummy_translation, dummy_scale, shear
        )
        M_inv.extend([0.0, 0.0, 1.0])
        M_inv = np.array(M_inv).reshape((3, 3))
        M = np.linalg.inv(M_inv)

        # Center in output patch
        img_corners = np.array([
            [0, 0, im_w, im_w],
            [0, im_h, 0, im_h],
            [1, 1, 1, 1],
        ])
        transformed_corners = M @ img_corners
        x_min = np.amin(transformed_corners[0])
        x_max = np.amax(transformed_corners[0])
        x_range = np.ceil(x_max - x_min)
        y_min = np.amin(transformed_corners[1])
        y_max = np.amax(transformed_corners[1])
        y_range = np.ceil(y_max - y_min)

        # Apply translation and scale after centering in output patch
        translation_adjustment = [(max_size - im_w) / 2, (max_size - im_h) / 2]
        translation[0] += translation_adjustment[0]
        translation[1] += translation_adjustment[1]
        scale_adjustment = min(max_size / x_range, max_size / y_range)
        scale *= scale_adjustment

        M_inv = T.functional._get_inverse_affine_matrix(
            center, angle, translation, scale, shear
        )
        # Convert to Numpy matrix so it can be inverted
        M_inv.extend([0.0, 0.0, 1.0])
        M_inv = np.array(M_inv).reshape((3, 3))
        M = np.linalg.inv(M_inv)

        do = self._rand_range() < self.prob
        if do:
            return AffineTransform(
                M_inv,
                max_size,
                max_size,
                flags=cv2.WARP_INVERSE_MAP + cv2.INTER_LINEAR,
                border_mode=cv2.BORDER_REPLICATE,
                is_inversed_M=True
            )
        else:
            return NoOpTransform()

# example repr: "RandomPivotScalingOp::[1.0, 0.75, 0.5]"
@TRANSFORM_OP_REGISTRY.register()
def RandomPivotScalingOp(cfg, arg_str, is_train):
    assert is_train
    scales = json.loads(arg_str)
    assert isinstance(scales, list)
    assert all(isinstance(scale, (float, int)) for scale in scales)
    return [RandomPivotScaling(scales=scales)]


@TRANSFORM_OP_REGISTRY.register()
def RandomAffineOp(cfg, arg_str, is_train):
    assert is_train
    kwargs = json.loads(arg_str) if arg_str is not None else {}
    assert isinstance(kwargs, dict)
    return [RandomAffine(**kwargs)]
