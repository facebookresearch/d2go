#!/usr/bin/env python3

import numpy as np
from scipy import ndimage


def generate_weight_mask(mask, ring_weight=2, ring_dist_threashold=0.075):
    """
    Geenrate the "trimap" from a binary mask.

    Arguments:
        mask (np.array): a float mask of shape [..., H, W].
        ring_weight (float): weight for ring area.
        ring_dist_threashold (float): threashold for identifying the ring area.

    Returns:
        mask_weighted (np.array): a float tensor of the same shape as input mask.
    """
    # Trimap for the groundtruth segmentation mask
    H, W = mask.shape[:2]
    size = np.sqrt(H * W)

    mask_weighted = np.ones(shape=mask.shape, dtype=np.float32)

    area_mask = mask_weighted.shape[0] * mask_weighted.shape[1]

    src_f = np.copy(mask)
    src_b = 1 - np.copy(mask)

    # Each nonzero element is replaced by it's Euclidean distance
    # in the ndarray to the nearest zero element
    dist_f = ndimage.distance_transform_edt(src_f)
    dist_b = ndimage.distance_transform_edt(src_b)

    position_a = np.where((dist_f < size * ring_dist_threashold) & (dist_f > 0))
    mask_weighted[position_a] = ring_weight
    position_b = np.where((dist_b < size * ring_dist_threashold) & (dist_b > 0))
    mask_weighted[position_b] = ring_weight

    # Computing ratio to scale the weights to make the loss comparable to a
    # non weighted loss
    num_pixels_in_ring = position_a[0].shape[0] + position_b[0].shape[0]
    area_weighted_mask = (
        area_mask - num_pixels_in_ring
    ) + ring_weight * num_pixels_in_ring

    ratio = area_mask / area_weighted_mask
    mask_weighted *= ratio

    return mask_weighted
