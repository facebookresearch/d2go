#!/usr/bin/env python3

import numpy as np
import torch
from detectron2.structures.boxes import Boxes


def get_box_union(boxes: Boxes):
    """ Merge all boxes into a single box """
    if len(boxes) == 0:
        return boxes
    bt = boxes.tensor
    union_bt = torch.cat(
        (torch.min(bt[:, :2], 0).values, torch.max(bt[:, 2:], 0).values)
    ).reshape(1, -1)
    return Boxes(union_bt)


def get_box_from_mask(mask: np.ndarray):
    """Find if there are non-zero elements per row/column first and then find
    min/max position of those elements.
    Only support 2d image (h x w)
    Return (x1, y1, w, h) if bbox found, otherwise None
    """
    assert len(mask.shape) == 2, f"Invalid shape {mask.shape}"
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    if bool(np.any(rows)) is False or bool(np.any(cols)) is False:
        return None
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]

    assert cmax >= cmin, f"cmax={cmax}, cmin={cmin}"
    assert rmax >= rmin, f"rmax={rmax}, rmin={rmin}"

    # x1, y1, w, h
    return cmin, rmin, cmax - cmin + 1, rmax - rmin + 1


def get_min_box_aspect_ratio(bbox_xywh, target_aspect_ratio):
    """Get a minimal bbox that matches the target_aspect_ratio
    target_aspect_ratio is representation by w/h
    bbox are represented by pixel coordinates"""
    bbox_xywh = torch.Tensor(bbox_xywh)
    box_w, box_h = bbox_xywh[2:]
    box_ar = float(box_w) / box_h
    if box_ar >= target_aspect_ratio:
        new_w = box_w
        new_h = float(new_w) / target_aspect_ratio
    else:
        new_h = box_h
        new_w = new_h * target_aspect_ratio
    new_wh = torch.Tensor([new_w, new_h])
    bbox_center = bbox_xywh[:2] + bbox_xywh[2:] / 2.0
    new_xy = bbox_center - new_wh / 2.0
    return torch.cat([new_xy, new_wh])


def get_box_center(bbox_xywh):
    """Get the center of the bbox"""
    return torch.Tensor(bbox_xywh[:2]) + torch.Tensor(bbox_xywh[2:]) / 2.0


def get_bbox_xywh_from_center_wh(bbox_center, bbox_wh):
    """Get a bbox from bbox center and the width and height"""
    bbox_wh = torch.Tensor(bbox_wh)
    bbox_xy = torch.Tensor(bbox_center) - bbox_wh / 2.0
    return torch.cat([bbox_xy, bbox_wh])


def get_bbox_xyxy_from_xywh(bbox_xywh):
    """Convert the bbox from xywh format to xyxy format
    bbox are represented by pixel coordinates,
    the center of pixels are (x + 0.5, y + 0.5)
    """
    return torch.Tensor(
        [
            bbox_xywh[0],
            bbox_xywh[1],
            bbox_xywh[0] + bbox_xywh[2],
            bbox_xywh[1] + bbox_xywh[3],
        ]
    )


def get_bbox_xywh_from_xyxy(bbox_xyxy):
    """Convert the bbox from xyxy format to xywh format"""
    return torch.Tensor(
        [
            bbox_xyxy[0],
            bbox_xyxy[1],
            bbox_xyxy[2] - bbox_xyxy[0],
            bbox_xyxy[3] - bbox_xyxy[1],
        ]
    )


def to_boxes_from_xywh(bbox_xywh):
    return Boxes(get_bbox_xyxy_from_xywh(bbox_xywh).unsqueeze(0))


def scale_bbox_center(bbox_xywh, target_scale):
    """Scale the bbox around the center of the bbox"""
    box_center = get_box_center(bbox_xywh)
    box_wh = torch.Tensor(bbox_xywh[2:]) * target_scale
    return get_bbox_xywh_from_center_wh(box_center, box_wh)


def offset_bbox(bbox_xywh, target_offset):
    """Offset the bbox based on target_offset"""
    box_center = get_box_center(bbox_xywh)
    new_center = box_center + torch.Tensor(target_offset)
    return get_bbox_xywh_from_center_wh(new_center, bbox_xywh[2:])


def clip_box_xywh(bbox_xywh, image_size_hw):
    """Clip the bbox based on image_size_hw"""
    h, w = image_size_hw
    bbox_xyxy = get_bbox_xyxy_from_xywh(bbox_xywh)
    bbox_xyxy[0] = max(bbox_xyxy[0], 0)
    bbox_xyxy[1] = max(bbox_xyxy[1], 0)
    bbox_xyxy[2] = min(bbox_xyxy[2], w)
    bbox_xyxy[3] = min(bbox_xyxy[3], h)
    return get_bbox_xywh_from_xyxy(bbox_xyxy)
