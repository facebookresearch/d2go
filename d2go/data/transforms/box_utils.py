#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import functools
from typing import Any, List, Tuple, Union

import detectron2.data.transforms.augmentation as aug
import numpy as np
import torch
from d2go.data.transforms.build import _json_load, TRANSFORM_OP_REGISTRY
from detectron2.config import CfgNode
from detectron2.data.transforms.transform import Transform
from detectron2.structures.boxes import Boxes


def get_box_union(boxes: Boxes):
    """Merge all boxes into a single box"""
    if len(boxes) == 0:
        return boxes
    bt = boxes.tensor
    union_bt = torch.cat(
        (torch.min(bt[:, :2], 0).values, torch.max(bt[:, 2:], 0).values)
    ).reshape(1, -1)
    return Boxes(union_bt)


def get_box_from_mask(mask: torch.Tensor) -> Tuple[int, int, int, int]:
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


def get_min_box_aspect_ratio(
    bbox_xywh: torch.Tensor, target_aspect_ratio: float
) -> torch.Tensor:
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


def get_box_center(bbox_xywh: torch.Tensor) -> torch.Tensor:
    """Get the center of the bbox"""
    return torch.Tensor(bbox_xywh[:2]) + torch.Tensor(bbox_xywh[2:]) / 2.0


def get_bbox_xywh_from_center_wh(
    bbox_center: torch.Tensor, bbox_wh: torch.Tensor
) -> torch.Tensor:
    """Get a bbox from bbox center and the width and height"""
    bbox_wh = torch.Tensor(bbox_wh)
    bbox_xy = torch.Tensor(bbox_center) - bbox_wh / 2.0
    return torch.cat([bbox_xy, bbox_wh])


def get_bbox_xyxy_from_xywh(bbox_xywh: torch.Tensor) -> torch.Tensor:
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


def get_bbox_xywh_from_xyxy(bbox_xyxy: torch.Tensor) -> torch.Tensor:
    """Convert the bbox from xyxy format to xywh format"""
    return torch.Tensor(
        [
            bbox_xyxy[0],
            bbox_xyxy[1],
            bbox_xyxy[2] - bbox_xyxy[0],
            bbox_xyxy[3] - bbox_xyxy[1],
        ]
    )


def to_boxes_from_xywh(bbox_xywh: torch.Tensor) -> torch.Tensor:
    return Boxes(get_bbox_xyxy_from_xywh(bbox_xywh).unsqueeze(0))


def scale_bbox_center(bbox_xywh: torch.Tensor, target_scale: float) -> torch.Tensor:
    """Scale the bbox around the center of the bbox"""
    box_center = get_box_center(bbox_xywh)
    box_wh = torch.Tensor(bbox_xywh[2:]) * target_scale
    return get_bbox_xywh_from_center_wh(box_center, box_wh)


def offset_bbox(bbox_xywh: torch.Tensor, target_offset: float) -> torch.Tensor:
    """Offset the bbox based on target_offset"""
    box_center = get_box_center(bbox_xywh)
    new_center = box_center + torch.Tensor(target_offset)
    return get_bbox_xywh_from_center_wh(new_center, bbox_xywh[2:])


def clip_box_xywh(bbox_xywh: torch.Tensor, image_size_hw: List[int]):
    """Clip the bbox based on image_size_hw"""
    h, w = image_size_hw
    bbox_xyxy = get_bbox_xyxy_from_xywh(bbox_xywh)
    bbox_xyxy[0] = max(bbox_xyxy[0], 0)
    bbox_xyxy[1] = max(bbox_xyxy[1], 0)
    bbox_xyxy[2] = min(bbox_xyxy[2], w)
    bbox_xyxy[3] = min(bbox_xyxy[3], h)
    return get_bbox_xywh_from_xyxy(bbox_xyxy)


def scale_coord(
    target: Union[torch.tensor, np.ndarray],
    source: Union[torch.tensor, np.ndarray],
    percentage: float,
):
    return [((a - b) * percentage + a) for a, b in zip(target, source)]


def pad_coord(
    target: Union[torch.tensor, np.ndarray],
    source: Union[torch.tensor, np.ndarray],
    fixed_pad: float,
):
    return [(np.sign(a - b) * fixed_pad + a) for a, b in zip(target, source)]


class EnlargeBoundingBox(Transform):
    """Enlarge bounding box based on fixed padding or percentage"""

    def __init__(
        self, percentage: float = None, fixed_pad: int = None, box_only: bool = False
    ):
        super().__init__()
        assert percentage is not None or fixed_pad is not None
        assert percentage is None or fixed_pad is None

        if percentage is not None:
            self.xfm_fn = functools.partial(scale_coord, percentage=percentage)

        elif fixed_pad is not None:
            self.xfm_fn = functools.partial(pad_coord, fixed_pad=fixed_pad)

        self.box_only = box_only

    def apply_image(self, img: torch.Tensor) -> np.ndarray:
        return img

    def apply_box(self, coords: Any) -> Any:
        # Takes boxes_xyxy
        center = (np.array(coords[0, 0:2]) + np.array(coords[0, 2:])) / 2
        new_coords = np.zeros_like(coords)
        new_coords[0, 0:2] = self.xfm_fn(coords[0, 0:2], center)
        new_coords[0, 2:] = self.xfm_fn(coords[0, 2:], center)
        return new_coords

    def apply_coords(self, coords: Any) -> Any:
        if self.box_only:
            return coords
        assert coords.shape[1] == 2, "Supported 2d inputs only"
        center = np.mean(coords, axis=0)
        for index in range(coords.shape[0]):
            coords[index] = self.xfm_fn(coords[index], center)
        return coords


@TRANSFORM_OP_REGISTRY.register()
def EnlargeBoundingBoxOp(
    cfg: CfgNode, arg_str: str, is_train: bool
) -> List[Union[aug.Augmentation, Transform]]:
    kwargs = _json_load(arg_str) if arg_str is not None else {}
    assert isinstance(kwargs, dict)
    return [EnlargeBoundingBox(**kwargs)]
