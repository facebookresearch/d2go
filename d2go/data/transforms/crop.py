#!/usr/bin/env python3

import math

import detectron2.data.transforms.augmentation as aug
import numpy as np
from detectron2.data.transforms import ExtentTransform, CropTransform
from detectron2.structures import BoxMode

from . import box_utils as bu
from fvcore.transforms.transform import CropTransform, NoOpTransform, Transform

from .build import TRANSFORM_OP_REGISTRY, _json_load


class CropBoundary(aug.Augmentation):
    """Crop the boundary of the image by `count` pixel on each side"""

    def __init__(self, count=3):
        super().__init__()
        self.count = count

    def get_transform(self, image):
        img_h, img_w = image.shape[:2]
        assert self.count < img_h and self.count < img_w
        assert img_h > self.count * 2
        assert img_w > self.count * 2
        box = [self.count, self.count, img_w - self.count * 2, img_h - self.count * 2]
        return CropTransform(*box)


class PadTransform(Transform):
    def __init__(self, x0, y0, w, h, org_w, org_h, pad_mode="constant"):
        super().__init__()
        assert x0 + w <= org_w
        assert y0 + h <= org_h
        self._set_attributes(locals())

    def apply_image(self, img):
        """img: HxWxC or HxW"""
        assert len(img.shape) == 2 or len(img.shape) == 3
        assert img.shape[0] == self.h and img.shape[1] == self.w
        pad_width = [
            (self.y0, self.org_h - self.h - self.y0),
            (self.x0, self.org_w - self.w - self.x0),
            *([(0, 0)] if len(img.shape) == 3 else []),
        ]
        pad_args = {"mode": self.pad_mode}
        if self.pad_mode == "constant":
            pad_args["constant_values"] = 0
        ret = np.pad(img, pad_width=tuple(pad_width), **pad_args)
        return ret

    def apply_coords(self, coords: np.ndarray) -> np.ndarray:
        raise NotImplementedError()

    def inverse(self) -> Transform:
        return CropTransform(self.x0, self.y0, self.w, self.h, self.org_w, self.org_h)


InvertibleCropTransform = CropTransform


class PadBorderDivisible(aug.Augmentation):
    def __init__(self, size_divisibility, pad_mode="constant"):
        super().__init__()
        self.size_divisibility = size_divisibility
        self.pad_mode = pad_mode

    def get_transform(self, image):
        """ image: HxWxC """
        assert len(image.shape) == 3 and image.shape[2] in [1, 3]
        H, W = image.shape[:2]
        new_h = int(math.ceil(H / self.size_divisibility) * self.size_divisibility)
        new_w = int(math.ceil(W / self.size_divisibility) * self.size_divisibility)
        return PadTransform(0, 0, W, H, new_w, new_h, pad_mode=self.pad_mode)


class RandomCropFixedAspectRatio(aug.Augmentation):
    def __init__(
        self, crop_aspect_ratios_list, scale_range=None, offset_scale_range=None
    ):
        super().__init__()
        assert isinstance(crop_aspect_ratios_list, (list, tuple))
        assert (
            scale_range is None
            or isinstance(scale_range, (list, tuple))
            and len(scale_range) == 2
        )
        assert (
            offset_scale_range is None
            or isinstance(offset_scale_range, (list, tuple))
            and len(offset_scale_range) == 2
        )
        # [w1/h1, w2/h2, ...]
        self.crop_aspect_ratios_list = crop_aspect_ratios_list
        # [low, high] or None
        self.scale_range = scale_range
        # [low, high] or None
        self.offset_scale_range = offset_scale_range

        self.rng = np.random.default_rng()

    def _pick_aspect_ratio(self):
        return self.rng.choice(self.crop_aspect_ratios_list)

    def _pick_scale(self):
        if self.scale_range is None:
            return 1.0
        return self.rng.uniform(*self.scale_range)

    def _pick_offset(self, box_w, box_h):
        if self.offset_scale_range is None:
            return [0, 0]
        offset_scale = self.rng.uniform(*self.offset_scale_range, size=2)
        return offset_scale[0] * box_w, offset_scale[1] * box_h

    def get_transform(self, image, sem_seg):
        # HWC or HW for image, HW for sem_seg
        assert len(image.shape) in [2, 3]
        assert len(sem_seg.shape) == 2

        mask_box_xywh = bu.get_box_from_mask(sem_seg)
        # do nothing if the mask is empty (the whole image is background)
        if mask_box_xywh is None:
            return NoOpTransform()

        crop_ar = self._pick_aspect_ratio()
        target_scale = self._pick_scale()
        target_offset = self._pick_offset(*mask_box_xywh[2:])

        mask_box_xywh = bu.offset_bbox(mask_box_xywh, target_offset)
        mask_box_xywh = bu.scale_bbox_center(mask_box_xywh, target_scale)

        target_box_xywh = bu.get_min_box_aspect_ratio(mask_box_xywh, crop_ar)
        target_bbox_xyxy = bu.get_bbox_xyxy_from_xywh(target_box_xywh)

        return ExtentTransform(
            src_rect=target_bbox_xyxy,
            output_size=(
                int(target_box_xywh[3].item()),
                int(target_box_xywh[2].item()),
            ),
        )


# example repr: "CropBoundaryOp::{'count': 3}"
@TRANSFORM_OP_REGISTRY.register()
def CropBoundaryOp(cfg, arg_str, is_train):
    assert is_train
    kwargs = _json_load(arg_str) if arg_str is not None else {}
    assert isinstance(kwargs, dict)
    return [CropBoundary(**kwargs)]


# example repr: "RandomCropFixedAspectRatioOp::{'crop_aspect_ratios_list': [0.5], 'scale_range': [0.8, 1.2], 'offset_scale_range': [-0.3, 0.3]}"
@TRANSFORM_OP_REGISTRY.register()
def RandomCropFixedAspectRatioOp(cfg, arg_str, is_train):
    assert is_train
    kwargs = _json_load(arg_str) if arg_str is not None else {}
    assert isinstance(kwargs, dict)
    return [RandomCropFixedAspectRatio(**kwargs)]


class RandomInstanceCrop(aug.Augmentation):
    def __init__(self, crop_scale=(0.8, 1.6)):
        """
        Generates a CropTransform centered around the instance.
        crop_scale: [low, high] relative crop scale around the instance, this
        determines how far to zoom in / out around the cropped instance
        """
        super().__init__()
        self.crop_scale = crop_scale
        assert (
            isinstance(crop_scale, (list, tuple)) and len(crop_scale) == 2
        ), crop_scale

    def get_transform(self, image, annotations):
        """
        This function will modify instances to set the iscrowd flag to 1 for
        annotations not picked. It relies on the dataset mapper to filter those
        items out
        """
        assert isinstance(annotations, (list, tuple)), annotations
        assert all("bbox" in x for x in annotations), annotations
        assert all("bbox_mode" in x for x in annotations), annotations

        image_size = image.shape[:2]

        # filter out iscrowd
        annotations = [x for x in annotations if x.get("iscrowd", 0) == 0]
        if len(annotations) == 0:
            return NoOpTransform()

        sel_index = np.random.randint(len(annotations))
        # set iscrowd flag of other annotations to 1 so that they will be
        #   filtered out by the datset mapper (https://fburl.com/diffusion/fg64cb4h)
        for idx, instance in enumerate(annotations):
            if idx != sel_index:
                instance["iscrowd"] = 1
        instance = annotations[sel_index]

        bbox_xywh = BoxMode.convert(
            instance["bbox"], instance["bbox_mode"], BoxMode.XYWH_ABS
        )

        scale = np.random.uniform(*self.crop_scale)
        bbox_xywh = bu.scale_bbox_center(bbox_xywh, scale)
        bbox_xywh = bu.clip_box_xywh(bbox_xywh, image_size).int()

        return CropTransform(*bbox_xywh.tolist())


# example repr: "RandomInstanceCropOp::{'crop_scale': [0.8, 1.6]}"
@TRANSFORM_OP_REGISTRY.register()
def RandomInstanceCropOp(cfg, arg_str, is_train):
    kwargs = _json_load(arg_str) if arg_str is not None else {}
    assert isinstance(kwargs, dict)
    return [RandomInstanceCrop(**kwargs)]



class CropBoxAug(aug.Augmentation):
    """ Augmentation to crop the image based on boxes
        Scale the box with `box_scale_factor` around the center before cropping
    """
    def __init__(self, box_scale_factor=1.0):
        super().__init__()
        self.box_scale_factor = box_scale_factor

    def get_transform(self, image: np.ndarray, boxes: np.ndarray):
        # boxes: 1 x 4 in xyxy format
        assert boxes.shape[0] == 1
        assert isinstance(image, np.ndarray)
        assert isinstance(boxes, np.ndarray)
        img_h, img_w = image.shape[0:2]

        box_xywh = bu.get_bbox_xywh_from_xyxy(boxes[0])
        if self.box_scale_factor != 1.0:
            box_xywh  = bu.scale_bbox_center(box_xywh, self.box_scale_factor)
            box_xywh = bu.clip_box_xywh(box_xywh, [img_h, img_w])
        box_xywh = box_xywh.int().tolist()
        return CropTransform(*box_xywh, orig_w=img_w, orig_h=img_h)
