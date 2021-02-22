#!/usr/bin/env python3

from typing import List, Optional, Union

import numpy as np
import torch
from detectron2.data.transforms.augmentation import AugmentationList, Augmentation
from detectron2.structures import Boxes
from fvcore.transforms.transform import Transform, TransformList


class AugInput:
    """
    Same as AugInput in vision/fair/detectron2/detectron2/data/transforms/augmentation.py
    but allows torch.Tensor as input
    """

    def __init__(
        self,
        image: Union[np.ndarray, torch.Tensor],
        *,
        boxes: Optional[Union[np.ndarray, torch.Tensor, Boxes]] = None,
        sem_seg: Optional[Union[np.ndarray, torch.Tensor]] = None,
    ):
        """
        Args:
            image (ndarray/torch.Tensor): (H,W) or (H,W,C) ndarray of type uint8 in range [0, 255], or
                floating point in range [0, 1] or [0, 255]. (C, H, W) for tensor.
            boxes (ndarray or None): Nx4 float32 boxes in XYXY_ABS mode
            sem_seg (ndarray or None): HxW uint8 semantic segmentation mask. Each element
                is an integer label of pixel.
        """
        self.image = image
        self.boxes = boxes
        self.sem_seg = sem_seg

    def transform(self, tfm: Transform) -> None:
        """
        In-place transform all attributes of this class.

        By "in-place", it means after calling this method, accessing an attribute such
        as ``self.image`` will return transformed data.
        """
        self.image = tfm.apply_image(self.image)
        if self.boxes is not None:
            self.boxes = tfm.apply_box(self.boxes)
        if self.sem_seg is not None:
            self.sem_seg = tfm.apply_segmentation(self.sem_seg)

    def apply_augmentations(
        self, augmentations: List[Union[Augmentation, Transform]]
    ) -> TransformList:
        """
        Equivalent of ``AugmentationList(augmentations)(self)``
        """
        return AugmentationList(augmentations)(self)


class Tensor2Array(Transform):
    """ Convert image tensor (CHW) to np array (HWC) """

    def __init__(self):
        super().__init__()

    def apply_image(self, img: torch.Tensor) -> np.ndarray:
        # CHW -> HWC
        assert isinstance(img, torch.Tensor)
        assert len(img.shape) == 3, img.shape
        return img.cpu().numpy().transpose(1, 2, 0)

    def apply_coords(self, coords):
        return coords

    def apply_segmentation(self, segmentation: torch.Tensor) -> np.ndarray:
        assert len(segmentation.shape) == 2, segmentation.shape
        return segmentation.cpu().numpy()

    def inverse(self):
        return Array2Tensor()


class Array2Tensor(Transform):
    """ Convert image np array (HWC) to torch tensor (CHW) """

    def __init__(self):
        super().__init__()

    def apply_image(self, img: np.ndarray) -> torch.Tensor:
        # HWC -> CHW
        assert isinstance(img, np.ndarray)
        assert len(img.shape) == 3, img.shape
        return torch.from_numpy(img.transpose(2, 0, 1).astype("float32"))

    def apply_coords(self, coords):
        return coords

    def apply_segmentation(self, segmentation: np.ndarray) -> torch.Tensor:
        assert len(segmentation.shape) == 2, segmentation.shape
        return torch.from_numpy(segmentation.astype("long"))

    def inverse(self):
        return Tensor2Array()
