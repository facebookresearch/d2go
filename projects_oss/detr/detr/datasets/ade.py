import math
import os
import random
import sys

import numpy as np
import skimage.morphology as morp
import torch
import torch.utils.data as data
import torchvision
import torchvision.transforms as transform
from detectron2.utils.file_io import PathManager
from PIL import Image, ImageFilter, ImageOps

from .coco import make_coco_transforms


class ADE20KParsing(torchvision.datasets.VisionDataset):
    def __init__(self, root, split, transforms=None):
        super(ADE20KParsing, self).__init__(root)
        # assert exists and prepare dataset automatically
        assert PathManager.exists(root), "Please setup the dataset"
        self.images, self.masks = _get_ade20k_pairs(root, split)
        assert len(self.images) == len(self.masks)
        if len(self.images) == 0:
            raise (
                RuntimeError(
                    "Found 0 images in subfolders of: \
                "
                    + root
                    + "\n"
                )
            )
        self._transforms = transforms

    def _mask_transform(self, mask):
        target = np.array(mask).astype("int64") - 1
        return target

    def __getitem__(self, index):
        with PathManager.open(self.images[index], "rb") as f:
            img = Image.open(f).convert("RGB")
        with PathManager.open(self.masks[index], "rb") as f:
            mask = Image.open(f).convert("P")
        w, h = img.size
        ## generating bbox and masks
        # get different classes
        mask = self._mask_transform(mask)
        classes = np.unique(mask)
        if -1 in classes:
            classes = classes[1:]
        segmasks = mask == classes[:, None, None]
        # find connected component
        detr_masks = []
        labels = []
        for i in range(len(classes)):
            mask = segmasks[i]
            mclass = classes[i]
            connected, nslice = morp.label(
                mask, connectivity=2, background=0, return_num=True
            )
            for j in range(1, nslice + 1):
                detr_masks.append(connected == j)
                labels.append(mclass)

        target = {}
        target["image_id"] = torch.tensor(
            int(os.path.basename(self.images[index])[10:-4])
        )
        if len(detr_masks) > 0:
            target["masks"] = torch.as_tensor(
                np.stack(detr_masks, axis=0), dtype=torch.uint8
            )
            target["boxes"] = masks_to_boxes(target["masks"])
        else:
            target["masks"] = torch.as_tensor(detr_masks, dtype=torch.uint8)
            target["boxes"] = target["masks"]
        target["labels"] = torch.tensor(labels)
        target["orig_size"] = torch.as_tensor([int(h), int(w)])
        target["size"] = torch.as_tensor([int(h), int(w)])

        if self._transforms is not None:
            img, target = self._transforms(img, target)
        return img, target

    def __len__(self):
        return len(self.images)

    @property
    def pred_offset(self):
        return 1


def masks_to_boxes(masks):
    """Compute the bounding boxes around the provided masks
    The masks should be in format [N, H, W] where N is the number of masks, (H, W) are the spatial dimensions.
    Returns a [N, 4] tensors, with the boxes in xyxy format
    """
    if masks.numel() == 0:
        return torch.zeros((0, 4), device=masks.device)

    h, w = masks.shape[-2:]

    y = torch.arange(0, h, dtype=torch.float)
    x = torch.arange(0, w, dtype=torch.float)
    y, x = torch.meshgrid(y, x)

    x_mask = masks * x.unsqueeze(0)
    x_max = x_mask.flatten(1).max(-1)[0]
    x_min = x_mask.masked_fill(~(masks.bool()), 1e8).flatten(1).min(-1)[0]

    y_mask = masks * y.unsqueeze(0)
    y_max = y_mask.flatten(1).max(-1)[0]
    y_min = y_mask.masked_fill(~(masks.bool()), 1e8).flatten(1).min(-1)[0]

    return torch.stack([x_min, y_min, x_max, y_max], 1)


def _get_ade20k_pairs(folder, split="train"):
    def get_path_pairs(img_folder, mask_folder):
        img_paths = []
        mask_paths = []
        print("Before listing", img_folder)
        filenames = PathManager.ls(img_folder)
        for filename in filenames:
            print("found: ", filename)
            basename, _ = os.path.splitext(filename)
            if filename.endswith(".jpg"):
                imgpath = os.path.join(img_folder, filename)
                maskname = basename + ".png"
                maskpath = os.path.join(mask_folder, maskname)
                img_paths.append(imgpath)
                mask_paths.append(maskpath)
                # if PathManager.isfile(maskpath):
                # else:
                #    print('cannot find the mask:', maskpath)
        return img_paths, mask_paths

    if split == "train":
        img_folder = os.path.join(folder, "images/training")
        mask_folder = os.path.join(folder, "annotations/training")
        img_paths, mask_paths = get_path_pairs(img_folder, mask_folder)
        print("len(img_paths):", len(img_paths))
        assert len(img_paths) == 20210
    elif split == "val":
        img_folder = os.path.join(folder, "images/validation")
        mask_folder = os.path.join(folder, "annotations/validation")
        img_paths, mask_paths = get_path_pairs(img_folder, mask_folder)
        assert len(img_paths) == 2000
    else:
        assert split == "trainval"
        train_img_folder = os.path.join(folder, "images/training")
        train_mask_folder = os.path.join(folder, "annotations/training")
        val_img_folder = os.path.join(folder, "images/validation")
        val_mask_folder = os.path.join(folder, "annotations/validation")
        train_img_paths, train_mask_paths = get_path_pairs(
            train_img_folder, train_mask_folder
        )
        val_img_paths, val_mask_paths = get_path_pairs(val_img_folder, val_mask_folder)
        img_paths = train_img_paths + val_img_paths
        mask_paths = train_mask_paths + val_mask_paths
        assert len(img_paths) == 22210
    return img_paths, mask_paths


def build(image_set, args):
    dataset = ADE20KParsing(
        args.ade_path, image_set, transforms=make_coco_transforms(image_set)
    )
    return dataset
