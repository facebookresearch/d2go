#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved


import copy
import logging

import numpy as np
import torch
from d2go.data.dataset_mappers.build import D2GO_DATA_MAPPER_REGISTRY
from d2go.data.dataset_mappers.d2go_dataset_mapper import D2GoDatasetMapper
from detectron2.data import detection_utils as utils, transforms as T
from detectron2.structures import BoxMode, Instances, RotatedBoxes


logger = logging.getLogger(__name__)


@D2GO_DATA_MAPPER_REGISTRY.register()
class RotatedDatasetMapper(D2GoDatasetMapper):
    def _original_call(self, dataset_dict):
        """
        Modified from detectron2's original __call__ in DatasetMapper
        """
        dataset_dict = copy.deepcopy(dataset_dict)  # it will be modified by code below

        image = self._read_image(dataset_dict, format=self.img_format)
        if not self.backfill_size:
            utils.check_image_size(dataset_dict, image)

        if "annotations" not in dataset_dict:
            image, transforms = T.apply_transform_gens(
                ([self.crop_gen] if self.crop_gen else []) + self.tfm_gens, image
            )
        else:
            # Crop around an instance if there are instances in the image.
            # USER: Remove if you don't use cropping
            if self.crop_gen:
                crop_tfm = utils.gen_crop_transform_with_instance(
                    self.crop_gen.get_crop_size(image.shape[:2]),
                    image.shape[:2],
                    np.random.choice(dataset_dict["annotations"]),
                )
                image = crop_tfm.apply_image(image)
            image, transforms = T.apply_transform_gens(self.tfm_gens, image)
            if self.crop_gen:
                transforms = crop_tfm + transforms

        image_shape = image.shape[:2]  # h, w
        dataset_dict["image"] = torch.as_tensor(
            image.transpose(2, 0, 1).astype("float32")
        )
        # Can use uint8 if it turns out to be slow some day

        assert not self.load_proposals, "Not supported!"

        if not self.is_train:
            dataset_dict.pop("annotations", None)
            dataset_dict.pop("sem_seg_file_name", None)
            return dataset_dict

        if "annotations" in dataset_dict:
            for anno in dataset_dict["annotations"]:
                if not self.mask_on:
                    anno.pop("segmentation", None)
                if not self.keypoint_on:
                    anno.pop("keypoints", None)

            # Convert dataset_dict["annotations"] to dataset_dict["instances"]
            annotations = [
                obj
                for obj in dataset_dict.pop("annotations")
                if obj.get("iscrowd", 0) == 0
            ]

            # Convert either rotated box or horizontal box to XYWHA_ABS format
            original_boxes = [
                BoxMode.convert(
                    box=obj["bbox"],
                    from_mode=obj["bbox_mode"],
                    to_mode=BoxMode.XYWHA_ABS,
                )
                for obj in annotations
            ]

            transformed_boxes = transforms.apply_rotated_box(
                np.array(original_boxes, dtype=np.float64)
            )

            instances = Instances(image_shape)
            instances.gt_classes = torch.tensor(
                [obj["category_id"] for obj in annotations], dtype=torch.int64
            )
            instances.gt_boxes = RotatedBoxes(transformed_boxes)
            instances.gt_boxes.clip(image_shape)

            dataset_dict["instances"] = instances[instances.gt_boxes.nonempty()]

        return dataset_dict
