#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved


import copy
import logging

import numpy as np
import torch
from d2go.data.dataset_mappers.build import D2GO_DATA_MAPPER_REGISTRY
from d2go.data.dataset_mappers.data_reading import (
    read_image_with_prefetch,
    read_sem_seg_file_with_prefetch,
)
from d2go.utils.helper import retryable
from detectron2.data import detection_utils as utils, transforms as T
from detectron2.data.transforms.augmentation import AugInput, AugmentationList

logger = logging.getLogger(__name__)

PREFETCHED_FILE_NAME = "prefetch_image"
PREFETCHED_SEM_SEG_FILE_NAME = "prefetch_sem_seg"


@D2GO_DATA_MAPPER_REGISTRY.register()
class D2GoDatasetMapper(object):
    def __init__(self, cfg, is_train=True, image_loader=None, tfm_gens=None):
        self.tfm_gens = (
            tfm_gens
            if tfm_gens is not None
            else utils.build_transform_gen(cfg, is_train)
        )

        if cfg.INPUT.CROP.ENABLED and is_train:
            self.crop_gen = T.RandomCrop(cfg.INPUT.CROP.TYPE, cfg.INPUT.CROP.SIZE)
            # D2GO NOTE: when INPUT.CROP.ENABLED, don't allow using RandomCropOp
            assert all(not isinstance(gen, T.RandomCrop) for gen in self.tfm_gens)
        else:
            self.crop_gen = None

        # fmt: off
        self.img_format     = cfg.INPUT.FORMAT  # noqa
        self.mask_on        = cfg.MODEL.MASK_ON  # noqa
        self.mask_format    = cfg.INPUT.MASK_FORMAT  # noqa
        self.keypoint_on    = cfg.MODEL.KEYPOINT_ON  # noqa
        # fmt: on
        if self.keypoint_on and is_train:
            # Flip only makes sense in training
            self.keypoint_hflip_indices = utils.create_keypoint_hflip_indices(
                cfg.DATASETS.TRAIN
            )
        else:
            self.keypoint_hflip_indices = None

        self.load_proposals = cfg.MODEL.LOAD_PROPOSALS
        if self.load_proposals:
            self.proposal_min_box_size = cfg.MODEL.PROPOSAL_GENERATOR.MIN_SIZE
            self.proposal_topk = (
                cfg.DATASETS.PRECOMPUTED_PROPOSAL_TOPK_TRAIN
                if is_train
                else cfg.DATASETS.PRECOMPUTED_PROPOSAL_TOPK_TEST
            )

        self.is_train = is_train

        # Setup image loader:
        self.image_loader = image_loader
        self.backfill_size = cfg.D2GO_DATA.MAPPER.BACKFILL_SIZE
        self.retry = cfg.D2GO_DATA.MAPPER.RETRY
        self.catch_exception = cfg.D2GO_DATA.MAPPER.CATCH_EXCEPTION

        if self.backfill_size:
            if cfg.DATALOADER.ASPECT_RATIO_GROUPING:
                logger.warning(
                    "ASPECT_RATIO_GROUPING may not work if image's width & height"
                    " are not given in json dataset when calling extended_coco_load,"
                    " if you encounter issue, consider disable ASPECT_RATIO_GROUPING."
                )

        self._error_count = 0
        self._total_counts = 0
        self._error_types = {}

    def _original_call(self, dataset_dict):
        """
        Modified from detectron2's original __call__ in DatasetMapper
        """
        dataset_dict = copy.deepcopy(dataset_dict)  # it will be modified by code below

        image = self._read_image(dataset_dict, format=self.img_format)
        if not self.backfill_size:
            utils.check_image_size(dataset_dict, image)

        image, dataset_dict = self._custom_transform(image, dataset_dict)

        inputs = AugInput(image=image)
        if "annotations" not in dataset_dict or dataset_dict["annotations"] == []:
            transforms = AugmentationList(
                ([self.crop_gen] if self.crop_gen else []) + self.tfm_gens
            )(inputs)
            image = inputs.image
        else:
            # pass additional arguments, will only be used when the Augmentation
            #   takes `annotations` as input
            inputs.annotations = dataset_dict["annotations"]
            inputs.boxes = [
                utils.get_bbox(obj)
                for obj in dataset_dict["annotations"]
                if obj.get("iscrowd", 0) == 0
            ]
            # Crop around an instance if there are instances in the image.
            if self.crop_gen:
                crop_tfm = utils.gen_crop_transform_with_instance(
                    self.crop_gen.get_crop_size(image.shape[:2]),
                    image.shape[:2],
                    np.random.choice(dataset_dict["annotations"]),
                )
                inputs.image = crop_tfm.apply_image(image)
            transforms = AugmentationList(self.tfm_gens)(inputs)
            image = inputs.image
            if self.crop_gen:
                transforms = crop_tfm + transforms

        # Cache identical transforms in dataset_dict for subclass mappers
        # TODO T122215878 Find more explicit way to expose transforms used
        dataset_dict["transforms"] = transforms

        image_shape = image.shape[:2]  # h, w
        if image.ndim == 2:
            image = np.expand_dims(image, 2)
        dataset_dict["image"] = torch.as_tensor(
            image.transpose(2, 0, 1).astype("float32")
        )
        # Can use uint8 if it turns out to be slow some day

        if self.load_proposals:
            utils.transform_proposals(
                dataset_dict,
                image_shape,
                transforms,
                proposal_topk=self.proposal_topk,
                min_box_size=self.proposal_min_box_size,
            )

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

            annos = [
                utils.transform_instance_annotations(
                    obj,
                    transforms,
                    image_shape,
                    keypoint_hflip_indices=self.keypoint_hflip_indices,
                )
                for obj in dataset_dict.pop("annotations")
                if obj.get("iscrowd", 0) == 0
            ]
            instances = utils.annotations_to_instances(
                annos, image_shape, mask_format=self.mask_format
            )
            # Create a tight bounding box from masks, useful when image is cropped
            if self.crop_gen and instances.has("gt_masks"):
                instances.gt_boxes = instances.gt_masks.get_bounding_boxes()
            dataset_dict["instances"] = utils.filter_empty_instances(instances)

        if "sem_seg_file_name" in dataset_dict:
            sem_seg_gt = read_sem_seg_file_with_prefetch(
                dataset_dict.pop("sem_seg_file_name"),
                prefetched=dataset_dict.get(PREFETCHED_SEM_SEG_FILE_NAME, None),
            )
            if len(sem_seg_gt.shape) > 2:
                sem_seg_gt = sem_seg_gt.squeeze(2)
            sem_seg_gt = transforms.apply_segmentation(sem_seg_gt)
            sem_seg_gt = torch.as_tensor(sem_seg_gt.astype("long"))
            dataset_dict["sem_seg"] = sem_seg_gt

        # extend standard D2 semantic segmentation to support multiple segmentation
        # files, each file can represent a class
        if "multi_sem_seg_file_names" in dataset_dict:
            raise NotImplementedError()

        if "_post_process_" in dataset_dict:
            proc_func = dataset_dict.pop("_post_process_")
            dataset_dict = proc_func(dataset_dict)

        return dataset_dict

    def __call__(self, dataset_dict):
        self._total_counts += 1

        @retryable(num_tries=self.retry, sleep_time=0.1)
        def _f():
            return self._original_call(dataset_dict)

        if not self.catch_exception:
            return _f()

        try:
            return _f()
        except Exception as e:
            self._error_count += 1
            # if self._error_count % 10 == 1:
            #     # print the stacktrace for easier debugging
            #     traceback.print_exc()
            error_type = type(e).__name__
            self._error_types[error_type] = self._error_types.get(error_type, 0) + 1

            if self._error_count % 100 == 0:
                logger.warning(
                    "{}Error when applying transform for dataset_dict: {};"
                    " error rate {}/{} ({:.2f}%), msg: {}".format(
                        self._get_logging_prefix(),
                        dataset_dict,
                        self._error_count,
                        self._total_counts,
                        100.0 * self._error_count / self._total_counts,
                        repr(e),
                    )
                )
                self._log_error_type_stats()

            # NOTE: the contract with MapDataset allows return `None` such that
            # it'll randomly use other element in the dataset. We use this
            # feature to handle error.
            return None

    def _get_logging_prefix(self):
        worker_info = torch.utils.data.get_worker_info()
        if not worker_info:
            return ""

        prefix = "[worker: {}/{}] ".format(worker_info.id, worker_info.num_workers)
        return prefix

    def _log_error_type_stats(self):
        error_type_count_msgs = [
            "{}: {}/{} ({}%)".format(
                k, v, self._total_counts, 100.0 * v / self._total_counts
            )
            for k, v in self._error_types.items()
        ]
        logger.warning(
            "{}Error statistics:\n{}".format(
                self._get_logging_prefix(), "\n".join(error_type_count_msgs)
            )
        )

    def _read_image(self, dataset_dict, format=None):
        if not (self.image_loader and self.image_loader.support(dataset_dict)):
            # fallback to use D2's read_image
            image = read_image_with_prefetch(
                dataset_dict["file_name"],
                format=format,
                prefetched=dataset_dict.get(PREFETCHED_FILE_NAME),
            )
            if self.backfill_size:
                h, w, _ = image.shape
                dataset_dict["width"] = w
                dataset_dict["height"] = h
            return image

        image = self.image_loader(dataset_dict)
        if self.backfill_size:
            dataset_dict["width"] = image.width
            dataset_dict["height"] = image.height

        return utils.convert_PIL_to_numpy(image, format)

    def _custom_transform(self, image, dataset_dict):
        """
        Override this method to inject custom transform.
        """
        return image, dataset_dict

    def __repr__(self):
        return (
            self.__class__.__name__
            + ":\n"
            + "\n".join(
                [
                    "  is_train: {}".format(self.is_train),
                    "  image_loader: {}".format(self.image_loader),
                    "  tfm_gens: \n{}".format(
                        "\n".join(["    - {}".format(x) for x in self.tfm_gens])
                    ),
                ]
            )
        )
