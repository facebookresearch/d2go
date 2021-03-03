#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved


import atexit
import contextlib
import json
import logging
import os
import shutil
import tempfile
from collections import defaultdict

import numpy as np
import torch.utils.data as data

logger = logging.getLogger(__name__)


from d2go.config import temp_defrost
from d2go.data.datasets import register_dataset_split, ANN_FN, IM_DIR
from detectron2.data import DatasetCatalog, MetadataCatalog
from fvcore.common.file_io import PathManager


class AdhocDatasetManager:
    # mapping from the new dataset name a AdhocDataset instance
    _REGISTERED = {}

    @staticmethod
    def add(adhoc_ds):
        assert isinstance(adhoc_ds, AdhocDataset)
        if adhoc_ds.new_ds_name in AdhocDatasetManager._REGISTERED:
            logger.warning(
                "Adhoc dataset {} has already been added, skip adding it".format(
                    adhoc_ds.new_ds_name
                )
            )
        else:
            logger.info("Adding new adhoc dataset {} ...".format(adhoc_ds.new_ds_name))
            AdhocDatasetManager._REGISTERED[adhoc_ds.new_ds_name] = adhoc_ds
            adhoc_ds.register_catalog()

    @staticmethod
    def remove(adhoc_ds):
        try:
            assert isinstance(adhoc_ds, AdhocDataset)
            if adhoc_ds.new_ds_name not in AdhocDatasetManager._REGISTERED:
                logger.warning(
                    "Adhoc dataset {} has already been removed, skip removing it".format(
                        adhoc_ds.new_ds_name
                    )
                )
            else:
                logger.info("Remove adhoc dataset {} ...".format(adhoc_ds.new_ds_name))
                del AdhocDatasetManager._REGISTERED[adhoc_ds.new_ds_name]
        finally:
            adhoc_ds.cleanup()

    @staticmethod
    @atexit.register
    def _atexit():
        for ds in AdhocDatasetManager._REGISTERED.values():
            logger.info("Remove remaining adhoc dataset: {}".format(ds.new_ds_name))
            ds.cleanup()


class AdhocDataset(object):
    def __init__(self, new_ds_name):
        assert isinstance(new_ds_name, str)
        self.new_ds_name = new_ds_name

    def register_catalog(self):
        raise NotImplementedError()

    def cleanup(self):
        raise NotImplementedError()


class CallFuncWithJsonFile(object):
    """
    The instance of this class is parameterless callable that calls its `func` using its
    `json_file`, it can be used to register in DatasetCatalog which later on provide
    access to the json file.
    """

    def __init__(self, func, json_file):
        self.func = func
        self.json_file = json_file

    def __call__(self):
        return self.func(self.json_file)


class AdhocCOCODataset(AdhocDataset):
    def __init__(self, src_ds_name, new_ds_name):
        super().__init__(new_ds_name)
        # NOTE: only support single source dataset now
        assert isinstance(src_ds_name, str)
        self.src_ds_name = src_ds_name

    def new_json_dict(self, json_dict):
        raise NotImplementedError()

    def register_catalog(self):
        """
        Adhoc COCO (json) dataset assumes the derived dataset can be created by only
        changing the json file, currently it supports two sources: 1) the dataset is
        registered using standard COCO registering functions in D2 or
        register_dataset_split from D2Go, this way it uses `json_file` from the metadata
        to access the json file. 2) the load func in DatasetCatalog is an instance of
        CallFuncWithJsonFile, which gives access to the json_file. In both cases,
        metadata will be the same except for the `name` and potentially `json_file`.
        """
        logger.info("Register {} from {}".format(self.new_ds_name, self.src_ds_name))
        metadata = MetadataCatalog.get(self.src_ds_name)

        load_func = DatasetCatalog[self.src_ds_name]
        src_json_file = (
            load_func.json_file
            if isinstance(load_func, CallFuncWithJsonFile)
            else metadata.json_file
        )

        # TODO cache ?
        with PathManager.open(src_json_file) as f:
            json_dict = json.load(f)
        assert "images" in json_dict, "Only support COCO-style json!"
        json_dict = self.new_json_dict(json_dict)
        self.tmp_dir = tempfile.mkdtemp(prefix="detectron2go_tmp_datasets")
        tmp_file = os.path.join(self.tmp_dir, "{}.json".format(self.new_ds_name))
        with open(tmp_file, "w") as f:
            json.dump(json_dict, f)

        # re-register DatasetCatalog
        if isinstance(load_func, CallFuncWithJsonFile):
            new_func = CallFuncWithJsonFile(func=load_func.func, json_file=tmp_file)
            DatasetCatalog.register(self.new_ds_name, new_func)
        else:
            # NOTE: only supports COCODataset as DS_TYPE since we cannot reconstruct
            # the split_dict
            register_dataset_split(
                self.new_ds_name,
                split_dict={ANN_FN: tmp_file, IM_DIR: metadata.image_root},
            )

        # re-regisister MetadataCatalog
        metadata_dict = metadata.as_dict()
        metadata_dict["name"] = self.new_ds_name
        if "json_file" in metadata_dict:
            metadata_dict["json_file"] = tmp_file
        MetadataCatalog.remove(self.new_ds_name)
        MetadataCatalog.get(self.new_ds_name).set(**metadata_dict)

    def cleanup(self):
        # remove temporarily registered dataset and json file
        DatasetCatalog.pop(self.new_ds_name, None)
        MetadataCatalog.pop(self.new_ds_name, None)
        if hasattr(self, "tmp_dir"):
            shutil.rmtree(self.tmp_dir)


class COCOSubsetWithNImages(AdhocCOCODataset):
    _SUPPORTED_SAMPLING = ["frontmost", "random"]

    def __init__(self, src_ds_name, num_images, sampling):
        super().__init__(
            src_ds_name=src_ds_name,
            new_ds_name="{}_{}{}".format(src_ds_name, sampling, num_images),
        )
        self.num_images = num_images
        self.sampling = sampling

    def new_json_dict(self, json_dict):
        all_images = json_dict["images"]
        if self.sampling == "frontmost":
            new_images = all_images[: self.num_images]
        elif self.sampling == "random":
            # use fixed seed so results are repeatable
            indices = np.random.RandomState(seed=42).permutation(len(all_images))
            new_images = [all_images[i] for i in indices[: self.num_images]]
        else:
            raise NotImplementedError(
                "COCOSubsetWithNImages doesn't support sampling method: {}".format(
                    self.sampling
                )
            )

        new_image_ids = {im["id"] for im in new_images}
        new_annotations = [
            ann for ann in json_dict["annotations"] if ann["image_id"] in new_image_ids
        ]
        json_dict["images"] = new_images
        json_dict["annotations"] = new_annotations
        return json_dict


class COCOSubsetWithGivenImages(AdhocCOCODataset):
    def __init__(self, src_ds_name, file_names, prefix="given"):
        super().__init__(
            src_ds_name=src_ds_name,
            new_ds_name="{}_{}{}".format(src_ds_name, prefix, len(file_names)),
        )

        self.file_names = file_names

    def new_json_dict(self, json_dict):
        all_images = json_dict["images"]

        file_name_to_im = {im["file_name"]: im for im in all_images}
        new_images = [file_name_to_im[file_name] for file_name in self.file_names]

        # re-assign image id to keep the order (COCO loads images by id order)
        old_id_to_new_id = {im["id"]: i for i, im in enumerate(new_images)}
        new_annotations = [
            ann
            for ann in json_dict["annotations"]
            if ann["image_id"] in old_id_to_new_id
        ]
        # update image id
        for im in new_images:
            im["id"] = old_id_to_new_id[im["id"]]
        for anno in new_annotations:
            anno["image_id"] = old_id_to_new_id[anno["image_id"]]
        json_dict["images"] = new_images
        json_dict["annotations"] = new_annotations
        return json_dict


class COCOWithClassesToUse(AdhocCOCODataset):
    def __init__(self, src_ds_name, classes_to_use):
        super().__init__(
            src_ds_name=src_ds_name,
            new_ds_name="{}@{}classes".format(src_ds_name, len(classes_to_use)),
        )
        self.classes_to_use = classes_to_use

    def new_json_dict(self, json_dict):
        categories = json_dict["categories"]
        new_categories = [
            cat for cat in categories if cat["name"] in self.classes_to_use
        ]
        new_category_ids = {cat["id"] for cat in new_categories}
        new_annotations = [
            ann
            for ann in json_dict["annotations"]
            if ann["category_id"] in new_category_ids
        ]
        json_dict["categories"] = new_categories
        json_dict["annotations"] = new_annotations
        return json_dict


class ClipLengthGroupedDataset(data.IterableDataset):
    """
    Batch data that have same clip length and similar aspect ratio.
    In this implementation, images with same length and whose aspect
    ratio < (or >) 1 will be batched together.
    This makes training with different clip length possible and improves
    training speed because the images then need less padding to form a batch.
    """

    def __init__(self, dataset, batch_size):
        """
        Args:
            dataset: an iterable. Each element must be a dict with keys
                "width" and "height", which will be used to batch data.
            batch_size (int):
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self._buckets = defaultdict(list)

    def __iter__(self):
        for d in self.dataset:
            clip_length = len(d["frames"])
            h, w = d["height"], d["width"]
            aspect_ratio_bucket_id = 0 if h > w else 1
            bucket = self._buckets[(clip_length, aspect_ratio_bucket_id)]
            bucket.append(d)

            if len(bucket) == self.batch_size:
                yield bucket[:]
                del bucket[:]


@contextlib.contextmanager
def register_sub_dataset_with_n_images(dataset_name, num_images, sampling):
    """
    Temporarily register a sub-dataset created from `dataset_name`, with the first
    `num_images` from it.
    """
    # when `num_images` is not larger than 0, return original dataset
    if num_images <= 0:
        yield dataset_name
        return

    # only support coco for now
    assert sampling in COCOSubsetWithNImages._SUPPORTED_SAMPLING

    new_dataset = COCOSubsetWithNImages(dataset_name, num_images, sampling)
    AdhocDatasetManager.add(new_dataset)
    try:
        yield new_dataset.new_ds_name
    finally:
        AdhocDatasetManager.remove(new_dataset)


@contextlib.contextmanager
def register_sub_dataset_with_given_images(*args, **kwargs):
    new_dataset = COCOSubsetWithGivenImages(*args, **kwargs)
    AdhocDatasetManager.add(new_dataset)
    AdhocDatasetManager.add(new_dataset)
    try:
        yield new_dataset.new_ds_name
    finally:
        AdhocDatasetManager.remove(new_dataset)


@contextlib.contextmanager
def maybe_subsample_n_images(cfg, is_train=False):
    """
    Create a new config whose train/test datasets only take a subsample of
    `max_images` image. Use all images (non-op) when `max_images` <= 0.
    """
    max_images = cfg.D2GO_DATA.TEST.MAX_IMAGES
    sampling = cfg.D2GO_DATA.TEST.SUBSET_SAMPLING
    with contextlib.ExitStack() as stack:  # python 3.3+
        new_splits = tuple(
            stack.enter_context(
                register_sub_dataset_with_n_images(ds, max_images, sampling)
            )
            for ds in (cfg.DATASETS.TRAIN if is_train else cfg.DATASETS.TEST)
        )
        new_cfg = cfg.clone()
        with temp_defrost(new_cfg):
            if is_train:
                new_cfg.DATASETS.TRAIN = new_splits
            else:
                new_cfg.DATASETS.TEST = new_splits
        yield new_cfg


def update_cfg_if_using_adhoc_dataset(cfg):
    if cfg.D2GO_DATA.DATASETS.TRAIN_CATEGORIES:
        new_train_datasets = [
            COCOWithClassesToUse(name, cfg.D2GO_DATA.DATASETS.TRAIN_CATEGORIES)
            for name in cfg.DATASETS.TRAIN
        ]
        [AdhocDatasetManager.add(new_ds) for new_ds in new_train_datasets]
        with temp_defrost(cfg):
            cfg.DATASETS.TRAIN = tuple(ds.new_ds_name for ds in new_train_datasets)

    if cfg.D2GO_DATA.DATASETS.TEST_CATEGORIES:
        new_test_datasets = [
            COCOWithClassesToUse(ds, cfg.D2GO_DATA.DATASETS.TEST_CATEGORIES)
            for ds in cfg.DATASETS.TEST
        ]
        [AdhocDatasetManager.add(new_ds) for new_ds in new_test_datasets]
        with temp_defrost(cfg):
            cfg.DATASETS.TEST = tuple(ds.new_ds_name for ds in new_test_datasets)

    return cfg
