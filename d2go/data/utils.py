#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved


import atexit
import contextlib
import json
import logging
import os
import pickle
import re
import shutil
import tempfile
import uuid
from collections import defaultdict
from unittest import mock

import numpy as np
import torch.utils.data as data
from d2go.config import temp_defrost
from d2go.data.datasets import (
    register_dataset_split,
    ANN_FN,
    IM_DIR,
    INJECTED_COCO_DATASETS_LUT,
)
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.build import (
    get_detection_dataset_dicts as d2_get_detection_dataset_dicts,
)
from detectron2.utils import comm
from detectron2.utils.file_io import PathManager
from detectron2.utils.logger import log_every_n_seconds

logger = logging.getLogger(__name__)


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
        elif self.src_ds_name in INJECTED_COCO_DATASETS_LUT:
            _src_func, _src_dict = INJECTED_COCO_DATASETS_LUT[self.src_ds_name]
            _src_func(
                self.new_ds_name,
                split_dict={**_src_dict, ANN_FN: tmp_file, IM_DIR: metadata.image_root},
            )
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
        if MetadataCatalog.get(self.new_ds_name):
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
        # check if name is already a derived class and try to reverse it
        res = re.match("(?P<src>.+)@(?P<num>[0-9]+)classes", src_ds_name)
        if res is not None:
            src_ds_name = res["src"]

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


def _local_master_gather(func, check_equal=False):
    if comm.get_local_rank() == 0:
        x = func()
        assert x is not None
    else:
        x = None
    x_all = comm.all_gather(x)
    x_local_master = [x for x in x_all if x is not None]

    if check_equal:
        master = x_local_master[0]
        assert all(x == master for x in x_local_master), x_local_master

    return x_local_master


class DiskCachedDatasetFromList(data.Dataset):
    """
    Wrap a list to a torch Dataset, the underlying storage is off-loaded to disk to
    save RAM usage.
    """

    CACHE_DIR = "/tmp/DatasetFromList_cache"
    _OCCUPIED_CACHE_DIRS = set()

    def __init__(self, lst, strategy="batched_static"):
        """
        Args:
            lst (list): a list which contains elements to produce.
            strategy (str): strategy of using diskcache, supported strategies:
                - native: saving each item individually.
                - batched_static: group N items together, where N is calculated from
                    the average item size.
        """
        self._lst = lst
        self._diskcache_strategy = strategy

        def _serialize(data):
            buffer = pickle.dumps(data, protocol=-1)
            return np.frombuffer(buffer, dtype=np.uint8)

        logger.info(
            "Serializing {} elements to byte tensors and concatenating them all ...".format(
                len(self._lst)
            )
        )
        self._lst = [_serialize(x) for x in self._lst]
        # TODO: only enabling DiskCachedDataset for large enough dataset
        logger.info(
            "Serialized dataset takes {:.2f} MiB".format(len(self._lst) / 1024 ** 2)
        )
        self._initialize_diskcache()

    def _initialize_diskcache(self):
        from mobile_cv.common.misc.local_cache import LocalCache

        cache_dir = "{}/{}".format(
            DiskCachedDatasetFromList.CACHE_DIR, uuid.uuid4().hex[:8]
        )
        cache_dir = comm.all_gather(cache_dir)[0]  # use same cache_dir
        logger.info("Creating diskcache database in: {}".format(cache_dir))
        self._cache = LocalCache(cache_dir=cache_dir, num_shards=8)
        # self._cache.cache.clear(retry=True)  # seems faster if index exists

        if comm.get_local_rank() == 0:
            DiskCachedDatasetFromList.get_all_cache_dirs().add(self._cache.cache_dir)

            if self._diskcache_strategy == "naive":
                for i, item in enumerate(self._lst):
                    ret = self._write_to_local_db((i, item))
                    assert ret, "Error writing index {} to local db".format(i)
                    pct = 100.0 * i / len(self._lst)
                    self._log_progress(pct)

            # NOTE: each item might be small in size (hundreds of bytes),
            # writing million of them can take a pretty long time (hours)
            # because of frequent disk access. One solution is grouping a batch
            # of items into larger blob.
            elif self._diskcache_strategy == "batched_static":
                TARGET_BYTES = 50 * 1024
                average_bytes = np.average(
                    [
                        self._lst[int(x)].size
                        for x in np.linspace(0, len(self._lst) - 1, 1000)
                    ]
                )
                self._chuck_size = max(1, int(TARGET_BYTES / average_bytes))
                logger.info(
                    "Average data size: {} bytes; target chuck data size {} KiB;"
                    " {} items per chuck; {} chucks in total".format(
                        average_bytes,
                        TARGET_BYTES / 1024,
                        self._chuck_size,
                        int(len(self._lst) / self._chuck_size),
                    )
                )
                for i in range(0, len(self._lst), self._chuck_size):
                    chunk = self._lst[i : i + self._chuck_size]
                    chunk_i = int(i / self._chuck_size)
                    ret = self._write_to_local_db((chunk_i, chunk))
                    assert ret, "Error writing index {} to local db".format(chunk_i)
                    pct = 100.0 * i / len(self._lst)
                    self._log_progress(pct)

            # NOTE: instead of using fixed chuck size, items can be grouped dynamically
            elif self._diskcache_strategy == "batched_dynamic":
                raise NotImplementedError()

            else:
                raise NotImplementedError(self._diskcache_strategy)

        comm.synchronize()
        logger.info(
            "Finished writing to local disk, db size: {:.2f} MiB".format(
                self._cache.cache.volume() / 1024 ** 2
            )
        )
        # Optional sync for some strategies
        if self._diskcache_strategy == "batched_static":
            # propagate chuck size and make sure all local rank 0 uses the same value
            self._chuck_size = _local_master_gather(
                lambda: self._chuck_size, check_equal=True
            )[0]

        # free the memory of self._lst
        self._size = _local_master_gather(lambda: len(self._lst), check_equal=True)[0]
        del self._lst

    def _write_to_local_db(self, task):
        index, record = task
        db_path = str(index)
        # suc = self._cache.load(lambda path, x: x, db_path, record)
        # record = BytesIO(np.random.bytes(np.random.randint(70000, 90000)))
        suc = self._cache.cache.set(db_path, record, retry=True)
        return suc

    def _log_progress(self, percentage):
        log_every_n_seconds(
            logging.INFO,
            "({:.2f}%) Wrote {} elements to local disk cache, db size: {:.2f} MiB".format(
                percentage,
                len(self._cache.cache),
                self._cache.cache.volume() / 1024 ** 2,
            ),
            n=10,
        )

    def __len__(self):
        if self._diskcache_strategy == "batched_static":
            return self._size
        else:
            raise NotImplementedError()

    def __getitem__(self, idx):
        if self._diskcache_strategy == "naive":
            bytes = memoryview(self._cache.cache[str(idx)])
            return pickle.loads(bytes)

        elif self._diskcache_strategy == "batched_static":
            chunk_i, residual = divmod(idx, self._chuck_size)
            chunk = self._cache.cache[str(chunk_i)]
            bytes = memoryview(chunk[residual])
            return pickle.loads(bytes)

        else:
            raise NotImplementedError()

    @classmethod
    def get_all_cache_dirs(cls):
        """return all the ocupied cache dirs of DiskCachedDatasetFromList"""
        return DiskCachedDatasetFromList._OCCUPIED_CACHE_DIRS

    def get_cache_dir(self):
        """return the current cache dirs of DiskCachedDatasetFromList instance"""
        return self._cache.cache_dir

    @staticmethod
    def _clean_up_cache_dir(cache_dir, **kwargs):
        print("Cleaning up cache dir: {}".format(cache_dir))
        shutil.rmtree(
            cache_dir,
            onerror=lambda func, path, ex: print(
                "Catch error when removing {}; func: {}; exc_info: {}".format(
                    path, func, ex
                )
            ),
        )

    @staticmethod
    @atexit.register
    def _clean_up_all():
        # in case the program exists unexpectly, clean all the cache dirs created by
        # this session.
        if comm.get_local_rank() == 0:
            for cache_dir in DiskCachedDatasetFromList.get_all_cache_dirs():
                DiskCachedDatasetFromList._clean_up_cache_dir(cache_dir)

    def __del__(self):
        # when data loader goes are GC-ed, remove the cache dir. This is needed to not
        # waste disk space in case that multiple data loaders are used, eg. running
        # evaluations on multiple datasets during training.
        if comm.get_local_rank() == 0:
            DiskCachedDatasetFromList._clean_up_cache_dir(self._cache.cache_dir)
            DiskCachedDatasetFromList.get_all_cache_dirs().remove(self._cache.cache_dir)


class _FakeListObj(list):
    def __init__(self, size):
        self.size = size

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        raise NotImplementedError()


def local_master_get_detection_dataset_dicts(*args, **kwargs):
    logger.info("Only load dataset dicts on local master process ...")

    dataset_dicts = (
        d2_get_detection_dataset_dicts(*args, **kwargs)
        if comm.get_local_rank() == 0
        else []
    )
    comm.synchronize()
    dataset_size = comm.all_gather(len(dataset_dicts))[0]

    if comm.get_local_rank() != 0:
        dataset_dicts = _FakeListObj(dataset_size)
    return dataset_dicts


@contextlib.contextmanager
def enable_disk_cached_dataset(cfg):
    """
    Context manager for enabling disk cache datasets, this is a experimental feature.

    - Replace D2's DatasetFromList with DiskCachedDatasetFromList, needs to patch all
        call sites.
    - Replace D2's get_detection_dataset_dicts with a local-master-only version.
    """

    if not cfg.D2GO_DATA.DATASETS.DISK_CACHE.ENABLED:
        yield
        return

    def _patched_dataset_from_list(lst, **kwargs):
        logger.info("Patch DatasetFromList with DiskCachedDatasetFromList")
        return DiskCachedDatasetFromList(lst)

    with contextlib.ExitStack() as stack:
        for ctx in [
            mock.patch(
                "detectron2.data.build.get_detection_dataset_dicts",
                side_effect=local_master_get_detection_dataset_dicts,
            ),
            mock.patch(
                "detectron2.data.build.DatasetFromList",
                side_effect=_patched_dataset_from_list,
            ),
            mock.patch(
                "d2go.data.build.DatasetFromList",
                side_effect=_patched_dataset_from_list,
            ),
            mock.patch(
                "d2go.data.build_fb.DatasetFromList",
                side_effect=_patched_dataset_from_list,
            ),
        ]:
            stack.enter_context(ctx)
        yield
