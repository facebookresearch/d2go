#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved


import os
import shutil
import tempfile
import unittest

import torch
from d2go.data.disk_cache import DiskCachedList, ROOT_CACHE_DIR
from d2go.data.utils import configure_dataset_creation
from d2go.runner import create_runner
from d2go.utils.testing.data_loader_helper import (
    create_detection_data_loader_on_toy_dataset,
    register_toy_coco_dataset,
)


class TestD2GoDatasetMapper(unittest.TestCase):
    """
    This class test D2GoDatasetMapper which is used to build
    data loader in GeneralizedRCNNRunner (the default runner) in Detectron2Go.
    """

    def setUp(self):
        self.output_dir = tempfile.mkdtemp(prefix="TestD2GoDatasetMapper_")
        self.addCleanup(shutil.rmtree, self.output_dir)

    def test_default_dataset(self):
        runner = create_runner("d2go.runner.GeneralizedRCNNRunner")
        cfg = runner.get_default_cfg()
        cfg.DATASETS.TRAIN = ["default_dataset_train"]
        cfg.DATASETS.TEST = ["default_dataset_test"]
        cfg.OUTPUT_DIR = self.output_dir

        with register_toy_coco_dataset("default_dataset_train", num_images=3):
            train_loader = runner.build_detection_train_loader(cfg)
            for i, data in enumerate(train_loader):
                self.assertIsNotNone(data)
                # for training loader, it has infinite length
                if i == 6:
                    break

        with register_toy_coco_dataset("default_dataset_test", num_images=3):
            test_loader = runner.build_detection_test_loader(
                cfg, dataset_name="default_dataset_test"
            )
            all_data = []
            for data in test_loader:
                all_data.append(data)
            self.assertEqual(len(all_data), 3)


class _MyClass(object):
    def __init__(self, x):
        self.x = x

    def do_something(self):
        return


class TestDiskCachedDataLoader(unittest.TestCase):
    def setUp(self):
        # make sure the ROOT_CACHE_DIR is empty when entering the test
        if os.path.exists(ROOT_CACHE_DIR):
            shutil.rmtree(ROOT_CACHE_DIR)

        self.output_dir = tempfile.mkdtemp(prefix="TestDiskCachedDataLoader_")
        self.addCleanup(shutil.rmtree, self.output_dir)

    def _count_cache_dirs(self):
        if not os.path.exists(ROOT_CACHE_DIR):
            return 0

        return len(os.listdir(ROOT_CACHE_DIR))

    def test_disk_cached_dataset_from_list(self):
        """Test the class of DiskCachedList"""
        # check the discache can handel different data types
        lst = [1, torch.tensor(2), _MyClass(3)]
        disk_cached_lst = DiskCachedList(lst)
        self.assertEqual(len(disk_cached_lst), 3)
        self.assertEqual(disk_cached_lst[0], 1)
        self.assertEqual(disk_cached_lst[1].item(), 2)
        self.assertEqual(disk_cached_lst[2].x, 3)

        # check the cache is created
        cache_dir = disk_cached_lst.cache_dir
        self.assertTrue(os.path.isdir(cache_dir))

        # check the cache is properly released
        del disk_cached_lst
        self.assertFalse(os.path.isdir(cache_dir))

    def test_disk_cached_dataloader(self):
        """Test the data loader backed by disk cache"""
        height = 6
        width = 8
        runner = create_runner("d2go.runner.GeneralizedRCNNRunner")
        cfg = runner.get_default_cfg()
        cfg.OUTPUT_DIR = self.output_dir
        cfg.DATALOADER.NUM_WORKERS = 2

        def _test_data_loader(data_loader):
            first_batch = next(iter(data_loader))
            self.assertTrue(first_batch[0]["height"], height)
            self.assertTrue(first_batch[0]["width"], width)

        # enable the disk cache
        cfg.merge_from_list(["D2GO_DATA.DATASETS.DISK_CACHE.ENABLED", "True"])
        with configure_dataset_creation(cfg):
            # no cache dir in the beginning
            self.assertEqual(self._count_cache_dirs(), 0)

            with create_detection_data_loader_on_toy_dataset(
                cfg, height, width, is_train=True
            ) as train_loader:
                # train loader should create one cache dir
                self.assertEqual(self._count_cache_dirs(), 1)

                _test_data_loader(train_loader)

                with create_detection_data_loader_on_toy_dataset(
                    cfg, height, width, is_train=False
                ) as test_loader:
                    # test loader should create another cache dir
                    self.assertEqual(self._count_cache_dirs(), 2)

                    _test_data_loader(test_loader)

                # test loader should release its cache
                del test_loader
                self.assertEqual(self._count_cache_dirs(), 1)

            # no cache dir in the end
            del train_loader
            self.assertEqual(self._count_cache_dirs(), 0)
