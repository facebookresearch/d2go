#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved


import unittest

from d2go.runner import create_runner
from d2go.utils.testing.data_loader_helper import register_toy_coco_dataset


class TestD2GoDatasetMapper(unittest.TestCase):
    """
    This class test D2GoDatasetMapper which is used to build
    data loader in GeneralizedRCNNRunner (the default runner) in Detectron2Go.
    """

    def test_default_dataset(self):
        runner = create_runner("d2go.runner.GeneralizedRCNNRunner")
        cfg = runner.get_default_cfg()
        cfg.DATASETS.TRAIN = ["default_dataset_train"]
        cfg.DATASETS.TEST = ["default_dataset_test"]

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
