#!/usr/bin/env python3

import os
import unittest

from d2go.runner import GeneralizedRCNNRunner, create_runner
from mobile_cv.common.misc.file_utils import make_temp_directory
from PIL import Image

from d2go.tests.data_loader_helper import LocalImageGenerator, register_toy_dataset


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

        with make_temp_directory("detectron2go_tmp_dataset") as dataset_dir:
            image_dir = os.path.join(dataset_dir, "images")
            os.makedirs(image_dir)
            image_generator = LocalImageGenerator(image_dir, width=80, height=60)

            with register_toy_dataset(
                "default_dataset_train", image_generator, num_images=3
            ):
                train_loader = runner.build_detection_train_loader(cfg)
                for i, data in enumerate(train_loader):
                    self.assertIsNotNone(data)
                    # for training loader, it has infinite length
                    if i == 6:
                        break

            with register_toy_dataset(
                "default_dataset_test", image_generator, num_images=3
            ):
                test_loader = runner.build_detection_test_loader(
                    cfg, dataset_name="default_dataset_test"
                )
                all_data = []
                for data in test_loader:
                    all_data.append(data)
                self.assertEqual(len(all_data), 3)
