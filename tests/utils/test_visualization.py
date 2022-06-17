#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved


import copy
import json
import os
import unittest
from typing import Dict, List, Optional, Tuple

import d2go.runner.default_runner as default_runner
import numpy as np
import torch
from d2go.registry.builtin import META_ARCH_REGISTRY
from d2go.utils.testing.data_loader_helper import (
    create_toy_dataset,
    LocalImageGenerator,
)
from d2go.utils.testing.helper import tempdir
from d2go.utils.visualization import DataLoaderVisWrapper, VisualizerWrapper
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.structures import Boxes, Instances
from detectron2.utils.events import EventStorage


def create_test_images_and_dataset_json(
    data_dir: str, img_w: int, img_h: int, num_images: int = 10, num_classes: int = -1
) -> Tuple[str, str]:
    # create image and json
    image_dir = os.path.join(data_dir, "images")
    os.makedirs(image_dir)
    json_dataset, meta_data = create_toy_dataset(
        LocalImageGenerator(image_dir, width=img_w, height=img_h),
        num_images=num_images,
        num_classes=num_classes,
    )
    json_file = os.path.join(data_dir, "annotation.json")

    with open(json_file, "w") as f:
        json.dump(json_dataset, f)

    return image_dir, json_file


def create_dummy_input_dict(
    img_w: int = 60, img_h: int = 60, bbox_list: Optional[List[List[int]]] = None
) -> Dict:
    # Create dummy data
    instance = Instances((img_w, img_h))
    if bbox_list is not None:
        instance.gt_boxes = Boxes(torch.tensor([[10, 10, 20, 20]]))
        instance.gt_classes = torch.tensor([0])
    input_dict = {"image": torch.zeros(3, img_w, img_h), "instances": instance}
    return input_dict


@META_ARCH_REGISTRY.register()
class DummyMetaArch(torch.nn.Module):
    @staticmethod
    def visualize_train_input(visualizer_wrapper, input_dict):
        return {"default": np.zeros((60, 60, 30)), "secondary": np.zeros((60, 60, 30))}


class ImageDictStore:
    def __init__(self):
        self.write_buffer = []

    def add_image(self, **kwargs):
        self.write_buffer.append(copy.deepcopy(kwargs))


class MockTbxWriter:
    def __init__(self):
        self._writer = ImageDictStore()


class TestVisualization(unittest.TestCase):
    def setUp(self):
        self._builtin_datasets = set(DatasetCatalog)

    def tearDown(self):
        # Need to remove injected dataset
        injected_dataset = set(DatasetCatalog) - self._builtin_datasets
        for ds in injected_dataset:
            DatasetCatalog.remove(ds)
            MetadataCatalog.remove(ds)

    @tempdir
    def test_visualizer_wrapper(self, tmp_dir: str):
        image_dir, json_file = create_test_images_and_dataset_json(tmp_dir, 60, 60)

        # Create config data
        runner = default_runner.Detectron2GoRunner()
        cfg = runner.get_default_cfg()
        cfg.merge_from_list(
            [
                "D2GO_DATA.DATASETS.COCO_INJECTION.NAMES",
                str(["inj_ds1"]),
                "D2GO_DATA.DATASETS.COCO_INJECTION.IM_DIRS",
                str([image_dir]),
                "D2GO_DATA.DATASETS.COCO_INJECTION.JSON_FILES",
                str([json_file]),
                "DATASETS.TRAIN",
                str(["inj_ds1"]),
            ]
        )

        # Register configs
        runner.register(cfg)
        DatasetCatalog.get("inj_ds1")

        # Create dummy data to pass to wrapper
        input_dict = create_dummy_input_dict(60, 60, [[10, 10, 20, 20]])
        vis_wrapper = VisualizerWrapper(cfg)
        vis_image = vis_wrapper.visualize_train_input(input_dict)
        # Visualize train by default scales input image by 2
        self.assertTrue(any(vis_image[20, 20, :] != 0))
        self.assertFalse(any(vis_image[30, 30, :] != 0))
        self.assertTrue(any(vis_image[40, 40, :] != 0))

    @tempdir
    def test_dataloader_visualizer_wrapper(self, tmp_dir: str):
        image_dir, json_file = create_test_images_and_dataset_json(tmp_dir, 60, 60)

        # Create config data
        runner = default_runner.Detectron2GoRunner()
        cfg = runner.get_default_cfg()
        cfg.merge_from_list(
            [
                "D2GO_DATA.DATASETS.COCO_INJECTION.NAMES",
                str(["inj_ds2"]),
                "D2GO_DATA.DATASETS.COCO_INJECTION.IM_DIRS",
                str([image_dir]),
                "D2GO_DATA.DATASETS.COCO_INJECTION.JSON_FILES",
                str([json_file]),
                "DATASETS.TRAIN",
                str(["inj_ds2"]),
            ]
        )

        # Register configs
        runner.register(cfg)
        DatasetCatalog.get("inj_ds2")

        with EventStorage():
            # Create mock storage for writer
            mock_tbx_writer = MockTbxWriter()
            # Create a wrapper around an iterable object and run once
            input_dict = create_dummy_input_dict(60, 60, [[1, 1, 2, 2]])
            dl_wrapper = DataLoaderVisWrapper(
                cfg, mock_tbx_writer, [[input_dict], [input_dict]]
            )
            for _ in dl_wrapper:
                break

            # Check data has been written to buffer
            self.assertTrue(len(mock_tbx_writer._writer.write_buffer) == 1)
            vis_image_dict = mock_tbx_writer._writer.write_buffer[0]
            self.assertTrue("tag" in vis_image_dict)
            self.assertTrue("img_tensor" in vis_image_dict)
            self.assertTrue("global_step" in vis_image_dict)

    @tempdir
    def test_dict_based_dataloader_visualizer_wrapper(self, tmp_dir: str):
        image_dir, json_file = create_test_images_and_dataset_json(tmp_dir, 60, 60)

        # Create config data
        runner = default_runner.Detectron2GoRunner()
        cfg = runner.get_default_cfg()
        cfg.merge_from_list(
            [
                "D2GO_DATA.DATASETS.COCO_INJECTION.NAMES",
                str(["inj_ds3"]),
                "D2GO_DATA.DATASETS.COCO_INJECTION.IM_DIRS",
                str([image_dir]),
                "D2GO_DATA.DATASETS.COCO_INJECTION.JSON_FILES",
                str([json_file]),
                "DATASETS.TRAIN",
                str(["inj_ds3"]),
                "MODEL.META_ARCHITECTURE",
                "DummyMetaArch",
            ]
        )

        # Register configs
        runner.register(cfg)
        DatasetCatalog.get("inj_ds3")

        with EventStorage():
            # Create mock storage for writer
            mock_tbx_writer = MockTbxWriter()
            # Create a wrapper around an iterable object and run once
            input_dict = create_dummy_input_dict(60, 60, [[1, 1, 2, 2]])
            dl_wrapper = DataLoaderVisWrapper(
                cfg, mock_tbx_writer, [[input_dict], [input_dict]]
            )
            for _ in dl_wrapper:
                break

            # Check data has been written to buffer
            self.assertTrue(len(mock_tbx_writer._writer.write_buffer) == 2)
            self.assertTrue(
                "train_loader_batch_0/default"
                in mock_tbx_writer._writer.write_buffer[0]["tag"]
            )
            self.assertTrue(
                "train_loader_batch_0/secondary"
                in mock_tbx_writer._writer.write_buffer[1]["tag"]
            )
