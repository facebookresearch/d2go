#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import copy
import json
import os
import tempfile
import unittest

import d2go.data.extended_coco as extended_coco
from d2go.data.datasets import ANN_FN, COCO_REGISTER_FUNCTION_REGISTRY, IM_DIR
from d2go.data.keypoint_metadata_registry import (
    get_keypoint_metadata,
    KEYPOINT_METADATA_REGISTRY,
    KeypointMetadata,
)
from d2go.data.utils import (
    AdhocDatasetManager,
    COCOWithClassesToUse,
    maybe_subsample_n_images,
)
from d2go.runner import Detectron2GoRunner
from d2go.utils.testing.data_loader_helper import (
    create_toy_dataset,
    LocalImageGenerator,
)
from d2go.utils.testing.helper import tempdir
from detectron2.data import DatasetCatalog, MetadataCatalog
from mobile_cv.common.misc.file_utils import make_temp_directory


def create_test_images_and_dataset_json(data_dir, num_images=10, num_classes=-1):
    # create image and json
    image_dir = os.path.join(data_dir, "images")
    os.makedirs(image_dir)
    json_dataset, meta_data = create_toy_dataset(
        LocalImageGenerator(image_dir, width=80, height=60),
        num_images=num_images,
        num_classes=num_classes,
    )
    json_file = os.path.join(data_dir, "annotation.json")
    with open(json_file, "w") as f:
        json.dump(json_dataset, f)

    return image_dir, json_file


class TestD2GoDatasets(unittest.TestCase):
    def setUp(self):
        self._builtin_datasets = set(DatasetCatalog)

    def tearDown(self):
        # Need to remove injected dataset
        injected_dataset = set(DatasetCatalog) - self._builtin_datasets
        for ds in injected_dataset:
            DatasetCatalog.remove(ds)
            MetadataCatalog.remove(ds)

    def test_coco_conversions(self):
        test_data_0 = {
            "info": {},
            "imgs": {
                "img_1": {
                    "file_name": "0.jpg",
                    "width": 600,
                    "height": 600,
                    "id": "img_1",
                }
            },
            "anns": {0: {"id": 0, "image_id": "img_1", "bbox": [30, 30, 60, 20]}},
            "imgToAnns": {"img_1": [0]},
            "cats": {},
        }
        test_data_1 = copy.deepcopy(test_data_0)
        test_data_1["imgs"][123] = test_data_1["imgs"].pop("img_1")
        test_data_1["imgs"][123]["id"] = 123
        test_data_1["anns"][0]["image_id"] = 123
        test_data_1["imgToAnns"][123] = test_data_1["imgToAnns"].pop("img_1")

        for test_data, exp_output in [(test_data_0, [0, 0]), (test_data_1, [123, 123])]:
            with make_temp_directory("detectron2go_tmp_dataset") as tmp_dir:
                src_json = os.path.join(tmp_dir, "source.json")
                out_json = os.path.join(tmp_dir, "output.json")

                with open(src_json, "w") as h_in:
                    json.dump(test_data, h_in)

                out_json = extended_coco.convert_coco_text_to_coco_detection_json(
                    src_json, out_json
                )

                self.assertEqual(out_json["images"][0]["id"], exp_output[0])
                self.assertEqual(out_json["annotations"][0]["image_id"], exp_output[1])

    def test_annotation_rejection(self):
        img_list = [
            {"id": 0, "width": 50, "height": 50, "file_name": "a.png"},
            {"id": 1, "width": 50, "height": 50, "file_name": "b.png"},
            {"id": 2, "width": 50, "height": 50, "file_name": "b.png"},
        ]
        ann_list = [
            [
                {
                    "id": 0,
                    "image_id": 0,
                    "category_id": 0,
                    "segmentation": [[0, 0, 10, 0, 10, 10, 0, 10]],
                    "area": 100,
                    "bbox": [0, 0, 10, 10],
                },
                {
                    "id": 1,
                    "image_id": 0,
                    "category_id": 0,
                    "segmentation": [[0, 0, 10, 0, 10, 10, 0, 10]],
                    "area": 100,
                    "bbox": [45, 45, 10, 10],
                },
                {
                    "id": 2,
                    "image_id": 0,
                    "category_id": 0,
                    "segmentation": [[0, 0, 10, 0, 10, 10, 0, 10]],
                    "area": 100,
                    "bbox": [-5, -5, 10, 10],
                },
                {
                    "id": 3,
                    "image_id": 0,
                    "category_id": 0,
                    "segmentation": [[0, 0, 10, 0, 10, 10, 0, 10]],
                    "area": 0,
                    "bbox": [5, 5, 0, 0],
                },
                {
                    "id": 4,
                    "image_id": 0,
                    "category_id": 0,
                    "segmentation": [[]],
                    "area": 25,
                    "bbox": [5, 5, 5, 5],
                },
            ],
            [
                {
                    "id": 5,
                    "image_id": 1,
                    "category_id": 0,
                    "segmentation": [[]],
                    "area": 100,
                    "bbox": [0, 0, 0, 0],
                },
            ],
            [],
        ]

        out_dict_list = extended_coco.convert_to_dict_list("", [0], img_list, ann_list)
        self.assertEqual(len(out_dict_list), 1)
        self.assertEqual(len(out_dict_list[0]["annotations"]), 1)

        out_dict_list = extended_coco.convert_to_dict_list(
            "", [0], img_list, ann_list, filter_empty_annotations=False
        )
        self.assertEqual(len(out_dict_list), 3)

    @tempdir
    def test_coco_injection(self, tmp_dir):
        image_dir, json_file = create_test_images_and_dataset_json(tmp_dir)

        runner = Detectron2GoRunner()
        cfg = runner.get_default_cfg()
        cfg.merge_from_list(
            [
                str(x)
                for x in [
                    "D2GO_DATA.DATASETS.COCO_INJECTION.NAMES",
                    ["inj_ds1", "inj_ds2"],
                    "D2GO_DATA.DATASETS.COCO_INJECTION.IM_DIRS",
                    [image_dir, "/mnt/fair"],
                    "D2GO_DATA.DATASETS.COCO_INJECTION.JSON_FILES",
                    [json_file, "inj_ds2"],
                ]
            ]
        )

        runner.register(cfg)
        inj_ds1 = DatasetCatalog.get("inj_ds1")
        self.assertEqual(len(inj_ds1), 10)
        for dic in inj_ds1:
            self.assertEqual(dic["width"], 80)
            self.assertEqual(dic["height"], 60)

    @tempdir
    def test_direct_copy_keys(self, tmp_dir):
        image_dir, json_file = create_test_images_and_dataset_json(tmp_dir)
        with tempfile.NamedTemporaryFile(prefix=tmp_dir, suffix=".json") as h_temp:
            new_json_file = h_temp.name
            with open(json_file, "r") as h_in:
                ds = json.load(h_in)
                for idx, x in enumerate(ds["images"]):
                    x["key1"] = idx
                    x["key2"] = idx
                with open(new_json_file, "w") as h_out:
                    json.dump(ds, h_out)

            loaded_ds = extended_coco.extended_coco_load(new_json_file, image_dir)
            self.assertTrue("key1" not in loaded_ds[0])
            self.assertTrue("key2" not in loaded_ds[0])

            loaded_ds = extended_coco.extended_coco_load(
                new_json_file, image_dir, image_direct_copy_keys=["key1"]
            )
            self.assertTrue("key1" in loaded_ds[0])
            self.assertTrue("key2" not in loaded_ds[0])

    @tempdir
    def test_sub_dataset(self, tmp_dir):
        image_dir, json_file = create_test_images_and_dataset_json(tmp_dir)

        runner = Detectron2GoRunner()
        cfg = runner.get_default_cfg()
        cfg.merge_from_list(
            [
                str(x)
                for x in [
                    "D2GO_DATA.DATASETS.COCO_INJECTION.NAMES",
                    ["inj_ds3"],
                    "D2GO_DATA.DATASETS.COCO_INJECTION.IM_DIRS",
                    [image_dir],
                    "D2GO_DATA.DATASETS.COCO_INJECTION.JSON_FILES",
                    [json_file],
                    "DATASETS.TEST",
                    ("inj_ds3",),
                    "D2GO_DATA.TEST.MAX_IMAGES",
                    1,
                ]
            ]
        )

        runner.register(cfg)
        with maybe_subsample_n_images(cfg) as new_cfg:
            test_loader = runner.build_detection_test_loader(
                new_cfg, new_cfg.DATASETS.TEST[0]
            )
            self.assertEqual(len(test_loader), 1)

    def test_coco_metadata_registry(self):
        @KEYPOINT_METADATA_REGISTRY.register()
        def TriangleMetadata():
            return KeypointMetadata(
                names=("A", "B", "C"),
                flip_map=(
                    ("A", "B"),
                    ("B", "C"),
                ),
                connection_rules=[
                    ("A", "B", (102, 204, 255)),
                    ("B", "C", (51, 153, 255)),
                ],
            )

        tri_md = get_keypoint_metadata("TriangleMetadata")
        self.assertEqual(tri_md["keypoint_names"][0], "A")
        self.assertEqual(tri_md["keypoint_flip_map"][0][0], "A")
        self.assertEqual(tri_md["keypoint_connection_rules"][0][0], "A")

    @tempdir
    def test_coco_metadata_register(self, tmp_dir):
        @KEYPOINT_METADATA_REGISTRY.register()
        def LineMetadata():
            return KeypointMetadata(
                names=("A", "B"),
                flip_map=(("A", "B"),),
                connection_rules=[
                    ("A", "B", (102, 204, 255)),
                ],
            )

        image_dir, json_file = create_test_images_and_dataset_json(tmp_dir)

        runner = Detectron2GoRunner()
        cfg = runner.get_default_cfg()
        cfg.merge_from_list(
            [
                str(x)
                for x in [
                    "D2GO_DATA.DATASETS.COCO_INJECTION.NAMES",
                    ["inj_ds"],
                    "D2GO_DATA.DATASETS.COCO_INJECTION.IM_DIRS",
                    [image_dir],
                    "D2GO_DATA.DATASETS.COCO_INJECTION.JSON_FILES",
                    [json_file],
                    "D2GO_DATA.DATASETS.COCO_INJECTION.KEYPOINT_METADATA",
                    ["LineMetadata"],
                ]
            ]
        )
        runner.register(cfg)
        inj_md = MetadataCatalog.get("inj_ds")
        self.assertEqual(inj_md.keypoint_names[0], "A")
        self.assertEqual(inj_md.keypoint_flip_map[0][0], "A")
        self.assertEqual(inj_md.keypoint_connection_rules[0][0], "A")

    @tempdir
    def test_coco_create_adhoc_class_to_use_dataset(self, tmp_dir):

        image_dir, json_file = create_test_images_and_dataset_json(
            tmp_dir, num_classes=2
        )

        runner = Detectron2GoRunner()
        cfg = runner.get_default_cfg()
        cfg.merge_from_list(
            [
                str(x)
                for x in [
                    "D2GO_DATA.DATASETS.COCO_INJECTION.NAMES",
                    ["test_adhoc_ds", "test_adhoc_ds2"],
                    "D2GO_DATA.DATASETS.COCO_INJECTION.IM_DIRS",
                    [image_dir, image_dir],
                    "D2GO_DATA.DATASETS.COCO_INJECTION.JSON_FILES",
                    [json_file, json_file],
                ]
            ]
        )
        runner.register(cfg)

        # Test adhoc classes to use
        AdhocDatasetManager.add(COCOWithClassesToUse("test_adhoc_ds", ["class_0"]))
        ds_list = DatasetCatalog.get("test_adhoc_ds@1classes")
        self.assertEqual(len(ds_list), 5)

        # Test adhoc classes to use with suffix removal
        AdhocDatasetManager.add(
            COCOWithClassesToUse("test_adhoc_ds2@1classes", ["class_0"])
        )
        ds_list = DatasetCatalog.get("test_adhoc_ds2@1classes")
        self.assertEqual(len(ds_list), 5)

    @tempdir
    def test_register_coco_dataset_registry(self, tmp_dir):

        dummy_buffer = []

        @COCO_REGISTER_FUNCTION_REGISTRY.register()
        def _register_dummy_function_coco(dataset_name, split_dict):
            dummy_buffer.append((dataset_name, split_dict))

        image_dir, json_file = create_test_images_and_dataset_json(tmp_dir)

        runner = Detectron2GoRunner()
        cfg = runner.get_default_cfg()
        cfg.merge_from_list(
            [
                str(x)
                for x in [
                    "D2GO_DATA.DATASETS.COCO_INJECTION.NAMES",
                    ["inj_test_registry"],
                    "D2GO_DATA.DATASETS.COCO_INJECTION.IM_DIRS",
                    [image_dir],
                    "D2GO_DATA.DATASETS.COCO_INJECTION.JSON_FILES",
                    [json_file],
                    "D2GO_DATA.DATASETS.COCO_INJECTION.REGISTER_FUNCTION",
                    "_register_dummy_function_coco",
                ]
            ]
        )
        runner.register(cfg)
        self.assertTrue(len(dummy_buffer) == 1)

    @tempdir
    def test_adhoc_register_coco_dataset_registry(self, tmp_dir):

        dummy_buffer = []

        def _dummy_load_func():
            return []

        @COCO_REGISTER_FUNCTION_REGISTRY.register()
        def _register_dummy_function_coco_adhoc(dataset_name, split_dict):

            json_file = split_dict[ANN_FN]
            image_root = split_dict[IM_DIR]

            DatasetCatalog.register(dataset_name, _dummy_load_func)

            MetadataCatalog.get(dataset_name).set(
                evaluator_type="coco",
                json_file=json_file,
                image_root=image_root,
            )
            dummy_buffer.append((dataset_name, split_dict))

        image_dir, json_file = create_test_images_and_dataset_json(tmp_dir)

        runner = Detectron2GoRunner()
        cfg = runner.get_default_cfg()
        cfg.merge_from_list(
            [
                str(x)
                for x in [
                    "D2GO_DATA.DATASETS.COCO_INJECTION.NAMES",
                    ["inj_test_registry_adhoc"],
                    "D2GO_DATA.DATASETS.COCO_INJECTION.IM_DIRS",
                    [image_dir],
                    "D2GO_DATA.DATASETS.COCO_INJECTION.JSON_FILES",
                    [json_file],
                    "D2GO_DATA.DATASETS.COCO_INJECTION.REGISTER_FUNCTION",
                    "_register_dummy_function_coco_adhoc",
                ]
            ]
        )
        runner.register(cfg)
        self.assertTrue(len(dummy_buffer) == 1)

        # Add adhoc class that uses only the first class
        AdhocDatasetManager.add(
            COCOWithClassesToUse("inj_test_registry_adhoc", ["class_0"])
        )

        # Check that the correct register function is used
        self.assertTrue(len(dummy_buffer) == 2)
