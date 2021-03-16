#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import copy
import json
import os
import unittest

import d2go.data.extended_coco as extended_coco
from d2go.data.utils import maybe_subsample_n_images
from d2go.runner import Detectron2GoRunner
from d2go.tests.data_loader_helper import LocalImageGenerator, create_toy_dataset
from detectron2.data import DatasetCatalog
from mobile_cv.common.misc.file_utils import make_temp_directory


def create_test_images_and_dataset_json(data_dir):
    # create image and json
    image_dir = os.path.join(data_dir, "images")
    os.makedirs(image_dir)
    json_dataset, meta_data = create_toy_dataset(
        LocalImageGenerator(image_dir, width=80, height=60), num_images=10
    )
    json_file = os.path.join(data_dir, "{}.json".format("inj_ds1"))
    with open(json_file, "w") as f:
        json.dump(json_dataset, f)

    return image_dir, json_file


class TestD2GoDatasets(unittest.TestCase):
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

    def test_coco_injection(self):

        with make_temp_directory("detectron2go_tmp_dataset") as tmp_dir:
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

    def test_sub_dataset(self):
        with make_temp_directory("detectron2go_tmp_dataset") as tmp_dir:
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
                        "DATASETS.TEST",
                        ("inj_ds",),
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
