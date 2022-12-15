#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved


import contextlib
import itertools
import json
import math
import os
import uuid

from d2go.data.datasets import register_dataset_split
from d2go.runner import create_runner
from detectron2.data import DatasetCatalog, MetadataCatalog
from mobile_cv.common.misc.file_utils import make_temp_directory
from PIL import Image


IM_DIR = "image_directory"
ANN_FN = "annotation_file"


def create_toy_dataset(
    image_generator, num_images, num_classes=-1, num_keypoints=0, is_rotated=False
):
    """given image_generator, create a dataset with toy annotations and catagories"""
    categories = []
    images = []
    annotations = []
    meta_data = {}

    if num_classes == -1:
        num_classes = num_images

    for i in range(num_images):
        image_generator.prepare_image(i)
        image_dict = image_generator.get_image_dict(i)
        width = image_dict["width"]
        height = image_dict["height"]
        images.append(image_dict)

        if i < num_classes:
            categories.append({"name": "class_{}".format(i), "id": i})

        bbox = (
            [width / 4, height / 4, width / 2, height / 2]  # XYWH_ABS
            if not is_rotated
            else [width / 2, height / 2, width / 2, height / 2, 45]  # cXcYWHO_ABS
        )

        keypoints = list(
            itertools.chain.from_iterable(
                [
                    math.cos(2 * math.pi * x / num_keypoints) * width / 4 + width / 2,
                    math.sin(2 * math.pi * x / num_keypoints) * height / 4 + height / 2,
                    1,
                ]
                for x in range(num_keypoints)
            )
        )

        no_pts = 10
        segmentation = list(
            itertools.chain.from_iterable(
                [
                    math.cos(2 * math.pi * x / no_pts) * width / 4 + width / 2,
                    math.sin(2 * math.pi * x / no_pts) * height / 4 + height / 2,
                ]
                for x in range(no_pts)
            )
        )

        annotations.append(
            {
                "image_id": i,
                "category_id": i % num_classes,
                "id": i + 1,
                "bbox": bbox,
                "keypoints": keypoints,
                "area": width * height,
                "iscrowd": 0,
                "ignore": 0,
                "segmentation": [segmentation],
            }
        )

    if num_keypoints > 0:
        keypoint_names = [f"kp_{idx}" for idx in range(num_keypoints)]
        meta_data.update({"keypoint_names": keypoint_names, "keypoint_flip_map": ()})

    return (
        {"categories": categories, "images": images, "annotations": annotations},
        meta_data,
    )


@contextlib.contextmanager
def _register_toy_dataset(
    dataset_name, image_generator, num_images, num_classes=-1, num_keypoints=0
):
    json_dataset, meta_data = create_toy_dataset(
        image_generator,
        num_images=num_images,
        num_classes=num_classes,
        num_keypoints=num_keypoints,
    )

    with make_temp_directory("detectron2go_tmp_dataset") as tmp_dir:
        json_file = os.path.join(tmp_dir, "{}.json".format(dataset_name))
        with open(json_file, "w") as f:
            json.dump(json_dataset, f)

        split_dict = {
            IM_DIR: image_generator.get_image_dir(),
            ANN_FN: json_file,
            "meta_data": meta_data,
        }
        register_dataset_split(dataset_name, split_dict)

        try:
            yield
        finally:
            DatasetCatalog.remove(dataset_name)
            MetadataCatalog.remove(dataset_name)


@contextlib.contextmanager
def register_toy_coco_dataset(
    dataset_name, num_images=3, image_size=(5, 10), num_classes=-1, num_keypoints=0
):
    width, height = image_size
    with make_temp_directory("detectron2go_tmp_dataset") as dataset_dir:
        image_dir = os.path.join(dataset_dir, "images")
        os.makedirs(image_dir)
        image_generator = LocalImageGenerator(image_dir, width=width, height=height)

        with _register_toy_dataset(
            dataset_name,
            image_generator,
            num_images=num_images,
            num_classes=num_classes,
            num_keypoints=num_keypoints,
        ):
            yield


def create_local_dataset(
    out_dir,
    num_images,
    image_width,
    image_height,
    num_classes=-1,
    num_keypoints=0,
    is_rotated=False,
):
    dataset_name = "_test_ds_" + str(uuid.uuid4())

    img_gen = LocalImageGenerator(out_dir, image_width, image_height)
    json_dataset, meta_data = create_toy_dataset(
        img_gen,
        num_images=num_images,
        num_classes=num_classes,
        num_keypoints=num_keypoints,
    )
    json_file = os.path.join(out_dir, "{}.json".format(dataset_name))
    with open(json_file, "w") as f:
        json.dump(json_dataset, f)

    split_dict = {
        IM_DIR: img_gen.get_image_dir(),
        ANN_FN: json_file,
        "meta_data": meta_data,
    }
    if is_rotated:
        split_dict["evaluator_type"] = "rotated_coco"
    register_dataset_split(dataset_name, split_dict)

    return dataset_name


class LocalImageGenerator:
    def __init__(self, image_dir, width, height):
        self._width = width
        self._height = height
        self._image_dir = image_dir

    def get_image_dir(self):
        return self._image_dir

    def get_image_dict(self, i):
        return {
            "file_name": "{}.jpg".format(i),
            "width": self._width,
            "height": self._height,
            "id": i,
        }

    def prepare_image(self, i):
        image = Image.new("RGB", (self._width, self._height))
        image.save(os.path.join(self._image_dir, self.get_image_dict(i)["file_name"]))


@contextlib.contextmanager
def create_detection_data_loader_on_toy_dataset(
    cfg, height, width, is_train, runner=None
):
    """
    Args:
        cfg (CfgNode): the config used to create data loader, it can control things like
            resizing, augmentation.
        height, width (int): the height/width of the image files (not the resized image size)
        is_train (bool): training or testing
    """
    if runner is None:
        runner = create_runner("d2go.runner.GeneralizedRCNNRunner")

    # change dataset name to toy dataset
    cfg.DATASETS.TRAIN = ["_toy_dataset_train_"]
    cfg.DATASETS.TEST = ["_toy_dataset_test_"]

    if is_train:
        with register_toy_coco_dataset(
            "_toy_dataset_train_", num_images=3, image_size=(width, height)
        ):
            train_loader = runner.build_detection_train_loader(cfg)
            yield train_loader
    else:
        with register_toy_coco_dataset(
            "_toy_dataset_test_", num_images=3, image_size=(width, height)
        ):
            test_loader = runner.build_detection_test_loader(
                cfg, dataset_name="_toy_dataset_test_"
            )
            yield test_loader
