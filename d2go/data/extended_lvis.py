#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved


import logging
import os

from d2go.data.extended_coco import _cache_json_file
from detectron2.data import MetadataCatalog
from detectron2.structures import BoxMode
from fvcore.common.timer import Timer

"""
This file contains functions to parse LVIS-format annotations into dicts in the
"Detectron2 format".
"""

logger = logging.getLogger(__name__)


def extended_lvis_load(json_file, image_root, dataset_name=None):
    """
    Load a json file in LVIS's annotation format.

    Args:
        json_file (str): full path to the LVIS json annotation file.
        image_root (str): the directory where the images in this json file exists.
        dataset_name (str): the name of the dataset (e.g., "lvis_v0.5_train").
            If provided, this function will put "thing_classes" into the metadata
            associated with this dataset.

    Returns:
        list[dict]: a list of dicts in "Detectron2 Dataset" format. (See DATASETS.md)

    Notes:
        1. This function does not read the image files.
           The results do not have the "image" field.
    """
    from lvis import LVIS

    json_file = _cache_json_file(json_file)

    timer = Timer()
    lvis_api = LVIS(json_file)
    if timer.seconds() > 1:
        logger.info(
            "Loading {} takes {:.2f} seconds.".format(json_file, timer.seconds())
        )

    # sort indices for reproducible results
    img_ids = sorted(list(lvis_api.imgs.keys()))
    # imgs is a list of dicts, each looks something like:
    # {'license': 4,
    #  'url': 'http://farm6.staticflickr.com/5454/9413846304_881d5e5c3b_z.jpg',
    #  'file_name': 'COCO_val2014_000000001268.jpg',
    #  'height': 427,
    #  'width': 640,
    #  'date_captured': '2013-11-17 05:57:24',
    #  'id': 1268}
    imgs = lvis_api.load_imgs(img_ids)
    # anns is a list[list[dict]], where each dict is an annotation
    # record for an object. The inner list enumerates the objects in an image
    # and the outer list enumerates over images. Example of anns[0]:
    # [{'segmentation': [[192.81,
    #     247.09,
    #     ...
    #     219.03,
    #     249.06]],
    #   'area': 1035.749,
    #   'image_id': 1268,
    #   'bbox': [192.81, 224.8, 74.73, 33.43],
    #   'category_id': 16,
    #   'id': 42986},
    #  ...]
    anns = [lvis_api.img_ann_map[img_id] for img_id in img_ids]

    # Sanity check that each annotation has a unique id
    ann_ids = [ann["id"] for anns_per_image in anns for ann in anns_per_image]
    assert len(set(ann_ids)) == len(
        ann_ids
    ), "Annotation ids in '{}' are not unique".format(json_file)

    imgs_anns = list(zip(imgs, anns))

    logger.info(
        "Loaded {} images in the LVIS format from {}".format(len(imgs_anns), json_file)
    )

    dataset_dicts = []

    count_ignore_image_root_warning = 0
    for img_dict, anno_dict_list in imgs_anns:
        record = {}
        if "://" not in img_dict["file_name"]:
            file_name = img_dict["file_name"]
            if img_dict["file_name"].startswith("COCO"):
                # Convert form the COCO 2014 file naming convention of
                # COCO_[train/val/test]2014_000000000000.jpg to the 2017 naming
                # convention of 000000000000.jpg (LVIS v1 will fix this naming issue)
                file_name = file_name[-16:]
            record["file_name"] = os.path.join(image_root, file_name)
        else:
            if image_root is not None:
                count_ignore_image_root_warning += 1
                if count_ignore_image_root_warning == 1:
                    logger.warning(
                        (
                            "Found '://' in file_name: {}, ignore image_root: {}"
                            "(logged once per dataset)."
                        ).format(img_dict["file_name"], image_root)
                    )
            record["file_name"] = img_dict["file_name"]
        record["height"] = img_dict["height"]
        record["width"] = img_dict["width"]
        record["not_exhaustive_category_ids"] = img_dict.get(
            "not_exhaustive_category_ids", []
        )
        record["neg_category_ids"] = img_dict.get("neg_category_ids", [])
        image_id = record["image_id"] = img_dict["id"]

        objs = []
        for anno in anno_dict_list:
            # Check that the image_id in this annotation is the same as
            # the image_id we're looking at.
            # Fails only when the data parsing logic or the annotation file is buggy.
            assert anno["image_id"] == image_id
            obj = {"bbox": anno["bbox"], "bbox_mode": BoxMode.XYWH_ABS}
            obj["category_id"] = (
                anno["category_id"] - 1
            )  # Convert 1-indexed to 0-indexed
            segm = anno["segmentation"]
            # filter out invalid polygons (< 3 points)
            valid_segm = [
                poly for poly in segm if len(poly) % 2 == 0 and len(poly) >= 6
            ]
            assert len(segm) == len(
                valid_segm
            ), "Annotation contains an invalid polygon with < 3 points"
            assert len(segm) > 0
            obj["segmentation"] = segm
            objs.append(obj)
        record["annotations"] = objs
        dataset_dicts.append(record)

    if dataset_name:
        meta = MetadataCatalog.get(dataset_name)
        meta.thing_classes = get_extended_lvis_instances_meta(lvis_api)["thing_classes"]

    return dataset_dicts


def get_extended_lvis_instances_meta(lvis_api):
    cat_ids = lvis_api.get_cat_ids()
    categories = lvis_api.load_cats(cat_ids)
    assert min(cat_ids) == 1 and max(cat_ids) == len(
        cat_ids
    ), "Category ids are not in [1, #categories], as expected"
    extended_lvis_categories = [k for k in sorted(categories, key=lambda x: x["id"])]
    thing_classes = [k["name"] for k in extended_lvis_categories]
    meta = {"thing_classes": thing_classes}
    return meta


if __name__ == "__main__":
    """
    Test the LVIS json dataset loader.

    Usage:
        python -m detectron2.data.datasets.lvis \
            path/to/json path/to/image_root dataset_name vis_limit
    """
    import sys

    import detectron2.data.datasets  # noqa  # add pre-defined metadata
    import numpy as np
    from detectron2.utils.logger import setup_logger
    from detectron2.utils.visualizer import Visualizer
    from PIL import Image

    logger = setup_logger(name=__name__)
    meta = MetadataCatalog.get(sys.argv[3])

    dicts = extended_lvis_load(sys.argv[1], sys.argv[2], sys.argv[3])
    logger.info("Done loading {} samples.".format(len(dicts)))

    dirname = "lvis-data-vis"
    os.makedirs(dirname, exist_ok=True)
    for d in dicts[: int(sys.argv[4])]:
        img = np.array(Image.open(d["file_name"]))
        visualizer = Visualizer(img, metadata=meta)
        vis = visualizer.draw_dataset_dict(d)
        fpath = os.path.join(dirname, os.path.basename(d["file_name"]))
        vis.save(fpath)
