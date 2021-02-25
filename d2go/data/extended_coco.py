#!/usr/bin/env python3

import json
import logging
import os
import shlex
import subprocess
from collections import defaultdict

import detectron2.utils.comm as comm
from detectron2.data import MetadataCatalog
from detectron2.structures import BoxMode
from pycocotools.coco import COCO

from .cache_util import _cache_json_file

logger = logging.getLogger(__name__)


class InMemoryCOCO(COCO):
    def __init__(self, loaded_json):
        """
        In this in-memory version of COCO we don't load json from the file,
        but direclty use a loaded_json instead. This approach improves
        both robustness and efficiency, as when we convert from other formats
        to COCO format, we don't need to save and re-load the json again.
        """
        # load dataset
        self.dataset = loaded_json
        self.anns = {}
        self.cats = {}
        self.imgs = {}
        self.imgToAnns, self.catToImgs = defaultdict(list), defaultdict(list)
        self.createIndex()


def extract_archive_file(archive_fn, im_dir):
    if not os.path.exists(im_dir) or not os.listdir(im_dir):
        # Dataset is not deployed. Deploy it.
        archive_fns = archive_fn
        # A dataset may be composed of several tgz files, or only one.
        # If one, make it into a list to make the code later more general
        if not isinstance(archive_fns, list):
            archive_fns = [archive_fns]
        logger.info(
            "Extracting datasets {} to local machine at {}".format(archive_fns, im_dir)
        )
        if not os.path.exists(im_dir):
            os.makedirs(im_dir)

        for archive_fn in archive_fns:
            # Extract the tgz file directly into the target directory,
            # without precopy.
            # Note that the tgz file contains a root directory that
            # we do not want, hence the strip-components=1
            commandUnpack = (
                "tar -mxzf {src_file} -C {tgt_dir} " "--strip-components=1"
            ).format(src_file=archive_fn, tgt_dir=im_dir)

            assert not subprocess.call(shlex.split(commandUnpack)), "Failed to unpack"
            logger.info("Extracted {}".format(archive_fn))


def convert_coco_text_to_coco_detection_json(
    source_json, target_json, set_type=None, min_img_size=100, text_cat_id=1
):
    """
    This function converts a COCOText style JSON to a COCODetection style
    JSON.
    For COCOText see: https://vision.cornell.edu/se3/coco-text-2/
    For COCODetection see: http://cocodataset.org/#overview
    """
    with open(source_json) as f:
        coco_text_json = json.load(f)

    coco_text_json["annotations"] = list(coco_text_json["anns"].values())
    coco_text_json["images"] = list(coco_text_json["imgs"].values())
    if set_type is not None:
        # COCO Text style JSONs often mix test, train, and val sets.
        # We need to make sure we only use the data type we want.
        coco_text_json["images"] = [
            x for x in coco_text_json["images"] if x["set"] == set_type
        ]
    coco_text_json["categories"] = [{"name": "text", "id": text_cat_id}]
    del coco_text_json["cats"]
    del coco_text_json["imgs"]
    del coco_text_json["anns"]
    for ann in coco_text_json["annotations"]:
        ann["category_id"] = text_cat_id
        ann["iscrowd"] = 0
        # Don't evaluate the model on illegible words
        if set_type == "val" and ann["legibility"] != "legible":
            ann["ignore"] = True
    # Some datasets seem to have extremely small images which break downstream
    # operations. If min_img_size is set, we can remove these.
    coco_text_json["images"] = [
        x
        for x in coco_text_json["images"]
        if x["height"] >= min_img_size and x["width"] >= min_img_size
    ]
    os.makedirs(os.path.dirname(target_json), exist_ok=True)
    with open(target_json, "w") as f:
        json.dump(coco_text_json, f)

    return coco_text_json


def convert_to_dict_list(image_root, id_map, imgs, anns, dataset_name=None):
    num_instances_without_valid_segmentation = 0
    dataset_dicts = []
    count_ignore_image_root_warning = 0
    for (img_dict, anno_dict_list) in zip(imgs, anns):
        record = {}
        # NOTE: besides using (relative path) in the "file_name" filed to represent
        # the image resource, "extended coco" also supports using uri which
        # represents an image using a single string, eg. "everstore_handle://xxx",
        if "://" not in img_dict["file_name"]:
            record["file_name"] = os.path.join(image_root, img_dict["file_name"])
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

        if "height" in img_dict or "width" in img_dict:
            record["height"] = img_dict["height"]
            record["width"] = img_dict["width"]
        image_id = record["image_id"] = img_dict["id"]

        objs = []
        for anno in anno_dict_list:
            # Check that the image_id in this annotation is the same. This fails
            # only when the data parsing logic or the annotation file is buggy.
            assert anno["image_id"] == image_id
            assert anno.get("ignore", 0) == 0

            obj = {
                field: anno[field]
                # NOTE: maybe use MetadataCatalog for this
                for field in ["iscrowd", "bbox", "keypoints", "category_id", "extras"]
                if field in anno
            }

            if obj.get("category_id", None) not in id_map:
                continue

            segm = anno.get("segmentation", None)
            if segm:  # either list[list[float]] or dict(RLE)
                if not isinstance(segm, dict):
                    # filter out invalid polygons (< 3 points)
                    segm = [
                        poly for poly in segm if len(poly) % 2 == 0 and len(poly) >= 6
                    ]
                    if len(segm) == 0:
                        num_instances_without_valid_segmentation += 1
                        continue  # ignore this instance
                obj["segmentation"] = segm

            if len(obj["bbox"]) == 5:
                obj["bbox_mode"] = BoxMode.XYWHA_ABS
            else:
                obj["bbox_mode"] = BoxMode.XYWH_ABS
            if id_map:
                obj["category_id"] = id_map[obj["category_id"]]
            objs.append(obj)
        record["annotations"] = objs
        if dataset_name is not None:
            record["dataset_name"] = dataset_name
        dataset_dicts.append(record)

    if count_ignore_image_root_warning > 0:
        logger.warning(
            "The 'ignore image_root: {}' warning occurred {} times".format(
                image_root, count_ignore_image_root_warning
            )
        )

    if num_instances_without_valid_segmentation > 0:
        logger.warning(
            "Filtered out {} instances without valid segmentation. "
            "There might be issues in your dataset generation process.".format(
                num_instances_without_valid_segmentation
            )
        )
    return dataset_dicts


def coco_text_load(
    coco_json_file,
    image_root,
    source_json_file=None,
    dataset_name=None,
    archive_file=None,
):
    if archive_file is not None:
        if comm.get_rank() == 0:
            extract_archive_file(archive_file, image_root)
        comm.synchronize()

    if source_json_file is not None:
        # Need to convert to coco detection format
        loaded_json = convert_coco_text_to_coco_detection_json(
            source_json_file, coco_json_file
        )
        return extended_coco_load(coco_json_file, image_root, dataset_name, loaded_json)

    return extended_coco_load(
        coco_json_file, image_root, dataset_name, loaded_json=None
    )


def extended_coco_load(json_file, image_root, dataset_name=None, loaded_json=None):
    """
    Load a json file with COCO's annotation format.
    Currently only supports instance segmentation annotations.

    Args:
        json_file (str): full path to the json file in COCO annotation format.
        image_root (str): the directory where the images in this json file exists.
        dataset_name (str): the name of the dataset (e.g., "coco", "cityscapes").
            If provided, this function will also put "thing_classes" into
            the metadata associated with this dataset.
        loaded_json (str): optional loaded json content, used in InMemoryCOCO to
            avoid loading from json_file again.
    Returns:
        list[dict]: a list of dicts in "Detectron2 Dataset" format. (See DATASETS.md)

    Notes:
        1. This function does not read the image files.
           The results do not have the "image" field.
        2. When `dataset_name=='coco'`,
           this function will translate COCO's
           incontiguous category ids to contiguous ids in [0, 80).
    """

    json_file = _cache_json_file(json_file)

    if loaded_json is None:
        coco_api = COCO(json_file)
    else:
        coco_api = InMemoryCOCO(loaded_json)

    id_map = None
    # Get filtered classes
    all_cat_ids = coco_api.getCatIds()
    all_cats = coco_api.loadCats(all_cat_ids)

    # Setup classes to use for creating id map
    classes_to_use = [c["name"] for c in sorted(all_cats, key=lambda x: x["id"])]

    # Setup id map
    id_map = {}
    for cat_id, cat in zip(all_cat_ids, all_cats):
        if cat["name"] in classes_to_use:
            id_map[cat_id] = classes_to_use.index(cat["name"])

    # Register dataset in metadata catalog
    if dataset_name is not None:
        # overwrite attrs
        meta_dict = MetadataCatalog.get(dataset_name).as_dict()
        meta_dict['thing_classes'] = classes_to_use
        meta_dict['thing_dataset_id_to_contiguous_id'] = id_map
        # update MetadataCatalog (cannot change inplace, has to remove)
        MetadataCatalog.remove(dataset_name)
        MetadataCatalog.get(dataset_name).set(**meta_dict)
        # assert the change
        assert MetadataCatalog.get(dataset_name).thing_classes == classes_to_use

    # sort indices for reproducible results
    img_ids = sorted(coco_api.imgs.keys())
    imgs = coco_api.loadImgs(img_ids)
    anns = [coco_api.imgToAnns[img_id] for img_id in img_ids]
    logger.info("Loaded {} images from {}".format(len(imgs), json_file))

    # Return the coco converted to record list
    return convert_to_dict_list(image_root, id_map, imgs, anns, dataset_name)


if __name__ == "__main__":
    """
    Test the COCO json dataset loader.

    Usage:
        python -m detectron2.data.datasets.coco \
            path/to/json path/to/image_root dataset_name
    """
    from detectron2.utils.logger import setup_logger
    from detectron2.utils.visualizer import Visualizer
    import cv2
    import sys

    logger = setup_logger(name=__name__)
    meta = MetadataCatalog.get(sys.argv[3])

    dicts = extended_coco_load(sys.argv[1], sys.argv[2], sys.argv[3], ["cat", "dog"])
    logger.info("Done loading {} samples.".format(len(dicts)))

    for d in dicts:
        img = cv2.imread(d["file_name"])[:, :, ::-1]
        visualizer = Visualizer(img, metadata=meta)
        vis = visualizer.draw_dataset_dict(d)
        fpath = os.path.join("coco-data-vis", os.path.basename(d["file_name"]))
        vis.save(fpath)
