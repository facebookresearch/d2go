#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import copy
import json
import logging
import os
import shlex
import subprocess
from collections import defaultdict
from typing import Callable, Dict, List, Optional

import detectron2.utils.comm as comm
from d2go.data.cache_util import _cache_json_file
from detectron2.data import MetadataCatalog
from detectron2.structures import BoxMode
from detectron2.utils.file_io import PathManager
from pycocotools.coco import COCO


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


def extract_archive_file(archive_fn: str, im_dir: str):
    if not PathManager.exists(im_dir) or not PathManager.ls(im_dir):
        # Dataset is not deployed. Deploy it.
        archive_fns = archive_fn
        # A dataset may be composed of several tgz files, or only one.
        # If one, make it into a list to make the code later more general
        if not isinstance(archive_fns, list):
            archive_fns = [archive_fns]
        logger.info(
            "Extracting datasets {} to local machine at {}".format(archive_fns, im_dir)
        )
        if not PathManager.exists(im_dir):
            PathManager.mkdirs(im_dir)

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


COCOTEXT_DATASET_CONVERSION_STATUS = {}


def save_converted_json(target_json, convert_coco_dict):

    if target_json in COCOTEXT_DATASET_CONVERSION_STATUS:
        return

    PathManager.mkdirs(os.path.dirname(target_json))
    if comm.get_local_rank() == 0:
        with PathManager.open(target_json, "w") as f:
            json.dump(convert_coco_dict, f)
    comm.synchronize()

    COCOTEXT_DATASET_CONVERSION_STATUS[target_json] = True


def convert_coco_text_to_coco_detection_json(
    source_json: str,
    target_json: str,
    set_type: Optional[str] = None,
    min_img_size: int = 100,
    text_cat_id: int = 1,
) -> Dict:
    """
    This function converts a COCOText style JSON to a COCODetection style
    JSON.
    For COCOText see: https://vision.cornell.edu/se3/coco-text-2/
    For COCODetection see: http://cocodataset.org/#overview
    """
    with PathManager.open(source_json, "r") as f:
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
    # Remap image_ids if necessary
    if isinstance(coco_text_json["images"][0]["id"], str):
        image_id_remap = {
            x["id"]: id_no for (id_no, x) in enumerate(coco_text_json["images"])
        }
        for x in coco_text_json["images"]:
            x["id"] = image_id_remap[x["id"]]
        for x in coco_text_json["annotations"]:
            if x["image_id"] in image_id_remap:
                x["image_id"] = image_id_remap[x["image_id"]]

    save_converted_json(target_json, coco_text_json)

    return coco_text_json


def valid_bbox(bbox_xywh: List[int], img_w: int, img_h: int) -> bool:
    if (
        bbox_xywh is None
        or not len(bbox_xywh) == 4
        or (bbox_xywh[3] == 0 or bbox_xywh[2] == 0)
        or not (0 <= bbox_xywh[0] <= img_w - bbox_xywh[2])
        or not (0 <= bbox_xywh[1] <= img_h - bbox_xywh[3])
    ):
        return False
    return True


def valid_bbox_rotated(bbox_xywha: List[int], img_w: int, img_h: int) -> bool:
    if (
        bbox_xywha is None
        or (bbox_xywha[3] == 0 or bbox_xywha[2] == 0)
        or not (
            0.4 * bbox_xywha[2] <= bbox_xywha[0] <= img_w - bbox_xywha[2] * 0.4
        )  # using 0.4*h and 0.4*w to give some leeway for rotation but still remove huge bboxes for training stability
        or not (0.4 * bbox_xywha[3] <= bbox_xywha[1] <= img_h - bbox_xywha[3] * 0.4)
    ):
        return False
    return True


def convert_coco_annotations(
    anno_dict_list: List[Dict],
    record: Dict,
    remapped_id: Dict,
    error_report: Dict,
    filter_invalid_bbox: Optional[bool] = True,
):
    """
    Converts annotations format of coco to internal format while applying
    some filtering
    """
    converted_annotations = []
    for anno in anno_dict_list:
        # Check that the image_id in this annotation is the same. This fails
        # only when the data parsing logic or the annotation file is buggy.
        assert anno["image_id"] == record["image_id"]
        assert anno.get("ignore", 0) == 0

        # Copy fields that do not need additional conversion
        fields_to_copy = [
            "iscrowd",
            "bbox",
            "bbox_mode",
            "keypoints",
            "category_id",
            "extras",
            "point_coords",
            "point_labels",
            "associations",
            "file_name",
        ]
        # NOTE: maybe use MetadataCatalog for this
        obj = {field: anno[field] for field in fields_to_copy if field in anno}

        # Filter out bad annotations where category do not match
        if obj.get("category_id", None) not in remapped_id:
            continue

        # Bounding boxes: convert and filter out bad bounding box annotations
        bbox_object = obj.get("bbox", None)
        if bbox_object:
            if "bbox_mode" in obj:
                bbox_object = BoxMode.convert(
                    bbox_object, obj["bbox_mode"], BoxMode.XYWH_ABS
                )
            else:
                # Assume default box mode is always (x, y, w h)
                error_report["without_bbox_mode"].cnt += 1
                obj["bbox_mode"] = (
                    BoxMode.XYWHA_ABS if len(obj["bbox"]) == 5 else BoxMode.XYWH_ABS
                )
        if obj["bbox_mode"] != BoxMode.XYWHA_ABS:  # for horizontal bboxes
            if (
                filter_invalid_bbox
                and record.get("width")
                and record.get("height")
                and not valid_bbox(bbox_object, record["width"], record["height"])
            ):
                error_report["without_valid_bounding_box"].cnt += 1
                continue
        else:  # for rotated bboxes in XYWHA format
            if (
                filter_invalid_bbox
                and record.get("width")
                and record.get("height")
                and not valid_bbox_rotated(
                    bbox_object, record["width"], record["height"]
                )
            ):
                error_report["without_valid_bounding_box"].cnt += 1
                continue

        # Segmentation: filter and add segmentation
        segm = anno.get("segmentation", None)
        if segm:  # either list[list[float]] or dict(RLE)
            if not isinstance(segm, dict):
                # filter out invalid polygons (< 3 points)
                segm = [poly for poly in segm if len(poly) % 2 == 0 and len(poly) >= 6]
                if len(segm) == 0:
                    error_report["without_valid_segmentation"].cnt += 1
                    continue  # ignore this instance
            obj["segmentation"] = segm

        # Remap ids
        obj["category_id"] = remapped_id[obj["category_id"]]

        converted_annotations.append(obj)
    return converted_annotations


# Error entry class for reporting coco conversion issues
class ErrorEntry:
    def __init__(self, error_name, msg, cnt=0):
        self.error_name = error_name
        self.cnt = cnt
        self.msg = msg

    def __repr__(self):
        return f"{self.msg} for {self.error_name}, count = {self.cnt}"


def print_conversion_report(ann_error_report, image_error_report, ex_warning_fn):
    # Report image errors
    report_str = ""
    for error_key in image_error_report:
        if image_error_report[error_key].cnt > 0:
            report_str += f"\t{image_error_report[error_key]}\n"
            if error_key == "ignore_image_root" and ex_warning_fn:
                report_str += f"\texample file name {ex_warning_fn}\n"

    # Report annotation errors
    for error_key in ann_error_report:
        if ann_error_report[error_key].cnt > 0:
            report_str += f"\t{ann_error_report[error_key]}\n"

    if len(report_str):
        logger.warning(f"Conversion issues:\n{report_str}")


def _assign_annotations_to_record(
    record: Dict, converted_anns: List[Dict], all_cat_names: Optional[List[str]]
) -> None:
    record["annotations"] = converted_anns
    if converted_anns and all(
        [ann.get("file_name", "").endswith(".png") for ann in converted_anns]
    ):
        if len(converted_anns) == 1:
            record["sem_seg_file_name"] = converted_anns[0]["file_name"]
            return
        assert (
            all_cat_names
        ), f"all_cat_names needs to be specified for MCS dataset: {converted_anns}"
        record["multi_sem_seg_file_names"] = {
            all_cat_names[ann["category_id"]]: ann["file_name"]
            for ann in converted_anns
        }


def _process_associations(
    record: Dict, converted_anns: List[Dict], _post_process_: Optional[Callable]
) -> None:
    post_process_dict = {"_post_process_": _post_process_} if _post_process_ else {}
    record.update(post_process_dict)

    if "associations" not in record or "associations" not in converted_anns[0]:
        return

    assert (
        len(converted_anns) == 1
    ), "Only one annotation expected when associated frames exist!"

    for key, associated_ann in converted_anns[0]["associations"].items():
        if key not in record["associations"]:
            continue
        record["associations"][key] = {
            "file_name": record["associations"][key],
            "sem_seg_file_name": associated_ann,
        }
        record["associations"][key].update(post_process_dict)
    # Following D23593142 to save memory
    record["associations"] = list(record["associations"])


def convert_to_dict_list(
    image_root: str,
    remapped_id: Dict,
    imgs: List[Dict],
    anns: List[Dict],
    dataset_name: Optional[str] = None,
    all_cat_names: Optional[List[str]] = None,
    image_direct_copy_keys: Optional[List[str]] = None,
    filter_invalid_bbox: Optional[bool] = True,
    filter_empty_annotations: Optional[bool] = True,
    _post_process_: Optional[Callable] = None,
) -> List[Dict]:

    ann_error_report = {
        name: ErrorEntry(name, msg, 0)
        for name, msg in [
            ("without_valid_segmentation", "Instance filtered"),
            ("without_valid_bounding_box", "Instance filtered"),
            ("without_bbox_mode", "Warning"),
        ]
    }
    image_error_report = {
        name: ErrorEntry(name, msg, 0)
        for name, msg in [
            ("ignore_image_root", f"Image root ignored {image_root}"),
            (
                "no_annotations",
                "Image filtered" if filter_empty_annotations else "Warning",
            ),
        ]
    }
    ex_warning_fn = None

    default_record = {"dataset_name": dataset_name} if dataset_name else {}

    converted_dict_list = []
    for img_dict, anno_dict_list in zip(imgs, anns):
        record = copy.deepcopy(default_record)

        # NOTE: besides using (relative path) in the "file_name" filed to represent
        # the image resource, "extended coco" also supports using uri which
        # represents an image using a single string, eg. "everstore_handle://xxx",
        if "://" not in img_dict["file_name"]:
            record["file_name"] = os.path.join(image_root, img_dict["file_name"])
        else:
            if image_root is not None:
                image_error_report["ignore_image_root"].cnt += 1
                ex_warning_fn = (
                    ex_warning_fn if ex_warning_fn else img_dict["file_name"]
                )
            record["file_name"] = img_dict["file_name"]

        # Setup image info and id
        if "height" in img_dict or "width" in img_dict:
            record["height"] = img_dict["height"]
            record["width"] = img_dict["width"]
        record["image_id"] = img_dict["id"]

        # Convert annotation for dataset_dict
        converted_anns = convert_coco_annotations(
            anno_dict_list,
            record,
            remapped_id,
            ann_error_report,
            filter_invalid_bbox=filter_invalid_bbox,
        )
        if len(converted_anns) == 0:
            image_error_report["no_annotations"].cnt += 1
            if filter_empty_annotations:
                continue
        _assign_annotations_to_record(record, converted_anns, all_cat_names)

        if "associations" in img_dict:
            record["associations"] = img_dict["associations"]
        _process_associations(record, converted_anns, _post_process_)

        # Copy keys if additionally asked
        if image_direct_copy_keys:
            for c_key in image_direct_copy_keys:
                assert c_key in img_dict, f"{c_key} not in coco image entry annotation"
                record[c_key] = img_dict[c_key]

        converted_dict_list.append(record)

    print_conversion_report(ann_error_report, image_error_report, ex_warning_fn)

    assert len(converted_dict_list) != 0, (
        f"Loaded zero entries from {dataset_name}. \n"
        f"  Size of inputs (imgs={len(imgs)}, anns={len(anns)})\n"
        f"  Image issues ({image_error_report})\n"
        f"  Instance issues ({ann_error_report})\n"
    )

    return converted_dict_list


def coco_text_load(
    coco_json_file: str,
    image_root: str,
    source_json_file: Optional[str] = None,
    dataset_name: Optional[str] = None,
    archive_file: Optional[str] = None,
) -> List[Dict]:
    if archive_file is not None:
        if comm.get_local_rank() == 0:
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


def extended_coco_load(
    json_file: str,
    image_root: str,
    dataset_name: Optional[str] = None,
    loaded_json: Optional[str] = None,
    image_direct_copy_keys: List[str] = None,
    filter_invalid_bbox: Optional[bool] = True,
    filter_empty_annotations: Optional[bool] = True,
    _post_process_: Optional[Callable] = None,
) -> List[Dict]:
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
    associations = coco_api.dataset.get("associations", {})

    # Collect classes and remap them starting from 0
    all_cat_ids = coco_api.getCatIds()
    all_cats = coco_api.loadCats(all_cat_ids)
    all_cat_names = [c["name"] for c in sorted(all_cats, key=lambda x: x["id"])]

    # Setup id remapping
    remapped_id = {}
    for cat_id, cat in zip(all_cat_ids, all_cats):
        remapped_id[cat_id] = all_cat_names.index(cat["name"])

    # Register dataset in metadata catalog
    if dataset_name is not None:
        # overwrite attrs
        meta_dict = MetadataCatalog.get(dataset_name).as_dict()
        meta_dict["thing_classes"] = all_cat_names
        meta_dict["thing_dataset_id_to_contiguous_id"] = remapped_id
        # update MetadataCatalog (cannot change inplace, have to remove)
        MetadataCatalog.remove(dataset_name)
        MetadataCatalog.get(dataset_name).set(**meta_dict)
        # assert the change
        assert MetadataCatalog.get(dataset_name).thing_classes == all_cat_names

    # Sort indices for reproducible results
    img_ids = sorted(coco_api.imgs.keys())
    imgs = coco_api.loadImgs(img_ids)
    anns = [coco_api.imgToAnns[img_id] for img_id in img_ids]
    logger.info("Loaded {} images from {}".format(len(imgs), json_file))

    for img in imgs:
        association = associations.get(img["file_name"], {})
        if association:
            img["associations"] = association

    # Return the coco converted to record list
    return convert_to_dict_list(
        image_root,
        remapped_id,
        imgs,
        anns,
        dataset_name=dataset_name,
        all_cat_names=all_cat_names,
        image_direct_copy_keys=image_direct_copy_keys,
        filter_invalid_bbox=filter_invalid_bbox,
        filter_empty_annotations=filter_empty_annotations,
        _post_process_=_post_process_,
    )


if __name__ == "__main__":
    """
    Test the COCO json dataset loader.

    Usage:
        python -m detectron2.data.datasets.coco \
            path/to/json path/to/image_root dataset_name
    """
    import sys

    import cv2
    from detectron2.utils.logger import setup_logger
    from detectron2.utils.visualizer import Visualizer

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
