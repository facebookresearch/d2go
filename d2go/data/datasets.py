#!/usr/bin/env python3

import functools
import importlib
import logging
import os

from detectron2.data import DatasetCatalog, MetadataCatalog
from d2go.utils.helper import get_dir_path

from .extended_coco import coco_text_load, extended_coco_load
from .extended_lvis import extended_lvis_load


logger = logging.getLogger(__name__)

D2GO_DATASETS_BASE_MODULE = "d2go.datasets"
IM_DIR = "image_directory"
ANN_FN = "annotation_file"


def _import_dataset(module_name):
    return importlib.import_module(
        "{}.{}".format(D2GO_DATASETS_BASE_MODULE, module_name)
    )


def _register_extended_coco(dataset_name, split_dict):
    json_file = split_dict[ANN_FN]
    image_root = split_dict[IM_DIR]

    # 1. register a function which returns dicts
    load_coco_json_func = functools.partial(
        extended_coco_load,
        json_file=json_file,
        image_root=image_root,
        dataset_name=dataset_name,
    )
    DatasetCatalog.register(dataset_name, load_coco_json_func)

    # 2. Optionally, add metadata about this split,
    # since they might be useful in evaluation, visualization or logging
    evaluator_type = split_dict.get("evaluator_type", "coco")
    meta_data = split_dict.get("meta_data", {})
    MetadataCatalog.get(dataset_name).set(
        evaluator_type=evaluator_type,
        json_file=json_file,
        image_root=image_root,
        **meta_data
    )


def _register_extended_lvis(dataset_name, split_dict):
    json_file = split_dict[ANN_FN]
    image_root = split_dict[IM_DIR]

    # 1. register a function which returns dicts
    load_lvis_json_func = functools.partial(
        extended_lvis_load,
        json_file=json_file,
        image_root=image_root,
        dataset_name=dataset_name,
    )
    DatasetCatalog.register(dataset_name, load_lvis_json_func)

    # 2. Optionally, add metadata about this split,
    # since they might be useful in evaluation, visualization or logging
    evaluator_type = split_dict.get("evaluator_type", "lvis")
    MetadataCatalog.get(dataset_name).set(
        evaluator_type=evaluator_type, json_file=json_file, image_root=image_root
    )


def _register_coco_text(dataset_name, split_dict):
    source_json_file = split_dict[ANN_FN]
    coco_json_file = "/tmp/{}.json".format(dataset_name)
    ARCHIVE_FN = "archive_file"

    # 1. register a function which returns dicts
    DatasetCatalog.register(
        dataset_name,
        functools.partial(
            coco_text_load,
            coco_json_file=coco_json_file,
            image_root=split_dict[IM_DIR],
            source_json_file=source_json_file,
            dataset_name=dataset_name,
            archive_file=split_dict.get(ARCHIVE_FN, None),
        ),
    )
    # 2. Optionally, add metadata about this split,
    # since they might be useful in evaluation, visualization or logging
    evaluator_type = split_dict.get("evaluator_type", "coco")
    MetadataCatalog.get(dataset_name).set(
        json_file=coco_json_file,
        image_root=split_dict[IM_DIR],
        evaluator_type=evaluator_type,
    )


def inject_coco_datasets(cfg):
    names = cfg.D2GO_DATA.DATASETS.COCO_INJECTION.NAMES
    im_dirs = cfg.D2GO_DATA.DATASETS.COCO_INJECTION.IM_DIRS
    json_files = cfg.D2GO_DATA.DATASETS.COCO_INJECTION.JSON_FILES

    assert len(names) == len(im_dirs) == len(json_files)
    for name, im_dir, json_file in zip(names, im_dirs, json_files):
        split_dict = {IM_DIR: im_dir, ANN_FN: json_file}
        logger.info("Inject coco dataset {}: {}".format(name, split_dict))
        _register_extended_coco(name, split_dict)


def register_dataset_split(dataset_name, split_dict):
    """
    Register a dataset to detectron2's DatasetCatalog and MetadataCatalog.
    """

    _DATASET_TYPE_LOAD_FUNC_MAP = {
        "COCODataset": _register_extended_coco,
        "COCOText": _register_coco_text,
        "COCOTextDataset": _register_coco_text,
        "LVISDataset": _register_extended_lvis,
    }

    factory = split_dict.get("DS_TYPE", "COCODataset")
    _DATASET_TYPE_LOAD_FUNC_MAP[factory](
        dataset_name=dataset_name, split_dict=split_dict
    )


def register_json_datasets():
    json_dataset_names = [
        os.path.splitext(filename)[0]
        for filename in os.listdir(
            get_dir_path(D2GO_DATASETS_BASE_MODULE.replace(".", "/"))
        )
        if filename.startswith("json_dataset_")
    ]
    json_dataset_names = [
        x
        for x in json_dataset_names
        if x
        not in [
            "json_dataset_lvis",
            "json_dataset_oculus_external",
            "json_dataset_people_ai_foot_tracking",
        ]
    ]

    # load all splits from json datasets
    all_splits = {}
    for dataset in json_dataset_names:
        module = _import_dataset(dataset)
        assert (
            len(set(all_splits).intersection(set(module.DATASETS))) == 0
        ), "Name confliction when loading {}".format(dataset)
        all_splits.update(module.DATASETS)

    # register all splits
    for split_name in all_splits:
        split_dict = all_splits[split_name]
        register_dataset_split(split_name, split_dict)


def register_builtin_datasets():
    builtin_dataset_names = [
        os.path.splitext(filename)[0]
        for filename in os.listdir(
            get_dir_path(D2GO_DATASETS_BASE_MODULE.replace(".", "/"))
        )
        if filename.startswith("builtin_dataset_")
    ]
    for dataset in builtin_dataset_names:
        _import_dataset(dataset)


def register_dynamic_datasets(cfg):
    for dataset in cfg.D2GO_DATA.DATASETS.DYNAMIC_DATASETS:
        assert dataset.startswith("dynamic_dataset_")
        _import_dataset(dataset)
