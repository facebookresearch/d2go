#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved


"""
This file contains utilities to load GANs datasets.

Similar to how COCO dataset is represented in Detectron2, a GANs dataset is represented
as a list of dicts, where each dict is in "standard dataset dict" format, which contains
raw data with fields such as:
    - input_path (str): filename of input image
    - fg_path (str): filename to the GT
    ...
"""

import json
import logging
import os
import tempfile
from pathlib import Path

from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.utils.file_io import PathManager


logger = logging.getLogger(__name__)


IMG_EXTENSIONS = [".jpg", ".JPG", ".png", ".PNG", ".ppm", ".PPM", ".bmp", ".BMP"]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def load_pix2pix_image_folder(image_root, input_folder="input", gt_folder="gt"):
    """
    Args:
        image_root (str): the directory where the images exist.
        gt_postfix (str): the postfix for the ground truth images

    Returns:
        list[dict]: a list of dicts in argos' "standard dataset dict" format
    """

    data = []

    # gt_postfix = "%s." % (gt_postfix)
    input_root = os.path.join(image_root, input_folder)
    for root, _, fnames in sorted(os.walk(input_root)):
        for fname in sorted(fnames):
            if is_image_file(fname):
                gt_fname = fname.replace("/%s/" % (gt_folder))
                input_path = os.path.join(root, fname)
                gt_path = os.path.join(root, gt_fname)
                if not os.path.isfile(gt_path):
                    logger.warning("{} is not exist".format(gt_fname))
                    continue
                # if len(gt_postfix) > 1 and fname.rfind(gt_postfix) != -1:  # skip GT file
                #     continue

                # gt_fname = fname[:-4] + gt_postfix + fname[-3:]
                # assert gt_fname in fnames, (
                #     "gt file %s is not exist in %s" % (gt_fname, root))

                f = {
                    "file_name": fname[:-4],
                    "input_path": input_path,
                    "gt_path": gt_path,
                }
                data.append(f)
                if image_root.rfind("test") != -1 and len(data) == 5000:
                    logger.info("Reach maxinum of test data: {} ".format(len(data)))
                    return data
    logger.info("Total number of data dicts: {} ".format(len(data)))
    return data


def load_pix2pix_json(
    json_path,
    input_folder,
    gt_folder,
    mask_folder,
    input_extras_folder,
    real_json_path=None,
    real_folder=None,
    max_num=1e10,
):
    """
    Args:
        json_path (str): the directory where the json file exists which saves the filenames and labels.
        input_folder (str): the directory for the input/source images
        input_folder (str): the directory for the ground_truth/target images
        mask_folder (str): the directory for the masks
        input_extras_folder (str): the directory for the input extras
    Returns:
        list[dict]: a list of dicts
    """
    real_filenames = {}
    if real_json_path is not None:
        with PathManager.open(real_json_path, "r") as f:
            real_filenames = json.load(f)

    data = []
    with PathManager.open(json_path, "r") as f:
        filenames = json.load(f)

        in_len = len(filenames)
        real_len = len(real_filenames)
        total_len = min(max(in_len, real_len), max_num)

        real_keys = [*real_filenames.keys()]
        in_keys = [*filenames.keys()]

        cnt = 0
        # for fname in filenames.keys():
        while cnt < total_len:
            fname = in_keys[cnt % in_len]
            input_label = filenames[fname]
            if isinstance(input_label, tuple) or isinstance(input_label, list):
                assert (
                    len(input_label) == 2
                ), "Save (real_name, label) as the value of the json dict for resampling"
                fname, input_label = input_label

            f = {
                "file_name": fname,
                "input_folder": input_folder,
                "gt_folder": gt_folder,
                "mask_folder": mask_folder,
                "input_extras_folder": input_extras_folder,
                "input_label": input_label,
                "real_folder": real_folder,
            }
            if real_len > 0:
                real_fname = real_keys[cnt % real_len]
                f["real_file_name"] = real_fname
            data.append(f)
            cnt += 1
            # 5000 is the general number of images used to calculate FID in GANs
            # if max_num > 0 and len(data) == max_num:
            #     logger.info("Reach maxinum of test data: {} ".format(len(data)))
            #     return data
    logger.info("Total number of data dicts: {} ".format(len(data)))
    return data


def register_folder_dataset(
    name,
    json_path,
    input_folder,
    gt_folder=None,
    mask_folder=None,
    input_extras_folder=None,
    input_src_path=None,
    gt_src_path=None,
    mask_src_path=None,
    input_extras_src_path=None,
    real_json_path=None,
    real_folder=None,
    real_src_path=None,
    max_num=1e10,
):
    DatasetCatalog.register(
        name,
        lambda: load_pix2pix_json(
            json_path,
            input_folder,
            gt_folder,
            mask_folder,
            input_extras_folder,
            real_json_path,
            real_folder,
            max_num,
        ),
    )
    metadata = {
        "input_src_path": input_src_path,
        "gt_src_path": gt_src_path,
        "mask_src_path": mask_src_path,
        "input_extras_src_path": input_extras_src_path,
        "real_src_path": real_src_path,
        "input_folder": input_folder,
        "gt_folder": gt_folder,
        "mask_folder": mask_folder,
        "input_extras_folder": input_extras_folder,
        "real_folder": real_folder,
    }
    MetadataCatalog.get(name).set(**metadata)


def load_lmdb_keys(max_num):
    """
    Args:
        max_num (str): the total number of
    Returns:
        list[dict]: a list of dicts
    """
    data = []
    for i in range(max_num):
        f = {"index": i}
        data.append(f)
    logger.info("Total number of data dicts: {} ".format(len(data)))
    return data


def register_lmdb_dataset(
    name,
    data_folder,
    src_data_folder,
    max_num,
):
    DatasetCatalog.register(name, lambda: load_lmdb_keys(max_num))
    metadata = {
        "data_folder": data_folder,
        "src_data_folder": src_data_folder,
        "max_num": max_num,
    }
    MetadataCatalog.get(name).set(**metadata)


def inject_gan_datasets(cfg):
    if cfg.D2GO_DATA.DATASETS.GAN_INJECTION.ENABLE:
        name = cfg.D2GO_DATA.DATASETS.GAN_INJECTION.NAME
        cfg.merge_from_list(
            [
                "DATASETS.TRAIN",
                list(cfg.DATASETS.TRAIN) + [name + "_train"],
                "DATASETS.TEST",
                list(cfg.DATASETS.TEST) + [name + "_test"],
            ]
        )

        json_path = cfg.D2GO_DATA.DATASETS.GAN_INJECTION.JSON_PATH
        assert PathManager.isfile(json_path), "{} is not valid!".format(json_path)

        if len(cfg.D2GO_DATA.DATASETS.GAN_INJECTION.LOCAL_DIR) > 0:
            image_dir = cfg.D2GO_DATA.DATASETS.GAN_INJECTION.LOCAL_DIR
        else:
            image_dir = Path(tempfile.mkdtemp())

        input_src_path = cfg.D2GO_DATA.DATASETS.GAN_INJECTION.INPUT_SRC_DIR
        assert PathManager.isfile(input_src_path), "{} is not valid!".format(
            input_src_path
        )
        input_folder = os.path.join(image_dir, name, "input")

        gt_src_path = cfg.D2GO_DATA.DATASETS.GAN_INJECTION.GT_SRC_DIR
        if PathManager.isfile(gt_src_path):
            gt_folder = os.path.join(image_dir, name, "gt")
        else:
            gt_src_path = None
            gt_folder = None

        mask_src_path = cfg.D2GO_DATA.DATASETS.GAN_INJECTION.MASK_SRC_DIR
        if PathManager.isfile(mask_src_path):
            mask_folder = os.path.join(image_dir, name, "mask")
        else:
            mask_src_path = None
            mask_folder = None

        input_extras_src_path = (
            cfg.D2GO_DATA.DATASETS.GAN_INJECTION.INPUT_EXTRAS_SRC_DIR
        )
        if PathManager.isfile(input_extras_src_path):
            input_extras_folder = os.path.join(image_dir, name, "input_extras")
        else:
            input_extras_src_path = None
            input_extras_folder = None

        real_src_path = cfg.D2GO_DATA.DATASETS.GAN_INJECTION.REAL_SRC_DIR
        if PathManager.isfile(real_src_path):
            real_folder = os.path.join(image_dir, name, "real")
            real_json_path = cfg.D2GO_DATA.DATASETS.GAN_INJECTION.REAL_JSON_PATH
            assert PathManager.isfile(real_json_path), "{} is not valid!".format(
                real_json_path
            )
        else:
            real_src_path = None
            real_folder = None
            real_json_path = None

        register_folder_dataset(
            name + "_train",
            json_path,
            input_folder,
            gt_folder,
            mask_folder,
            input_extras_folder,
            input_src_path,
            gt_src_path,
            mask_src_path,
            input_extras_src_path,
            real_json_path,
            real_folder,
            real_src_path,
        )

        register_folder_dataset(
            name + "_test",
            json_path,
            input_folder,
            gt_folder,
            mask_folder,
            input_extras_folder,
            input_src_path,
            gt_src_path,
            mask_src_path,
            input_extras_src_path,
            real_json_path,
            real_folder,
            real_src_path,
            max_num=cfg.D2GO_DATA.DATASETS.GAN_INJECTION.MAX_TEST_IMAGES,
        )
