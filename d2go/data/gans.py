#!/usr/bin/env python3

"""
This file contains utilities to load GANs datasets.

Similar to how COCO dataset is represented in Detectron2, a GANs dataset is represented
as a list of dicts, where each dict is in "standard dataset dict" format, which contains
raw data with fields such as:
    - input_path (str): filename of input image
    - fg_path (str): filename to the GT
    ...
"""

import os
import json
import logging
from fvcore.common.file_io import PathManager


logger = logging.getLogger(__name__)


IMG_EXTENSIONS = ['.jpg', '.JPG', '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP']


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
                    "gt_path": gt_path
                }
                data.append(f)
                if image_root.rfind("test") != -1 and len(data) == 5000:
                    logger.info("Reach maxinum of test data: {} ".format(len(data)))
                    return data
    logger.info("Total number of data dicts: {} ".format(len(data)))
    return data


def load_pix2pix_json(json_path, input_folder, gt_folder, mask_folder, max_num=-1):
    """
    Args:
        json_path (str): the directory where the json file exists which saves the filenames and labels.
        input_folder (str): the directory for the input/source images
        input_folder (str): the directory for the ground_truth/target images
        mask_folder (str): the directory for the masks
    Returns:
        list[dict]: a list of dicts
    """
    data = []
    with PathManager.open(json_path, 'r') as f:
        filenames = json.load(f)
        for fname in filenames.keys():
            f = {
                "file_name": fname,
                "input_folder": input_folder,
                "gt_folder": gt_folder,
                "mask_folder": mask_folder,
                "input_label": filenames[fname],
            }
            data.append(f)
            # 5000 is the general number of images used to calculate FID in GANs
            if max_num > 0 and len(data) == max_num:
                logger.info("Reach maxinum of test data: {} ".format(len(data)))
                return data
    logger.info("Total number of data dicts: {} ".format(len(data)))
    return data
