# Copyright (c) Facebook, Inc. and its affiliates.
import os
from typing import Optional

import pkg_resources
import torch
from d2go.runner import create_runner
from d2go.utils.launch_environment import MODEL_ZOO_STORAGE_PREFIX
from detectron2.checkpoint import DetectionCheckpointer


class _ModelZooUrls(object):
    """
    Mapping from names to officially released D2Go pre-trained models.
    """

    CONFIG_PATH_TO_URL_SUFFIX = {
        "faster_rcnn_fbnetv3a_C4.yaml": "268421013/model_final.pth",
        "faster_rcnn_fbnetv3a_dsmask_C4.yaml": "268412271/model_0499999.pth",
        "faster_rcnn_fbnetv3g_fpn.yaml": "250356938/model_0374999.pth",
        "mask_rcnn_fbnetv3a_C4.yaml": "268421013/model_final.pth",
        "mask_rcnn_fbnetv3a_dsmask_C4.yaml": "268412271/model_0499999.pth",
        "mask_rcnn_fbnetv3g_fpn.yaml": "287445123/model_0409999.pth",
        "keypoint_rcnn_fbnetv3a_dsmask_C4.yaml": "250430934/model_0389999.pth",
    }


def get_checkpoint_url(config_path):
    """
    Returns the URL to the model trained using the given config
    Args:
        config_path (str): config file name relative to d2go's "configs/"
            directory, e.g., "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_1x.yaml"
    Returns:
        str: a URL to the model
    """
    name = config_path.replace(".yaml", "")
    if config_path in _ModelZooUrls.CONFIG_PATH_TO_URL_SUFFIX:
        suffix = _ModelZooUrls.CONFIG_PATH_TO_URL_SUFFIX[config_path]
        return MODEL_ZOO_STORAGE_PREFIX + suffix
    raise RuntimeError("{} not available in Model Zoo!".format(name))


def get_config_file(config_path):
    """
    Returns path to a builtin config file.
    Args:
        config_path (str): config file name relative to d2go's "configs/"
            directory, e.g., "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_1x.yaml"
    Returns:
        str: the real path to the config file.
    """
    cfg_file = pkg_resources.resource_filename(
        "d2go", os.path.join("configs", config_path)
    )
    if not os.path.exists(cfg_file):
        raise RuntimeError("{} not available in Model Zoo!".format(config_path))
    return cfg_file


def get_config(
    config_path, trained: bool = False, runner="d2go.runner.GeneralizedRCNNRunner"
):
    """
    Returns a config object for a model in model zoo.
    Args:
        config_path (str): config file name relative to d2go's "configs/"
            directory, e.g., "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_1x.yaml"
        trained (bool): If True, will set ``MODEL.WEIGHTS`` to trained model zoo weights.
            If False, the checkpoint specified in the config file's ``MODEL.WEIGHTS`` is used
            instead; this will typically (though not always) initialize a subset of weights using
            an ImageNet pre-trained model, while randomly initializing the other weights.
    Returns:
        CfgNode: a config object
    """
    cfg_file = get_config_file(config_path)
    runner = create_runner(runner)
    cfg = runner.get_default_cfg()
    cfg.merge_from_file(cfg_file)
    if trained:
        cfg.MODEL.WEIGHTS = get_checkpoint_url(config_path)
    return cfg


def get(
    config_path,
    trained: bool = False,
    device: Optional[str] = None,
    runner="d2go.runner.GeneralizedRCNNRunner",
):
    """
    Get a model specified by relative path under Detectron2's official ``configs/`` directory.
    Args:
        config_path (str): config file name relative to d2go's "configs/"
            directory, e.g., "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_1x.yaml"
        trained (bool): see :func:`get_config`.
        device (str or None): overwrite the device in config, if given.
    Returns:
        nn.Module: a d2go model. Will be in training mode.
    Example:
    ::
        from d2go import model_zoo
        model = model_zoo.get("faster_rcnn_fbnetv3a_C4.yaml", trained=True)
    """
    cfg = get_config(config_path, trained)
    if device is not None:
        cfg.MODEL.DEVICE = device
    elif not torch.cuda.is_available():
        cfg.MODEL.DEVICE = "cpu"

    runner = create_runner(runner)
    model = runner.build_model(cfg)
    DetectionCheckpointer(model).load(cfg.MODEL.WEIGHTS)
    return model
