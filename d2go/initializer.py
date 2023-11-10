#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import logging

from d2go.registry.bootstrap import bootstrap_registries
from mobile_cv.common.misc.oss_utils import fb_overwritable

logger = logging.getLogger(__name__)

_INITIALIZED = False


def initialize_all(boostrap_registries: bool = False) -> None:
    global _INITIALIZED
    if _INITIALIZED:
        return
    _INITIALIZED = True

    _initialize_all(boostrap_registries=boostrap_registries)


def _initialize_all(boostrap_registries: bool) -> None:
    _setup_env()
    _register_builtin_datasets()

    _populate_registries()
    if boostrap_registries:
        bootstrap_registries(enable_cache=True, catch_exception=True)


# fmt: off


@fb_overwritable()
def _setup_env():
    # register torch vision ops
    from torchvision.ops import nms  # noqa

    # setup Detectron2 environments
    from detectron2.utils.env import setup_environment as setup_d2_environment # isort:skip
    setup_d2_environment()


@fb_overwritable()
def _register_builtin_datasets():
    # Register D2 builtin datasets
    import detectron2.data  # noqa F401


@fb_overwritable()
def _populate_registries():
    logger.info("Populating registries ...")
    from d2go import optimizer  # noqa
    from d2go.data import dataset_mappers  # noqa
    from d2go.modeling.backbone import fbnet_v2  # noqa
    logger.info(f"populated registries for following modules: {optimizer} {dataset_mappers} {fbnet_v2}")


# fmt: on
