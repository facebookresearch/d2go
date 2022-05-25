#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from mobile_cv.common.misc.oss_utils import fb_overwritable


@fb_overwritable()
def _setup_env():
    # Set up custom environment before nearly anything else is imported
    # NOTE: this should be the first import (no not reorder)
    from detectron2.utils.env import (  # noqa F401 isort:skip
        setup_environment as d2_setup_environment,  # noqa
    )


@fb_overwritable()
def _register_d2_datasets():
    # this will register D2 builtin datasets
    import detectron2.data  # noqa F401


@fb_overwritable()
def _register():
    from d2go.data import dataset_mappers  # noqa
    from d2go.data.datasets import (  # noqa
        register_builtin_datasets,  # noqa
        register_json_datasets,  # noqa
    )
    from d2go.modeling.backbone import fbnet_v2  # noqa


@fb_overwritable()
def initialize_all():
    # exclude torch from timing
    from torchvision.ops import nms  # noqa

    _setup_env()
    _register_d2_datasets()
    _register()


_INITIALIZED = False
if not _INITIALIZED:
    initialize_all()
    _INITIALIZED = True
