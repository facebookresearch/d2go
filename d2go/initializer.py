#!/usr/bin/env python3

import time


SETUP_ENV_TIME = []
REGISTER_D2_DATASETS_TIME = []
REGISTER_TIME = []


def _record_times(time_list):
    def warp(f):
        def timed_f(*args, **kwargs):
            start = time.perf_counter()
            ret = f(*args, **kwargs)
            time_list.append(time.perf_counter() - start)
            return ret

        return timed_f

    return warp


@_record_times(SETUP_ENV_TIME)
def _setup_env():
    # Set up custom environment before nearly anything else is imported
    # NOTE: this should be the first import (no not reorder)
    from detectron2.utils.env import (  # noqa F401 isort:skip
        setup_environment as d2_setup_environment,
    )


@_record_times(REGISTER_D2_DATASETS_TIME)
def _register_d2_datasets():
    # this will register D2 builtin datasets
    import detectron2.data  # noqa F401


@_record_times(REGISTER_TIME)
def _register():
    from d2go.modeling.backbone import (  # NOQA
        fbnet_v2,
    )
    from d2go.data import dataset_mappers # NOQA
    from d2go.data.datasets import (
        register_json_datasets,
        register_builtin_datasets,
    )

    #register_json_datasets()
    #register_builtin_datasets()


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
