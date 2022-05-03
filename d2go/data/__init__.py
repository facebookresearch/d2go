#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved


# Logger
import logging

from d2go.data.dataset_mappers.build import build_dataset_mapper

# Mapped train loader OSS/FB switching
from .build import build_mapped_train_loader # @oss-only
from .fb.build import build_mapped_train_loader  # @fb-only #noqa

# Extended COCO OSS/FB switching
from .datasets import _register_extended_coco # @oss-only
from .fb.datasets import _register_extended_coco  # @fb-only # noqa

logger = logging.getLogger(__name__)

# In __init__ because the behaviour depends on whether it is on OSS or on FB code
def build_d2go_train_loader(cfg, mapper=None):
    """
    Build the dataloader for training in D2Go. This is the main entry and customizations
    will be done by using Registry.

    This interface is currently experimental.
    """
    logger.info("Building D2Go's train loader ...")
    # TODO: disallow passing mapper and use registry for all mapper registering
    mapper = mapper or build_dataset_mapper(cfg, is_train=True)
    logger.info("Using dataset mapper:\n{}".format(mapper))

    data_loader = build_mapped_train_loader(cfg, mapper)

    # TODO: decide if move vis_wrapper inside this interface
    return data_loader
