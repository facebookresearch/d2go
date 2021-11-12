#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

from .build import D2GO_DATA_MAPPER_REGISTRY
from .d2go_dataset_mapper_impl import (
    D2GoDatasetMapper,
    PREFETCHED_FILE_NAME,
    PREFETCHED_SEM_SEG_FILE_NAME,
    read_image_with_prefetch,
    read_sem_seg_file_with_prefetch,
)

__all__ = [
    "D2GoDatasetMapper",
    "PREFETCHED_FILE_NAME",
    "PREFETCHED_SEM_SEG_FILE_NAME",
    "read_image_with_prefetch",
    "read_sem_seg_file_with_prefetch",
]


# NOTE: D2GoDatasetMapper might have different versions between internal and oss code,
# which causes double-registration if the class is registered at declaration.
D2GO_DATA_MAPPER_REGISTRY.register(D2GoDatasetMapper)
