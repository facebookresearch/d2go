#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

# NOTE: making necessary imports to register with Registry
# @fb-only: from d2go.modeling.meta_arch import fb  # isort:skip  # noqa 
from d2go.modeling.meta_arch import (  # noqa
    fcos,
    panoptic_fpn,
    rcnn,
    retinanet,
    semantic_seg,
)
