#!/usr/bin/env python3

import logging
import torch

from caffe2.python import dyndep
dyndep.InitOpsLibrary("//caffe2/caffe2/share/fb/mask_rcnn:bbox_concat_batch_splits_op")

logger = logging.getLogger(__name__)


def concat_bbox_with_batch_splits(boxes, batch_splits=None):
    inputs = [boxes, batch_splits] if batch_splits is not None else [boxes]
    return torch.ops._caffe2.BBoxConcatBatchSplits(inputs)
