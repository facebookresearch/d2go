#!/usr/bin/env python3

import logging

from detectron2.export.caffe2_inference import ProtobufDetectionModel
from d2go.config import temp_defrost

logger = logging.getLogger(__name__)


def infer_mask_on(model: ProtobufDetectionModel):
    # the real self.assembler should tell about this, currently use heuristic
    possible_blob_names = {"mask_fcn_probs"}
    return any(
        possible_blob_names.intersection(op.output)
        for op in model.protobuf_model.net.Proto().op
    )


def infer_keypoint_on(model: ProtobufDetectionModel):
    # the real self.assembler should tell about this, currently use heuristic
    possible_blob_names = {"kps_score"}
    return any(
        possible_blob_names.intersection(op.output)
        for op in model.protobuf_model.net.Proto().op
    )


def infer_densepose_on(model: ProtobufDetectionModel):
    possible_blob_names = {"AnnIndex", "Index_UV", "U_estimated", "V_estimated"}
    return any(
        possible_blob_names.intersection(op.output)
        for op in model.protobuf_model.net.Proto().op
    )


def _update_if_true(cfg, key, value):
    if not value:
        return

    keys = key.split(".")
    ref_value = cfg
    while len(keys):
        ref_value = getattr(ref_value, keys.pop(0))

    if ref_value != value:
        logger.warning(
            "There's conflict between cfg and model, overwrite config {} from {} to {}"
            .format(key, ref_value, value)
        )
        cfg.merge_from_list([key, value])


def update_cfg_from_pb_model(cfg, model):
    """
    Update cfg statically based given caffe2 model, in cast that there's conflict
    between caffe2 model and the cfg, caffe2 model has higher priority.
    """
    with temp_defrost(cfg):
        _update_if_true(cfg, "MODEL.MASK_ON", infer_mask_on(model))
        _update_if_true(cfg, "MODEL.KEYPOINT_ON", infer_keypoint_on(model))
        _update_if_true(cfg, "MODEL.DENSEPOSE_ON", infer_densepose_on(model))
    return cfg
