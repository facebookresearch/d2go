#!/usr/bin/env python3

import torch.nn as nn
from detectron2.export.c10 import Caffe2Boxes, InstancesList
from detectron2.export.caffe2_modeling import Caffe2GeneralizedRCNN
from detectron2.export.shared import mock_torch_nn_functional_interpolate


class TraceableTracker(Caffe2GeneralizedRCNN):
    """
    This is a copy of Caffe2GeneralizedRCNN detector but uses the
    input proposals instead of running proposal_generator in the
    forward pass in order to create a tracker.

    Similar to Caffe2GeneralizedRCNN, the input is a PyTorch detection
    model which this stores as self._wrapped_model.

    The c2_postprocess step is copied out of the proposal_generator
    (Caffe2RPN: fbcode/vision/fair/detectron2/detectron2/export/c10.py)
    so that the input to roi_heads has the same format (InstancesLIst)
    as when running the detector.
    """

    def c2_postprocess(self, im_info, rpn_rois):
        proposals = InstancesList(
            im_info=im_info,
            indices=rpn_rois[:, 0],
            extra_fields={"proposal_boxes": Caffe2Boxes(rpn_rois)},
        )
        if not self.tensor_mode:
            proposals = InstancesList.to_d2_instances_list(proposals)
        else:
            proposals = [proposals]
        return proposals

    @mock_torch_nn_functional_interpolate()
    def forward(self, inputs):
        if not self.tensor_mode:
            return self._wrapped_model.inference(inputs)

        data, im_info, rois = inputs
        images = super()._caffe2_preprocess_image((data, im_info))
        features = self._wrapped_model.backbone(images.tensor)
        proposals = self.c2_postprocess(im_info, rois)
        with self.roi_heads_patcher.mock_roi_heads():
            detector_results, _ = self._wrapped_model.roi_heads(
                images, features, proposals
            )
        return tuple(detector_results[0].flatten())


class DetectAndTrack(nn.Module):
    def __init__(self, script_detector, script_tracker):
        super().__init__()
        self.detect = script_detector
        self.track = script_tracker

    def forward(self, image, im_info, run_track, rois):
        if run_track:
            return self.track((image, im_info, rois))
        else:
            return self.detect((image, im_info))
