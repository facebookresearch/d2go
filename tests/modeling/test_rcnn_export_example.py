#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import os
import unittest

import torch
from d2go.runner.default_runner import GeneralizedRCNNRunner
from d2go.tools.exporter import main
from d2go.utils.testing.data_loader_helper import create_local_dataset
from d2go.utils.testing.rcnn_helper import get_quick_test_config_opts
from mobile_cv.common.misc.file_utils import make_temp_directory


def maskrcnn_export_caffe2_vs_torchvision_opset_format_example():
    with make_temp_directory("export_demo") as tmp_dir:
        # use a fake dataset for ci
        dataset_name = create_local_dataset(tmp_dir, 5, 224, 224)
        config_list = [
            "DATASETS.TRAIN",
            (dataset_name,),
            "DATASETS.TEST",
            (dataset_name,),
        ]
        # START_WIKI_EXAMPLE_TAG
        runner = GeneralizedRCNNRunner()
        cfg = runner.get_default_cfg()
        cfg.merge_from_file("detectron2go://mask_rcnn_fbnetv3a_dsmask_C4.yaml")
        cfg.merge_from_list(get_quick_test_config_opts())
        cfg.merge_from_list(config_list)

        # equivalent to running:
        #   exporter.par --runner GeneralizedRCNNRunner --config-file config.yaml --predictor-types torchscript tourchscript@c2_ops --output-dir tmp_dir
        _ = main(
            cfg,
            tmp_dir,
            runner,
            predictor_types=["torchscript@c2_ops", "torchscript"],
        )

        # the path can be fetched from the return of main, here just use hard-coded values
        torchvision_ops_model = torch.jit.load(
            os.path.join(tmp_dir, "torchscript", "model.jit")
        )
        caffe2_ops_model = torch.jit.load(
            os.path.join(tmp_dir, "torchscript@c2_ops", "model.jit")
        )

        # Running inference using torchvision-style format
        image = torch.zeros(1, 64, 96)  # chw 3D tensor
        torchvision_style_outputs = torchvision_ops_model(
            image
        )  # suppose N instances are detected
        # NOTE: the output are flattened tensors of the real output (which is a dict), they're
        # ordered by the key in dict, which is deterministic for the given model, but it might
        # be difficult to figure out just from model.jit file. The predictor_info.json from
        # the same directory contains the `outputs_schema`, which indicate how the final output
        # is constructed from flattened tensors.
        pred_boxes = torchvision_style_outputs[0]  # torch.Size([N, 4])
        pred_classes = torchvision_style_outputs[1]  # torch.Size([N])
        pred_masks = torchvision_style_outputs[2]  # torch.Size([N, 1, Hmask, Wmask])
        scores = torchvision_style_outputs[3]  # torch.Size([N])

        # Running inference using caffe2-style format
        data = torch.zeros(1, 1, 64, 96)
        im_info = torch.tensor([[64, 96, 1.0]])
        caffe2_style_outputs = caffe2_ops_model([data, im_info])
        # NOTE: the output order is determined in the order of creating the tensor during
        # forward function, it's also follow the order of original Caffe2 model.
        roi_bbox_nms = caffe2_style_outputs[0]  # torch.Size([N, 4])
        roi_score_nms = caffe2_style_outputs[1]  # torch.Size([N])
        roi_class_nms = caffe2_style_outputs[2]  # torch.Size([N])
        mask_fcn_probs = caffe2_style_outputs[3]  # torch.Size([N, Cmask, Hmask, Wmask])

        # relations between torchvision-style outputs and caffe2-style outputs
        torch.testing.assert_allclose(pred_boxes, roi_bbox_nms)
        torch.testing.assert_allclose(pred_classes, roi_class_nms)
        torch.testing.assert_allclose(
            pred_masks, mask_fcn_probs[:, roi_class_nms.to(torch.int64), :, :]
        )
        torch.testing.assert_allclose(scores, roi_score_nms)
        # END_WIKI_EXAMPLE_TAG


class TestOptimizer(unittest.TestCase):
    @unittest.skipIf(os.getenv("OSSRUN") == "1", "Caffe2 is not available for OSS")
    def test_maskrcnn_export_caffe2_vs_torchvision_opset_format_example(self):
        maskrcnn_export_caffe2_vs_torchvision_opset_format_example()
