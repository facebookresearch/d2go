#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved


import copy
import os
import unittest

import torch
from d2go.export.api import convert_and_export_predictor
from d2go.export.d2_meta_arch import patch_d2_meta_arch
from d2go.runner import GeneralizedRCNNRunner
from d2go.utils.testing.data_loader_helper import create_fake_detection_data_loader
from d2go.utils.testing.rcnn_helper import RCNNBaseTestCases, get_quick_test_config_opts
from mobile_cv.common.misc.file_utils import make_temp_directory

# Add APIs to D2's meta arch, this is usually called in runner's setup, however in
# unittest it needs to be called sperarately. (maybe we should apply this by default)
patch_d2_meta_arch()


class TestFBNetV3MaskRCNNFP32(RCNNBaseTestCases.TemplateTestCase):
    def setup_custom_test(self):
        super().setup_custom_test()
        self.cfg.merge_from_file("detectron2go://mask_rcnn_fbnetv3a_dsmask_C4.yaml")

    def test_inference(self):
        self._test_inference()

    @RCNNBaseTestCases.expand_parameterized_test_export(
        [
            ["torchscript@c2_ops", True],
            ["torchscript", True],
            ["torchscript_int8@c2_ops", False],
            ["torchscript_int8", False],
        ]
    )
    def test_export(self, predictor_type, compare_match):
        if os.getenv("OSSRUN") == "1" and "@c2_ops" in predictor_type:
            self.skipTest("Caffe2 is not available for OSS")
        self._test_export(predictor_type, compare_match=compare_match)


class TestFBNetV3MaskRCNNFPNFP32(RCNNBaseTestCases.TemplateTestCase):
    def setup_custom_test(self):
        super().setup_custom_test()
        self.cfg.merge_from_file("detectron2go://mask_rcnn_fbnetv3g_fpn.yaml")

    def test_inference(self):
        self._test_inference()

    @RCNNBaseTestCases.expand_parameterized_test_export(
        [
            ["torchscript@c2_ops", True],
            ["torchscript", True],
            ["torchscript_int8@c2_ops", False],
            ["torchscript_int8", False],
        ]
    )
    def test_export(self, predictor_type, compare_match):
        if os.getenv("OSSRUN") == "1" and "@c2_ops" in predictor_type:
            self.skipTest("Caffe2 is not available for OSS")
        self._test_export(predictor_type, compare_match=compare_match)


class TestFBNetV3MaskRCNNQATEager(RCNNBaseTestCases.TemplateTestCase):
    def setup_custom_test(self):
        super().setup_custom_test()
        self.cfg.merge_from_file("detectron2go://mask_rcnn_fbnetv3a_dsmask_C4.yaml")
        # enable QAT
        self.cfg.merge_from_list(
            [
                "QUANTIZATION.BACKEND",
                "qnnpack",
                "QUANTIZATION.QAT.ENABLED",
                "True",
            ]
        )
        # FIXME: NaiveSyncBN is not supported
        self.cfg.merge_from_list(["MODEL.FBNET_V2.NORM", "bn"])

    def test_inference(self):
        self._test_inference()

    @RCNNBaseTestCases.expand_parameterized_test_export(
        [
            ["torchscript_int8@c2_ops", False],  # TODO: fix mismatch
            ["torchscript_int8", False],  # TODO: fix mismatch
        ]
    )
    def test_export(self, predictor_type, compare_match):
        if os.getenv("OSSRUN") == "1" and "@c2_ops" in predictor_type:
            self.skipTest("Caffe2 is not available for OSS")
        self._test_export(predictor_type, compare_match=compare_match)


class TestFBNetV3KeypointRCNNFP32(RCNNBaseTestCases.TemplateTestCase):
    def setup_custom_test(self):
        super().setup_custom_test()
        self.cfg.merge_from_file("detectron2go://keypoint_rcnn_fbnetv3a_dsmask_C4.yaml")

        # FIXME: have to use qnnpack due to follow error:
        # Per Channel Quantization is currently disabled for transposed conv
        self.cfg.merge_from_list(
            [
                "QUANTIZATION.BACKEND",
                "qnnpack",
            ]
        )

    def test_inference(self):
        self._test_inference()

    @RCNNBaseTestCases.expand_parameterized_test_export(
        [
            ["torchscript_int8@c2_ops", False],  # TODO: fix mismatch
            ["torchscript_int8", False],  # TODO: fix mismatch
        ]
    )
    def test_export(self, predictor_type, compare_match):
        if os.getenv("OSSRUN") == "1" and "@c2_ops" in predictor_type:
            self.skipTest("Caffe2 is not available for OSS")
        self._test_export(predictor_type, compare_match=compare_match)


class TestTorchVisionExport(unittest.TestCase):
    def test_export_torchvision_format(self):
        runner = GeneralizedRCNNRunner()
        cfg = runner.get_default_cfg()
        cfg.merge_from_file("detectron2go://mask_rcnn_fbnetv3a_dsmask_C4.yaml")
        cfg.merge_from_list(get_quick_test_config_opts())

        cfg.merge_from_list(["MODEL.DEVICE", "cpu"])
        pytorch_model = runner.build_model(cfg, eval_only=True)

        from typing import List, Dict

        class Wrapper(torch.nn.Module):
            def __init__(self, model):
                super().__init__()
                self.model = model

            def forward(self, inputs: List[torch.Tensor]):
                x = inputs[0].unsqueeze(0) * 255
                scale = 320.0 / min(x.shape[-2], x.shape[-1])
                x = torch.nn.functional.interpolate(
                    x,
                    scale_factor=scale,
                    mode="bilinear",
                    align_corners=True,
                    recompute_scale_factor=True,
                )
                out = self.model(x[0])
                res: Dict[str, torch.Tensor] = {}
                res["boxes"] = out[0] / scale
                res["labels"] = out[2]
                res["scores"] = out[1]
                return inputs, [res]

        size_divisibility = max(pytorch_model.backbone.size_divisibility, 10)
        h, w = size_divisibility, size_divisibility * 2
        with create_fake_detection_data_loader(h, w, is_train=False) as data_loader:
            with make_temp_directory("test_export_torchvision_format") as tmp_dir:
                predictor_path = convert_and_export_predictor(
                    cfg,
                    copy.deepcopy(pytorch_model),
                    "torchscript",
                    tmp_dir,
                    data_loader,
                )

                orig_model = torch.jit.load(os.path.join(predictor_path, "model.jit"))
                wrapped_model = Wrapper(orig_model)
                # optionally do a forward
                wrapped_model([torch.rand(3, 600, 600)])
                scripted_model = torch.jit.script(wrapped_model)
                scripted_model.save(os.path.join(tmp_dir, "new_file.pt"))


if __name__ == "__main__":
    unittest.main()
