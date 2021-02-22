#!/usr/bin/env python3

import contextlib
import copy
import os
import unittest

import torch
from d2go.export.api import convert_and_export_predictor
from d2go.export.d2_meta_arch import patch_d2_meta_arch
from d2go.runner import create_runner, GeneralizedRCNNRunner
from mobile_cv.common.misc.file_utils import make_temp_directory
from mobile_cv.predictor.api import create_predictor

from .data_loader_helper import LocalImageGenerator, register_toy_dataset

# Add APIs to D2's meta arch, this is usually called in runner's setup, however in
# unittest it needs to be called sperarately. (maybe we should apply this by default)
patch_d2_meta_arch()


@contextlib.contextmanager
def create_fake_detection_data_loader(height, width, is_train):
    with make_temp_directory("detectron2go_tmp_dataset") as dataset_dir:
        runner = create_runner("d2go.runner.GeneralizedRCNNRunner")
        cfg = runner.get_default_cfg()
        cfg.DATASETS.TRAIN = ["default_dataset_train"]
        cfg.DATASETS.TEST = ["default_dataset_test"]

        with make_temp_directory("detectron2go_tmp_dataset") as dataset_dir:
            image_dir = os.path.join(dataset_dir, "images")
            os.makedirs(image_dir)
            image_generator = LocalImageGenerator(image_dir, width=width, height=height)

            if is_train:
                with register_toy_dataset(
                    "default_dataset_train", image_generator, num_images=3
                ):
                    train_loader = runner.build_detection_train_loader(cfg)
                    yield train_loader
            else:
                with register_toy_dataset(
                    "default_dataset_test", image_generator, num_images=3
                ):
                    test_loader = runner.build_detection_test_loader(
                        cfg, dataset_name="default_dataset_test"
                    )
                    yield test_loader


def _validate_outputs(inputs, outputs):
    assert len(inputs) == len(outputs)
    # TODO: figure out how to validate outputs


class BaseTestCases:
    class TemplateTestCase(unittest.TestCase):  # TODO: maybe subclass from TestMetaArch
        def setup_custom_test(self):
            raise NotImplementedError()

        def setUp(self):
            runner = GeneralizedRCNNRunner()
            self.cfg = runner.get_default_cfg()
            self.is_mcs = False
            self.setup_custom_test()

            # NOTE: change some config to make the model run fast
            epsilon = 1e-4
            self.cfg.merge_from_list(
                [
                    str(x)
                    for x in [
                        "MODEL.RPN.POST_NMS_TOPK_TEST",
                        1,
                        "TEST.DETECTIONS_PER_IMAGE",
                        1,
                        "MODEL.PROPOSAL_GENERATOR.MIN_SIZE",
                        0,
                        "MODEL.RPN.NMS_THRESH",
                        1.0 + epsilon,
                        "MODEL.ROI_HEADS.NMS_THRESH_TEST",
                        1.0 + epsilon,
                        "MODEL.ROI_HEADS.SCORE_THRESH_TEST",
                        0.0 - epsilon,
                    ]
                ]
            )

            self.cfg.merge_from_list(["MODEL.DEVICE", "cpu"])
            self.test_model = runner.build_model(self.cfg, eval_only=True)

        def _test_export(self, predictor_type, compare_match=True):
            size_divisibility = max(self.test_model.backbone.size_divisibility, 10)
            h, w = size_divisibility, size_divisibility * 2
            with create_fake_detection_data_loader(h, w, is_train=False) as data_loader:
                inputs = next(iter(data_loader))

                with make_temp_directory(
                    "test_export_{}".format(predictor_type)
                ) as tmp_dir:
                    # TODO: the export may change model it self, need to fix this
                    model_to_export = copy.deepcopy(self.test_model)
                    predictor_path = convert_and_export_predictor(
                        self.cfg, model_to_export, predictor_type, tmp_dir, data_loader
                    )

                    predictor = create_predictor(predictor_path)
                    predicotr_outputs = predictor(inputs)
                    _validate_outputs(inputs, predicotr_outputs)

                    def _check_close(a, b):
                        self.assertLess(torch.abs(a - b).sum().sum(), 1e-3)

                    if compare_match:
                        with torch.no_grad():
                            pytorch_outputs = self.test_model(inputs)

                        pred = predicotr_outputs[0]["instances"].pred_boxes.tensor
                        ref = pytorch_outputs[0]["instances"].pred_boxes.tensor
                        _check_close(ref, pred)

        def test_inference(self):
            size_divisibility = max(self.test_model.backbone.size_divisibility, 10)
            h, w = size_divisibility, size_divisibility * 2

            with create_fake_detection_data_loader(h, w, is_train=False) as data_loader:
                inputs = next(iter(data_loader))

            with torch.no_grad():
                outputs = self.test_model(inputs)
            _validate_outputs(inputs, outputs)


class TestMaskRCNNNormal(BaseTestCases.TemplateTestCase):
    def setup_custom_test(self):
        self.cfg.merge_from_file("detectron2go://e2e_mask_rcnn_fbnet.yaml")

    def test_export(self):
        self._test_export("caffe2", compare_match=True)
        self._test_export("torchscript_int8", compare_match=False)


class TestMaskRCNNQATEager(BaseTestCases.TemplateTestCase):
    def setup_custom_test(self):
        self.cfg.merge_from_file("detectron2go://e2e_mask_rcnn_fbnet.yaml")
        # enable QAT
        self.cfg.merge_from_list(
            [
                "QUANTIZATION.BACKEND",
                "qnnpack",
                "QUANTIZATION.QAT.ENABLED",
                "True",
            ]
        )

    def test_export(self):
        self._test_export("torchscript_int8", compare_match=False)  # TODO: fix mismatch


class TestKeypointRCNNNormal(BaseTestCases.TemplateTestCase):
    def setup_custom_test(self):
        self.cfg.merge_from_file(
            "detectron2go://e2e_keypoint_mask_rcnn_fbnet_default.yaml"
        )

        # FIXME: have to use qnnpack due to follow error:
        # Per Channel Quantization is currently disabled for transposed conv
        self.cfg.merge_from_list(
            [
                "QUANTIZATION.BACKEND",
                "qnnpack",
            ]
        )

    def test_export(self):
        self._test_export("caffe2", compare_match=True)
        self._test_export("torchscript_int8", compare_match=False)


if __name__ == "__main__":
    unittest.main()
