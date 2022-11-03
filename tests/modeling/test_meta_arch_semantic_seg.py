#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved


import os
import shutil
import tempfile
import unittest

import torch
from d2go.export.exporter import convert_and_export_predictor
from d2go.runner import Detectron2GoRunner
from mobile_cv.predictor.api import create_predictor


def _get_batch(height, width, is_train):
    def _get_frame():
        random_image = torch.rand(3, height, width).to(torch.float32)
        ret = {"image": random_image}
        if is_train:
            mask_size = (height, width)
            random_mask = torch.randint(low=0, high=2, size=mask_size).to(torch.int64)
            ret["sem_seg"] = random_mask
        return ret

    batch_size = 2 if is_train else 1
    return [
        {"filename": "some_file", "width": 100, "height": 100, **_get_frame()}
        for _ in range(batch_size)
    ]


def _get_data_loader(height, width, is_train):
    inputs = _get_batch(height, width, is_train)

    def get_data_loader():
        while True:
            yield inputs

    return get_data_loader()


def _get_input_dim(model):
    h = w = max(model.backbone.size_divisibility, 1)
    return h, w


class BaseSemanticSegTestCase:
    class TemplateTestCase(unittest.TestCase):
        def setUp(self):
            self.test_dir = tempfile.mkdtemp(prefix="test_meta_arch_semantic_seg_")
            self.addCleanup(shutil.rmtree, self.test_dir)

            runner = Detectron2GoRunner()
            self.cfg = runner.get_default_cfg()
            self.setup_custom_test()

            self.cfg.merge_from_list(["MODEL.DEVICE", "cpu"])
            self.test_model = runner.build_model(self.cfg, eval_only=True)

        def setup_custom_test(self):
            raise NotImplementedError()

        def test_inference(self):
            h, w = _get_input_dim(self.test_model)
            inputs = _get_batch(h, w, False)
            with torch.no_grad():
                self.test_model(inputs)

        def test_train(self):
            h, w = _get_input_dim(self.test_model)
            inputs = _get_batch(h, w, True)
            self.test_model.train()
            loss_dict = self.test_model(inputs)
            losses = sum(loss_dict.values())
            losses.backward()

        def _test_export(self, predictor_type, compare_match=True):
            h, w = _get_input_dim(self.test_model)
            dl = _get_data_loader(h, w, False)
            inputs = next(iter(dl))

            output_dir = os.path.join(self.test_dir, "test_export")
            predictor_path = convert_and_export_predictor(
                self.cfg, self.test_model, predictor_type, output_dir, dl
            )

            predictor = create_predictor(predictor_path)
            predicotr_outputs = predictor(inputs)
            self.assertEqual(len(predicotr_outputs), len(inputs))

            with torch.no_grad():
                pytorch_outputs = self.test_model(inputs)
                self.assertEqual(len(pytorch_outputs), len(inputs))

            if compare_match:
                for predictor_output, pytorch_output in zip(
                    predicotr_outputs, pytorch_outputs
                ):
                    torch.testing.assert_close(
                        predictor_output["sem_seg"], pytorch_output["sem_seg"]
                    )


class TestR50FPN(BaseSemanticSegTestCase.TemplateTestCase):
    def setup_custom_test(self):
        self.cfg.merge_from_file("detectron2://Misc/semantic_R_50_FPN_1x.yaml")
        # discard pretrained backbone weights
        self.cfg.merge_from_list(["MODEL.WEIGHTS", ""])

    def test_export_torchscript(self):
        self._test_export("torchscript", compare_match=True)
