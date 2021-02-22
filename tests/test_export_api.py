#!/usr/bin/env python3

import os
import torch
import torch.nn as nn
import unittest

from d2go.export.api import convert_and_export_predictor, PredictorExportConfig
from mobile_cv.common.misc.file_utils import make_temp_directory
from mobile_cv.predictor.api import create_predictor


class SimpleModel(nn.Module):
    def forward(self, x):
        return 2 * x

    def prepare_for_export(self, cfg, inputs, export_scheme):
        # pre/post processing and run_func are default values
        return PredictorExportConfig(
            model=self,
            # model(x) -> model(*(x,))
            data_generator=lambda x: (x, ),
        )


class TestExportAPI(unittest.TestCase):
    def test_simple_model(self):
        with make_temp_directory("TestExportAPI") as tmp_dir:
            cfg = None  # cfg is not used for simple model
            model = SimpleModel()
            predictor_path = convert_and_export_predictor(
                cfg,
                model,
                predictor_type="torchscript",
                output_dir=tmp_dir,
                data_loader=iter([
                    torch.tensor(1),
                    torch.tensor(2),
                    torch.tensor(3)
                ]),
            )
            self.assertTrue(os.path.isdir(predictor_path))

            # also test loading predictor
            predictor = create_predictor(predictor_path)
            x = torch.tensor(42)
            self.assertEqual(predictor(x), model(x))
