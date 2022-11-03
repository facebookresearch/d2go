#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved


import os
import unittest
from typing import List

import torch
import torch.nn as nn
from d2go.export.api import FuncInfo, PredictorExportConfig
from d2go.export.exporter import convert_and_export_predictor
from d2go.export.torchscript import (
    DefaultTorchscriptExport,
    TracingAdaptedTorchscriptExport,
)
from mobile_cv.common.misc.file_utils import make_temp_directory
from mobile_cv.predictor.api import create_predictor
from parameterized import parameterized


class SimpleModel(nn.Module):
    def forward(self, x):
        return 2 * x

    def prepare_for_export(self, cfg, inputs, predictor_type):
        # pre/post processing and run_func are default values
        return PredictorExportConfig(
            model=self,
            # model(x) -> model(*(x,))
            data_generator=lambda x: (x,),
        )


class TwoPartSimpleModel(nn.Module):
    """
    Suppose there're some function in the middle that can't be traced, therefore we
    need to export the model as two parts.
    """

    def __init__(self):
        super().__init__()
        self.part1 = SimpleModel()
        self.part2 = SimpleModel()

    def forward(self, x):
        x = self.part1(x)
        x = TwoPartSimpleModel.non_traceable_func(x)
        x = self.part2(x)
        return x

    def prepare_for_export(self, cfg, inputs, predictor_type):
        def data_generator(x):
            part1_args = (x,)
            x = self.part1(x)
            x = TwoPartSimpleModel.non_traceable_func(x)
            part2_args = (x,)
            return {"part1": part1_args, "part2": part2_args}

        return PredictorExportConfig(
            model={"part1": self.part1, "part2": self.part2},
            data_generator=data_generator,
            run_func_info=FuncInfo.gen_func_info(TwoPartSimpleModel.RunFunc, params={}),
        )

    @staticmethod
    def non_traceable_func(x):
        return x + 1 if len(x.shape) > 3 else x - 1

    class RunFunc(object):
        def __call__(self, model, x):
            assert isinstance(model, dict)
            x = model["part1"](x)
            x = TwoPartSimpleModel.non_traceable_func(x)
            x = model["part2"](x)
            return x


class ScriptingOnlyModel(nn.Module):
    """
    Example of a model that requires scripting (eg. having control loop).
    """

    def forward(self, inputs: List[torch.Tensor]) -> List[torch.Tensor]:
        outputs = []
        for i, t in enumerate(inputs):
            outputs.append(t * i)
        return outputs

    def prepare_for_export(self, cfg, inputs, predictor_type):
        if cfg == "explicit":
            return PredictorExportConfig(
                model=self,
                data_generator=None,  # data is not needed for scripting
                model_export_kwargs={
                    "jit_mode": "script"
                },  # explicitly using script mode
            )
        elif cfg == "implicit":
            # Sometime user wants to switch between scripting and tracing without
            # touching the PredictorExportConfig
            return PredictorExportConfig(
                model=self,
                data_generator=None,  # data is not needed for scripting
            )
        raise NotImplementedError()


class TestExportAPI(unittest.TestCase):
    def _export_simple_model(self, cfg, model, data, output_dir, predictor_type):
        predictor_path = convert_and_export_predictor(
            cfg,
            model,
            predictor_type=predictor_type,
            output_dir=output_dir,
            data_loader=iter([data] * 3),
        )
        self.assertTrue(os.path.isdir(predictor_path))

        # also test loading predictor
        predictor = create_predictor(predictor_path)
        return predictor

    def test_simple_model(self):
        with make_temp_directory("test_simple_model") as tmp_dir:
            model = SimpleModel()
            predictor = self._export_simple_model(
                None, model, torch.tensor(1), tmp_dir, predictor_type="torchscript"
            )
            x = torch.tensor(42)
            self.assertEqual(predictor(x), model(x))

    def test_simple_two_part_model(self):
        with make_temp_directory("test_simple_two_part_model") as tmp_dir:
            model = TwoPartSimpleModel()
            predictor = self._export_simple_model(
                None, model, torch.tensor(1), tmp_dir, predictor_type="torchscript"
            )
            x = torch.tensor(42)
            self.assertEqual(predictor(x), model(x))

    def test_script_only_model(self):
        def _validate(predictor):
            outputs = predictor([torch.tensor(1), torch.tensor(2), torch.tensor(3)])
            self.assertEqual(len(outputs), 3)
            self.assertEqual(
                outputs, [torch.tensor(0), torch.tensor(2), torch.tensor(6)]
            )

        # Method 1: explicitly set jit_mode to "trace"
        with make_temp_directory("test_test_script_only_model") as tmp_dir:
            model = ScriptingOnlyModel()
            predictor = self._export_simple_model(
                "explicit", model, None, tmp_dir, predictor_type="torchscript"
            )
            _validate(predictor)

        # Method 2: using torchscript@scripting as predictor type
        with make_temp_directory("test_test_script_only_model") as tmp_dir:
            model = ScriptingOnlyModel()
            predictor = self._export_simple_model(
                "implicit", model, None, tmp_dir, predictor_type="torchscript@scripting"
            )
            _validate(predictor)


class MultiTensorInSingleTensorOut(nn.Module):
    def forward(self, x, y):
        return x + y

    @staticmethod
    def get_input_args():
        return (torch.tensor([2]), torch.tensor([3]))

    @staticmethod
    def check_outputs(new_output, original_output):
        torch.testing.assert_close(new_output, torch.tensor([5]))


# NOTE: caffe2 wrapper assumes tensors are fp32
class SingleListInSingleListOut(nn.Module):
    def forward(self, inputs):
        x, y = inputs
        return [x + y]

    @staticmethod
    def get_input_args():
        inputs = [torch.tensor([2.0]), torch.tensor([3.0])]
        return (inputs,)

    @staticmethod
    def check_outputs(new_output, original_output):
        assert len(new_output) == 1
        torch.testing.assert_close(new_output[0], torch.tensor([5.0]))


class MultiDictInMultiDictOut(nn.Module):
    def forward(self, x, y):
        first = {"add": x["first"] + y["first"], "sub": x["first"] - y["first"]}
        second = {"add": x["second"] + y["second"], "sub": x["second"] - y["second"]}
        return [first, second]

    @staticmethod
    def get_input_args():
        return (
            {"first": torch.tensor([1]), "second": torch.tensor([2])},  # x
            {"first": torch.tensor([3]), "second": torch.tensor([4])},  # y
        )

    @staticmethod
    def check_outputs(new_output, original_output):
        first, second = original_output
        torch.testing.assert_close(first["add"], torch.tensor([4]))
        torch.testing.assert_close(first["sub"], torch.tensor([-2]))
        torch.testing.assert_close(second["add"], torch.tensor([6]))
        torch.testing.assert_close(second["sub"], torch.tensor([-2]))


MODEL_EXPORT_METHOD_TEST_CASES = [
    [DefaultTorchscriptExport, MultiTensorInSingleTensorOut],
    [DefaultTorchscriptExport, SingleListInSingleListOut],
    [TracingAdaptedTorchscriptExport, MultiTensorInSingleTensorOut],
    [TracingAdaptedTorchscriptExport, SingleListInSingleListOut],
    [TracingAdaptedTorchscriptExport, MultiDictInMultiDictOut],
]


try:
    from d2go.export.fb.caffe2 import DefaultCaffe2Export

    MODEL_EXPORT_METHOD_TEST_CASES.extend(
        [
            # [DefaultCaffe2Export, MultiTensorInSingleTensorOut],  # TODO: make caffe2 support this
            [DefaultCaffe2Export, SingleListInSingleListOut],
        ]
    )
except ImportError:
    pass


class TestModelExportMethods(unittest.TestCase):
    @parameterized.expand(
        MODEL_EXPORT_METHOD_TEST_CASES,
        name_func=lambda testcase_func, param_num, param: (
            "{}_{}_{}".format(
                testcase_func.__name__, param.args[0].__name__, param.args[1].__name__
            )
        ),
    )
    def test_interface(self, model_export_method, test_model_class):
        model = test_model_class()
        input_args = test_model_class.get_input_args()
        output_checker = test_model_class.check_outputs
        model_export_method.test_export_and_load(
            model, input_args, None, {}, output_checker
        )
