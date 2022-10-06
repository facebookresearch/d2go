#!/usr/bin/env fbpython
# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

import unittest

import torch
from d2go.quantization.learnable_qat import iterate_module_named_parameters


class TestLearnableQat(unittest.TestCase):
    @staticmethod
    def get_test_model():
        class TestModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight_norm = torch.nn.Parameter(
                    torch.tensor([[1, 2, 3], [4, 5, 6]]).to(dtype=torch.float32),
                    requires_grad=True,
                )
                self.weight_rnd = torch.nn.Parameter(
                    torch.tensor([[2, 2, 2], [1, 4, 16]]).to(dtype=torch.float32),
                    requires_grad=True,
                )
                self.weight_frozen = torch.nn.Parameter(
                    torch.tensor([[1, 2, 3], [4, 5, 6]]).to(dtype=torch.float32),
                    requires_grad=False,
                )

        model = TestModel()
        model.sub_module_L1 = TestModel()
        model.sub_module_L1.sub_sub_module = TestModel()
        return model

    def _fetch_params(self, model, check_requires_grad=True, reg_exps=None):
        result = set()
        for module_name, _, param_name, _ in iterate_module_named_parameters(
            model, check_requires_grad, reg_exps
        ):
            result.add(("" if not module_name else module_name + ".") + param_name)
        return result

    def test_iterate_module_named_parameters(self) -> None:
        test_model = TestLearnableQat.get_test_model()

        EXPECTED_RESULT_ALL_TRAIN_PARAM = {
            "weight_norm",
            "weight_rnd",
            "sub_module_L1.weight_norm",
            "sub_module_L1.weight_rnd",
            "sub_module_L1.sub_sub_module.weight_norm",
            "sub_module_L1.sub_sub_module.weight_rnd",
        }
        self.assertSetEqual(
            self._fetch_params(test_model), EXPECTED_RESULT_ALL_TRAIN_PARAM
        )

        result = set()
        for module_name, _, param_name, _ in iterate_module_named_parameters(
            test_model,
        ):
            result.add(module_name + "." + param_name)

        EXPECTED_RESULT_NOT_REQUIRE_GRAD = {
            "weight_norm",
            "weight_rnd",
            "weight_frozen",
            "sub_module_L1.weight_norm",
            "sub_module_L1.weight_rnd",
            "sub_module_L1.weight_frozen",
            "sub_module_L1.sub_sub_module.weight_norm",
            "sub_module_L1.sub_sub_module.weight_rnd",
            "sub_module_L1.sub_sub_module.weight_frozen",
        }
        self.assertSetEqual(
            self._fetch_params(test_model, check_requires_grad=False),
            EXPECTED_RESULT_NOT_REQUIRE_GRAD,
        )
        self.assertSetEqual(
            self._fetch_params(test_model, check_requires_grad=True, reg_exps=[]),
            EXPECTED_RESULT_ALL_TRAIN_PARAM,
        )

        EXPECTED_RESULT_SUB_MODULE_ONLY = {
            "sub_module_L1.weight_norm",
            "sub_module_L1.weight_rnd",
            "sub_module_L1.sub_sub_module.weight_norm",
            "sub_module_L1.sub_sub_module.weight_rnd",
        }
        self.assertSetEqual(
            self._fetch_params(
                test_model, check_requires_grad=True, reg_exps=["sub_module"]
            ),
            EXPECTED_RESULT_SUB_MODULE_ONLY,
        )

        EXPECTED_RESULT_WEIGHT_NORM_ONLY = {
            "weight_norm",
        }
        self.assertSetEqual(
            self._fetch_params(test_model, reg_exps=["weight_norm"]),
            EXPECTED_RESULT_WEIGHT_NORM_ONLY,
        )

        EXPECTED_RESULT_ALL_WEIGHT_NORM_ONLY = {
            "weight_norm",
            "sub_module_L1.weight_norm",
            "sub_module_L1.sub_sub_module.weight_norm",
        }
        self.assertSetEqual(
            self._fetch_params(test_model, reg_exps=[".*weight_norm"]),
            EXPECTED_RESULT_ALL_WEIGHT_NORM_ONLY,
        )

        EXPECTED_RESULT_REGEXP = {
            "weight_norm",
            "sub_module_L1.weight_norm",
            "sub_module_L1.sub_sub_module.weight_norm",
            "sub_module_L1.sub_sub_module.weight_rnd",
        }
        self.assertSetEqual(
            self._fetch_params(
                test_model,
                reg_exps=["sub_module_L1.sub_sub_module", ".*weight_norm"],
            ),
            EXPECTED_RESULT_REGEXP,
        )

        EXPECTED_RESULT_REGEXP_2 = {
            "weight_norm",
            "sub_module_L1.weight_norm",
            "sub_module_L1.sub_sub_module.weight_norm",
        }
        self.assertSetEqual(
            self._fetch_params(
                test_model,
                reg_exps=["sub_sub_module", ".*weight_norm"],
            ),
            EXPECTED_RESULT_REGEXP_2,
        )
