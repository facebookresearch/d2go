#!/usr/bin/env python3

import os
import tempfile
import unittest
from collections import defaultdict

import torch
from d2go.evaluation.evaluator import inference_on_dataset, ResultCache
from detectron2.evaluation import DatasetEvaluator, DatasetEvaluators


class EvaluatorForTest(DatasetEvaluator):
    def __init__(self):
        self.results = []

    def reset(self):
        self.results.clear()

    def process(self, inputs, outputs):
        self.results.append(outputs)

    def evaluate(self):
        return sum(self.results)


class EvaluatorWithCheckpointForTest(DatasetEvaluator):
    def __init__(self, save_dir):
        self.results = []
        self.result_cache = ResultCache(save_dir)
        self._call_count = defaultdict(int)

    def reset(self):
        self.results.clear()
        self._call_count["reset"] += 1

    def has_finished_process(self):
        return self.result_cache.has_cache()

    def process(self, inputs, outputs):
        assert not self.result_cache.has_cache()
        self.results.append(outputs)
        self._call_count["process"] += 1

    def evaluate(self):
        if not self.result_cache.has_cache():
            self.result_cache.save(self.results)
        else:
            self.results = self.result_cache.load()
        self._call_count["evaluate"] += 1

        return sum(self.results)


class Model(torch.nn.Module):
    def forward(self, x):
        return x


class TestEvaluator(unittest.TestCase):
    def test_inference(self):
        model = Model()
        evaluator = EvaluatorForTest()
        data_loader = [1, 2, 3, 4, 5]
        results = inference_on_dataset(model, data_loader, evaluator)
        self.assertEqual(results, 15)

    def test_inference_with_checkpoint(self):
        with tempfile.TemporaryDirectory() as save_dir:
            model = Model()
            evaluator = EvaluatorWithCheckpointForTest(save_dir)
            self.assertFalse(evaluator.has_finished_process())
            data_loader = [1, 2, 3, 4, 5]
            results = inference_on_dataset(model, data_loader, evaluator)
            self.assertEqual(results, 15)
            self.assertEqual(evaluator._call_count["reset"], 1)
            self.assertEqual(evaluator._call_count["process"], 5)
            self.assertEqual(evaluator._call_count["evaluate"], 1)

            # run again with cache
            self.assertTrue(evaluator.has_finished_process())
            results = inference_on_dataset(model, data_loader, evaluator)
            self.assertEqual(results, 15)
            self.assertEqual(evaluator._call_count["reset"], 2)
            self.assertEqual(evaluator._call_count["process"], 5)
            self.assertEqual(evaluator._call_count["evaluate"], 2)
            self.assertTrue(os.path.isfile(evaluator.result_cache.cache_file))

    def test_evaluators_patch(self):
        with tempfile.TemporaryDirectory() as save_dir:
            cp_evaluator = EvaluatorWithCheckpointForTest(save_dir)
            evaluator = DatasetEvaluators([cp_evaluator])
            self.assertFalse(evaluator.has_finished_process())

            cp_evaluator.reset()
            cp_evaluator.process(1, 1)
            cp_evaluator.evaluate()

            self.assertTrue(evaluator.has_finished_process())
