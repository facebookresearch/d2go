#!/usr/bin/env python3

import unittest

import torch
from d2go.evaluation.prediction_count_evaluation import PredictionCountEvaluator
from detectron2.structures.instances import Instances


class TestPredictionCountEvaluation(unittest.TestCase):
    def setUp(self):
        self.evaluator = PredictionCountEvaluator()
        image_size = (224, 224)
        self.mock_outputs = [
            {"instances": Instances(image_size, scores=torch.Tensor([0.9, 0.8, 0.7]))},
            {"instances": Instances(image_size, scores=torch.Tensor([0.9, 0.8, 0.7]))},
            {"instances": Instances(image_size, scores=torch.Tensor([0.9, 0.8]))},
            {"instances": Instances(image_size, scores=torch.Tensor([0.9, 0.8]))},
            {"instances": Instances(image_size, scores=torch.Tensor([0.9]))},
        ]
        # PredictionCountEvaluator does not depend on inputs
        self.mock_inputs = [None] * len(self.mock_outputs)

    def test_process_evaluate_reset(self):
        self.assertEqual(len(self.evaluator.prediction_counts), 0)
        self.assertEqual(len(self.evaluator.confidence_scores), 0)

        # Test that `process` registers the outputs.
        self.evaluator.process(self.mock_inputs, self.mock_outputs)
        self.assertListEqual(self.evaluator.prediction_counts, [3, 3, 2, 2, 1])
        self.assertEqual(len(self.evaluator.confidence_scores), 11)

        # Test that `evaluate` returns the correct metrics.
        output_metrics = self.evaluator.evaluate()
        self.assertDictAlmostEqual(
            output_metrics,
            {
                "false_positives": {
                    "predictions_per_image": 11 / 5,
                    "confidence_per_prediction": (0.9 * 5 + 0.8 * 4 + 0.7 * 2) / 11,
                }
            },
        )

        # Test that `reset` clears the evaluator state.
        self.evaluator.reset()
        self.assertEqual(len(self.evaluator.prediction_counts), 0)
        self.assertEqual(len(self.evaluator.confidence_scores), 0)

    def assertDictAlmostEqual(self, dict1, dict2):
        keys1 = list(dict1.keys())
        keys2 = list(dict2.keys())
        # Assert lists are equal, irrespective of ordering
        self.assertCountEqual(keys1, keys2)

        for k, v1 in dict1.items():
            v2 = dict2[k]
            if isinstance(v2, list):
                self.assertListEqual(v1, v2)
            elif isinstance(v2, dict):
                self.assertDictAlmostEqual(v1, v2)
            else:
                self.assertAlmostEqual(v1, v2)
