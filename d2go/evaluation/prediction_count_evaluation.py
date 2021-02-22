#!/usr/bin/env python3

import logging
from collections import OrderedDict

import numpy as np
from detectron2.evaluation import DatasetEvaluator


logger = logging.getLogger(__name__)


class PredictionCountEvaluator(DatasetEvaluator):
    """
    Custom Detectron2 evaluator class to simply count the number of predictions
    e.g. on a dataset of hard negatives where there are no annotations, and
    summarize results into interpretable metrics.

    See class pattern from detectron2.evaluation.evaluator.py, especially
    :func:`inference_on_dataset` to see how this class will be called.
    """

    def reset(self):
        self.prediction_counts = []
        self.confidence_scores = []

    def process(self, inputs, outputs):
        """
        Params:
            input: the input that's used to call the model.
            output: the return value of `model(output)`
        """
        # outputs format:
        # [{
        #   "instances": Instances(
        #                  num_instances=88,
        #                  fields=[scores = tensor([list of len num_instances])]
        #   ), ...
        #  },
        #  ... other dicts
        # ]
        for output_dict in outputs:
            instances = output_dict["instances"]
            self.prediction_counts.append(len(instances))
            self.confidence_scores.extend(instances.get("scores").tolist())

    def evaluate(self):
        """
        Returns:
            In detectron2.tools.train_net.py, following format expected:
            dict:
                * key: the name of the task (e.g., bbox)
                * value: a dict of {metric name: score}, e.g.: {"AP50": 80}
        """
        mpi = np.mean(self.prediction_counts)
        mcp = np.mean(self.confidence_scores)
        output_metrics = OrderedDict(
            {
                "false_positives": {
                    "predictions_per_image": mpi,
                    "confidence_per_prediction": mcp,
                }
            }
        )
        logger.info(f"mean predictions per image: {mpi}")
        logger.info(f"mean confidence per prediction: {mcp}")
        return output_metrics
