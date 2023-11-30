# Copyright (c) Facebook, Inc. and its affiliates.
import logging
import os
from collections import abc
from typing import Any, Iterable, List, Union

import torch

from detectron2.evaluation import (
    DatasetEvaluator,
    DatasetEvaluators,
    inference_on_dataset as inference_on_dataset_d2,
)
from detectron2.utils import comm
from detectron2.utils.file_io import PathManager


logger = logging.getLogger(__name__)


def DatasetEvaluators_has_finished_process(self):
    ret = True
    for x in self._evaluators:
        if hasattr(x, "has_finished_process"):
            ret &= x.has_finished_process()
        else:
            ret &= False
    return ret


# patch evaluators defined in d2
DatasetEvaluators.has_finished_process = DatasetEvaluators_has_finished_process


def inference_on_dataset(
    model: torch.nn.Module,
    data_loader: Iterable,
    evaluator: Union[DatasetEvaluator, List[DatasetEvaluator], None],
    **kwargs,
):
    """
    A drop-in replacement for d2's inference_on_dataset to run inference on datasets,
    supports customization for checkpointing
    * has_finished_process(self) -> bool: return True if `self.process()` could be skipped
    """
    if evaluator is None:
        return inference_on_dataset_d2(model, data_loader, evaluator, **kwargs)

    if isinstance(evaluator, abc.MutableSequence):
        evaluator = DatasetEvaluators(evaluator)

    if not (
        hasattr(evaluator, "has_finished_process") and evaluator.has_finished_process()
    ):
        return inference_on_dataset_d2(model, data_loader, evaluator, **kwargs)

    evaluator.reset()
    results = evaluator.evaluate()
    if results is None:
        results = {}
    return results


class ResultCache(object):
    def __init__(self, cache_dir: str):
        """A utility class to handle save/load cache data across processes"""
        self.cache_str = cache_dir

    @property
    def cache_file(self):
        if self.cache_str is None:
            return None
        return os.path.join(self.cache_str, f"_result_cache_.{comm.get_rank()}.pkl")

    def has_cache(self):
        return PathManager.isfile(self.cache_file)

    def load(self, gather: bool = False):
        """
        Load cache results.
        gather (bool): gather cache results arcoss ranks to a list
        """
        if self.cache_file is None or not PathManager.exists(self.cache_file):
            ret = None
        else:
            with PathManager.open(self.cache_file, "rb") as fp:
                ret = torch.load(fp)
            logger.info(f"Loaded from checkpoint {self.cache_file}")

        if gather:
            ret = comm.all_gather(ret)

        return ret

    def save(self, data: Any):
        if self.cache_file is None:
            return

        PathManager.mkdirs(os.path.dirname(self.cache_file))
        with PathManager.open(self.cache_file, "wb") as fp:
            torch.save(data, fp)
        logger.info(f"Saved checkpoint to {self.cache_file}")
