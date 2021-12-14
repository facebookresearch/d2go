#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved


import heapq
import itertools
import logging
from contextlib import contextmanager

from detectron2.data import MetadataCatalog
from detectron2.evaluation import DatasetEvaluator, SemSegEvaluator
from detectron2.utils.comm import all_gather, synchronize


logger = logging.getLogger(__name__)


class MultiSemSegEvaluator(DatasetEvaluator):
    """
    Evaluate multiple results for the same target. SemSegEvaluator requires the
    outputs of model to be like:
    [
        {"sem_seg": Tensor},
    ]
    This evaluator allows evaluating mutliple predictions, it may takes outputs like:
    [
        {
            "prediction_1": {"sem_seg": Tensor},
            "prediction_2": {"sem_seg": Tensor},
        }
    ]
    """

    _DUMMY_KEY_PREFIX = "dummy_eval"

    def __init__(self, dataset_name, *args, distributed, output_dir=None, **kwargs):
        self._distributed = distributed
        self._output_dir = output_dir

        self.evaluators = {}
        self.dataset_name = dataset_name
        self.init_args = args
        self.init_kwargs = kwargs

    def _get_evaluator(self, key, superclass_name=None):
        if key in self.evaluators:
            return self.evaluators[key]

        def create_evaluator_and_reset(dataset_name):
            logger.info(
                "Create an instance of SemSegEvaluator for {} on dataset {} ...".format(
                    key, dataset_name
                )
            )
            evaluator = SemSegEvaluator(
                dataset_name,
                *self.init_args,
                **self.init_kwargs,
                distributed=self._distributed,
                output_dir=self._output_dir,
            )
            evaluator.reset()
            return evaluator

        if superclass_name is None:
            self.evaluators[key] = create_evaluator_and_reset(self.dataset_name)
        else:
            # NOTE: create temporary single-super-class dataset and use standard
            # evaluator for the dataset
            metadata = MetadataCatalog.get(self.dataset_name)
            tmp_dataset_name = "__AUTOGEN__{}@{}".format(
                self.dataset_name, superclass_name
            )
            from d2go.data.fb.semantic_seg import register_sem_seg

            if tmp_dataset_name not in MetadataCatalog:
                if superclass_name in metadata.pseudo_gt_classes:
                    mask_dir = metadata.pseudo_gt_mask_dir
                else:
                    mask_dir = metadata.mask_dir

                register_sem_seg(
                    tmp_dataset_name,
                    metadata=metadata.mcs_metadata[superclass_name],
                    image_root=metadata.image_root,
                    sem_seg_root=metadata.sem_seg_root,
                    instances_json=metadata.json_file,
                    mask_dir=mask_dir.format(superclass_name),
                )
            self.evaluators[key] = create_evaluator_and_reset(tmp_dataset_name)

        return self.evaluators[key]

    def reset(self):
        for evaluator in self.evaluators.values():
            evaluator.reset()

    def process(self, inputs, outputs):
        if "sem_seg" in outputs[0].keys():
            # normal eval is compatible with SemSegEvaluator
            self._get_evaluator("sem_seg").process(inputs, outputs)
        else:
            # only the file_name of inputs is needed for SemSegEvaluator
            inputs_ = [{"file_name": inp["file_name"]} for inp in inputs]
            for frame_name in outputs[0].keys():
                if isinstance(outputs[0]["detect"]["sem_seg"], dict):  # multi-class
                    for superclass_name in outputs[0]["detect"]["sem_seg"]:
                        outputs_ = []
                        for outp in outputs:
                            x = outp[frame_name]
                            x = {"sem_seg": x["sem_seg"][superclass_name]}
                            outputs_.append(x)
                        self._get_evaluator(
                            "sem_seg-{}-{}".format(frame_name, superclass_name),
                            superclass_name=superclass_name,
                        ).process(inputs_, outputs_)
                else:
                    # convert the output to SemSegEvaluator's format
                    outputs_ = [outp[frame_name] for outp in outputs]
                    self._get_evaluator("sem_seg-{}".format(frame_name)).process(
                        inputs_, outputs_
                    )

    def evaluate(self):
        results = {}

        # The evaluation will get stuck sometimes if the follwoing code is not used.
        # `SemSegEvaluator` will do synchronization between processes when computing
        # the metrics. In some cases the number of self.evaluators will not be the
        # same between processes and the code will stuck in synchronization.
        # For example, evaluate 10 images on 8 GPUs, only 5 GPUs
        # will be used for evaluation, each has 2 images, the rest 3 GPUs will have
        # zero self.evaluators as they are constructed on-the-fly when calling
        # self.process())
        # We create additional evaluators so that all processes have the same size
        # of evaluators so that the synchronization will not get stuck.
        evaluator_size = len(self.evaluators)
        synchronize()
        evaluator_size_list = all_gather(evaluator_size)
        max_evaluator_size = max(evaluator_size_list)
        if evaluator_size < max_evaluator_size:
            # create additional evaluators so that all processes have the same
            # size of evaluators
            metadata = MetadataCatalog.get(self.dataset_name)
            mcs_metadata = metadata.get("mcs_metadata")
            for idx in range(max_evaluator_size - evaluator_size):
                dummy_key = f"{self._DUMMY_KEY_PREFIX}_{idx}"
                assert dummy_key not in self.evaluators
                if mcs_metadata:
                    for k in mcs_metadata:
                        self._get_evaluator(dummy_key, superclass_name=k).reset()
                else:
                    self._get_evaluator(dummy_key).reset()

        for name, evaluator in self.evaluators.items():
            result = evaluator.evaluate()
            # NOTE: .evaluate() returns None for non-main process
            if result is not None:
                results[name] = result["sem_seg"]
        return results


class MultiSemSegVidEvaluator(MultiSemSegEvaluator):
    """
    Evaluate semantic segmentation results for video clips. MultiSemSegVidEvaluator
    requires the outputs of model to be like:
    [
        {"file_names": Tensor},
    ]
    """

    def process(self, inputs, outputs):
        assert "file_names" in inputs[0]
        inputs_ = []
        for batch_id in range(len(inputs)):
            for frame_i in range(len(inputs[batch_id]["file_names"])):
                inputs_.append({"file_name": inputs[batch_id]["file_names"][frame_i]})
        for name in outputs[0].keys():
            # convert the output to SemSegEvaluator's format
            outputs_ = [outp[name] for outp in outputs]
            self.evaluators["sem_seg_{}".format(name)].process(inputs_, outputs_)


@contextmanager
def all_logging_disabled(highest_level=logging.CRITICAL):
    """
    A context manager that will prevent any logging messages
    triggered during the body from being processed.
    :param highest_level: the maximum logging level in use.
      This would only need to be changed if a custom level greater than CRITICAL
      is defined.
    """
    # two kind-of hacks here:
    #    * can't get the highest logging level in effect => delegate to the user
    #    * can't get the current module-level override => use an undocumented
    #       (but non-private!) interface

    previous_level = logging.root.manager.disable

    logging.disable(highest_level)

    try:
        yield
    finally:
        logging.disable(previous_level)


class PerImageEvaluator(object):
    def __init__(
        self,
        evaluator,
        callback,
        distributed=True,
        playback_criterion=None,
        playback_limit=0,
    ):
        self._evaluator = evaluator
        self._evaluator._distributed = False
        self._evaluator._output_dir = None

        self._distributed = distributed
        self.callback = callback
        self.results_per_image = []

        # record the N most interesting results for playback
        self.playback_heap = []
        self.playback_criterion = playback_criterion
        self.playback_limit = playback_limit

    def reset(self):
        self._evaluator.reset()

    def process(self, inputs, outputs):
        self._evaluator.process(inputs, outputs)

        assert len(inputs) == 1
        with all_logging_disabled():
            result = self._evaluator.evaluate()
        self.results_per_image.append((inputs[0], result))

        if self.playback_criterion:
            score = self.playback_criterion(result)
            heapq.heappush(self.playback_heap, (score, inputs[0], outputs[0], result))
            if len(self.playback_heap) > self.playback_limit:
                heapq.heappop(self.playback_heap)

        self._evaluator.reset()

    def evaluate(self):
        if self._distributed:
            synchronize()
            results_per_image = all_gather(self.results_per_image)
            self.results_per_image = list(itertools.chain(*results_per_image))

            playback_heap = all_gather(self.playback_heap)
            playback_heap = list(itertools.chain(*playback_heap))
            # each GPU has local N mininums, sort and take global mininums
            playback_heap = sorted(playback_heap, key=lambda x: x[0])
            self.playback_heap = playback_heap[: self.playback_limit]

        self.callback(self)
        return {}
