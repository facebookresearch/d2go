#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved


import itertools
import logging
import operator
from collections import defaultdict, OrderedDict
from typing import Dict

import torch
from d2go.config import CfgNode
from d2go.data.dataset_mappers.build import build_dataset_mapper
from d2go.data.utils import ClipLengthGroupedDataset
from detectron2.data import (
    build_batch_data_loader,
    build_detection_train_loader,
    get_detection_dataset_dicts,
)
from detectron2.data.build import worker_init_reset_seed
from detectron2.data.common import DatasetFromList, MapDataset
from detectron2.data.dataset_mapper import DatasetMapper
from detectron2.data.samplers import RepeatFactorTrainingSampler
from detectron2.utils.comm import get_world_size
from mobile_cv.common.misc.oss_utils import fb_overwritable
from tabulate import tabulate

logger = logging.getLogger(__name__)


def add_weighted_training_sampler_default_configs(cfg: CfgNode):
    """
    The CfgNode under cfg.DATASETS.TRAIN_REPEAT_FACTOR should be a list of
    tuples (dataset_name, scalar-repeat-factor) specifying upsampled frequencies
    for each dataset when using RepeatFactorTrainingSampler. An example looks like:
    DATASETS:
      TRAIN:
        - "train_1"
        - "train_2"
        - "small_train_3"
      TEST: ...
      TRAIN_REPEAT_FACTOR:
        - ["small_train_3", 2.5]
    """
    cfg.DATASETS.TRAIN_REPEAT_FACTOR = []


def add_random_subset_training_sampler_default_configs(cfg: CfgNode):
    """
    Add default cfg.DATALOADER.RANDOM_SUBSET_RATIO for RandomSubsetTrainingSampler
    The CfgNode under cfg.DATALOADER.RANDOM_SUBSET_RATIO should be a float > 0 and <= 1
    """
    cfg.DATALOADER.RANDOM_SUBSET_RATIO = 1.0


def get_train_datasets_repeat_factors(cfg: CfgNode) -> Dict[str, float]:
    repeat_factors = cfg.DATASETS.TRAIN_REPEAT_FACTOR
    assert all(len(tup) == 2 for tup in repeat_factors)
    name_to_weight = defaultdict(lambda: 1, dict(repeat_factors))
    # The sampling weights map should only contain datasets in train config
    unrecognized = set(name_to_weight.keys()) - set(cfg.DATASETS.TRAIN)
    assert not unrecognized, f"unrecognized datasets: {unrecognized}"

    logger.info(f"Found repeat factors: {list(name_to_weight.items())}")

    # pyre-fixme[7]: Expected `Dict[str, float]` but got `DefaultDict[typing.Any, int]`.
    return name_to_weight


def get_sampling_probability_table(
    dataset_sizes: Dict[str, int], dataset_repeat_factors: Dict[str, float]
) -> str:
    total_sum = sum(
        dataset_repeat_factors.get(dsname, 1.0) * size
        for dsname, size in dataset_sizes.items()
    )
    sample_prob_data = [
        (
            dsname,
            size,
            dataset_repeat_factors.get(dsname, 1.0),
            (dataset_repeat_factors.get(dsname, 1.0) * size) * 100 / total_sum,
        )
        for dsname, size in dataset_sizes.items()
    ]
    headers = ["Dataset", "Samples", "Repeat factor", "Sample Prob (%)"]
    table = tabulate(sample_prob_data, headers=headers, tablefmt="pipe")
    return table


def build_weighted_detection_train_loader(
    cfg: CfgNode, mapper=None, enable_category_balance=False
):
    dataset_repeat_factors = get_train_datasets_repeat_factors(cfg)
    # OrderedDict to guarantee order of values() consistent with repeat factors
    dataset_name_to_dicts = OrderedDict(
        {
            name: get_detection_dataset_dicts(
                [name],
                filter_empty=cfg.DATALOADER.FILTER_EMPTY_ANNOTATIONS,
                min_keypoints=(
                    cfg.MODEL.ROI_KEYPOINT_HEAD.MIN_KEYPOINTS_PER_IMAGE
                    if cfg.MODEL.KEYPOINT_ON
                    else 0
                ),
                proposal_files=(
                    cfg.DATASETS.PROPOSAL_FILES_TRAIN
                    if cfg.MODEL.LOAD_PROPOSALS
                    else None
                ),
            )
            for name in cfg.DATASETS.TRAIN
        }
    )
    # Repeat factor for every sample in the dataset
    repeat_factors = [
        [dataset_repeat_factors[dsname]] * len(dataset_name_to_dicts[dsname])
        for dsname in cfg.DATASETS.TRAIN
    ]
    sampling_prob_table = get_sampling_probability_table(
        {dsname: len(dataset_name_to_dicts[dsname]) for dsname in cfg.DATASETS.TRAIN},
        dataset_repeat_factors,
    )
    logger.info("Dataset TRAIN sampling probability: \n" + sampling_prob_table)
    repeat_factors = list(itertools.chain.from_iterable(repeat_factors))

    dataset_dicts = dataset_name_to_dicts.values()
    dataset_dicts = list(itertools.chain.from_iterable(dataset_dicts))
    dataset = DatasetFromList(dataset_dicts, copy=False)
    if mapper is None:
        mapper = DatasetMapper(cfg, True)
    dataset = MapDataset(dataset, mapper)

    repeat_factors = torch.tensor(repeat_factors)
    if enable_category_balance:
        """
        1. Calculate repeat factors using category frequency for each dataset and then merge them.
        2. Element wise dot producting the dataset frequency repeat factors with
            the category frequency repeat factors gives the final repeat factors.
        """
        category_repeat_factors = [
            RepeatFactorTrainingSampler.repeat_factors_from_category_frequency(
                dataset_dict, cfg.DATALOADER.REPEAT_THRESHOLD
            )
            for dataset_dict in dataset_name_to_dicts.values()
        ]
        # flatten the category repeat factors from all datasets
        category_repeat_factors = list(
            itertools.chain.from_iterable(category_repeat_factors)
        )
        category_repeat_factors = torch.tensor(category_repeat_factors)
        repeat_factors = torch.mul(category_repeat_factors, repeat_factors)
        repeat_factors = repeat_factors / torch.min(repeat_factors)
        logger.info(
            "Using WeightedCategoryTrainingSampler with repeat_factors={}".format(
                cfg.DATASETS.TRAIN_REPEAT_FACTOR
            )
        )
    else:
        logger.info(
            "Using WeightedTrainingSampler with repeat_factors={}".format(
                cfg.DATASETS.TRAIN_REPEAT_FACTOR
            )
        )

    sampler = RepeatFactorTrainingSampler(repeat_factors, seed=cfg.SEED)

    return build_batch_data_loader(
        dataset,
        sampler,
        cfg.SOLVER.IMS_PER_BATCH,
        aspect_ratio_grouping=cfg.DATALOADER.ASPECT_RATIO_GROUPING,
        num_workers=cfg.DATALOADER.NUM_WORKERS,
        seed=cfg.SEED,
    )


def build_clip_grouping_data_loader(dataset, sampler, total_batch_size, num_workers=0):
    """
    Build a batched dataloader for training with video clips.

    Args:
        dataset (torch.utils.data.Dataset): map-style PyTorch dataset. Can be indexed.
        sampler (torch.utils.data.sampler.Sampler): a sampler that produces indices
        total_batch_size (int): total batch size across GPUs.
        num_workers (int): number of parallel data loading workers

    Returns:
        iterable[list]. Length of each list is the batch size of the current
            GPU. Each element in the list comes from the dataset.
    """
    world_size = get_world_size()
    assert (
        total_batch_size > 0 and total_batch_size % world_size == 0
    ), "Total batch size ({}) must be divisible by the number of gpus ({}).".format(
        total_batch_size, world_size
    )
    batch_size = total_batch_size // world_size
    data_loader = torch.utils.data.DataLoader(
        dataset,
        sampler=sampler,
        num_workers=num_workers,
        batch_sampler=None,
        collate_fn=operator.itemgetter(0),  # don't batch, but yield individual elements
        worker_init_fn=worker_init_reset_seed,
    )  # yield individual mapped dict
    return ClipLengthGroupedDataset(data_loader, batch_size)


@fb_overwritable()
def build_mapped_train_loader(cfg, mapper):
    if cfg.DATALOADER.SAMPLER_TRAIN == "WeightedTrainingSampler":
        # balancing only datasets frequencies
        data_loader = build_weighted_detection_train_loader(cfg, mapper=mapper)
    elif cfg.DATALOADER.SAMPLER_TRAIN == "WeightedCategoryTrainingSampler":
        # balancing both datasets and its categories
        data_loader = build_weighted_detection_train_loader(
            cfg, mapper=mapper, enable_category_balance=True
        )
    else:
        data_loader = build_detection_train_loader(cfg, mapper=mapper)
    return data_loader


def build_d2go_train_loader(cfg, mapper=None):
    """
    Build the dataloader for training in D2Go. This is the main entry and customizations
    will be done by using Registry.

    This interface is currently experimental.
    """
    logger.info("Building D2Go's train loader ...")
    # TODO: disallow passing mapper and use registry for all mapper registering
    mapper = mapper or build_dataset_mapper(cfg, is_train=True)
    logger.info("Using dataset mapper:\n{}".format(mapper))

    data_loader = build_mapped_train_loader(cfg, mapper)

    # TODO: decide if move vis_wrapper inside this interface
    return data_loader
