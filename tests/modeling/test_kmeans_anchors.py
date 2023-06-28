#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved


import unittest

import numpy as np
import torch
from d2go.modeling.kmeans_anchors import (
    add_kmeans_anchors_cfg,
    compute_kmeans_anchors,
    compute_kmeans_anchors_hook,
)
from d2go.runner import GeneralizedRCNNRunner
from d2go.utils.testing.data_loader_helper import register_toy_coco_dataset
from detectron2.data import DatasetCatalog, DatasetFromList, MapDataset
from detectron2.engine.train_loop import SimpleTrainer
from torch.utils.data.sampler import BatchSampler, Sampler


class IntervalSampler(Sampler):
    def __init__(self, size: int, interval: int):
        self._local_indices = range(0, size, interval)

    def __iter__(self):
        yield from self._local_indices

    def __len__(self):
        return len(self._local_indices)


def build_sequence_loader(cfg, dataset_name, mapper, total_samples, batch_size=1):
    """
    Similar to `build_detection_test_loader` in the way that its sampler
        samples dataset_dicts in order and only loops once.
    """
    dataset_dicts = DatasetCatalog.get(dataset_name)
    dataset = DatasetFromList(dataset_dicts)
    dataset = MapDataset(dataset, mapper)

    interval = max(1, int(len(dataset) / total_samples))
    sampler = IntervalSampler(len(dataset), interval)
    batch_sampler = BatchSampler(sampler, batch_size, drop_last=False)

    def _trivial_batch_collator(batch):
        return batch

    data_loader = torch.utils.data.DataLoader(
        dataset,
        num_workers=cfg.DATALOADER.NUM_WORKERS,
        batch_sampler=batch_sampler,
        collate_fn=_trivial_batch_collator,
    )
    return data_loader


class TestKmeansAnchors(unittest.TestCase):
    def setUp(self):
        self.runner = GeneralizedRCNNRunner()

    def _get_default_cfg(self):
        cfg = self.runner.get_default_cfg()
        add_kmeans_anchors_cfg(cfg)
        return cfg

    @unittest.skip("This can only run locally and takes significant of time")
    def test_matching_previous_results(self):
        cfg = self._get_default_cfg()
        cfg.INPUT.MIN_SIZE_TRAIN = (144,)
        cfg.MODEL.KMEANS_ANCHORS.KMEANS_ANCHORS_ON = True
        cfg.MODEL.KMEANS_ANCHORS.NUM_CLUSTERS = 10
        cfg.MODEL.KMEANS_ANCHORS.NUM_TRAINING_IMG = 512
        cfg.MODEL.KMEANS_ANCHORS.DATASETS = ()

        # NOTE: create a data loader that samples exact the same as previous
        # implementation. In D2Go, we will rely on the train loader instead.
        # NOTE: in order to load OV580_XRM dataset, change the IM_DIR to:
        # "/mnt/vol/gfsai-east/aml/mobile-vision//dataset/oculus/hand_tracking//torch/Segmentation/OV580_XRM_640x480_V3_new_rerun/images"  # noqa
        data_loader = build_sequence_loader(
            cfg,
            # dataset_name="coco_2014_valminusminival",
            # dataset_name="OV580_XRM_640x480_V3_train",
            dataset_name="OV580_XRM_640x480_V3_heldOut_small_512",
            mapper=self.runner.get_mapper(cfg, is_train=True),
            total_samples=cfg.MODEL.KMEANS_ANCHORS.NUM_TRAINING_IMG,
            batch_size=3,
        )

        kmeans_anchors = compute_kmeans_anchors(
            cfg, data_loader, sort_by_area=False, _stride=16, _legacy_plus_one=True
        )

        # Taken from D9849940
        reference_anchors = np.array(
            [
                [-15.33554182, -15.29361029, 31.33554182, 31.29361029],  # noqa
                [-9.34156693, -9.32553548, 25.34156693, 25.32553548],  # noqa
                [-6.03052776, -6.02034167, 22.03052776, 22.02034167],  # noqa
                [-2.25951741, -2.182888, 18.25951741, 18.182888],  # noqa
                [-18.93553378, -18.93553403, 34.93553378, 34.93553403],  # noqa
                [-12.69068356, -12.73989029, 28.69068356, 28.73989029],  # noqa
                [-24.73489189, -24.73489246, 40.73489189, 40.73489246],  # noqa
                [-4.06014466, -4.06014469, 20.06014466, 20.06014469],  # noqa
                [-7.61036119, -7.60467538, 23.61036119, 23.60467538],  # noqa
                [-10.88200579, -10.87634414, 26.88200579, 26.87634414],  # noqa
            ]
        )
        np.testing.assert_allclose(kmeans_anchors, reference_anchors, atol=1e-6)

    def test_build_model(self):
        cfg = self._get_default_cfg()
        cfg.INPUT.MIN_SIZE_TRAIN = (60,)
        cfg.MODEL.KMEANS_ANCHORS.KMEANS_ANCHORS_ON = True
        cfg.MODEL.KMEANS_ANCHORS.NUM_CLUSTERS = 3
        cfg.MODEL.KMEANS_ANCHORS.NUM_TRAINING_IMG = 5
        cfg.MODEL.KMEANS_ANCHORS.DATASETS = ("toy_dataset",)

        cfg.MODEL.DEVICE = "cpu"
        cfg.MODEL.ANCHOR_GENERATOR.NAME = "KMeansAnchorGenerator"

        with register_toy_coco_dataset(
            "toy_dataset",
            image_size=(80, 60),  # w, h
            num_images=cfg.MODEL.KMEANS_ANCHORS.NUM_TRAINING_IMG,
        ):
            model = self.runner.build_model(cfg)
            trainer = SimpleTrainer(model, data_loader=[], optimizer=None)
            trainer_hooks = [compute_kmeans_anchors_hook(self.runner, cfg)]
            trainer.register_hooks(trainer_hooks)
            trainer.before_train()
            anchor_generator = model.proposal_generator.anchor_generator
            cell_anchors = list(anchor_generator.cell_anchors)
            gt_anchors = np.array(
                [
                    [-20, -15, 20, 15]  # toy_dataset's bbox is half size of image
                    for _ in range(cfg.MODEL.KMEANS_ANCHORS.NUM_CLUSTERS)
                ]
            )
            np.testing.assert_allclose(cell_anchors[0], gt_anchors)


if __name__ == "__main__":
    unittest.main()
