#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved


import logging
from typing import List

import detectron2.utils.comm as comm
import numpy as np
import torch
from d2go.config import CfgNode as CN, temp_defrost
from detectron2.engine import hooks
from detectron2.layers import ShapeSpec
from detectron2.modeling import GeneralizedRCNN
from detectron2.modeling.anchor_generator import (
    ANCHOR_GENERATOR_REGISTRY,
    BufferList,
    DefaultAnchorGenerator,
)
from detectron2.modeling.proposal_generator.rpn import RPN
from detectron2.structures.boxes import Boxes

logger = logging.getLogger(__name__)


def add_kmeans_anchors_cfg(_C):
    _C.MODEL.KMEANS_ANCHORS = CN()
    _C.MODEL.KMEANS_ANCHORS.KMEANS_ANCHORS_ON = False
    _C.MODEL.KMEANS_ANCHORS.NUM_CLUSTERS = 0
    _C.MODEL.KMEANS_ANCHORS.NUM_TRAINING_IMG = 0
    _C.MODEL.KMEANS_ANCHORS.DATASETS = ()
    _C.MODEL.ANCHOR_GENERATOR.OFFSET = 0.0
    _C.MODEL.KMEANS_ANCHORS.RNG_SEED = 3

    return _C


def compute_kmeans_anchors_hook(runner, cfg):
    """
    This function will create a before_train hook, it will:
        1: create a train loader using provided KMEANS_ANCHORS.DATASETS.
        2: collecting statistics of boxes using outputs from train loader, use up
            to KMEANS_ANCHORS.NUM_TRAINING_IMG images.
        3: compute K-means using KMEANS_ANCHORS.NUM_CLUSTERS clusters
        4: update the buffers in anchor_generator.
    """

    def before_train_callback(trainer):
        if not cfg.MODEL.KMEANS_ANCHORS.KMEANS_ANCHORS_ON:
            return

        new_cfg = cfg.clone()
        with temp_defrost(new_cfg):
            new_cfg.DATASETS.TRAIN = cfg.MODEL.KMEANS_ANCHORS.DATASETS
            data_loader = runner.build_detection_train_loader(new_cfg)

        anchors = compute_kmeans_anchors(cfg, data_loader)
        anchors = anchors.tolist()

        assert isinstance(trainer.model, GeneralizedRCNN)
        assert isinstance(trainer.model.proposal_generator, RPN)
        anchor_generator = trainer.model.proposal_generator.anchor_generator
        assert isinstance(anchor_generator, KMeansAnchorGenerator)
        anchor_generator.update_cell_anchors(anchors)

    return hooks.CallbackHook(before_train=before_train_callback)


@ANCHOR_GENERATOR_REGISTRY.register()
class KMeansAnchorGenerator(DefaultAnchorGenerator):
    """Generate anchors using pre-computed KMEANS_ANCHORS.COMPUTED_ANCHORS"""

    def __init__(self, cfg, input_shape: List[ShapeSpec]):
        torch.nn.Module.__init__(self)
        self.strides = [x.stride for x in input_shape]
        self.offset = cfg.MODEL.ANCHOR_GENERATOR.OFFSET

        assert 0.0 <= self.offset < 1.0, self.offset

        # kmeans anchors
        num_features = len(cfg.MODEL.RPN.IN_FEATURES)
        assert num_features == 1, "Doesn't support multiple feature map"

        # NOTE: KMEANS anchors are only computed at training time, when initialized,
        # set anchors to correct shape but invalid value as place holder.
        computed_anchors = [[float("Inf")] * 4] * cfg.MODEL.KMEANS_ANCHORS.NUM_CLUSTERS

        cell_anchors = [torch.Tensor(computed_anchors)]
        self.cell_anchors = BufferList(cell_anchors)

    def update_cell_anchors(self, computed_anchors):
        assert len(self.cell_anchors) == 1
        for buf in self.cell_anchors.buffers():
            assert len(buf) == len(computed_anchors)
            buf.data = torch.Tensor(computed_anchors).to(buf.device)
        logger.info("Updated cell anchors")

    def forward(self, *args, **kwargs):
        for base_anchors in self.cell_anchors:
            assert torch.isfinite(base_anchors).all(), (
                "The anchors are not initialized yet, please providing COMPUTED_ANCHORS"
                " when creating the model and/or loading the valid weights."
            )
        return super().forward(*args, **kwargs)


def collect_boxes_size_stats(data_loader, max_num_imgs, _legacy_plus_one=False):
    logger.info(
        "Collecting size of boxes, loading up to {} images from data loader ...".format(
            max_num_imgs
        )
    )
    # data_loader might be infinite length, thus can't loop all images, using
    # max_num_imgs == 0 stands for 0 images instead of whole dataset
    assert max_num_imgs >= 0

    box_sizes = []
    remaining_num_imgs = max_num_imgs
    total_batches = 0
    for i, batched_inputs in enumerate(data_loader):
        total_batches += len(batched_inputs)
        batch_size = min(remaining_num_imgs, len(batched_inputs))
        batched_inputs = batched_inputs[:batch_size]
        for x in batched_inputs:
            boxes = x["instances"].gt_boxes  # xyxy
            assert isinstance(boxes, Boxes)
            for t in boxes.tensor:
                box_sizes += [[t[2] - t[0], t[3] - t[1]]]

                # NOTE: previous implementation didn't apply +1, thus to match
                # previous (incorrect) results we have to minus the im_scale
                if _legacy_plus_one:  # only for matching old tests
                    im_scale = x["image"].shape[1] / x["height"]  # image is chw
                    box_sizes[-1][0] -= im_scale
                    box_sizes[-1][1] -= im_scale

        estimated_iters = max_num_imgs / total_batches * (i + 1)
        remaining_num_imgs -= batch_size
        if i % max(1, int(estimated_iters / 20)) == 0:
            # log 20 times at most
            percentage = 100.0 * i / estimated_iters
            logger.info(
                "Processed batch {} ({:.2f}%) from data_loader, got {} boxes,"
                " remaining number of images: {}/{}".format(
                    i, percentage, len(box_sizes), remaining_num_imgs, max_num_imgs
                )
            )
        if remaining_num_imgs <= 0:
            assert remaining_num_imgs == 0
            break

    box_sizes = np.array(box_sizes)
    logger.info(
        "Collected {} boxes from {} images".format(len(box_sizes), max_num_imgs)
    )
    return box_sizes


def compute_kmeans_anchors(
    cfg, data_loader, sort_by_area=True, _stride=0, _legacy_plus_one=False
):
    assert (
        cfg.MODEL.KMEANS_ANCHORS.NUM_TRAINING_IMG > 0
    ), "Please provide positive MODEL.KMEANS_ANCHORS.NUM_TRAINING_IMG"

    num_training_img = cfg.MODEL.KMEANS_ANCHORS.NUM_TRAINING_IMG
    div_i, mod_i = divmod(num_training_img, comm.get_world_size())
    num_training_img_i = div_i + (comm.get_rank() < mod_i)

    box_sizes_i = collect_boxes_size_stats(
        data_loader,
        num_training_img_i,
        _legacy_plus_one=_legacy_plus_one,
    )

    all_box_sizes = comm.all_gather(box_sizes_i)
    box_sizes = np.concatenate(all_box_sizes)
    logger.info("Collected {} boxes from all gpus".format(len(box_sizes)))

    assert (
        cfg.MODEL.KMEANS_ANCHORS.NUM_CLUSTERS > 0
    ), "Please provide positive MODEL.KMEANS_ANCHORS.NUM_CLUSTERS"
    from sklearn.cluster import KMeans  # delayed import

    default_anchors = (
        KMeans(
            n_clusters=cfg.MODEL.KMEANS_ANCHORS.NUM_CLUSTERS,
            random_state=cfg.MODEL.KMEANS_ANCHORS.RNG_SEED,
        )
        .fit(box_sizes)
        .cluster_centers_
    )

    anchors = []
    for anchor in default_anchors:
        w, h = anchor
        # center anchor boxes at (stride/2,stride/2)
        new_anchors = np.hstack(
            (
                _stride / 2 - 0.5 * w,
                _stride / 2 - 0.5 * h,
                _stride / 2 + 0.5 * w,
                _stride / 2 + 0.5 * h,
            )
        )
        anchors.append(new_anchors)
    anchors = np.array(anchors)

    # sort anchors by area
    areas = (anchors[:, 2] - anchors[:, 0]) * (anchors[:, 3] - anchors[:, 1])
    sqrt_areas = np.sqrt(areas)
    if sort_by_area:
        indices = np.argsort(sqrt_areas)
        anchors = anchors[indices]
        sqrt_areas = sqrt_areas[indices].tolist()

    display_str = "\n".join(
        [
            s + "\t sqrt area: {:.2f}".format(a)
            for s, a in zip(str(anchors).split("\n"), sqrt_areas)
        ]
    )
    logger.info(
        "Compuated kmeans anchors (sorted by area: {}):\n{}".format(
            sort_by_area, display_str
        )
    )
    return anchors
