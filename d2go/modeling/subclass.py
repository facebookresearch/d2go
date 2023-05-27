#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List

import numpy as np
import torch
from d2go.config import CfgNode as CN
from d2go.data.dataset_mappers.build import D2GO_DATA_MAPPER_REGISTRY
from d2go.data.dataset_mappers.d2go_dataset_mapper import D2GoDatasetMapper
from detectron2.layers import cat
from detectron2.modeling import ROI_HEADS_REGISTRY, StandardROIHeads
from detectron2.utils.registry import Registry
from mobile_cv.torch.utils_toffee.alias import alias
from torch import nn
from torch.nn import functional as F


logger = logging.getLogger(__name__)

SUBCLASS_FETCHER_REGISTRY = Registry("SUBCLASS_FETCHER")


def add_subclass_configs(cfg):
    _C = cfg
    _C.MODEL.SUBCLASS = CN()
    _C.MODEL.SUBCLASS.SUBCLASS_ON = False
    _C.MODEL.SUBCLASS.NUM_SUBCLASSES = 0  # must be set
    _C.MODEL.SUBCLASS.NUM_LAYERS = 1
    _C.MODEL.SUBCLASS.SUBCLASS_ID_FETCHER = "SubclassFetcher"  # ABC, must be set
    _C.MODEL.SUBCLASS.SUBCLASS_MAPPING = (
        []
    )  # subclass mapping from model output to annotation


class SubclassFetcher(ABC):
    """Fetcher class to read subclass id annotations from dataset and prepare for train/eval.
    Subclass this and register with `@SUBCLASS_FETCHER_REGISTRY.register()` decorator
    to use with custom projects.
    """

    def __init__(self, cfg):
        raise NotImplementedError()

    @property
    @abstractmethod
    def subclass_names(self) -> List[str]:
        """Overwrite this member with any new mappings' subclass names, which
        may be useful for specific evaluation purposes.
        len(self.subclass_names) should be equal to the expected number
        of subclass head outputs (cfg.MODEL.SUBCLASS.NUM_SUBCLASSES + 1).
        """
        pass

    def remap(self, subclass_id: int) -> int:
        """Map subclass ids read from dataset to new label id"""
        return subclass_id

    def fetch_subclass_ids(self, dataset_dict: Dict[str, Any]) -> List[int]:
        """Get all the subclass_ids in a dataset dict"""
        extras_list = [anno.get("extras") for anno in dataset_dict["annotations"]]
        subclass_ids = [extras["subclass_id"] for extras in extras_list]
        return subclass_ids


@D2GO_DATA_MAPPER_REGISTRY.register()
class SubclassDatasetMapper(D2GoDatasetMapper):
    """
    Wrap any dataset mapper, encode gt_subclasses to the instances.
    """

    def __init__(self, cfg, is_train, tfm_gens=None, subclass_fetcher=None):
        super().__init__(cfg, is_train=is_train, tfm_gens=tfm_gens)
        if subclass_fetcher is None:
            fetcher_name = cfg.MODEL.SUBCLASS.SUBCLASS_ID_FETCHER
            self.subclass_fetcher = SUBCLASS_FETCHER_REGISTRY.get(fetcher_name)(cfg)
            logger.info(
                f"Initialized {self.__class__.__name__} with "
                f"subclass fetcher '{self.subclass_fetcher.__class__.__name__}'"
            )
        else:
            assert isinstance(subclass_fetcher, SubclassFetcher), subclass_fetcher
            self.subclass_fetcher = subclass_fetcher
            logger.info(f"Set subclass fetcher to {self.subclass_fetcher}")

        # NOTE: field doesn't exist when loading a (old) caffe2 model.
        # self.subclass_on = cfg.MODEL.SUBCLASS.SUBCLASS_ON
        self.subclass_on = True

    def _original_call(self, dataset_dict):
        """
        Map the dataset dict with D2GoDatasetMapper, then augment with subclass gt tensors.
        """
        # Transform removes key 'annotations' from the dataset dict
        mapped_dataset_dict = super()._original_call(dataset_dict)

        if self.is_train and self.subclass_on:
            subclass_ids = self.subclass_fetcher.fetch_subclass_ids(dataset_dict)
            subclasses = torch.tensor(subclass_ids, dtype=torch.int64)
            mapped_dataset_dict["instances"].gt_subclasses = subclasses
        return mapped_dataset_dict


def build_subclass_head(cfg, in_chann, out_chann):
    #  fully connected layers: n-1 in_chann x in_chann layers, and 1 in_chann x out_chann layer
    layers = [
        nn.Linear(in_chann, in_chann) for _ in range(cfg.MODEL.SUBCLASS.NUM_LAYERS - 1)
    ]
    layers.append(nn.Linear(in_chann, out_chann))

    return nn.Sequential(*layers)


@ROI_HEADS_REGISTRY.register()
class StandardROIHeadsWithSubClass(StandardROIHeads):
    """
    A Standard ROIHeads which contains an addition of subclass head.
    """

    def __init__(self, cfg, input_shape):
        super().__init__(cfg, input_shape)
        self.subclass_on = cfg.MODEL.SUBCLASS.SUBCLASS_ON
        if not self.subclass_on:
            return

        self.num_subclasses = cfg.MODEL.SUBCLASS.NUM_SUBCLASSES
        self.subclass_head = build_subclass_head(
            cfg, self.box_head.output_shape.channels, self.num_subclasses + 1
        )

        for layer in self.subclass_head:
            nn.init.normal_(layer.weight, std=0.01)
            nn.init.constant_(layer.bias, 0.0)

    def forward(self, images, features, proposals, targets=None):
        """
        Same as StandardROIHeads.forward but add logic for subclass.
        """
        if not self.subclass_on:
            return super().forward(images, features, proposals, targets)

        # --- start copy -------------------------------------------------------
        del images

        if self.training:
            proposals = self.label_and_sample_proposals(proposals, targets)
            # NOTE: `has_gt` = False for negatives and we must manually register `gt_subclasses`,
            #  because custom gt_* fields will not be automatically registered in sampled proposals.
            for pp_per_im in proposals:
                if not pp_per_im.has("gt_subclasses"):
                    background_subcls_idx = 0
                    pp_per_im.gt_subclasses = torch.cuda.LongTensor(
                        len(pp_per_im)
                    ).fill_(background_subcls_idx)
        del targets

        features_list = [features[f] for f in self.in_features]

        box_features = self.box_pooler(
            features_list, [x.proposal_boxes for x in proposals]
        )
        box_features = self.box_head(box_features)
        predictions = self.box_predictor(box_features)
        # --- end copy ---------------------------------------------------------

        # NOTE: don't delete box_features, keep it temporarily
        # del box_features
        box_features = box_features.view(
            box_features.shape[0], np.prod(box_features.shape[1:])
        )
        pred_subclass_logits = self.subclass_head(box_features)

        if self.training:
            losses = self.box_predictor.losses(predictions, proposals)
            # During training the proposals used by the box head are
            # used by the mask, keypoint (and densepose) heads.
            losses.update(self._forward_mask(features, proposals))
            losses.update(self._forward_keypoint(features, proposals))

            # subclass head
            gt_subclasses = cat([p.gt_subclasses for p in proposals], dim=0)
            loss_subclass = F.cross_entropy(
                pred_subclass_logits, gt_subclasses, reduction="mean"
            )
            losses.update({"loss_subclass": loss_subclass})

            return proposals, losses
        else:
            pred_instances, kept_indices = self.box_predictor.inference(
                predictions, proposals
            )
            # During inference cascaded prediction is used: the mask and keypoints
            # heads are only applied to the top scoring box detections.
            pred_instances = self.forward_with_given_boxes(features, pred_instances)

            # subclass head
            probs = F.softmax(pred_subclass_logits, dim=-1)
            for pred_instances_i, kept_indices_i in zip(pred_instances, kept_indices):
                pred_instances_i.pred_subclass_prob = torch.index_select(
                    probs,
                    dim=0,
                    index=kept_indices_i.to(torch.int64),
                )

            if torch.onnx.is_in_onnx_export():
                assert len(pred_instances) == 1
                pred_instances[0].pred_subclass_prob = alias(
                    pred_instances[0].pred_subclass_prob, "subclass_prob_nms"
                )

            return pred_instances, {}
