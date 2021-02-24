#!/usr/bin/env python3

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from detectron2.layers import cat
from detectron2.modeling import ROI_HEADS_REGISTRY, StandardROIHeads
from d2go.config import CfgNode as CN
from d2go.data.dataset_mappers import (
    D2GO_DATA_MAPPER_REGISTRY,
    D2GoDatasetMapper,
)
from d2go.utils.helper import alias


def add_subclass_configs(cfg):
    _C = cfg
    _C.MODEL.SUBCLASS = CN()
    _C.MODEL.SUBCLASS.SUBCLASS_ON = False
    _C.MODEL.SUBCLASS.NUM_SUBCLASSES = 0  # must be set


def fetch_subclass_from_extras(dataset_dict):
    """
    Retrieve subclass (eg. hand gesture per RPN region) info from dataset dict.
    """
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
        self.subclass_fetcher = subclass_fetcher or fetch_subclass_from_extras
        # NOTE: field doesn't exist when loading a (old) caffe2 model.
        # self.subclass_on = cfg.MODEL.SUBCLASS.SUBCLASS_ON
        self.subclass_on = True

    def _original_call(self, dataset_dict):
        """
        Map the dataset dict with D2GoDatasetMapper, then augment with subclass gt tensors.
        """
        # Transform removes key 'annotations' from the dataset dict
        mapped_dataset_dict = super()._original_call(dataset_dict)

        if (self.is_train and self.subclass_on):
            subclass_ids = self.subclass_fetcher(dataset_dict)
            subclasses = torch.tensor(subclass_ids, dtype=torch.int64)
            mapped_dataset_dict["instances"].gt_subclasses = subclasses
        return mapped_dataset_dict


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

        self.subclass_head = nn.Linear(
            self.box_head.output_shape.channels, cfg.MODEL.SUBCLASS.NUM_SUBCLASSES + 1
        )
        nn.init.normal_(self.subclass_head.weight, std=0.01)
        nn.init.constant_(self.subclass_head.bias, 0.0)

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
                    pp_per_im.gt_subclasses = torch.cuda.LongTensor(len(pp_per_im)).fill_(background_subcls_idx)
        del targets

        features_list = [features[f] for f in self.in_features]

        box_features = self.box_pooler(features_list, [x.proposal_boxes for x in proposals])
        box_features = self.box_head(box_features)
        predictions = self.box_predictor(box_features)
        # --- end copy ---------------------------------------------------------

        # NOTE: don't delete box_features, keep it temporarily
        # del box_features
        box_features = box_features.view(
            box_features.shape[0],
            np.prod(box_features.shape[1:])
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
                    pred_instances[0].pred_subclass_prob,
                    "subclass_prob_nms"
                )

            return pred_instances, {}
