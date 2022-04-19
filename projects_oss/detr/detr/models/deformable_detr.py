# ------------------------------------------------------------------------
# Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------

"""
Deformable DETR model and criterion classes.
"""
import copy
import math

import torch
import torch.nn.functional as F
from detectron2.config import configurable
from torch import nn

from ..util import box_ops
from ..util.misc import (
    accuracy,
    get_world_size,
    interpolate,
    is_dist_avail_and_initialized,
    nested_tensor_from_tensor_list,
    NestedTensor,
)
from .backbone import build_backbone
from .build import build_detr_backbone, DETR_MODEL_REGISTRY
from .deformable_transformer import DeformableTransformer
from .matcher import build_matcher
from .segmentation import (
    DETRsegm,
    dice_loss,
    PostProcessPanoptic,
    PostProcessSegm,
    sigmoid_focal_loss,
)
from .setcriterion import FocalLossSetCriterion


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


@DETR_MODEL_REGISTRY.register()
class DeformableDETR(nn.Module):
    """This is the Deformable DETR module that performs object detection"""

    @configurable
    def __init__(
        self,
        backbone,
        transformer,
        num_classes,
        num_queries,
        num_feature_levels,
        aux_loss=True,
        with_box_refine=False,
        two_stage=False,
        bbox_embed_num_layers=3,
    ):
        """Initializes the model.
        Parameters:
            backbone: torch module of the backbone to be used. See backbone.py
            transformer: torch module of the transformer architecture. See transformer.py
            num_classes: number of object classes
            num_queries: number of object queries, ie detection slot. This is the maximal number of objects
                         DETR can detect in a single image. For COCO, we recommend 100 queries.
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
            with_box_refine: iterative bounding box refinement
            two_stage: two-stage Deformable DETR
            bbox_embed_num_layers: number of FC layers in bbox_embed MLP
        """
        super().__init__()
        self.num_queries = num_queries
        self.transformer = transformer
        hidden_dim = transformer.d_model
        # We will use sigmoid activation and focal loss
        self.class_embed = nn.Linear(hidden_dim, num_classes)
        self.bbox_embed = MLP(hidden_dim, hidden_dim, 4, bbox_embed_num_layers)
        self.num_feature_levels = num_feature_levels
        if not two_stage:
            self.query_embed = nn.Embedding(num_queries, hidden_dim * 2)
        if num_feature_levels > 1:
            num_backbone_outs = len(backbone.strides)
            input_proj_list = []
            for _ in range(num_backbone_outs):
                in_channels = backbone.num_channels[_]
                input_proj_list.append(
                    nn.Sequential(
                        nn.Conv2d(in_channels, hidden_dim, kernel_size=1),
                        nn.GroupNorm(32, hidden_dim),
                    )
                )
            for _ in range(num_feature_levels - num_backbone_outs):
                input_proj_list.append(
                    nn.Sequential(
                        nn.Conv2d(
                            in_channels, hidden_dim, kernel_size=3, stride=2, padding=1
                        ),
                        nn.GroupNorm(32, hidden_dim),
                    )
                )
                in_channels = hidden_dim
            self.input_proj = nn.ModuleList(input_proj_list)
        else:
            self.input_proj = nn.ModuleList(
                [
                    nn.Sequential(
                        nn.Conv2d(backbone.num_channels[0], hidden_dim, kernel_size=1),
                        nn.GroupNorm(32, hidden_dim),
                    )
                ]
            )
        self.backbone = backbone
        self.aux_loss = aux_loss
        self.with_box_refine = with_box_refine
        self.two_stage = two_stage

        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        self.class_embed.bias.data = torch.ones(num_classes) * bias_value

        for proj in self.input_proj:
            nn.init.xavier_uniform_(proj[0].weight, gain=1)
            nn.init.constant_(proj[0].bias, 0)

        num_pred = transformer.decoder.num_layers
        if with_box_refine:
            self.class_embed = _get_clones(self.class_embed, num_pred)
            self.bbox_embed = _get_clones(self.bbox_embed, num_pred)
            # initialize the box scale height/width at the 1st scale to be 0.1
            nn.init.constant_(self.bbox_embed[0].layers[-1].bias.data[2:], -2.0)
            # hack implementation for iterative bounding box refinement
            self.transformer.decoder.bbox_embed = self.bbox_embed
        else:
            nn.init.constant_(self.bbox_embed.layers[-1].bias.data[2:], -2.0)
            self.class_embed = nn.ModuleList(
                [self.class_embed for _ in range(num_pred)]
            )
            self.bbox_embed = nn.ModuleList([self.bbox_embed for _ in range(num_pred)])
            self.transformer.decoder.bbox_embed = None
        if two_stage:
            # hack implementation for two-stage
            # We only predict foreground/background at the output of encoder
            class_embed = nn.Linear(hidden_dim, 1)
            class_embed.bias.data = torch.ones(1) * bias_value
            self.transformer.encoder.class_embed = class_embed

            for box_embed in self.bbox_embed:
                nn.init.constant_(box_embed.layers[-1].bias.data[2:], 0.0)

            self.transformer.encoder.bbox_embed = MLP(
                hidden_dim, hidden_dim, 4, bbox_embed_num_layers
            )

    @classmethod
    def from_config(cls, cfg):
        num_classes = cfg.MODEL.DETR.NUM_CLASSES
        hidden_dim = cfg.MODEL.DETR.HIDDEN_DIM
        num_queries = cfg.MODEL.DETR.NUM_OBJECT_QUERIES
        # Transformer parameters:
        nheads = cfg.MODEL.DETR.NHEADS
        dropout = cfg.MODEL.DETR.DROPOUT
        dim_feedforward = cfg.MODEL.DETR.DIM_FEEDFORWARD
        enc_layers = cfg.MODEL.DETR.ENC_LAYERS
        dec_layers = cfg.MODEL.DETR.DEC_LAYERS
        pre_norm = cfg.MODEL.DETR.PRE_NORM

        # Loss parameters:
        deep_supervision = cfg.MODEL.DETR.DEEP_SUPERVISION
        num_feature_levels = cfg.MODEL.DETR.NUM_FEATURE_LEVELS

        use_focal_loss = cfg.MODEL.DETR.USE_FOCAL_LOSS

        backbone = build_detr_backbone(cfg)
        transformer = DeformableTransformer(
            d_model=hidden_dim,
            nhead=nheads,
            num_encoder_layers=enc_layers,
            num_decoder_layers=dec_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="relu",
            return_intermediate_dec=True,
            num_feature_levels=num_feature_levels,
            dec_n_points=4,
            enc_n_points=4,
            two_stage=cfg.MODEL.DETR.TWO_STAGE,
            two_stage_num_proposals=num_queries,
        )

        return {
            "backbone": backbone,
            "transformer": transformer,
            "num_classes": num_classes,
            "num_queries": num_queries,
            "num_feature_levels": num_feature_levels,
            "aux_loss": deep_supervision,
            "with_box_refine": cfg.MODEL.DETR.WITH_BOX_REFINE,
            "two_stage": cfg.MODEL.DETR.TWO_STAGE,
        }

    def forward(self, samples: NestedTensor):
        """The forward expects a NestedTensor, which consists of:
           - samples.tensor: batched images, of shape [batch_size x 3 x H x W]
           - samples.mask: a binary mask of shape [batch_size x H x W], containing 1 on padded pixels

        It returns a dict with the following elements:
           - "pred_logits": the classification logits (including no-object) for all queries.
                            Shape= [batch_size x num_queries x (num_classes + 1)]
           - "pred_boxes": The normalized boxes coordinates for all queries, represented as
                           (center_x, center_y, height, width). These values are normalized in [0, 1],
                           relative to the size of each individual image (disregarding possible padding).
                           See PostProcess for information on how to retrieve the unnormalized bounding box.
           - "aux_outputs": Optional, only returned when auxilary losses are activated. It is a list of
                            dictionnaries containing the two above keys for each decoder layer.
        """
        if isinstance(samples, (list, torch.Tensor)):
            samples = nested_tensor_from_tensor_list(samples)
        # features is a list of num_levels NestedTensor.
        # pos is a list of num_levels tensors. Each one has shape (B, H_l, W_l).
        features, pos = self.backbone(samples)
        # srcs is a list of num_levels tensor. Each one has shape (B, C, H_l, W_l)
        srcs = []
        # masks is a list of num_levels tensor. Each one has shape (B, H_l, W_l)
        masks = []
        for l, feat in enumerate(features):
            # src shape: (N, C, H_l, W_l)
            # mask shape: (N, H_l, W_l)
            src, mask = feat.decompose()
            srcs.append(self.input_proj[l](src))
            masks.append(mask)
            assert mask is not None

        if self.num_feature_levels > len(srcs):
            N, C, H, W = samples.tensor.size()
            sample_mask = torch.ones((N, H, W), dtype=torch.bool, device=src.device)
            for idx in range(N):
                image_size = samples.image_sizes[idx]
                h, w = image_size
                sample_mask[idx, :h, :w] = False
            # sample_mask shape (1, N, H, W)
            sample_mask = sample_mask[None].float()

            _len_srcs = len(srcs)
            for l in range(_len_srcs, self.num_feature_levels):
                if l == _len_srcs:
                    src = self.input_proj[l](features[-1].tensors)
                else:
                    src = self.input_proj[l](srcs[-1])
                b, _, h, w = src.size()
                # mask shape (batch_size, h_l, w_l)
                mask = F.interpolate(sample_mask, size=src.shape[-2:]).to(torch.bool)[0]
                pos_l = self.backbone[1](NestedTensor(src, mask)).to(src.dtype)
                srcs.append(src)
                masks.append(mask)
                pos.append(pos_l)

        query_embeds = None
        if not self.two_stage:
            # shape (num_queries, hidden_dim*2)
            query_embeds = self.query_embed.weight

        # hs shape: (num_layers, batch_size, num_queries, c)
        # init_reference shape: (batch_size, num_queries, 2)
        # inter_references shape: (num_layers, bs, num_queries, num_levels, 2)
        (
            hs,
            init_reference,
            inter_references,
            enc_outputs_class,
            enc_outputs_coord_unact,
        ) = self.transformer(srcs, masks, pos, query_embeds)

        outputs_classes = []
        outputs_coords = []
        for lvl in range(hs.shape[0]):
            # reference shape: (num_queries, 2)
            if lvl == 0:
                reference = init_reference
            else:
                reference = inter_references[lvl - 1]
            # shape (batch_size, num_queries, num_classes)
            outputs_class = self.class_embed[lvl](hs[lvl])
            # shape (batch_size, num_queries, 4). 4-tuple (cx, cy, w, h)
            tmp = self.bbox_embed[lvl](hs[lvl])
            if reference.shape[-1] == 4:
                tmp += reference
            else:
                assert reference.shape[-1] == 2
                tmp[..., :2] += reference
            # shape (batch_size, num_queries, 4). 4-tuple (cx, cy, w, h)
            outputs_coord = tmp.sigmoid()

            outputs_classes.append(outputs_class)
            outputs_coords.append(outputs_coord)
        # shape (num_levels, batch_size, num_queries, num_classes)
        outputs_class = torch.stack(outputs_classes)
        # shape (num_levels, batch_size, num_queries, 4)
        outputs_coord = torch.stack(outputs_coords)

        out = {"pred_logits": outputs_class[-1], "pred_boxes": outputs_coord[-1]}
        if self.aux_loss:
            out["aux_outputs"] = self._set_aux_loss(outputs_class, outputs_coord)

        if self.two_stage:
            enc_outputs_coord = enc_outputs_coord_unact.sigmoid()
            out["enc_outputs"] = {
                "pred_logits": enc_outputs_class,
                "pred_boxes": enc_outputs_coord,
            }
        return out

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_coord):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        return [
            {"pred_logits": a, "pred_boxes": b}
            for a, b in zip(outputs_class[:-1], outputs_coord[:-1])
        ]


class PostProcess(nn.Module):
    """This module converts the model's output into the format expected by the coco api"""

    @torch.no_grad()
    def forward(self, outputs, target_sizes):
        """Perform the computation
        Parameters:
            outputs: raw outputs of the model
            target_sizes: tensor of dimension [batch_size x 2] containing the size of each images of the batch
                          For evaluation, this must be the original image size (before any data augmentation)
                          For visualization, this should be the image size after data augment, but before padding
        """
        out_logits, out_bbox = outputs["pred_logits"], outputs["pred_boxes"]

        assert len(out_logits) == len(target_sizes)
        assert target_sizes.shape[1] == 2

        prob = out_logits.sigmoid()
        topk_values, topk_indexes = torch.topk(
            prob.view(out_logits.shape[0], -1), 100, dim=1
        )
        scores = topk_values
        topk_boxes = topk_indexes // out_logits.shape[2]
        labels = topk_indexes % out_logits.shape[2]
        boxes = box_ops.box_cxcywh_to_xyxy(out_bbox)
        boxes = torch.gather(boxes, 1, topk_boxes.unsqueeze(-1).repeat(1, 1, 4))

        # and from relative [0, 1] to absolute [0, height] coordinates
        img_h, img_w = target_sizes.unbind(1)
        scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1)
        boxes = boxes * scale_fct[:, None, :]

        results = [
            {"scores": s, "labels": l, "boxes": b}
            for s, l, b in zip(scores, labels, boxes)
        ]

        return results


class MLP(nn.Module):
    """Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
        )
        # initialize FC weights and bias
        for i, layer in enumerate(self.layers):
            if i < num_layers - 1:
                nn.init.kaiming_uniform_(layer.weight, a=1)
            else:
                nn.init.constant_(layer.weight, 0)
            nn.init.constant_(layer.bias, 0)

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x
