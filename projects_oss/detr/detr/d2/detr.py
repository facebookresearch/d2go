#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import numpy as np
import torch
import torch.nn.functional as F
from d2go.registry.builtin import META_ARCH_REGISTRY
from detectron2.modeling import detector_postprocess
from detectron2.structures import BitMasks, Boxes, ImageList, Instances
from detr.datasets.coco import convert_coco_poly_to_mask
from detr.models.backbone import Joiner
from detr.models.build import build_detr_model
from detr.models.deformable_detr import DeformableDETR
from detr.models.deformable_transformer import DeformableTransformer
from detr.models.detr import DETR
from detr.models.matcher import HungarianMatcher
from detr.models.position_encoding import PositionEmbeddingSine
from detr.models.segmentation import DETRsegm, PostProcessSegm
from detr.models.setcriterion import FocalLossSetCriterion, SetCriterion
from detr.util.box_ops import box_cxcywh_to_xyxy, box_xyxy_to_cxcywh
from detr.util.misc import NestedTensor
from torch import nn

__all__ = ["Detr"]


@META_ARCH_REGISTRY.register()
class Detr(nn.Module):
    """
    Implement Detr
    """

    def __init__(self, cfg):
        super().__init__()

        self.device = torch.device(cfg.MODEL.DEVICE)
        self.use_focal_loss = cfg.MODEL.DETR.USE_FOCAL_LOSS
        self.num_classes = cfg.MODEL.DETR.NUM_CLASSES
        self.mask_on = cfg.MODEL.MASK_ON
        dec_layers = cfg.MODEL.DETR.DEC_LAYERS

        # Loss parameters:
        giou_weight = cfg.MODEL.DETR.GIOU_WEIGHT
        l1_weight = cfg.MODEL.DETR.L1_WEIGHT
        cls_weight = cfg.MODEL.DETR.CLS_WEIGHT
        deep_supervision = cfg.MODEL.DETR.DEEP_SUPERVISION
        no_object_weight = cfg.MODEL.DETR.NO_OBJECT_WEIGHT

        self.detr = build_detr_model(cfg)

        if self.mask_on:
            frozen_weights = cfg.MODEL.DETR.FROZEN_WEIGHTS
            if frozen_weights != "":
                print("LOAD pre-trained weights")
                weight = torch.load(
                    frozen_weights, map_location=lambda storage, loc: storage
                )["model"]
                new_weight = {}
                for k, v in weight.items():
                    if "detr." in k:
                        new_weight[k.replace("detr.", "")] = v
                    else:
                        print(f"Skipping loading weight {k} from frozen model")
                del weight
                self.detr.load_state_dict(new_weight)
                del new_weight
            self.detr = DETRsegm(self.detr, freeze_detr=(frozen_weights != ""))
            self.seg_postprocess = PostProcessSegm

        self.detr.to(self.device)

        # building criterion
        matcher = HungarianMatcher(
            cost_class=cls_weight,
            cost_bbox=l1_weight,
            cost_giou=giou_weight,
            use_focal_loss=self.use_focal_loss,
        )
        weight_dict = {"loss_ce": cls_weight, "loss_bbox": l1_weight}
        weight_dict["loss_giou"] = giou_weight
        if deep_supervision:
            aux_weight_dict = {}
            for i in range(dec_layers - 1):
                aux_weight_dict.update({k + f"_{i}": v for k, v in weight_dict.items()})
            weight_dict.update(aux_weight_dict)
        losses = ["labels", "boxes", "cardinality"]
        if self.mask_on:
            losses += ["masks"]
        if self.use_focal_loss:
            self.criterion = FocalLossSetCriterion(
                self.num_classes,
                matcher=matcher,
                weight_dict=weight_dict,
                losses=losses,
            )
        else:
            self.criterion = SetCriterion(
                self.num_classes,
                matcher=matcher,
                weight_dict=weight_dict,
                eos_coef=no_object_weight,
                losses=losses,
            )
        self.criterion.to(self.device)

        pixel_mean = torch.Tensor(cfg.MODEL.PIXEL_MEAN).to(self.device).view(3, 1, 1)
        pixel_std = torch.Tensor(cfg.MODEL.PIXEL_STD).to(self.device).view(3, 1, 1)
        self.normalizer = lambda x: (x - pixel_mean) / pixel_std
        self.to(self.device)

    def forward(self, batched_inputs):
        """
        Args:
            batched_inputs: a list, batched outputs of :class:`DatasetMapper` .
                Each item in the list contains the inputs for one image.
                For now, each item in the list is a dict that contains:

                * image: Tensor, image in (C, H, W) format.
                * instances: Instances

                Other information that's included in the original dicts, such as:

                * "height", "width" (int): the output resolution of the model, used in inference.
                  See :meth:`postprocess` for details.
        Returns:
            dict[str: Tensor]:
                mapping from a named loss to a tensor storing the loss. Used during training only.
        """
        images_lists = self.preprocess_image(batched_inputs)
        # convert images_lists to Nested Tensor?
        nested_images = self.imagelist_to_nestedtensor(images_lists)
        output = self.detr(nested_images)

        if self.training:
            gt_instances = [x["instances"].to(self.device) for x in batched_inputs]

            # targets: List[Dict[str, torch.Tensor]]. Keys
            # "labels": [NUM_BOX,]
            # "boxes": [NUM_BOX, 4]
            targets = self.prepare_targets(gt_instances)
            loss_dict = self.criterion(output, targets)
            weight_dict = self.criterion.weight_dict
            for k in loss_dict.keys():
                if k in weight_dict:
                    loss_dict[k] *= weight_dict[k]
            return loss_dict
        else:
            box_cls = output["pred_logits"]
            box_pred = output["pred_boxes"]
            mask_pred = output["pred_masks"] if self.mask_on else None
            results = self.inference(
                box_cls, box_pred, mask_pred, images_lists.image_sizes
            )
            processed_results = []
            for results_per_image, input_per_image, image_size in zip(
                results, batched_inputs, images_lists.image_sizes
            ):
                height = input_per_image.get("height", image_size[0])
                width = input_per_image.get("width", image_size[1])
                r = detector_postprocess(results_per_image, height, width)
                processed_results.append({"instances": r})
            return processed_results

    def prepare_targets(self, targets):
        new_targets = []
        for targets_per_image in targets:
            h, w = targets_per_image.image_size
            image_size_xyxy = torch.as_tensor(
                [w, h, w, h], dtype=torch.float, device=self.device
            )
            gt_classes = targets_per_image.gt_classes  # shape (NUM_BOX,)
            gt_boxes = targets_per_image.gt_boxes.tensor / image_size_xyxy
            gt_boxes = box_xyxy_to_cxcywh(gt_boxes)  # shape (NUM_BOX, 4)
            new_targets.append({"labels": gt_classes, "boxes": gt_boxes})
            if self.mask_on and hasattr(targets_per_image, "gt_masks"):
                gt_masks = targets_per_image.gt_masks
                gt_masks = convert_coco_poly_to_mask(gt_masks.polygons, h, w)
                new_targets[-1].update({"masks": gt_masks})
        return new_targets

    def inference(self, box_cls, box_pred, mask_pred, image_sizes):
        """
        Arguments:
            box_cls (Tensor): tensor of shape (batch_size, num_queries, K).
                The tensor predicts the classification probability for each query.
            box_pred (Tensor): tensors of shape (batch_size, num_queries, 4).
                The tensor predicts 4-vector (x,y,w,h) box
                regression values for every queryx
            image_sizes (List[torch.Size]): the input image sizes

        Returns:
            results (List[Instances]): a list of #images elements.
        """
        assert len(box_cls) == len(image_sizes)
        results = []

        # For each box we assign the best class or the second best if the best on is `no_object`.
        if self.use_focal_loss:
            prob = box_cls.sigmoid()
            # TODO make top-100 as an option for non-focal-loss as well
            scores, topk_indexes = torch.topk(
                prob.view(box_cls.shape[0], -1), 100, dim=1
            )
            topk_boxes = topk_indexes // box_cls.shape[2]
            labels = topk_indexes % box_cls.shape[2]
        else:
            scores, labels = F.softmax(box_cls, dim=-1)[:, :, :-1].max(-1)

        for i, (
            scores_per_image,
            labels_per_image,
            box_pred_per_image,
            image_size,
        ) in enumerate(zip(scores, labels, box_pred, image_sizes)):
            result = Instances(image_size)
            boxes = box_cxcywh_to_xyxy(box_pred_per_image)
            if self.use_focal_loss:
                boxes = torch.gather(boxes, 0, topk_boxes[i].unsqueeze(-1).repeat(1, 4))

            result.pred_boxes = Boxes(boxes)
            result.pred_boxes.scale(scale_x=image_size[1], scale_y=image_size[0])
            if self.mask_on:
                mask = F.interpolate(
                    mask_pred[i].unsqueeze(0),
                    size=image_size,
                    mode="bilinear",
                    align_corners=False,
                )
                mask = mask[0].sigmoid() > 0.5
                B, N, H, W = mask_pred.shape
                mask = BitMasks(mask.cpu()).crop_and_resize(
                    result.pred_boxes.tensor.cpu(), 32
                )
                result.pred_masks = mask.unsqueeze(1).to(mask_pred[0].device)

            result.scores = scores_per_image
            result.pred_classes = labels_per_image
            results.append(result)
        return results

    def preprocess_image(self, batched_inputs):
        """
        Normalize, pad and batch the input images.
        """
        images = [self.normalizer(x["image"].to(self.device)) for x in batched_inputs]
        images = ImageList.from_tensors(images)
        return images

    def imagelist_to_nestedtensor(self, images):
        tensor = images.tensor
        device = tensor.device
        N, _, H, W = tensor.shape
        masks = torch.ones((N, H, W), dtype=torch.bool, device=device)
        for idx, (h, w) in enumerate(images.image_sizes):
            masks[idx, :h, :w] = False
        return NestedTensor(tensor, masks)
