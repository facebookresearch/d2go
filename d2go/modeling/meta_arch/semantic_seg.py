#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

from typing import Any, Dict, List

import torch
import torch.nn as nn
from d2go.export.api import PredictorExportConfig
from d2go.registry.builtin import META_ARCH_REGISTRY
from detectron2.modeling import SemanticSegmentor as _SemanticSegmentor
from detectron2.modeling.postprocessing import sem_seg_postprocess
from detectron2.structures import ImageList
from mobile_cv.predictor.api import FuncInfo


# Re-register D2's meta-arch in D2Go with updated APIs
@META_ARCH_REGISTRY.register()
class SemanticSegmentor(_SemanticSegmentor):
    def prepare_for_export(self, cfg, inputs, predictor_type):
        preprocess_info = FuncInfo.gen_func_info(
            PreprocessFunc,
            params={
                "size_divisibility": self.backbone.size_divisibility,
                "device": str(self.device),
            },
        )
        postprocess_info = FuncInfo.gen_func_info(
            PostprocessFunc,
            params={},
        )

        preprocess_func = preprocess_info.instantiate()

        return PredictorExportConfig(
            model=ModelWrapper(self),
            data_generator=lambda x: (preprocess_func(x),),
            preprocess_info=preprocess_info,
            postprocess_info=postprocess_info,
        )


class ModelWrapper(nn.Module):
    def __init__(self, segmentor):
        super().__init__()
        self.segmentor = segmentor

    def forward(self, x):
        x = (x - self.segmentor.pixel_mean) / self.segmentor.pixel_std
        features = self.segmentor.backbone(x)
        results, losses = self.segmentor.sem_seg_head(features, targets=None)
        return results


class PreprocessFunc(object):
    """
    A common preprocessing module for semantic segmentation models.
    """

    def __init__(self, size_divisibility, device):
        self.size_divisibility = size_divisibility
        self.device = device

    def __call__(self, batched_inputs: List[Dict[str, Any]]) -> torch.Tensor:
        """
        Retreive image tensor from dataloader batches.

        Args:
            batched_inputs: (List[Dict[str, Tensor]]): output from a
                D2Go train or test data loader.

        Returns:
            input images (torch.Tensor): ImageList-wrapped NCHW tensor
                (i.e. with padding and divisibility alignment) of batches' images.
        """
        images = [x["image"].to(self.device) for x in batched_inputs]
        images = ImageList.from_tensors(images, self.size_divisibility)
        return images.tensor


class PostprocessFunc(object):
    """
    A common postprocessing module for semantic segmentation models.
    """

    def __call__(
        self,
        batched_inputs: List[Dict[str, Any]],
        tensor_inputs: torch.Tensor,
        tensor_outputs: torch.Tensor,
    ) -> List[Dict[str, Any]]:
        """
        Rescales sem_seg logits to original image input resolution,
        and packages the logits into D2Go's expected output format.

        Args:
            inputs (List[Dict[str, Tensor]]): batched inputs from the dataloader.
            tensor_inputs (Tensor): tensorized inputs, e.g. from `PreprocessFunc`.
            tensor_outputs (Tensor): sem seg logits tensor from the model to process.

        Returns:
            processed_results (List[Dict]): List of D2Go output dicts ready to be used
                downstream in an Evaluator, for export, etc.
        """
        results = tensor_outputs  # nchw

        processed_results = []
        for result, input_per_image in zip(results, batched_inputs):
            height = input_per_image.get("height")
            width = input_per_image.get("width")
            image_tensor_shape = input_per_image["image"].shape
            image_size = (image_tensor_shape[1], image_tensor_shape[2])

            # D2's sem_seg_postprocess rescales sem seg masks to the
            # provided original input resolution.
            r = sem_seg_postprocess(result, image_size, height, width)
            processed_results.append({"sem_seg": r})
        return processed_results
