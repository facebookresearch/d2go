#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import torch.nn as nn
from d2go.export.api import PredictorExportConfig
from detectron2.modeling.postprocessing import sem_seg_postprocess
from detectron2.structures import ImageList
from mobile_cv.predictor.api import FuncInfo


class SemanticSegmentorPatch:
    METHODS_TO_PATCH = [
        "prepare_for_export",
    ]

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
    def __init__(self, size_divisibility, device):
        self.size_divisibility = size_divisibility
        self.device = device

    def __call__(self, inputs):
        images = [x["image"].to(self.device) for x in inputs]
        images = ImageList.from_tensors(images, self.size_divisibility)
        return images.tensor


class PostprocessFunc(object):
    def __call__(self, inputs, tensor_inputs, tensor_outputs):
        results = tensor_outputs  # nchw

        processed_results = []
        for result, input_per_image in zip(results, inputs):
            height = input_per_image.get("height")
            width = input_per_image.get("width")
            image_tensor_shape = input_per_image["image"].shape
            image_size = (image_tensor_shape[1], image_tensor_shape[2])

            r = sem_seg_postprocess(result, image_size, height, width)
            processed_results.append({"sem_seg": r})
        return processed_results
