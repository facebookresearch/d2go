#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved


import torch
from d2go.quantization.qconfig import set_backend_and_create_qconfig
from d2go.registry.builtin import META_ARCH_REGISTRY
from d2go.utils.testing.data_loader_helper import create_local_dataset
from detectron2.structures import Boxes, ImageList, Instances
from torch.ao.quantization.quantize_fx import convert_fx, prepare_qat_fx


@META_ARCH_REGISTRY.register()
class DetMetaArchForTest(torch.nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 4, kernel_size=3, stride=1, padding=1)
        self.bn = torch.nn.BatchNorm2d(4)
        self.relu = torch.nn.ReLU(inplace=True)
        self.avgpool = torch.nn.AdaptiveAvgPool2d((1, 1))
        # weights that will be updated in forward() during training, use to simulate
        # weight udpates in optimization step
        self.register_buffer("scale_weight", torch.Tensor([0.0]))

    @property
    def device(self):
        return self.conv.weight.device

    def forward(self, inputs):
        if not self.training:
            return self.inference(inputs)

        images = [x["image"].to(self.device) for x in inputs]
        images = ImageList.from_tensors(images, 1)
        ret = self.conv(images.tensor)
        ret = self.bn(ret)
        ret = self.relu(ret)
        ret = self.avgpool(ret)

        # simulate weight updates
        self.scale_weight.fill_(1.0)

        return {"loss": ret.norm()}

    def inference(self, inputs):
        instance = Instances((10, 10))
        instance.pred_boxes = Boxes(
            torch.tensor([[2.5, 2.5, 7.5, 7.5]], device=self.device) * self.scale_weight
        )
        instance.scores = torch.tensor([0.9], device=self.device)
        instance.pred_classes = torch.tensor([1], dtype=torch.int32, device=self.device)
        ret = [{"instances": instance}]
        return ret

    def custom_prepare_fx(self, cfg, is_qat, example_input=None):
        example_inputs = (torch.rand(1, 3, 3, 3),)
        self.avgpool = prepare_qat_fx(
            self.avgpool,
            {"": set_backend_and_create_qconfig(cfg, is_train=self.training)},
            example_inputs,
        )

        def convert_fx_callback(model):
            model.avgpool = convert_fx(model.avgpool)
            return model

        return self, convert_fx_callback


def get_det_meta_arch_cfg(cfg, dataset_name, output_dir):
    cfg.MODEL.DEVICE = "cpu"
    cfg.MODEL.META_ARCHITECTURE = "DetMetaArchForTest"

    cfg.DATASETS.TRAIN = (dataset_name,)
    cfg.DATASETS.TEST = (dataset_name,)

    cfg.INPUT.MIN_SIZE_TRAIN = (10,)
    cfg.INPUT.MIN_SIZE_TEST = (10,)

    cfg.SOLVER.MAX_ITER = 5
    cfg.SOLVER.STEPS = [2]
    cfg.SOLVER.WARMUP_ITERS = 1
    cfg.SOLVER.CHECKPOINT_PERIOD = 1
    cfg.SOLVER.IMS_PER_BATCH = 2
    cfg.SOLVER.REFERENCE_WORLD_SIZE = 0

    cfg.OUTPUT_DIR = output_dir

    return cfg


def create_detection_cfg(runner, output_dir):
    ds_name = create_local_dataset(output_dir, 5, 10, 10)
    cfg = runner.get_default_cfg()
    return get_det_meta_arch_cfg(cfg, ds_name, output_dir)
