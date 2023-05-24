#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import os
import unittest

import d2go.runner.default_runner as default_runner
import torch
from d2go.registry.builtin import META_ARCH_REGISTRY
from d2go.utils.testing.data_loader_helper import create_local_dataset
from d2go.utils.testing.helper import tempdir
from detectron2.structures import ImageList


TEST_CUDA: bool = torch.cuda.is_available()


@META_ARCH_REGISTRY.register()
class MetaArchForTestSimple(torch.nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 4, kernel_size=3, stride=1, padding=1)
        self.bn = torch.nn.BatchNorm2d(4)
        self.relu = torch.nn.ReLU(inplace=True)
        self.avgpool = torch.nn.AdaptiveAvgPool2d((1, 1))

    @property
    def device(self):
        return self.conv.weight.device

    def forward(self, inputs):
        images = [x["image"] for x in inputs]
        images = ImageList.from_tensors(images, 1).to(self.device)
        ret = self.conv(images.tensor)
        ret = self.bn(ret)
        ret = self.relu(ret)
        ret = self.avgpool(ret)
        return {"loss": ret.norm()}


def train_with_memory_profiler(output_dir, device="cpu"):
    ds_name = create_local_dataset(output_dir, 5, 10, 10)

    runner = default_runner.Detectron2GoRunner()
    cfg = runner.get_default_cfg()

    cfg.MODEL.DEVICE = device
    cfg.MODEL.META_ARCHITECTURE = "MetaArchForTestSimple"
    cfg.SOLVER.MAX_ITER = 10
    cfg.DATASETS.TRAIN = (ds_name,)
    cfg.DATASETS.TEST = (ds_name,)
    cfg.OUTPUT_DIR = output_dir
    cfg.MEMORY_PROFILER.ENABLED = True
    cfg.MEMORY_PROFILER.LOG_N_STEPS = 3
    cfg.MEMORY_PROFILER.LOG_DURING_TRAIN_AT = 5

    # Register configs
    runner.register(cfg)

    # Create dummy data to pass to wrapper
    model = runner.build_model(cfg)
    runner.do_train(cfg, model, resume=True)
    return cfg


class TestGPUMemoryProfiler(unittest.TestCase):
    @tempdir
    def test_gpu_memory_profiler_no_gpu(self, tmp_dir: str):
        # GPU memory profiler should silently pass if no CUDA is available
        train_with_memory_profiler(tmp_dir, device="cpu")

    @tempdir
    @unittest.skipIf(not TEST_CUDA, "no CUDA detected")
    def test_gpu_memory_profiler_with_gpu(self, tmp_dir: str):
        cfg = train_with_memory_profiler(tmp_dir, device="cuda")
        n = cfg.MEMORY_PROFILER.LOG_N_STEPS
        s = cfg.MEMORY_PROFILER.LOG_DURING_TRAIN_AT

        save_dir = os.path.join(tmp_dir, "memory_snapshot")
        self.assertTrue(os.path.exists(save_dir))
        for i in [n - 1, s + n - 1]:
            trace_dir = os.path.join(save_dir, f"iter{i}_rank0")
            self.assertTrue(os.path.exists(trace_dir))
            self.assertTrue(os.path.exists(os.path.join(trace_dir, "snapshot.pickle")))
            self.assertTrue(os.path.exists(os.path.join(trace_dir, "trace_plot.html")))
            self.assertTrue(
                os.path.exists(os.path.join(trace_dir, "segment_plot.html"))
            )
