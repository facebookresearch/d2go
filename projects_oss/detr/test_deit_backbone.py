import logging
import unittest

import torch
from d2go.config import CfgNode as CN
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.modeling import BACKBONE_REGISTRY
from detectron2.utils.file_io import PathManager
from detr.backbone.deit import add_deit_backbone_config
from detr.backbone.pit import add_pit_backbone_config

logger = logging.getLogger(__name__)

# avoid testing on sandcastle due to access to manifold
USE_CUDA = torch.cuda.device_count() > 0


class TestTransformerBackbone(unittest.TestCase):
    @unittest.skipIf(
        not USE_CUDA, "avoid testing on sandcastle due to access to manifold"
    )
    def test_deit_model(self):
        cfg = CN()
        cfg.MODEL = CN()
        add_deit_backbone_config(cfg)
        build_model = BACKBONE_REGISTRY.get("deit_d2go_model_wrapper")
        deit_models = {
            "8X-7-RM_4": 170,
            "DeiT-Tiny": 224,
            "DeiT-Small": 224,
            "32X-1-RM_2": 221,
            "8X-7": 160,
            "32X-1": 256,
        }
        deit_model_weights = {
            "8X-7-RM_4": "manifold://mobile_vision_workflows/tree/workflows/kyungminkim/20210511/deit_[model]deit_scaling_distill_[bs]128_[mcfg]8X-7-RM_4_.OIXarYpbZw/checkpoint_best.pth",
            "DeiT-Tiny": "manifold://mobile_vision_workflows/tree/workflows/cl114/DeiT-official-ckpt/deit_tiny_distilled_patch16_224-b40b3cf7.pth",
            "DeiT-Small": "manifold://mobile_vision_workflows/tree/workflows/cl114/DeiT-official-ckpt/deit_small_distilled_patch16_224-649709d9.pth",
            "32X-1-RM_2": "manifold://mobile_vision_workflows/tree/workflows/kyungminkim/20210511/deit_[model]deit_scaling_distill_[bs]64_[mcfg]32X-1-RM_2_.xusuFyNMdD/checkpoint_best.pth",
            "8X-7": "manifold://mobile_vision_workflows/tree/workflows/cl114/scaled_best/8X-7.pth",
            "32X-1": "manifold://mobile_vision_workflows/tree/workflows/cl114/scaled_best/32X-1.pth",
        }

        for model_name, org_size in deit_models.items():
            print("model_name", model_name)
            cfg.MODEL.DEIT.MODEL_CONFIG = f"manifold://mobile_vision_workflows/tree/workflows/wbc/deit/model_cfgs/{model_name}.json"
            cfg.MODEL.DEIT.WEIGHTS = deit_model_weights[model_name]
            model = build_model(cfg, None)
            model.eval()
            for input_size_h in [org_size, 192, 224, 256, 320]:
                for input_size_w in [org_size, 192, 224, 256, 320]:
                    x = torch.rand(1, 3, input_size_h, input_size_w)
                    y = model(x)
                    print(f"x.shape: {x.shape}, y.shape: {y.shape}")

    @unittest.skipIf(
        not USE_CUDA, "avoid testing on sandcastle due to access to manifold"
    )
    def test_pit_model(self):
        cfg = CN()
        cfg.MODEL = CN()
        add_pit_backbone_config(cfg)
        build_model = BACKBONE_REGISTRY.get("pit_d2go_model_wrapper")
        pit_models = {
            "pit_ti_ours": 160,
            "pit_ti": 224,
            "pit_s_ours_v1": 256,
            "pit_s": 224,
        }
        pit_model_weights = {
            "pit_ti_ours": "manifold://mobile_vision_workflows/tree/workflows/kyungminkim/20210515/deit_[model]pit_scalable_distilled_[bs]128_[mcfg]pit_ti_ours_.HImkjNCpJI/checkpoint_best.pth",
            "pit_ti": "manifold://mobile_vision_workflows/tree/workflows/kyungminkim/20210515/deit_[model]pit_scalable_distilled_[bs]128_[mcfg]pit_ti_.QJeFNUfYOD/checkpoint_best.pth",
            "pit_s_ours_v1": "manifold://mobile_vision_workflows/tree/workflows/kyungminkim/20210515/deit_[model]pit_scalable_distilled_[bs]64_[mcfg]pit_s_ours_v1_.LXdwyBDaNY/checkpoint_best.pth",
            "pit_s": "manifold://mobile_vision_workflows/tree/workflows/kyungminkim/20210515/deit_[model]pit_scalable_distilled_[bs]128_[mcfg]pit_s_.zReQLPOuJe/checkpoint_best.pth",
        }
        for model_name, org_size in pit_models.items():
            print("model_name", model_name)
            cfg.MODEL.PIT.MODEL_CONFIG = f"manifold://mobile_vision_workflows/tree/workflows/wbc/deit/model_cfgs/{model_name}.json"
            cfg.MODEL.PIT.WEIGHTS = pit_model_weights[model_name]
            cfg.MODEL.PIT.DILATED = True
            model = build_model(cfg, None)
            model.eval()
            for input_size_h in [org_size, 192, 224, 256, 320]:
                for input_size_w in [org_size, 192, 224, 256, 320]:
                    x = torch.rand(1, 3, input_size_h, input_size_w)
                    y = model(x)
                    print(f"x.shape: {x.shape}, y.shape: {y.shape}")
