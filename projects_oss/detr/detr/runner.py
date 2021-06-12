#!/usr/bin/env python3

from detr.d2 import DetrDatasetMapper, add_detr_config

from detectron2.solver.build import maybe_add_gradient_clipping
from d2go.config import CfgNode as CN
from d2go.runner import GeneralizedRCNNRunner
from d2go.data.dataset_mappers.build import D2GO_DATA_MAPPER_REGISTRY
from d2go.data.dataset_mappers.d2go_dataset_mapper import D2GoDatasetMapper


@D2GO_DATA_MAPPER_REGISTRY.register()
class DETRDatasetMapper(DetrDatasetMapper, D2GoDatasetMapper):
    def __init__(self, cfg, is_train=True, image_loader=None, tfm_gens=None):
        self.image_loader = None
        self.backfill_size = False
        self.retry = 3
        self.catch_exception = True
        self._error_count = 0
        self._total_counts = 0
        self._error_types = {}
        super().__init__(cfg, is_train)

    def _original_call(self, dataset_dict):
        return DetrDatasetMapper.__call__(self, dataset_dict)

    def __call__(self, dataset_dict):
        return D2GoDatasetMapper.__call__(self, dataset_dict)

class DETRRunner(GeneralizedRCNNRunner):
    def get_default_cfg(self):
        _C = super().get_default_cfg()
        add_detr_config(_C)
        _C.MODEL.DETR = CN(_C.MODEL.DETR)
        return _C

    # TODO rm this after update optimizer
    @classmethod
    def build_optimizer(cls, cfg, model):
        import torch
        import itertools
        from typing import Any, Dict, List, Set
        from detectron2.solver.build import maybe_add_gradient_clipping
        params: List[Dict[str, Any]] = []
        memo: Set[torch.nn.parameter.Parameter] = set()
        for key, value in model.named_parameters(recurse=True):
            if not value.requires_grad:
                continue
            # Avoid duplicating parameters
            if value in memo:
                continue
            memo.add(value)
            lr = cfg.SOLVER.BASE_LR
            weight_decay = cfg.SOLVER.WEIGHT_DECAY
            if "backbone.0" in key or "reference_points" in key or "sampling_offsets" in key:
                lr = lr * 0.1
            params += [{"params": [value], "lr": lr, "weight_decay": weight_decay}]

        def maybe_add_full_model_gradient_clipping(optim):  # optim: the optimizer class
            # detectron2 doesn't have full model gradient clipping now
            clip_norm_val = cfg.SOLVER.CLIP_GRADIENTS.CLIP_VALUE
            enable = (
                cfg.SOLVER.CLIP_GRADIENTS.ENABLED
                and cfg.SOLVER.CLIP_GRADIENTS.CLIP_TYPE == "full_model"
                and clip_norm_val > 0.0
            )

            class FullModelGradientClippingOptimizer(optim):
                def step(self, closure=None):
                    all_params = itertools.chain(*[x["params"] for x in self.param_groups])
                    torch.nn.utils.clip_grad_norm_(all_params, clip_norm_val)
                    super().step(closure=closure)

            return FullModelGradientClippingOptimizer if enable else optim

        optimizer_type = cfg.SOLVER.OPTIMIZER
        if optimizer_type == "SGD":
            optimizer = maybe_add_full_model_gradient_clipping(torch.optim.SGD)(
                params, cfg.SOLVER.BASE_LR, momentum=cfg.SOLVER.MOMENTUM
            )
        elif optimizer_type == "ADAMW":
            optimizer = maybe_add_full_model_gradient_clipping(torch.optim.AdamW)(
                params, cfg.SOLVER.BASE_LR
            )
        else:
            raise NotImplementedError(f"no optimizer type {optimizer_type}")
        if not cfg.SOLVER.CLIP_GRADIENTS.CLIP_TYPE == "full_model":
            optimizer = maybe_add_gradient_clipping(cfg, optimizer)
        return optimizer
