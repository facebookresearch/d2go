#!/usr/bin/env python3

from d2go.config import CfgNode as CN
from d2go.data.dataset_mappers.build import D2GO_DATA_MAPPER_REGISTRY
from d2go.data.dataset_mappers.d2go_dataset_mapper import D2GoDatasetMapper
from d2go.runner import GeneralizedRCNNRunner
from detr.backbone.deit import add_deit_backbone_config
from detr.backbone.pit import add_pit_backbone_config
from detr.d2 import add_detr_config, DetrDatasetMapper


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
    @classmethod
    def get_default_cfg(cls):
        _C = super().get_default_cfg()
        add_detr_config(_C)
        add_deit_backbone_config(_C)
        add_pit_backbone_config(_C)
        return _C
