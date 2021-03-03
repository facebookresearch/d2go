#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved


from detectron2.utils.registry import Registry

D2GO_DATA_MAPPER_REGISTRY = Registry("D2GO_DATA_MAPPER")


def build_dataset_mapper(cfg, is_train, *args, **kwargs):
    name = cfg.D2GO_DATA.MAPPER.NAME
    return D2GO_DATA_MAPPER_REGISTRY.get(name)(cfg, is_train, *args, **kwargs)
