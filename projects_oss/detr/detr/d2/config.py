#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# -*- coding: utf-8 -*-
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from d2go.config import CfgNode as CN


def add_detr_config(cfg):
    """
    Add config for DETR.
    """
    cfg.MODEL.DETR = CN()
    cfg.MODEL.DETR.NAME = "DETR"
    cfg.MODEL.DETR.NUM_CLASSES = 80

    # simple backbone
    cfg.MODEL.BACKBONE.SIMPLE = False
    cfg.MODEL.BACKBONE.STRIDE = 1
    cfg.MODEL.BACKBONE.CHANNEL = 0

    # FBNet
    cfg.MODEL.FBNET_V2.OUT_FEATURES = ["trunk3"]

    # For Segmentation
    cfg.MODEL.DETR.FROZEN_WEIGHTS = ""

    # LOSS
    cfg.MODEL.DETR.DEFORMABLE = False
    cfg.MODEL.DETR.USE_FOCAL_LOSS = False
    cfg.MODEL.DETR.CENTERED_POSITION_ENCODIND = False
    cfg.MODEL.DETR.CLS_WEIGHT = 1.0
    cfg.MODEL.DETR.NUM_FEATURE_LEVELS = 4
    cfg.MODEL.DETR.GIOU_WEIGHT = 2.0
    cfg.MODEL.DETR.L1_WEIGHT = 5.0
    cfg.MODEL.DETR.DEEP_SUPERVISION = True
    cfg.MODEL.DETR.NO_OBJECT_WEIGHT = 0.1
    cfg.MODEL.DETR.WITH_BOX_REFINE = False
    cfg.MODEL.DETR.TWO_STAGE = False
    cfg.MODEL.DETR.DECODER_BLOCK_GRAD = True

    # TRANSFORMER
    cfg.MODEL.DETR.NHEADS = 8
    cfg.MODEL.DETR.DROPOUT = 0.1
    cfg.MODEL.DETR.DIM_FEEDFORWARD = 2048
    cfg.MODEL.DETR.ENC_LAYERS = 6
    cfg.MODEL.DETR.DEC_LAYERS = 6
    cfg.MODEL.DETR.BBOX_EMBED_NUM_LAYERS = 3
    cfg.MODEL.DETR.PRE_NORM = False

    cfg.MODEL.DETR.HIDDEN_DIM = 256
    cfg.MODEL.DETR.NUM_OBJECT_QUERIES = 100

    # solver
    cfg.SOLVER.OPTIMIZER = "ADAMW"
    cfg.SOLVER.BACKBONE_MULTIPLIER = 0.1

    # tgt & embeddings
    cfg.MODEL.DETR.LEARNABLE_TGT = False
