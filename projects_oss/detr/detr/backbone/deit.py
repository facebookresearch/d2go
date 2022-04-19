# code adapt from https://www.internalfb.com/intern/diffusion/FBS/browse/master/fbcode/mobile-vision/experimental/deit/models.py
# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
import json
import math
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from aml.multimodal_video.utils.einops.lib import rearrange
from detectron2.modeling import Backbone, BACKBONE_REGISTRY
from detectron2.utils.file_io import PathManager
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.models.layers import trunc_normal_
from timm.models.registry import register_model
from timm.models.vision_transformer import PatchEmbed, VisionTransformer


def monkey_patch_forward(self, x):
    x = self.proj(x).flatten(2).transpose(1, 2)
    return x


PatchEmbed.forward = monkey_patch_forward


class DistilledVisionTransformer(VisionTransformer, Backbone):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dist_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        num_patches = self.patch_embed.num_patches
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 2, self.embed_dim))
        self.head_dist = (
            nn.Linear(self.embed_dim, self.num_classes)
            if self.num_classes > 0
            else nn.Identity()
        )

        trunc_normal_(self.dist_token, std=0.02)
        trunc_normal_(self.pos_embed, std=0.02)
        self.head_dist.apply(self._init_weights)
        self.norm = None

    def _get_pos_embed(self, H, W):
        embed_size = self.pos_embed.shape[-1]
        # get ride of extra tokens
        pos_tokens = self.pos_embed[:, 2:, :]
        npatchs = pos_tokens.shape[1]
        H0 = W0 = int(math.sqrt(npatchs))
        if H0 == H and W0 == W:
            return self.pos_embed
        # reshape to 2D
        pos_tokens = pos_tokens.transpose(1, 2).reshape(-1, embed_size, H0, W0)
        # interp
        pos_tokens = F.interpolate(
            pos_tokens,
            size=(H, W),
            mode="bilinear",
            align_corners=False,
        )
        # flatten and reshape back
        pos_tokens = pos_tokens.reshape(-1, embed_size, H * W).transpose(1, 2)
        pos_embed = torch.cat((self.pos_embed[:, :2, :], pos_tokens), dim=1)
        return pos_embed

    def forward_features(self, x):
        # taken from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
        # with slight modifications to add the dist_token
        patch_size = self.patch_embed.patch_size[0]
        H, W = x.shape[-2:]
        H, W = H // patch_size, W // patch_size

        B = x.shape[0]
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(
            B, -1, -1
        )  # stole cls_tokens impl from Phil Wang, thanks
        dist_token = self.dist_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, dist_token, x), dim=1)

        # pick the spatial embed and do iterp
        pos_embed = self._get_pos_embed(H, W)
        x = x + pos_embed
        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)

        # x = self.norm(x)
        spatial = rearrange(x[:, 2:], "b (h w) c -> b c h w", h=H, w=W)
        return x[:, 0], x[:, 1], spatial

    def forward(self, x):
        x, x_dist, x0 = self.forward_features(x)
        return x0
        # x = self.head(x)
        # x_dist = self.head_dist(x_dist)
        # if self.training:
        #     return x, x_dist
        # else:
        #     # during inference, return the average of both classifier predictions
        #     return (x + x_dist) / 2


def _cfg(input_size=224, url="", **kwargs):
    return {
        "url": url,
        "num_classes": 1000,
        "input_size": (3, input_size, input_size),
        "pool_size": None,
        "crop_pct": 0.9,
        "interpolation": "bilinear",
        "mean": IMAGENET_DEFAULT_MEAN,
        "std": IMAGENET_DEFAULT_STD,
        "first_conv": "patch_embed.proj",
        "classifier": "head",
        **kwargs,
    }


def deit_scalable_distilled(model_config, pretrained=False, **kwargs):
    assert not pretrained
    model = DistilledVisionTransformer(
        img_size=model_config["I"],
        patch_size=model_config["p"],
        embed_dim=model_config["h"] * model_config["e"],
        depth=model_config["d"],
        num_heads=model_config["h"],
        mlp_ratio=model_config["r"],
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs,
    )
    model.default_cfg = _cfg(input_size=model_config["I"])
    print("model arch config: {}".format(model_config))
    print("model train config: {}".format(model.default_cfg))
    return model


def add_deit_backbone_config(cfg):
    cfg.MODEL.DEIT = type(cfg)()
    cfg.MODEL.DEIT.MODEL_CONFIG = None
    cfg.MODEL.DEIT.WEIGHTS = None


@BACKBONE_REGISTRY.register()
def deit_d2go_model_wrapper(cfg, _):
    assert cfg.MODEL.DEIT.MODEL_CONFIG is not None
    with PathManager.open(cfg.MODEL.DEIT.MODEL_CONFIG) as f:
        model_config = json.load(f)
    model = deit_scalable_distilled(
        model_config,
        num_classes=0,  # set num_classes=0 to avoid building cls head
        drop_rate=0,
        drop_path_rate=0.1,
    )
    # load weights
    if cfg.MODEL.DEIT.WEIGHTS is not None:
        with PathManager.open(cfg.MODEL.DEIT.WEIGHTS, "rb") as f:
            state_dict = torch.load(f, map_location="cpu")["model"]
        rm_keys = [k for k in state_dict if "head" in k]
        rm_keys = rm_keys + ["norm.weight", "norm.bias"]
        print(rm_keys)
        for k in rm_keys:
            del state_dict[k]
        model.load_state_dict(state_dict)
        print(f"loaded weights from {cfg.MODEL.DEIT.WEIGHTS}")
    return model
