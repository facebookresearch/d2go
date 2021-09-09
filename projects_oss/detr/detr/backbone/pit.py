# https://www.internalfb.com/intern/diffusion/FBS/browse/master/fbcode/mobile-vision/experimental/deit/pit_models.py
# PiT
# Copyright 2021-present NAVER Corp.
# Apache License v2.0

import json
import math
from functools import partial

import torch
import torch.nn.functional as F
from aml.multimodal_video.utils.einops.lib import rearrange
from detectron2.modeling import Backbone, BACKBONE_REGISTRY
from detectron2.utils.file_io import PathManager
from timm.models.layers import trunc_normal_
from timm.models.registry import register_model
from timm.models.vision_transformer import Block as transformer_block
from torch import nn


class Transformer(nn.Module):
    def __init__(
        self,
        base_dim,
        depth,
        heads,
        mlp_ratio,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_prob=None,
    ):
        super(Transformer, self).__init__()
        self.layers = nn.ModuleList([])
        embed_dim = base_dim * heads

        if drop_path_prob is None:
            drop_path_prob = [0.0 for _ in range(depth)]

        self.blocks = nn.ModuleList(
            [
                transformer_block(
                    dim=embed_dim,
                    num_heads=heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=True,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=drop_path_prob[i],
                    norm_layer=partial(nn.LayerNorm, eps=1e-6),
                )
                for i in range(depth)
            ]
        )

    def forward(self, x, cls_tokens):
        h, w = x.shape[2:4]
        x = rearrange(x, "b c h w -> b (h w) c")

        token_length = cls_tokens.shape[1]
        x = torch.cat((cls_tokens, x), dim=1)
        for blk in self.blocks:
            x = blk(x)

        cls_tokens = x[:, :token_length]
        x = x[:, token_length:]
        x = rearrange(x, "b (h w) c -> b c h w", h=h, w=w)

        return x, cls_tokens


class conv_head_pooling(nn.Module):
    def __init__(
        self,
        in_feature,
        out_feature,
        stride,
        conv_type,
        padding_mode="zeros",
        dilation=1,
    ):
        super(conv_head_pooling, self).__init__()
        if conv_type == "depthwise":
            _groups = in_feature
        else:
            _groups = 1
        print("_groups in conv_head_pooling: ", _groups)
        self.conv = nn.Conv2d(
            in_feature,
            out_feature,
            kernel_size=3,
            padding=dilation,
            dilation=dilation,
            stride=stride,
            padding_mode=padding_mode,
            groups=_groups,
        )
        self.fc = nn.Linear(in_feature, out_feature)

    def forward(self, x, cls_token):

        x = self.conv(x)
        cls_token = self.fc(cls_token)

        return x, cls_token


class conv_embedding(nn.Module):
    def __init__(self, in_channels, out_channels, patch_size, stride, padding):
        super(conv_embedding, self).__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=patch_size,
            stride=stride,
            padding=padding,
            bias=True,
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class PoolingTransformer(Backbone):
    def __init__(
        self,
        image_size,
        patch_size,
        stride,
        base_dims,
        depth,
        heads,
        mlp_ratio,
        conv_type="depthwise",
        num_classes=1000,
        in_chans=3,
        attn_drop_rate=0.0,
        drop_rate=0.0,
        drop_path_rate=0.0,
        dilated=False,
    ):
        super(PoolingTransformer, self).__init__()

        total_block = sum(depth)
        padding = 0
        block_idx = 0
        self.padding = padding
        self.stride = stride

        width = math.floor((image_size + 2 * padding - patch_size) / stride + 1)

        self.conv_type = conv_type
        self.base_dims = base_dims
        self.heads = heads
        self.num_classes = num_classes

        self.patch_size = patch_size
        self.pos_embed = nn.Parameter(
            torch.randn(1, base_dims[0] * heads[0], width, width), requires_grad=True
        )
        self.patch_embed = conv_embedding(
            in_chans, base_dims[0] * heads[0], patch_size, stride, padding
        )

        self.cls_token = nn.Parameter(
            torch.randn(1, 1, base_dims[0] * heads[0]), requires_grad=True
        )
        self.pos_drop = nn.Dropout(p=drop_rate)

        self.transformers = nn.ModuleList([])
        self.pools = nn.ModuleList([])

        for stage in range(len(depth)):
            drop_path_prob = [
                drop_path_rate * i / total_block
                for i in range(block_idx, block_idx + depth[stage])
            ]
            block_idx += depth[stage]

            self.transformers.append(
                Transformer(
                    base_dims[stage],
                    depth[stage],
                    heads[stage],
                    mlp_ratio,
                    drop_rate,
                    attn_drop_rate,
                    drop_path_prob,
                )
            )
            if stage < len(heads) - 1:
                if stage == len(heads) - 2 and dilated:
                    pool_dilation = 2
                    pool_stride = 1
                else:
                    pool_dilation = 1
                    pool_stride = 2
                self.pools.append(
                    conv_head_pooling(
                        base_dims[stage] * heads[stage],
                        base_dims[stage + 1] * heads[stage + 1],
                        stride=pool_stride,
                        dilation=pool_dilation,
                        conv_type=self.conv_type,
                    )
                )

        # self.norm = nn.LayerNorm(base_dims[-1] * heads[-1], eps=1e-6)
        self.embed_dim = base_dims[-1] * heads[-1]

        # Classifier head
        if num_classes > 0:
            self.head = nn.Linear(base_dims[-1] * heads[-1], num_classes)
        else:
            self.head = nn.Identity()

        trunc_normal_(self.pos_embed, std=0.02)
        trunc_normal_(self.cls_token, std=0.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {"pos_embed", "cls_token"}

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=""):
        self.num_classes = num_classes
        if num_classes > 0:
            self.head = nn.Linear(self.embed_dim, num_classes)
        else:
            self.head = nn.Identity()

    def _get_pos_embed(self, H, W):
        H0, W0 = self.pos_embed.shape[-2:]
        if H0 == H and W0 == W:
            return self.pos_embed
        # interp
        pos_embed = F.interpolate(
            self.pos_embed,
            size=(H, W),
            mode="bilinear",
            align_corners=False,
        )
        return pos_embed

    def forward_features(self, x):
        H, W = x.shape[-2:]

        x = self.patch_embed(x)

        # featuremap size after patch embeding
        H = math.floor((H + 2 * self.padding - self.patch_size) / self.stride + 1)
        W = math.floor((W + 2 * self.padding - self.patch_size) / self.stride + 1)

        pos_embed = self._get_pos_embed(H, W)

        x = self.pos_drop(x + pos_embed)
        cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)

        for stage in range(len(self.pools)):
            x, cls_tokens = self.transformers[stage](x, cls_tokens)
            x, cls_tokens = self.pools[stage](x, cls_tokens)
        x, cls_tokens = self.transformers[-1](x, cls_tokens)

        # cls_tokens = self.norm(cls_tokens) # no gradient for layer norm, which cause failure

        return cls_tokens, x

    def forward(self, x):
        cls_token, _ = self.forward_features(x)
        cls_token = self.head(cls_token[:, 0])
        return cls_token


class DistilledPoolingTransformer(PoolingTransformer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cls_token = nn.Parameter(
            torch.randn(1, 2, self.base_dims[0] * self.heads[0]), requires_grad=True
        )
        if self.num_classes > 0:
            self.head_dist = nn.Linear(
                self.base_dims[-1] * self.heads[-1], self.num_classes
            )
        else:
            self.head_dist = nn.Identity()

        trunc_normal_(self.cls_token, std=0.02)
        self.head_dist.apply(self._init_weights)

    def forward(self, x):
        cls_token, x = self.forward_features(x)
        return x
        # x_cls = self.head(cls_token[:, 0])
        # x_dist = self.head_dist(cls_token[:, 1])
        # if self.training:
        #    return x_cls, x_dist
        # else:
        #    return (x_cls + x_dist) / 2


def pit_scalable_distilled(model_config, pretrained=False, print_info=True, **kwargs):
    if "conv_type" in model_config:
        conv_type = model_config["conv_type"]
    else:
        conv_type = "depthwise"
    model = DistilledPoolingTransformer(
        image_size=model_config["I"],
        patch_size=model_config["p"],
        stride=model_config["s"],
        base_dims=model_config["e"],
        depth=model_config["d"],
        heads=model_config["h"],
        mlp_ratio=model_config["r"],
        conv_type=conv_type,
        **kwargs,
    )
    if print_info:
        print("model arch config: {}".format(model_config))
    assert pretrained == False, "pretrained must be False"
    return model


def add_pit_backbone_config(cfg):
    cfg.MODEL.PIT = type(cfg)()
    cfg.MODEL.PIT.MODEL_CONFIG = None
    cfg.MODEL.PIT.WEIGHTS = None
    cfg.MODEL.PIT.DILATED = True


@BACKBONE_REGISTRY.register()
def pit_d2go_model_wrapper(cfg, _):
    assert cfg.MODEL.PIT.MODEL_CONFIG is not None
    dilated = cfg.MODEL.PIT.DILATED
    with PathManager.open(cfg.MODEL.PIT.MODEL_CONFIG) as f:
        model_config = json.load(f)
    model = pit_scalable_distilled(
        model_config,
        num_classes=0,  # set num_classes=0 to avoid building cls head
        drop_rate=0,
        drop_path_rate=0.1,
        dilated=dilated,
    )
    # load weights
    if cfg.MODEL.PIT.WEIGHTS is not None:
        with PathManager.open(cfg.MODEL.PIT.WEIGHTS, "rb") as f:
            state_dict = torch.load(f, map_location="cpu")["model"]
        rm_keys = [k for k in state_dict if "head" in k]
        rm_keys = rm_keys + ["norm.weight", "norm.bias"]
        print(rm_keys)
        for k in rm_keys:
            del state_dict[k]
        model.load_state_dict(state_dict)
        print(f"loaded weights from {cfg.MODEL.PIT.WEIGHTS}")
    return model
