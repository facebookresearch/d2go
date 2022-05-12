# ------------------------------------------------------------------------
# Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------
import copy
import math

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.init import constant_, normal_, xavier_uniform_

from ..modules import MSDeformAttn
from ..util.misc import inverse_sigmoid

# we do not use float("-inf") to avoid potential NaN during training
NEG_INF = -10000.0


class DeformableTransformer(nn.Module):
    def __init__(
        self,
        d_model=256,
        nhead=8,
        num_encoder_layers=6,
        num_decoder_layers=6,
        dim_feedforward=1024,
        dropout=0.1,
        activation="relu",
        return_intermediate_dec=False,
        num_feature_levels=4,
        dec_n_points=4,
        enc_n_points=4,
        two_stage=False,
        two_stage_num_proposals=300,
        decoder_block_grad=True,
    ):
        super().__init__()

        self.d_model = d_model
        self.nhead = nhead
        self.two_stage = two_stage
        self.two_stage_num_proposals = two_stage_num_proposals

        encoder_layer = DeformableTransformerEncoderLayer(
            d_model,
            dim_feedforward,
            dropout,
            activation,
            num_feature_levels,
            nhead,
            enc_n_points,
        )
        self.encoder = DeformableTransformerEncoder(encoder_layer, num_encoder_layers)

        decoder_layer = DeformableTransformerDecoderLayer(
            d_model,
            dim_feedforward,
            dropout,
            activation,
            num_feature_levels,
            nhead,
            dec_n_points,
        )
        self.decoder = DeformableTransformerDecoder(
            decoder_layer,
            num_decoder_layers,
            return_intermediate_dec,
            decoder_block_grad,
        )

        self.level_embed = nn.Parameter(torch.Tensor(num_feature_levels, d_model))

        if two_stage:
            self.enc_output = nn.Linear(d_model, d_model)
            self.enc_output_norm = nn.LayerNorm(d_model)
            self.pos_trans = nn.Linear(d_model * 2, d_model * 2)
            self.pos_trans_norm = nn.LayerNorm(d_model * 2)
        else:
            self.reference_points = nn.Linear(d_model, 2)

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                xavier_uniform_(p)
        for m in self.modules():
            if isinstance(m, MSDeformAttn):
                m._reset_parameters()
        if not self.two_stage:
            xavier_uniform_(self.reference_points.weight, gain=1.0)
            constant_(self.reference_points.bias, 0.0)
        normal_(self.level_embed)

    def get_proposal_pos_embed(self, proposals):
        """
        Args
            proposals: shape (bs, top_k, 4). Last dimension of size 4 denotes (cx, cy, w, h)
        """
        num_pos_feats = 128
        temperature = 10000
        scale = 2 * math.pi
        # shape (num_pos_feats)
        dim_t = torch.arange(
            num_pos_feats, dtype=torch.float32, device=proposals.device
        )
        dim_t = temperature ** (2 * (dim_t // 2) / num_pos_feats)
        # N, L, 4
        proposals = proposals.sigmoid() * scale
        # pos shape: (bs, top_k, 4, num_pos_feats)
        pos = proposals[:, :, :, None] / dim_t
        # pos shape: (bs, top_k, 4, num_pos_feats/2, 2) -> (bs, top_k, 4 * num_pos_feats)
        pos = torch.stack(
            (pos[:, :, :, 0::2].sin(), pos[:, :, :, 1::2].cos()), dim=4
        ).flatten(2)
        # pos shape: (bs, top_k, 4 * num_pos_feats) = (bs, top_k, 512)
        return pos

    def gen_encoder_output_proposals(self, memory, memory_padding_mask, spatial_shapes):
        """
        Args:
            memory: shape (bs, K, C) where K = \sum_l H_l * w_l
            memory_padding_mask: shape (bs, K)
            spatial_shapes: shape (num_levels, 2)
        """
        N_, S_, C_ = memory.shape
        proposals = []
        _cur = 0
        base_object_scale = 0.05
        for lvl, (H_, W_) in enumerate(spatial_shapes):
            # shape (bs, H_l * W_l)
            mask_flatten_ = memory_padding_mask[:, _cur : (_cur + H_ * W_)].view(
                N_, H_, W_, 1
            )
            # shape (bs, )
            valid_H = torch.sum(~mask_flatten_[:, :, 0, 0], 1)
            # shape (bs, )
            valid_W = torch.sum(~mask_flatten_[:, 0, :, 0], 1)
            # grid_y, grid_x shape (H_l, W_l)
            grid_y, grid_x = torch.meshgrid(
                torch.linspace(
                    0, H_ - 1, H_, dtype=torch.float32, device=memory.device
                ),
                torch.linspace(
                    0, W_ - 1, W_, dtype=torch.float32, device=memory.device
                ),
            )
            # grid shape (H_l, W_l, 2)
            grid = torch.cat([grid_x.unsqueeze(-1), grid_y.unsqueeze(-1)], -1)
            # scale shape (bs, 1, 1, 2)
            scale = torch.cat([valid_W.unsqueeze(-1), valid_H.unsqueeze(-1)], 1).view(
                N_, 1, 1, 2
            )
            # grid shape (bs, H_l, W_l, 2). Value could be > 1
            grid = (grid.unsqueeze(0).expand(N_, -1, -1, -1) + 0.5) / scale
            # wh shape (bs, H_l, W_l, 2)
            wh = torch.ones_like(grid) * base_object_scale * (2.0**lvl)
            # proposal shape (bs, H_l * W_l, 4)
            proposal = torch.cat((grid, wh), -1).view(N_, -1, 4)
            proposals.append(proposal)
            _cur += H_ * W_
        # shape (bs, K, 4) where K = \sum_l H_l * W_l
        output_proposals = torch.cat(proposals, 1)
        # shape (bs, K, 1)
        output_proposals_valid = (
            (output_proposals > 0.01) & (output_proposals < 0.99)
        ).all(-1, keepdim=True)

        output_proposals = inverse_sigmoid(output_proposals)
        # memory: shape (bs, K, C)
        output_memory = memory
        # memory_padding_mask: shape (bs, K)
        output_memory = output_memory.masked_fill(
            memory_padding_mask.unsqueeze(-1), float(0)
        )
        output_memory = output_memory.masked_fill(~output_proposals_valid, float(0))
        output_memory = self.enc_output_norm(self.enc_output(output_memory))
        return output_memory, output_proposals, output_proposals_valid

    def get_valid_ratio(self, mask):
        _, H, W = mask.shape
        # shape (bs,)
        valid_H = torch.sum(~mask[:, :, 0], 1)
        valid_W = torch.sum(~mask[:, 0, :], 1)
        valid_ratio_h = valid_H.float() / H
        valid_ratio_w = valid_W.float() / W
        valid_ratio = torch.stack([valid_ratio_w, valid_ratio_h], -1)
        # shape (bs, 2)
        return valid_ratio

    def forward(self, srcs, masks, pos_embeds, query_embed=None):
        """
        Args:
            srcs: a list of num_levels tensors. Each has shape (N, C, H_l, W_l)
            masks: a list of num_levels tensors. Each has shape (N, H_l, W_l)
            pos_embeds: a list of num_levels tensors. Each has shape (N, C, H_l, W_l)
            query_embed: a tensor has shape (num_queries, C)
        """
        assert self.two_stage or query_embed is not None

        # prepare input for encoder
        src_flatten = []
        mask_flatten = []
        lvl_pos_embed_flatten = []
        spatial_shapes = []
        for lvl, (src, mask, pos_embed) in enumerate(zip(srcs, masks, pos_embeds)):
            bs, c, h, w = src.shape
            spatial_shape = (h, w)
            spatial_shapes.append(spatial_shape)
            # src shape (bs, h_l*w_l, c)
            src = src.flatten(2).transpose(1, 2)
            # mask shape (bs, h_l*w_l)
            mask = mask.flatten(1)
            # pos_embed shape (bs, h_l*w_l, c)
            pos_embed = pos_embed.flatten(2).transpose(1, 2)
            # lvl_pos_embed shape (bs, h_l*w_l, c)
            lvl_pos_embed = pos_embed + self.level_embed[lvl].view(1, 1, -1)
            lvl_pos_embed_flatten.append(lvl_pos_embed)
            src_flatten.append(src)
            mask_flatten.append(mask)
        # src_flatten shape: (bs, K, c) where K = \sum_l H_l * w_l
        src_flatten = torch.cat(src_flatten, 1)
        # mask_flatten shape: (bs, K)
        mask_flatten = torch.cat(mask_flatten, 1)
        # lvl_pos_embed_flatten shape: (bs, K, c)
        lvl_pos_embed_flatten = torch.cat(lvl_pos_embed_flatten, 1)
        # spatial_shapes shape: (num_levels, 2)
        spatial_shapes = torch.as_tensor(
            spatial_shapes, dtype=torch.long, device=src_flatten.device
        )
        # level_start_index shape: (num_levels)
        level_start_index = torch.cat(
            (spatial_shapes.new_zeros((1,)), spatial_shapes.prod(1).cumsum(0)[:-1])
        )
        # valid_ratios shape: (bs, num_levels, 2)
        valid_ratios = torch.stack([self.get_valid_ratio(m) for m in masks], 1)
        # encoder
        # memory shape (bs, K, C) where K = \sum_l H_l * w_l
        memory = self.encoder(
            src_flatten,
            spatial_shapes,
            level_start_index,
            valid_ratios,
            lvl_pos_embed_flatten,
            mask_flatten,
        )

        # prepare input for decoder
        bs, _, c = memory.shape
        if self.two_stage:
            # output_memory shape (bs, K, C)
            # output_proposals shape (bs, K, 4)
            # output_proposals_valid shape (bs, K, 1)
            (
                output_memory,
                output_proposals,
                output_proposals_valid,
            ) = self.gen_encoder_output_proposals(memory, mask_flatten, spatial_shapes)
            # hack implementation for two-stage Deformable DETR
            # shape (bs, K, 1)
            enc_outputs_class = self.encoder.class_embed(output_memory)
            # fill in -inf foreground logit at invalid positions so that we will never pick
            # top-scored proposals at those positions
            enc_outputs_class.masked_fill(mask_flatten.unsqueeze(-1), NEG_INF)
            enc_outputs_class.masked_fill(~output_proposals_valid, NEG_INF)
            # shape (bs, K, 4)
            enc_outputs_coord_unact = (
                self.encoder.bbox_embed(output_memory) + output_proposals
            )
            topk = self.two_stage_num_proposals
            # topk_proposals: indices of top items. Shape (bs, top_k)
            topk_proposals = torch.topk(enc_outputs_class[..., 0], topk, dim=1)[1]

            # topk_coords_unact shape (bs, top_k, 4)
            topk_coords_unact = torch.gather(
                enc_outputs_coord_unact, 1, topk_proposals.unsqueeze(-1).repeat(1, 1, 4)
            )
            topk_coords_unact = topk_coords_unact.detach()

            init_reference_out = topk_coords_unact
            # shape (bs, top_k, C=512)
            pos_trans_out = self.pos_trans_norm(
                self.pos_trans(self.get_proposal_pos_embed(topk_coords_unact))
            )
            # query_embed shape (bs, top_k, c)
            # tgt shape (bs, top_k, c)
            query_embed, tgt = torch.split(pos_trans_out, c, dim=2)
        else:
            # query_embed (or tgt) shape: (num_queries, c)
            query_embed, tgt = torch.split(query_embed, c, dim=1)
            # query_embed shape: (batch_size, num_queries, c)
            query_embed = query_embed.unsqueeze(0).expand(bs, -1, -1)
            # tgt shape: (batch_size, num_queries, c)
            tgt = tgt.unsqueeze(0).expand(bs, -1, -1)
            # init_reference_out shape: (batch_size, num_queries, 2)
            init_reference_out = self.reference_points(query_embed)

        # decoder
        # hs shape: (num_layers, batch_size, num_queries, c)
        # inter_references shape: (num_layers, batch_size, num_queries, num_levels, 2)
        hs, inter_references = self.decoder(
            tgt,
            init_reference_out,
            memory,
            spatial_shapes,
            level_start_index,
            valid_ratios,
            query_embed,
            mask_flatten,
        )

        inter_references_out = inter_references
        if self.two_stage:
            return (
                hs,
                init_reference_out,
                inter_references_out,
                enc_outputs_class,
                enc_outputs_coord_unact,
            )
        # hs shape: (num_layers, batch_size, num_queries, c)
        # init_reference_out shape: (batch_size, num_queries, 2)
        # inter_references_out shape: (num_layers, bs, num_queries, num_levels, 2)
        return hs, init_reference_out, inter_references_out, None, None


class DeformableTransformerEncoderLayer(nn.Module):
    def __init__(
        self,
        d_model=256,
        d_ffn=1024,
        dropout=0.1,
        activation="relu",
        n_levels=4,
        n_heads=8,
        n_points=4,
    ):
        super().__init__()

        # self attention
        self.self_attn = MSDeformAttn(d_model, n_levels, n_heads, n_points)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        # ffn
        self.linear1 = nn.Linear(d_model, d_ffn)
        self.activation = _get_activation_fn(activation)
        self.dropout2 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.dropout3 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_ffn(self, src):
        src2 = self.linear2(self.dropout2(self.activation(self.linear1(src))))
        src = src + self.dropout3(src2)
        src = self.norm2(src)
        return src

    def forward(
        self,
        src,
        pos,
        reference_points,
        spatial_shapes,
        level_start_index,
        padding_mask=None,
    ):
        """
        Args:
            src: tensor, shape (bs, K, c) where K = \sum_l H_l * w_l
            pos: tensor, shape (bs, K, c)
            reference_points: tensor, shape (bs, K, num_levels, 2)
            spatial_shapes: tensor, shape (num_levels, 2)
            level_start_index: tensor, shape (num_levels,)
            padding_mask: tensor, shape: (bs, K)
        """
        # self attention
        # shape: (bs, \sum_l H_l * w_l, c)
        src2 = self.self_attn(
            self.with_pos_embed(src, pos),
            reference_points,
            src,
            spatial_shapes,
            level_start_index,
            padding_mask,
        )
        src = src + self.dropout1(src2)
        src = self.norm1(src)

        # ffn
        src = self.forward_ffn(src)

        return src


class DeformableTransformerEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers

    @staticmethod
    def get_reference_points(spatial_shapes, valid_ratios, device):
        reference_points_list = []
        for lvl, (H_, W_) in enumerate(spatial_shapes):
            # ref_y shape: (H_l, W_l)
            # ref_x shape: (H_l, W_l)
            ref_y, ref_x = torch.meshgrid(
                torch.linspace(0.5, H_ - 0.5, H_, dtype=torch.float32, device=device),
                torch.linspace(0.5, W_ - 0.5, W_, dtype=torch.float32, device=device),
            )
            # ref_y
            #   shape (None, H_l*W_l) / (N, None) = (N, H_l*W_l)
            #   value could be >1
            ref_y = ref_y.reshape(-1)[None] / (valid_ratios[:, None, lvl, 1] * H_)
            ref_x = ref_x.reshape(-1)[None] / (valid_ratios[:, None, lvl, 0] * W_)
            # ref shape (N, H_l*W_l, 2)
            ref = torch.stack((ref_x, ref_y), -1)
            reference_points_list.append(ref)
        # shape (N, K, 2) where K = \sum_l (H_l * W_l)
        reference_points = torch.cat(reference_points_list, 1)
        # reference_points
        #   shape (N, K, 1, 2) * (N, 1, num_levels, 2) = (N, K, num_levels, 2)
        #   ideally, value should be <1. In practice, value coule be >= 1. Thus, clamp max to 1
        reference_points = reference_points[:, :, None] * valid_ratios[:, None]
        reference_points = reference_points.clamp(max=1.0)
        return reference_points

    def forward(
        self,
        src,
        spatial_shapes,
        level_start_index,
        valid_ratios,
        pos=None,
        padding_mask=None,
    ):
        output = src
        reference_points = self.get_reference_points(
            spatial_shapes, valid_ratios, device=src.device
        )
        for _, layer in enumerate(self.layers):
            output = layer(
                output,
                pos,
                reference_points,
                spatial_shapes,
                level_start_index,
                padding_mask,
            )
        # shape (bs, K, c) where K = \sum_l H_l * w_l
        return output


class DeformableTransformerDecoderLayer(nn.Module):
    def __init__(
        self,
        d_model=256,
        d_ffn=1024,
        dropout=0.1,
        activation="relu",
        n_levels=4,
        n_heads=8,
        n_points=4,
    ):
        super().__init__()

        # cross attention
        self.cross_attn = MSDeformAttn(d_model, n_levels, n_heads, n_points)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        # self attention
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

        # ffn
        self.linear1 = nn.Linear(d_model, d_ffn)
        self.activation = _get_activation_fn(activation)
        self.dropout3 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.dropout4 = nn.Dropout(dropout)
        self.norm3 = nn.LayerNorm(d_model)

    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_ffn(self, tgt):
        tgt2 = self.linear2(self.dropout3(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout4(tgt2)
        tgt = self.norm3(tgt)
        return tgt

    def forward(
        self,
        tgt,
        query_pos,
        reference_points,
        src,
        src_spatial_shapes,
        level_start_index,
        src_padding_mask=None,
    ):
        """
        Args:
            tgt: tensor, shape (batch_size, num_queries, c)
            query_pos: tensor, shape: (batch_size, num_queries, c)
            reference_points: tensor, shape: (batch_size, num_queries, num_levels, 2/4). values \in (0, 1)
            src: tensor, shape (batch_size, K, c) where K = \sum_l H_l * w_l
            src_spatial_shapes: tensor, shape (num_levels, 2)
            level_start_index: tensor, shape (num_levels,)
            src_padding_mask: tensor, (batch_size, K)
        """
        # self attention
        q = k = self.with_pos_embed(tgt, query_pos)
        tgt2 = self.self_attn(
            q.transpose(0, 1), k.transpose(0, 1), tgt.transpose(0, 1)
        )[0].transpose(0, 1)
        # tgt shape: (batch_size, num_queries, c)
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)

        # cross attention
        tgt2 = self.cross_attn(
            self.with_pos_embed(tgt, query_pos),
            reference_points,
            src,
            src_spatial_shapes,
            level_start_index,
            src_padding_mask,
        )
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)

        # ffn
        tgt = self.forward_ffn(tgt)
        # tgt shape: (batch_size, num_queries, c)
        return tgt


class DeformableTransformerDecoder(nn.Module):
    def __init__(
        self, decoder_layer, num_layers, return_intermediate=False, block_grad=True
    ):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.return_intermediate = return_intermediate
        # hack implementation for iterative bounding box refinement and two-stage Deformable DETR
        self.bbox_embed = None
        self.block_grad = block_grad

    def forward(
        self,
        tgt,
        reference_points_unact,
        src,
        src_spatial_shapes,
        src_level_start_index,
        src_valid_ratios,
        query_pos=None,
        src_padding_mask=None,
    ):
        """
        Args:
            tgt: tensor, shape (batch_size, num_queries, c)
            reference_points_unact: tensor, shape (batch_size, num_queries, 2 or 4).
                values \in (0, 1)
            src: tensor, shape (batch_size, K, c) where K = \sum_l H_l * w_l
            src_spatial_shapes: tensor, shape (num_levels, 2)
            src_level_start_index: tensor, shape (num_levels,)
            src_valid_ratios: tensor, shape (batch_size, num_levels, 2)
            query_pos: tensor, shape: (batch_size, num_queries, c)
            src_padding_mask: tensor, (bs, K)
        """
        output = tgt

        intermediate = []
        intermediate_reference_points = []
        for lid, layer in enumerate(self.layers):
            reference_points = reference_points_unact.sigmoid()

            if reference_points.shape[-1] == 4:
                # shape: (bs, num_queries, 1, 4) * (bs, 1, num_levels, 4) = (bs, num_queries, num_levels, 4)
                reference_points_input = (
                    reference_points[:, :, None]
                    * torch.cat([src_valid_ratios, src_valid_ratios], -1)[:, None]
                )
            else:
                assert reference_points.shape[-1] == 2
                # shape (bs, num_queries, 1, 2) * (bs, 1, num_levels, 2) = (bs, num_queries, num_levels, 2)
                reference_points_input = (
                    reference_points[:, :, None] * src_valid_ratios[:, None]
                )
            # shape: (bs, num_queries, c)
            output = layer(
                output,
                query_pos,
                reference_points_input,
                src,
                src_spatial_shapes,
                src_level_start_index,
                src_padding_mask,
            )
            # hack implementation for iterative bounding box refinement
            if self.bbox_embed is not None:
                tmp = self.bbox_embed[lid](output)
                if reference_points.shape[-1] == 4:
                    new_reference_points_unact = tmp + reference_points_unact
                else:
                    assert reference_points.shape[-1] == 2
                    new_reference_points_unact = tmp
                    new_reference_points_unact[..., :2] = (
                        tmp[..., :2] + reference_points_unact
                    )
                # block gradient backpropagation here to stabilize optimization
                if self.block_grad:
                    new_reference_points_unact = new_reference_points_unact.detach()

                reference_points_unact = new_reference_points_unact
            else:
                new_reference_points_unact = reference_points_unact

            if self.return_intermediate:
                intermediate.append(output)
                intermediate_reference_points.append(new_reference_points_unact)

        if self.return_intermediate:
            # shape 1: (num_layers, batch_size, num_queries, c)
            # shape 2: (num_layers, bs, num_queries, num_levels, 2)
            return torch.stack(intermediate), torch.stack(intermediate_reference_points)
        # output shape: (batch_size, num_queries, c)
        # new_reference_points_unact shape: (bs, num_queries, num_levels, 2)
        return output, new_reference_points_unact


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(f"activation should be relu/gelu, not {activation}.")


def build_deforamble_transformer(args):
    return DeformableTransformer(
        d_model=args.hidden_dim,
        nhead=args.nheads,
        num_encoder_layers=args.enc_layers,
        num_decoder_layers=args.dec_layers,
        dim_feedforward=args.dim_feedforward,
        dropout=args.dropout,
        activation="relu",
        return_intermediate_dec=True,
        num_feature_levels=args.num_feature_levels,
        dec_n_points=args.dec_n_points,
        enc_n_points=args.enc_n_points,
        two_stage=args.two_stage,
        two_stage_num_proposals=args.num_queries,
    )
