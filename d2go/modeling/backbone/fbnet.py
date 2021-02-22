#!/usr/bin/env python3

import copy
import functools
import itertools
import json
import logging
from collections import OrderedDict
from typing import List

import torch.nn as nn
from detectron2.layers import ShapeSpec
from detectron2.modeling import (
    BACKBONE_REGISTRY,
    RPN_HEAD_REGISTRY,
    Backbone,
    build_anchor_generator,
)
from detectron2.modeling.backbone.fpn import FPN, LastLevelMaxPool
from detectron2.modeling.roi_heads import box_head, keypoint_head, mask_head
from mobile_cv.arch.fbnet import fbnet_builder as mbuilder, fbnet_modeldef as modeldef

from .modules import (
    KeypointRCNNPredictor,
    KeypointRCNNPredictorNoUpscale,
    MaskRCNNConv1x1Predictor,
    RPNHeadConvRegressor,
)


logger = logging.getLogger(__name__)


@functools.lru_cache()
def create_fbnet_builder(fbnet_cfg):
    bn_type = fbnet_cfg.BN_TYPE
    if bn_type == "gn":
        bn_type = (bn_type, fbnet_cfg.NUM_GROUPS)
    factor = fbnet_cfg.SCALE_FACTOR

    arch = fbnet_cfg.ARCH
    arch_def = fbnet_cfg.ARCH_DEF
    if len(arch_def) > 0:
        arch_def = json.loads(arch_def)
    if arch in modeldef.MODEL_ARCH:
        if len(arch_def) > 0:
            assert (
                arch_def == modeldef.MODEL_ARCH[arch]
            ), "Two architectures with the same name {},\n{},\n{}".format(
                arch, arch_def, modeldef.MODEL_ARCH[arch]
            )
        arch_def = modeldef.MODEL_ARCH[arch]
    else:
        assert arch_def is not None and len(arch_def) > 0
    arch_def = mbuilder.unify_arch_def(arch_def)

    if "rpn_stride" in arch_def:
        logger.warning(
            'Found "rpn_stride" in arch_def, this is deprecated,'
            " please set it using cfg.MODEL.RPN.ANCHOR_STRIDE"
        )
    width_divisor = fbnet_cfg.WIDTH_DIVISOR
    dw_skip_bn = fbnet_cfg.DW_CONV_SKIP_BN
    dw_skip_relu = fbnet_cfg.DW_CONV_SKIP_RELU

    def _arch_def_to_str_list(arch_def):
        """ Return a list of strings that represents arch_def in readable format """
        arch_def = copy.deepcopy(arch_def)
        lines = []
        for k, v in arch_def.items():
            lines.append(k)
            if k == "stages":
                for stage in v:
                    lines.append("- {}".format(stage))
            else:
                lines.append("  {}".format(v))
        return lines

    logger.info(
        'Creating FBNetBuilder with arch "{}" (without scaling):\n{}'.format(
            arch, "\n".join(_arch_def_to_str_list(arch_def))
        )
    )

    builder = mbuilder.FBNetBuilder(
        width_ratio=factor,
        bn_type=bn_type,
        width_divisor=width_divisor,
        dw_skip_bn=dw_skip_bn,
        dw_skip_relu=dw_skip_relu,
    )

    return builder, arch_def


def _get_trunk_cfg(arch_def):
    """ Get all stages except the last one """
    num_stages = mbuilder.get_num_stages(arch_def)
    trunk_stages = arch_def.get("backbone", range(num_stages - 1))
    ret = mbuilder.get_blocks(arch_def, stage_indices=trunk_stages)
    return [ret, len(trunk_stages)]


def _get_stage_strides(stride_per_stage_block, flattened_stages):
    first_stride = stride_per_stage_block[0]
    stride_per_stage_block = stride_per_stage_block[1:]
    assert len(stride_per_stage_block) == len(flattened_stages)
    stage_idx_set = {s["stage_idx"] for s in flattened_stages}
    # assume stage idx are 0, 1, 2, ...
    assert max(stage_idx_set) + 1 == len(stage_idx_set)
    ids_per_stage = [
        [i for i, s in enumerate(flattened_stages) if s["stage_idx"] == stage_idx]
        for stage_idx in range(len(stage_idx_set))
    ]  # eg. [[0], [1, 2], [3, 4, 5, 6], ...]
    block_stride_per_stage = [
        [stride_per_stage_block[i] for i in ids] for ids in ids_per_stage
    ]  # eg. [[1], [2, 1], [2, 1, 1, 1], ...]
    stride_per_stage = [
        list(itertools.accumulate(s, lambda x, y: x * y))[-1]
        for s in block_stride_per_stage
    ]  # eg. [1, 2, 2, ...]
    stride_till_stage = [
        first_stride * stride
        for stride in itertools.accumulate(stride_per_stage, lambda x, y: x * y)
    ]  # eg. [first*1, first*2, first*4, ...]
    return stride_till_stage


def _update_state_dict_with_key_mapping(state_dict, map_key_func):
    """
        Given state_dict, map all its keys using map_key_func,
        this is in-place and keeps the order.
    """
    new_state_dict = OrderedDict()
    for old_key in state_dict:
        new_key = map_key_func(old_key)
        new_state_dict[new_key] = state_dict[old_key]
    # update state_dict in-place
    state_dict.clear()
    for key in new_state_dict:
        state_dict[key] = new_state_dict[key]


class FBNetTrunk(Backbone):
    """
    Backbone (bottom-up) for FBNet.

    Versions:
        Version 1:
            first.{conv/bn}
            stages.xifX_Y.{pw/dw/pwl}.{conv/bn}
        Version 2 (D17001516):
            first.{conv/bn}
            stageX.xifX_Y.{pw/dw/pwl}.{conv/bn}
    """

    _version = 2

    def __init__(self, builder, arch_def, dim_in, quant_input=False):
        super(FBNetTrunk, self).__init__()
        trunk_cfg, num_stages = _get_trunk_cfg(arch_def)
        stride_per_block = mbuilder.count_stride_each_block(trunk_cfg)
        stride_till_stage = _get_stage_strides(stride_per_block, trunk_cfg["stages"])
        assert len(stride_till_stage) == num_stages
        assert stride_till_stage[-1] == mbuilder.count_strides(trunk_cfg)

        if not quant_input:
            self.first = builder.add_first(arch_def["first"], dim_in=dim_in)
        else:
            # HACK: we should let QAT supprts adding the first PACT, then
            # remove this branch
            first = builder.add_first(arch_def["first"], dim_in=dim_in)
            from silicon.quantization.quantPack import PACT
            self.first = nn.Sequential(PACT(clipping_type=2), first)

        self._out_feature_strides = {"first": stride_per_block[0]}
        self._out_feature_channels = {"first": builder.last_depth}

        self._stages = OrderedDict()
        for i in range(num_stages):
            name = "stage{}".format(i)
            stage = builder.add_blocks(trunk_cfg["stages"], [i])
            self.add_module(name, stage)

            self._stages[name] = stage
            self._out_feature_channels[name] = builder.last_depth
            self._out_feature_strides[name] = stride_till_stage[i]

        # returned features are the final output of each stage
        self._out_features = list(self._stages.keys())

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        version = local_metadata.get("version", None)

        if version is None or version < 2:
            logger.info(
                "FBNetTrunk {} is upgraded to version 2".format(prefix.rstrip("."))
            )

            # [prefix]stages.xifX_Y.[remaining] -> [prefix]stageX.xifX_Y.[remaining]
            def _map_key_func(key):
                stages_prefix = prefix + "stages."
                if key.startswith(stages_prefix):
                    after_stage_str = key[len(stages_prefix) :]
                    xif_block = after_stage_str.split(".")[0]
                    assert xif_block.startswith("xif")
                    stage, block = xif_block[len("xif") :].split("_")
                    assert stage.isdigit()
                    assert block.isdigit()
                    new_key = prefix + "stage" + stage + "." + after_stage_str
                    return new_key
                else:
                    return key

            _update_state_dict_with_key_mapping(state_dict, _map_key_func)

        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    # return features for each stage
    def forward(self, x):
        x = self.first(x)
        features = {}
        for name, stage in self._stages.items():
            x = stage(x)
            features[name] = x
        return features


class FBNetFPN(FPN):
    """
    FPN module for FBNet.

    Versions:
        Version 1:
            bottom_up.body.{first/stages.X}.REMAINING
            ...
        Version 2 (D17324937):
            bottom_up.{first/stageX}.REMAINING
            ...
    """

    _version = 2

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        version = local_metadata.get("version", None)

        if version is None or version < 2:
            logger.info(
                "FBNetFPN {} is upgraded to version 2".format(prefix.rstrip("."))
            )

            # [prefix]body.{stages.X}.[remaining] -> [prefix]{stages}.[remaining]
            # NOTE: the mapped key name will be Version 1 for FBNetTrunk, and
            # we let FBNetTrunk to the further mapping.
            def _map_key_func(key):
                bottom_up_prefix = prefix + "bottom_up."
                if key.startswith(bottom_up_prefix):
                    after_bottom_up_str = key[len(bottom_up_prefix) :]
                    if after_bottom_up_str.startswith("body."):
                        after_bottom_up_str = after_bottom_up_str[len("body.") :]
                        segments = after_bottom_up_str.split(".")
                        if segments and segments[0] == "stages":
                            # remove .X from stages.X.xifX_Y
                            del segments[1]
                            after_bottom_up_str = ".".join(segments)
                        new_key = bottom_up_prefix + after_bottom_up_str
                        return new_key
                return key

            _update_state_dict_with_key_mapping(state_dict, _map_key_func)

        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )


def build_fbnet_backbone(cfg):
    builder, arch_def = create_fbnet_builder(cfg.MODEL.FBNET)
    dim_in = cfg.MODEL.FBNET.STEM_IN_CHANNELS
    return FBNetTrunk(builder, arch_def, dim_in, cfg.QUANTIZATION.SILICON_QAT.ENABLED)


@BACKBONE_REGISTRY.register()
class FBNetBackbone(Backbone):
    def __init__(self, cfg, _):
        super(FBNetBackbone, self).__init__()
        self.body = build_fbnet_backbone(cfg)
        self._out_features = self.body._out_features
        self._out_feature_strides = self.body._out_feature_strides
        self._out_feature_channels = self.body._out_feature_channels

    def forward(self, x):
        return self.body(x)


@BACKBONE_REGISTRY.register()
def FBNetFpnBackbone(cfg, _):
    # TODO: write more efficient FPN module
    backbone = FBNetFPN(
        bottom_up=build_fbnet_backbone(cfg),
        in_features=cfg.MODEL.FPN.IN_FEATURES,
        out_channels=cfg.MODEL.FPN.OUT_CHANNELS,
        norm=cfg.MODEL.FPN.NORM,
        top_block=LastLevelMaxPool(),
    )

    return backbone


def _get_rpn_stage(arch_def, num_blocks):
    rpn_stage = arch_def.get("rpn")
    ret = mbuilder.get_blocks(arch_def, stage_indices=rpn_stage)
    if num_blocks > 0:
        logger.warning("Use last {} blocks in {} as rpn".format(num_blocks, ret))
        block_count = len(ret["stages"])
        assert num_blocks <= block_count, "use block {}, block count {}".format(
            num_blocks, block_count
        )
        blocks = range(block_count - num_blocks, block_count)
        ret = mbuilder.get_blocks(ret, block_indices=blocks)
    return ret["stages"]


class FBNetRpnFeatureHead(nn.Module):
    def __init__(self, cfg, in_channels, builder, arch_def):
        super(FBNetRpnFeatureHead, self).__init__()
        assert in_channels == builder.last_depth

        rpn_bn_type = cfg.MODEL.FBNET.RPN_BN_TYPE
        if len(rpn_bn_type) > 0:
            builder.bn_type = rpn_bn_type

        use_blocks = cfg.MODEL.FBNET.RPN_HEAD_BLOCKS
        stages = _get_rpn_stage(arch_def, use_blocks)

        self.head = builder.add_blocks(stages)
        self.out_channels = builder.last_depth

    def forward(self, x):
        x = [self.head(y) for y in x]
        return x


@RPN_HEAD_REGISTRY.register()
class FBNetRpnHead(nn.Module):
    def __init__(self, cfg, input_shape: List[ShapeSpec]):
        super(FBNetRpnHead, self).__init__()

        in_channels = [x.channels for x in input_shape]
        assert len(set(in_channels)) == 1
        in_channels = in_channels[0]
        anchor_generator = build_anchor_generator(cfg, input_shape)
        num_cell_anchors = anchor_generator.num_cell_anchors
        box_dim = anchor_generator.box_dim
        assert len(set(num_cell_anchors)) == 1
        num_cell_anchors = num_cell_anchors[0]

        builder, model_arch = create_fbnet_builder(cfg.MODEL.FBNET)
        builder.last_depth = in_channels

        assert in_channels == builder.last_depth
        # builder.name_prefix = "[rpn]"

        self.rpn_feature = FBNetRpnFeatureHead(cfg, in_channels, builder, model_arch)
        self.rpn_regressor = RPNHeadConvRegressor(
            in_channels=self.rpn_feature.out_channels,
            num_anchors=num_cell_anchors,
            box_dim=box_dim,
        )

    def forward(self, x):
        x = self.rpn_feature(x)
        return self.rpn_regressor(x)


def _get_head_stage(arch, head_name, blocks):
    head_stage = arch[head_name]
    ret = mbuilder.get_blocks(arch, stage_indices=head_stage, block_indices=blocks)
    return ret["stages"]


class FBNetGenericRoIHead(nn.Module):
    def __init__(
        self,
        cfg,
        in_channels,
        builder,
        arch_def,
        head_name,
        use_blocks,
        stride_init,
        last_layer_scale,
    ):
        super(FBNetGenericRoIHead, self).__init__()
        assert in_channels == builder.last_depth
        assert isinstance(use_blocks, list)

        stage = _get_head_stage(arch_def, head_name, use_blocks)

        assert stride_init in [0, 1, 2]
        if stride_init != 0:
            stage[0]["block"][3] = stride_init
        blocks = builder.add_blocks(stage)

        last_info = copy.deepcopy(arch_def["last"])
        # last_info[1] = last_layer_scale
        # adapt to the new builder in mobile_cv/arch
        last_info[1][1] = last_layer_scale
        last = builder.add_last(last_info)

        self.head = nn.Sequential(OrderedDict([("blocks", blocks), ("last", last)]))

        self.out_channels = builder.last_depth

    def forward(self, x):
        x = self.head(x)
        return x


@box_head.ROI_BOX_HEAD_REGISTRY.register()
class FBNetRoIBoxHead(nn.Module):
    def __init__(self, cfg, input_shape: ShapeSpec):
        super(FBNetRoIBoxHead, self).__init__()

        in_channels = input_shape.channels
        builder, model_arch = create_fbnet_builder(cfg.MODEL.FBNET)
        builder.last_depth = in_channels
        # builder.name_prefix = "_[bbox]_"

        self.roi_box_conv = FBNetGenericRoIHead(
            cfg,
            in_channels,
            builder,
            model_arch,
            head_name="bbox",
            use_blocks=cfg.MODEL.FBNET.DET_HEAD_BLOCKS,
            stride_init=cfg.MODEL.FBNET.DET_HEAD_STRIDE,
            last_layer_scale=cfg.MODEL.FBNET.DET_HEAD_LAST_SCALE,
        )

        self.avgpool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        x = self.roi_box_conv(x)
        x = self.avgpool(x)
        return x

    @property
    def output_shape(self):
        return ShapeSpec(channels=self.roi_box_conv.out_channels)


@keypoint_head.ROI_KEYPOINT_HEAD_REGISTRY.register()
class FBNetRoIKeypointHead(keypoint_head.BaseKeypointRCNNHead):
    def __init__(self, cfg, input_shape: ShapeSpec):
        super().__init__(cfg, input_shape)

        in_channels = input_shape.channels
        builder, model_arch = create_fbnet_builder(cfg.MODEL.FBNET)
        builder.last_depth = in_channels
        # builder.name_prefix = "_[kpts]_"

        self.feature_extractor = FBNetGenericRoIHead(
            cfg,
            in_channels,
            builder,
            model_arch,
            head_name="kpts",
            use_blocks=cfg.MODEL.FBNET.KPTS_HEAD_BLOCKS,
            stride_init=cfg.MODEL.FBNET.KPTS_HEAD_STRIDE,
            last_layer_scale=cfg.MODEL.FBNET.KPTS_HEAD_LAST_SCALE,
        )

        self.predictor = KeypointRCNNPredictor(
            in_channels=self.feature_extractor.out_channels,
            num_keypoints=cfg.MODEL.ROI_KEYPOINT_HEAD.NUM_KEYPOINTS,
        )

    def layers(self, x):
        x = self.feature_extractor(x)
        x = self.predictor(x)
        return x


@keypoint_head.ROI_KEYPOINT_HEAD_REGISTRY.register()
class FBNetRoIKeypointHeadKRCNNPredictorNoUpscale(keypoint_head.BaseKeypointRCNNHead):
    def __init__(self, cfg, input_shape: ShapeSpec):
        super().__init__(cfg, input_shape)

        in_channels = input_shape.channels
        builder, model_arch = create_fbnet_builder(cfg.MODEL.FBNET)
        builder.last_depth = in_channels
        # builder.name_prefix = "_[kpts]_"

        self.feature_extractor = FBNetGenericRoIHead(
            cfg,
            in_channels,
            builder,
            model_arch,
            head_name="kpts",
            use_blocks=cfg.MODEL.FBNET.KPTS_HEAD_BLOCKS,
            stride_init=cfg.MODEL.FBNET.KPTS_HEAD_STRIDE,
            last_layer_scale=cfg.MODEL.FBNET.KPTS_HEAD_LAST_SCALE,
        )

        self.predictor = KeypointRCNNPredictorNoUpscale(
            in_channels=self.feature_extractor.out_channels,
            num_keypoints=cfg.MODEL.ROI_KEYPOINT_HEAD.NUM_KEYPOINTS,
        )

    def layers(self, x):
        x = self.feature_extractor(x)
        x = self.predictor(x)
        return x


@mask_head.ROI_MASK_HEAD_REGISTRY.register()
class FBNetRoIMaskHead(mask_head.BaseMaskRCNNHead):
    def __init__(self, cfg, input_shape: ShapeSpec):
        super().__init__(cfg, input_shape)

        in_channels = input_shape.channels
        builder, model_arch = create_fbnet_builder(cfg.MODEL.FBNET)
        builder.last_depth = in_channels
        # builder.name_prefix = "_[mask]_"

        self.feature_extractor = FBNetGenericRoIHead(
            cfg,
            in_channels,
            builder,
            model_arch,
            head_name="mask",
            use_blocks=cfg.MODEL.FBNET.MASK_HEAD_BLOCKS,
            stride_init=cfg.MODEL.FBNET.MASK_HEAD_STRIDE,
            last_layer_scale=cfg.MODEL.FBNET.MASK_HEAD_LAST_SCALE,
        )

        num_classes = cfg.MODEL.ROI_HEADS.NUM_CLASSES
        self.predictor = MaskRCNNConv1x1Predictor(
            self.feature_extractor.out_channels, num_classes
        )

    def layers(self, x):
        x = self.feature_extractor(x)
        x = self.predictor(x)
        return x
