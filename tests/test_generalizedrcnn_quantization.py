#!/usr/bin/env python3

import unittest

import copy
import os
import tempfile

import torch
from d2go.export.d2_meta_arch import d2_meta_arch_prepare_for_quant
from d2go.setup import prepare_for_launch, setup_after_launch
from d2go.tools.exporter import get_parser
from d2go.runner import GeneralizedRCNNRunner

# from mobile_cv.common.misc.file_utils import make_temp_directory
from mobile_cv.arch.utils.quantize_utils import swap_syncbn_to_bn
from torch.quantization.quantize_fx import convert_fx, prepare_qat_fx

torch.manual_seed(0)


def get_example_inputs():
    # see GeneralizedRCNN.forward for expected format
    batched_inputs = [{"image": torch.randn(3, 600, 600)}]
    return batched_inputs


def prepare_eager_all_steps(model_eager, qconfig, cfg):
    # prepare for eager mode quantization
    model_eager = d2_meta_arch_prepare_for_quant(model_eager, cfg)
    # Eager mode post training quant - prepare
    eager_prepare_fn = torch.quantization.prepare_qat
    # backbone
    model_eager.backbone.qconfig = qconfig
    model_eager.backbone = eager_prepare_fn(model_eager.backbone)
    # RPN
    model_eager.proposal_generator.qconfig = qconfig
    model_eager.proposal_generator = eager_prepare_fn(model_eager.proposal_generator)
    # head
    model_eager.roi_heads.qconfig = qconfig
    model_eager.roi_heads = eager_prepare_fn(model_eager.roi_heads)
    return model_eager


def prepare_fx_all_steps(model_fx, qconfig_dict):
    fx_prepare_fn = prepare_qat_fx
    # need to preserve some attributes which are not part of forward
    # to fix https://www.internalfb.com/intern/paste/P150232942/
    # hacky solution unbreaks for now, need to think of a more general
    # solution for a wider rollout
    old_size_divisibility = model_fx.backbone.size_divisibility
    model_fx.backbone = fx_prepare_fn(model_fx.backbone, qconfig_dict)
    model_fx.backbone.size_divisibility = old_size_divisibility

    # proposal generator not symbolically traceable: https://www.internalfb.com/intern/paste/P150230844/
    # example of Eager quantized proposal generator: https://www.internalfb.com/intern/paste/P155558257/
    # only model.proposal_generator.rpn_head is quantized in Eager,
    #   see https://www.internalfb.com/diff/D25467258 for details
    # rpn_head is not traceable because of [y for y in x], so trace
    # the pieces instead
    model_fx.proposal_generator.rpn_head.rpn_feature = fx_prepare_fn(
        model_fx.proposal_generator.rpn_head.rpn_feature, qconfig_dict
    )
    model_fx.proposal_generator.rpn_head.rpn_regressor.cls_logits = fx_prepare_fn(
        model_fx.proposal_generator.rpn_head.rpn_regressor.cls_logits, qconfig_dict
    )
    model_fx.proposal_generator.rpn_head.rpn_regressor.bbox_pred = fx_prepare_fn(
        model_fx.proposal_generator.rpn_head.rpn_regressor.bbox_pred, qconfig_dict
    )

    # heads not symbollicaly traceable: https://www.internalfb.com/intern/paste/P150231525/
    # example of Eager quantized head: https://www.internalfb.com/intern/paste/P155610062/
    # box_head is an FBNetV2RoIBoxHead, with self.roi_box_conv needing
    # and the rest of the model not being traceable and not needing
    # quantization. So, we only quantize self.roi_box_conv
    # TODO(future): only do this when keypoint head exists
    # TODO(future): test for roi_heads.mask_head, in a model which has it
    model_fx.roi_heads.box_head.roi_box_conv = fx_prepare_fn(
        model_fx.roi_heads.box_head.roi_box_conv,
        qconfig_dict,
        prepare_custom_config_dict={
            "output_quantized_idxs": [0],
        },
    )
    model_fx.roi_heads.box_head.avgpool = fx_prepare_fn(
        model_fx.roi_heads.box_head.avgpool,
        qconfig_dict,
        prepare_custom_config_dict={
            "input_quantized_idxs": [0],
        },
    )
    # box_predictor is an instance of FastRCNNOutputLayers.  This class currently
    # has a symbolically traceable `forward` function, as well as other functions
    # unrelated to the forward but essential to inference logic.  Because symbolic
    # tracing only preserves the forward, we cannot symbolically trace the entire
    # object.  Therefore, we trace just the inner layers.
    # model_fx.roi_heads.box_predictor = fx_prepare_fn(
    #     model_fx.roi_heads.box_predictor, qconfig_dict)
    model_fx.roi_heads.box_predictor.cls_score = fx_prepare_fn(
        model_fx.roi_heads.box_predictor.cls_score, qconfig_dict
    )
    model_fx.roi_heads.box_predictor.bbox_pred = fx_prepare_fn(
        model_fx.roi_heads.box_predictor.bbox_pred, qconfig_dict
    )

    # keypoint head
    model_fx.roi_heads.keypoint_head.feature_extractor = fx_prepare_fn(
        model_fx.roi_heads.keypoint_head.feature_extractor, qconfig_dict
    )
    model_fx.roi_heads.keypoint_head.predictor = fx_prepare_fn(
        model_fx.roi_heads.keypoint_head.predictor, qconfig_dict
    )

    return model_fx


# Eager mode post training quant - convert
def convert_eager_all_steps(model_eager):
    model_eager.backbone = torch.quantization.convert(model_eager.backbone)
    model_eager.proposal_generator = torch.quantization.convert(
        model_eager.proposal_generator
    )
    model_eager.roi_heads = torch.quantization.convert(model_eager.roi_heads)
    return model_eager


# FX post training quant - convert
def convert_fx_all_steps(model_fx):

    # backbone
    # preserve size_divisibility
    old_size_divisibility = model_fx.backbone.size_divisibility
    model_fx.backbone = convert_fx(model_fx.backbone)
    model_fx.backbone.size_divisibility = old_size_divisibility

    # rpn
    model_fx.proposal_generator.rpn_head.rpn_feature = convert_fx(
        model_fx.proposal_generator.rpn_head.rpn_feature
    )
    model_fx.proposal_generator.rpn_head.rpn_regressor.cls_logits = convert_fx(
        model_fx.proposal_generator.rpn_head.rpn_regressor.cls_logits
    )
    model_fx.proposal_generator.rpn_head.rpn_regressor.bbox_pred = convert_fx(
        model_fx.proposal_generator.rpn_head.rpn_regressor.bbox_pred
    )

    # box head
    model_fx.roi_heads.box_head.roi_box_conv = convert_fx(
        model_fx.roi_heads.box_head.roi_box_conv
    )
    model_fx.roi_heads.box_head.avgpool = convert_fx(
        model_fx.roi_heads.box_head.avgpool
    )
    model_fx.roi_heads.box_predictor.cls_score = convert_fx(
        model_fx.roi_heads.box_predictor.cls_score
    )
    model_fx.roi_heads.box_predictor.bbox_pred = convert_fx(
        model_fx.roi_heads.box_predictor.bbox_pred
    )

    # keypoint head
    model_fx.roi_heads.keypoint_head.feature_extractor = convert_fx(
        model_fx.roi_heads.keypoint_head.feature_extractor
    )
    model_fx.roi_heads.keypoint_head.predictor = convert_fx(
        model_fx.roi_heads.keypoint_head.predictor
    )

    return model_fx


# This is the config from f238933809, with unnecessary info removed.
CONFIG = """
MODEL:
  ANCHOR_GENERATOR:
    ANGLES:
    - - -90
      - 0
      - 90
    ASPECT_RATIOS:
    - - 0.5
      - 1.0
      - 2.0
    NAME: DefaultAnchorGenerator
    OFFSET: 0.0
    SIZES:
    - - 32
      - 64
      - 128
      - 256
      - 512
  BACKBONE:
    FREEZE_AT: 2
    NAME: FBNetV2C4Backbone
  BIFPN:
    DEPTH_MULTIPLIER: 1
    NORM: naiveSyncBN
    NORM_ARGS: []
    SCALE_FACTOR: 1
    TOP_BLOCK_BEFORE_FPN: false
    WIDTH_DIVISOR: 8
  DDP_FIND_UNUSED_PARAMETERS: false
  DEVICE: cpu
  FBNET:
    ARCH: default
    ARCH_DEF: ''
    BN_TYPE: bn
    DET_HEAD_BLOCKS: []
    DET_HEAD_LAST_SCALE: 1.0
    DET_HEAD_STRIDE: 0
    DW_CONV_SKIP_BN: true
    DW_CONV_SKIP_RELU: true
    KPTS_HEAD_BLOCKS: []
    KPTS_HEAD_LAST_SCALE: 0.0
    KPTS_HEAD_STRIDE: 0
    MASK_HEAD_BLOCKS: []
    MASK_HEAD_LAST_SCALE: 0.0
    MASK_HEAD_STRIDE: 0
    NUM_GROUPS: 32
    RPN_BN_TYPE: ''
    RPN_HEAD_BLOCKS: 0
    SCALE_FACTOR: 1.0
    STEM_IN_CHANNELS: 3
    WIDTH_DIVISOR: 1
  FBNET_V2:
    ARCH: FBNetV3_B_light_large
    ARCH_DEF: []
    NORM: naiveSyncBN
    NORM_ARGS: []
    SCALE_FACTOR: 1.0
    STEM_IN_CHANNELS: 3
    WIDTH_DIVISOR: 8
  FPN:
    FUSE_TYPE: sum
    IN_FEATURES: []
    NORM: naiveSyncBN
    OUT_CHANNELS: 256
  FROZEN_LAYER_REG_EXP: []
  KEYPOINT_ON: true
  KMEANS_ANCHORS:
    DATASETS: []
    KMEANS_ANCHORS_ON: false
    NUM_CLUSTERS: 0
    NUM_TRAINING_IMG: 0
    RNG_SEED: 3
  LOAD_PROPOSALS: false
  MASK_ON: false
  META_ARCHITECTURE: GeneralizedRCNN
  PANOPTIC_FPN:
    COMBINE:
      ENABLED: true
      INSTANCES_CONFIDENCE_THRESH: 0.5
      OVERLAP_THRESH: 0.5
      STUFF_AREA_LIMIT: 4096
    INSTANCE_LOSS_WEIGHT: 1.0
  PIXEL_MEAN:
  - 103.53
  - 116.28
  - 123.675
  PIXEL_STD:
  - 57.375
  - 57.12
  - 58.395
  PROPOSAL_GENERATOR:
    MIN_SIZE: 0
    NAME: RPN
  RESNETS:
    DEFORM_MODULATED: false
    DEFORM_NUM_GROUPS: 1
    DEFORM_ON_PER_STAGE:
    - false
    - false
    - false
    - false
    DEPTH: 50
    NORM: naiveSyncBN
    NUM_GROUPS: 1
    OUT_FEATURES:
    - res4
    RES2_OUT_CHANNELS: 256
    RES5_DILATION: 1
    STEM_OUT_CHANNELS: 64
    STRIDE_IN_1X1: true
    WIDTH_PER_GROUP: 64
  RETINANET:
    BBOX_REG_LOSS_TYPE: smooth_l1
    BBOX_REG_WEIGHTS: &id001
    - 1.0
    - 1.0
    - 1.0
    - 1.0
    FOCAL_LOSS_ALPHA: 0.25
    FOCAL_LOSS_GAMMA: 2.0
    IN_FEATURES:
    - p3
    - p4
    - p5
    - p6
    - p7
    IOU_LABELS:
    - 0
    - -1
    - 1
    IOU_THRESHOLDS:
    - 0.4
    - 0.5
    NMS_THRESH_TEST: 0.5
    NORM: ''
    NUM_CLASSES: 80
    NUM_CONVS: 4
    PRIOR_PROB: 0.01
    SCORE_THRESH_TEST: 0.05
    SMOOTH_L1_LOSS_BETA: 0.1
    TOPK_CANDIDATES_TEST: 1000
  ROI_BOX_CASCADE_HEAD:
    BBOX_REG_WEIGHTS:
    - - 10.0
      - 10.0
      - 5.0
      - 5.0
    - - 20.0
      - 20.0
      - 10.0
      - 10.0
    - - 30.0
      - 30.0
      - 15.0
      - 15.0
    IOUS:
    - 0.5
    - 0.6
    - 0.7
  ROI_BOX_HEAD:
    BBOX_REG_LOSS_TYPE: smooth_l1
    BBOX_REG_LOSS_WEIGHT: 1.0
    BBOX_REG_WEIGHTS:
    - 10.0
    - 10.0
    - 5.0
    - 5.0
    CLS_AGNOSTIC_BBOX_REG: false
    CONV_DIM: 256
    FC_DIM: 1024
    NAME: FBNetV2RoIBoxHead
    NORM: naiveSyncBN
    NUM_CONV: 0
    NUM_FC: 0
    POOLER_RESOLUTION: 6
    POOLER_SAMPLING_RATIO: 0
    POOLER_TYPE: ROIAlignV2
    SMOOTH_L1_BETA: 0.5
    TRAIN_ON_PRED_BOXES: false
  ROI_HEADS:
    BATCH_SIZE_PER_IMAGE: 512
    IN_FEATURES:
    - trunk3
    IOU_LABELS:
    - 0
    - 1
    IOU_THRESHOLDS:
    - 0.5
    NAME: StandardROIHeads
    NMS_THRESH_TEST: 0.5
    NUM_CLASSES: 1
    POSITIVE_FRACTION: 0.25
    PROPOSAL_APPEND_GT: true
    SCORE_THRESH_TEST: 0.05
  ROI_KEYPOINT_HEAD:
    CONV_DIMS:
    - 512
    - 512
    - 512
    - 512
    - 512
    - 512
    - 512
    - 512
    LOSS_WEIGHT: 1.0
    MIN_KEYPOINTS_PER_IMAGE: 1
    NAME: FBNetV2RoIKeypointHeadKPRCNNIRFPredictorNoUpscale
    NORMALIZE_LOSS_BY_VISIBLE_KEYPOINTS: true
    NUM_KEYPOINTS: 17
    POOLER_RESOLUTION: 6
    POOLER_SAMPLING_RATIO: 0
    POOLER_TYPE: ROIAlignV2
  ROI_MASK_HEAD:
    CLS_AGNOSTIC_MASK: false
    CONV_DIM: 256
    NAME: FBNetV2RoIMaskHead
    NORM: naiveSyncBN
    NUM_CONV: 0
    POOLER_RESOLUTION: 6
    POOLER_SAMPLING_RATIO: 0
    POOLER_TYPE: ROIAlignV2
  RPN:
    BATCH_SIZE_PER_IMAGE: 256
    BBOX_REG_LOSS_TYPE: smooth_l1
    BBOX_REG_LOSS_WEIGHT: 1.0
    BBOX_REG_WEIGHTS: *id001
    BOUNDARY_THRESH: -1
    HEAD_NAME: FBNetV2RpnHead
    IN_FEATURES:
    - trunk3
    IOU_LABELS:
    - 0
    - -1
    - 1
    IOU_THRESHOLDS:
    - 0.3
    - 0.7
    LOSS_WEIGHT: 1.0
    NMS_THRESH: 0.5
    POSITIVE_FRACTION: 0.5
    POST_NMS_TOPK_TEST: 100
    POST_NMS_TOPK_TRAIN: 1000
    PRE_NMS_TOPK_TEST: 6000
    PRE_NMS_TOPK_TRAIN: 4000
    SMOOTH_L1_BETA: 0.0
  SEM_SEG_HEAD:
    COMMON_STRIDE: 4
    CONVS_DIM: 128
    IGNORE_VALUE: 255
    IN_FEATURES:
    - p2
    - p3
    - p4
    - p5
    LOSS_WEIGHT: 1.0
    NAME: SemSegFPNHead
    NORM: naiveSyncBN
    NUM_CLASSES: 54
  SUBCLASS:
    NUM_SUBCLASSES: 0
    SUBCLASS_ON: false
  VT_FPN:
    HEADS: 16
    IN_FEATURES:
    - res2
    - res3
    - res4
    - res5
    LAYERS: 3
    MIN_GROUP_PLANES: 64
    NORM: BN
    OUT_CHANNELS: 256
    POS_HWS: []
    POS_N_DOWNSAMPLE: []
    TOKEN_C: 1024
    TOKEN_LS:
    - 16
    - 16
    - 8
    - 8
  WEIGHTS: ''
  XRAYMOBILE_V1:
    SCALE_CHANNELS: 1
QUANTIZATION:
  BACKEND: fbgemm
  CUSTOM_QSCHEME: ''
  EAGER_MODE: true
  PTQ:
    CALIBRATION_FORCE_ON_GPU: false
    CALIBRATION_NUM_IMAGES: 1
  QAT:
    BATCH_SIZE_FACTOR: 1.0
    DISABLE_OBSERVER_ITER: 38000
    ENABLED: false
    ENABLE_OBSERVER_ITER: 35000
    FREEZE_BN_ITER: 37000
    START_ITER: 35000
    UPDATE_OBSERVER_STATS_PERIOD: 1
    UPDATE_OBSERVER_STATS_PERIODICALLY: false
  SILICON_QAT:
    ENABLED: false
SEED: -1
"""


def _write_yaml_to_file(out_dir, prefix, config_str):
    temp_name = next(tempfile._get_candidate_names())
    file_name = f"{prefix}_{temp_name}.yaml"
    out_file = os.path.join(out_dir, file_name)
    with open(out_file, "w") as wf:
        wf.write(config_str)
    return out_file


def _get_config(out_dir):
    return _write_yaml_to_file(out_dir, "generalized_rcnn", CONFIG)


class TestGeneralizedRCNNQuantization(unittest.TestCase):
    def test_eager_vs_fx(self):
        """
        buck run @mode/dev-nosan mobile-vision/d2go/tests:test_generalizedrcnn_quantization -- d2go.tests.test_generalizedrcnn_quantization.TestGeneralizedRCNNQuantization.test_eager_vs_fx
        """

        runner = GeneralizedRCNNRunner()
        cfg = runner.get_default_cfg()
        with tempfile.TemporaryDirectory() as cfg_dir:
            cfg_filename = _get_config(cfg_dir)
            cfg.merge_from_file(cfg_filename)
            cfg.freeze()
        model = runner.build_model(cfg).cpu()
        # swap SyncBNs
        swap_syncbn_to_bn(model)

        # model def (approx): https://www.internalfb.com/intern/paste/P150227062/
        # GeneralizedRCNN
        #   - backbone: FBNetV2C4Backbone
        #     - body: FBNetV2Backbone
        #   - proposal_generator: RPN (https://fburl.com/diffusion/l61ea4s6)
        #     - rpn_head: FBNetV2RpnHead
        #     - anchor_generator: DefaultAnchorGenerator
        #   - roi_heads: StandardROIHeads
        #     - box_pooler: ROIPooler
        #     - box_head: FBNetV2RoIBoxHead
        #     - box_predictor: FastRCNNOutputLayers
        #     - keypoint_pooler: ROIPooler
        #     - keypoint_head: FBNetV2RoIKeypointHeadKRCNNPredictorNoUpscale

        qconfig = torch.quantization.get_default_qat_qconfig("qnnpack")
        qconfig_dict = {"": qconfig}

        model_eager = copy.deepcopy(model)
        model_eager = prepare_eager_all_steps(model_eager, qconfig, cfg)

        model_fx = copy.deepcopy(model)
        model_fx = prepare_fx_all_steps(model_fx, qconfig_dict)

        batched_inputs = get_example_inputs()

        # set to eval, as some logic in train mode is non-deterministic
        model_eager.eval()
        model_fx.eval()

        # compare eager vs graph outputs of various parts of the model
        results = {
            "eager": {
                "qat_fp32": {},
                "qat_int8": {},
            },
            "fx": {
                "qat_fp32": {},
                "qat_int8": {},
            },
        }

        # this if reproducting the inference logic from
        # fbcode/vision/fair/detectron2/detectron2/modeling/meta_arch/rcnn.py,
        # GeneralizedRCNN.inference
        for idx, m in enumerate([model_eager, model_fx]):
            images = m.preprocess_image(batched_inputs)
            features = m.backbone(images.tensor)
            proposals, _ = m.proposal_generator(images, features, None)
            results_head = m.roi_heads(images, features, proposals, None)

            model_type = "eager" if idx == 0 else "fx"
            results[model_type]["qat_fp32"] = {
                "features": features,
                "proposals": proposals,
                "head_output": results_head,
            }

        model_eager = convert_eager_all_steps(model_eager)
        model_fx = convert_fx_all_steps(model_fx)

        # compare quantized results

        # this if reproducting the inference logic from
        # fbcode/vision/fair/detectron2/detectron2/modeling/meta_arch/rcnn.py,
        # GeneralizedRCNN.inference
        for idx, m in enumerate([model_eager, model_fx]):
            images = m.preprocess_image(batched_inputs)
            features = m.backbone(images.tensor)
            proposals, _ = m.proposal_generator(images, features, None)
            results_head = m.roi_heads(images, features, proposals, None)

            model_type = "eager" if idx == 0 else "fx"
            results[model_type]["qat_int8"] = {
                "features": features,
                "proposals": proposals,
                "head_output": results_head,
            }

        # compare v2
        for stage in ("qat_fp32", "qat_int8"):
            r_eager = results["eager"][stage]
            r_fx = results["fx"][stage]

            # trunk
            for key in ("trunk0", "trunk1", "trunk2", "trunk3"):
                features_eager = r_eager["features"][key]
                features_fx = r_fx["features"][key]
                self.assertTrue(torch.allclose(features_eager, features_fx))

            # proposals
            proposal_boxes_eager = r_eager["proposals"][0].proposal_boxes.tensor
            proposal_boxes_fx = r_fx["proposals"][0].proposal_boxes.tensor
            # note: some elements here fail with the default atol=1e-8. With manual
            # inspection, seems fine, just fp32 inaccuracies. Relax the tolerance.
            self.assertTrue(torch.allclose(proposal_boxes_eager, proposal_boxes_fx, atol=1e-5))

            proposal_objectness_logits_eager = r_eager["proposals"][0].objectness_logits
            proposal_objectness_logits_fx = r_fx["proposals"][0].objectness_logits
            self.assertTrue(torch.allclose(
                proposal_objectness_logits_eager,
                proposal_objectness_logits_fx,
            ))

            # head
            head_pred_boxes_eager = r_eager["head_output"][0][0].pred_boxes.tensor
            head_pred_boxes_fx = r_fx["head_output"][0][0].pred_boxes.tensor
            self.assertTrue(torch.allclose(head_pred_boxes_eager, head_pred_boxes_fx))

            head_pred_keypoint_heatmaps_eager = r_eager["head_output"][0][
                0
            ].pred_keypoint_heatmaps
            head_pred_keypoint_heatmaps_fx = r_fx["head_output"][0][0].pred_keypoint_heatmaps
            self.assertTrue(torch.allclose(
                head_pred_keypoint_heatmaps_eager,
                head_pred_keypoint_heatmaps_fx,
            ))
