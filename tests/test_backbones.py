#!/usr/bin/env python3

import json
import math
import torch
import unittest
# import modules to to register backbones
from detectron2.modeling import BACKBONE_REGISTRY, Backbone
from detectron2.layers import ShapeSpec
from d2go.runner import GeneralizedRCNNRunner
from d2go.modeling.backbone.fbnet_v2 import _parse_arch_def


# overwrite configs if specified, otherwise default config is used
BACKBONE_CFGS = {
    "build_hrfpn_backbone": None,
    "build_pose_hrnet_backbone": None,
    "build_resnet_fpn_backbone": "detectron2://Base-RCNN-FPN.yaml",
    "build_retinanet_resnet_fpn_backbone": "detectron2://Base-RetinaNet.yaml",
    "FBNetFpnBackbone": "detectron2go://deprecated_fbnet_v1/e2e_mask_rcnn_fbnet_fpn.yaml",  # noqa
    "FBNetV2FpnBackbone": "detectron2go://e2e_mask_rcnn_fbnet_fpn.yaml",
    "FBNetV2RetinaNetBackbone": "detectron2go://retinanet_fbnet_default_600.yaml",
    "FBNetV2HRBackbone": "mv_experimental://people_ai/segmentation/configs/semantic_fbnet_hr_xirp5_192.yaml",  # noqa, TODO: swtich to use baseline config
    "FBNetV2BiFpnBackbone": "detectron2go://e2e_mask_rcnn_fbnet_bifpn.yaml",
    "FBNetV2RetinaNetBiFpnBackbone": "detectron2go://retinanet_fbnet_eff_d0_bifpn.yaml",
    # segmentation
    # TODO: swtich to use baseline config
    "FBNetV2HRBackbone": "mv_experimental://people_ai/segmentation/configs/semantic_fbnet_hr_xirp5_96.yaml",  # noqa
    "build_resnet_vt_fpn_backbone": "mv_experimental://det_cv/configs/semantic_coco_R_50_VT_FPN_1x.yaml",  # noqa
    "build_resnet_vt_fpn_det_backbone": "mv_experimental://det_cv/configs/faster_rcnn_coco_R_50_VT_FPN_1x.yaml",  # noqa
}


class TestBackbones(unittest.TestCase):
    def test_build_backbones(self):
        """ Make sure backbones run """

        self.assertGreater(len(BACKBONE_REGISTRY._obj_map), 0)

        for name, backbone_builder in BACKBONE_REGISTRY._obj_map.items():
            print("Testing {}...".format(name))
            cfg = GeneralizedRCNNRunner().get_default_cfg()
            if name in BACKBONE_CFGS:
                config_file = BACKBONE_CFGS[name]
                if config_file is None:
                    print("Skip {}...".format(name))
                    continue
                cfg.merge_from_file(config_file)
            backbone = backbone_builder(
                cfg,
                ShapeSpec(channels=len(cfg.MODEL.PIXEL_MEAN))
            )

            # make sures the backbone has `out_channels`
            self.assertIsInstance(backbone, Backbone)
            self.assertIsInstance(backbone.output_shape(), dict)

            N, C_in, H, W = 2, 3, 256, 384
            input = torch.rand([N, C_in, H, W], dtype=torch.float32)
            out = backbone(input)
            self.assertIsInstance(out, dict)
            for feature_name, cur_out in out.items():
                # Check feature channels
                self.assertEqual(
                    cur_out.shape[:2],
                    torch.Size([N, backbone.output_shape()[feature_name].channels])
                )
                # Check feature strides
                ratio_h = H / cur_out.shape[2]
                ratio_w = W / cur_out.shape[3]
                stride = backbone.output_shape()[feature_name].stride
                self.assertEqual(round(math.log2(ratio_h)), round(math.log2(stride)))
                self.assertEqual(round(math.log2(ratio_w)), round(math.log2(stride)))


class TestFBNetV2(unittest.TestCase):
    def test_parse_arch_def(self):
        """Check that parse can deal with arch name and arch def"""
        cfg = GeneralizedRCNNRunner().get_default_cfg()
        arch_def_A = _parse_arch_def(cfg)
        cfg.MODEL.FBNET_V2.ARCH = ""
        cfg.MODEL.FBNET_V2.ARCH_DEF = [arch_def_A]
        arch_def_B = _parse_arch_def(cfg)

        self.assertEqual(arch_def_A, arch_def_B)


if __name__ == "__main__":
    unittest.main()
