import unittest

from d2go.config import CfgNode


class TestConfigNode(unittest.TestCase):
    @staticmethod
    def _get_default_config():
        cfg = CfgNode()
        cfg.INPUT = CfgNode()
        cfg.INPUT.CROP = CfgNode()
        cfg.INPUT.CROP.ENABLED = False
        cfg.INPUT.CROP.SIZE = (0.9, 0.9)
        cfg.INPUT.CROP.TYPE = "relative_range"
        cfg.MODEL = CfgNode()
        cfg.MODEL.MIN_DIM_SIZE = 360
        cfg.INFERENCE_SDK = CfgNode()
        cfg.INFERENCE_SDK.MODEL = CfgNode()
        cfg.INFERENCE_SDK.MODEL.SCORE_THRESHOLD = 0.8
        cfg.INFERENCE_SDK.IOU_TRACKER = CfgNode()
        cfg.INFERENCE_SDK.IOU_TRACKER.IOU_THRESHOLD = 0.15
        cfg.INFERENCE_SDK.ENABLE_ID_TRACKING = True
        return cfg

    def test_get_field_or_none(self):
        cfg = self._get_default_config()
        self.assertEqual(cfg.get_field_or_none("MODEL.MIN_DIM_SIZE"), 360)
        self.assertEqual(
            cfg.get_field_or_none("INFERENCE_SDK.ENABLE_ID_TRACKING"), True
        )
        self.assertEqual(cfg.get_field_or_none("MODEL.INPUT_SIZE"), None)
        self.assertEqual(cfg.get_field_or_none("MODEL.INPUT_SIZE.HEIGHT"), None)

    def test_as_flattened_dict(self):
        cfg = self._get_default_config()
        cfg_dict = cfg.as_flattened_dict()
        target_cfg_dict = {
            "INPUT.CROP.ENABLED": False,
            "INPUT.CROP.SIZE": (0.9, 0.9),
            "INPUT.CROP.TYPE": "relative_range",
            "MODEL.MIN_DIM_SIZE": 360,
            "INFERENCE_SDK.MODEL.SCORE_THRESHOLD": 0.8,
            "INFERENCE_SDK.IOU_TRACKER.IOU_THRESHOLD": 0.15,
            "INFERENCE_SDK.ENABLE_ID_TRACKING": True,
        }
        self.assertEqual(target_cfg_dict, cfg_dict)
