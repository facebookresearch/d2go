#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved


import glob
import logging
import os
import unittest

from d2go.config import auto_scale_world_size, reroute_config_path
from d2go.config.utils import (
    config_dict_to_list_str,
    flatten_config_dict,
    get_cfg_diff_table,
    get_from_flattened_config_dict,
)
from d2go.runner import GeneralizedRCNNRunner
from d2go.utils.testing.helper import get_resource_path
from mobile_cv.common.misc.file_utils import make_temp_directory


logger = logging.getLogger(__name__)


class TestConfig(unittest.TestCase):
    def test_load_configs(self):
        """Make sure configs are loadable"""

        for location in ["detectron2", "detectron2go"]:
            root_dir = os.path.abspath(reroute_config_path(f"{location}://."))
            files = glob.glob(os.path.join(root_dir, "**/*.yaml"), recursive=True)
            files = [f for f in files if "fbnas" not in f]
            self.assertGreater(len(files), 0)
            for fn in sorted(files):
                logger.info("Loading {}...".format(fn))
                GeneralizedRCNNRunner().get_default_cfg().merge_from_file(fn)

    def test_load_arch_defs(self):
        """Test arch def str-to-dict conversion compatible with merging"""
        default_cfg = GeneralizedRCNNRunner().get_default_cfg()
        cfg = default_cfg.clone()
        cfg.merge_from_file(get_resource_path("arch_def_merging.yaml"))

        with make_temp_directory("detectron2go_tmp") as tmp_dir:
            # Dump out config with arch def
            file_name = os.path.join(tmp_dir, "test_archdef_config.yaml")
            with open(file_name, "w") as f:
                f.write(cfg.dump())

            # Attempt to reload the config
            another_cfg = default_cfg.clone()
            another_cfg.merge_from_file(file_name)

    def test_base_reroute(self):
        default_cfg = GeneralizedRCNNRunner().get_default_cfg()

        # use rerouted file as base
        cfg = default_cfg.clone()
        cfg.merge_from_file(get_resource_path("rerouted_base.yaml"))
        self.assertEqual(cfg.MODEL.MASK_ON, True)  # base is loaded
        self.assertEqual(cfg.MODEL.FBNET_V2.ARCH, "test")  # non-base is loaded

        # use multiple files as base
        cfg = default_cfg.clone()
        cfg.merge_from_file(get_resource_path("rerouted_multi_base.yaml"))
        self.assertEqual(cfg.MODEL.MASK_ON, True)  # base is loaded
        self.assertEqual(cfg.MODEL.FBNET_V2.ARCH, "FBNetV3_A")  # second base is loaded
        self.assertEqual(cfg.OUTPUT_DIR, "test")  # non-base is loaded

    def test_default_cfg_dump_and_load(self):
        default_cfg = GeneralizedRCNNRunner().get_default_cfg()

        cfg = default_cfg.clone()
        with make_temp_directory("detectron2go_tmp") as tmp_dir:
            file_name = os.path.join(tmp_dir, "config.yaml")
            # this is same as the one in fblearner_launch_utils_detectron2go.py
            with open(file_name, "w") as f:
                f.write(cfg.dump(default_flow_style=False))

            # check if the dumped config file can be merged
            cfg.merge_from_file(file_name)

    def test_default_cfg_deprecated_keys(self):
        default_cfg = GeneralizedRCNNRunner().get_default_cfg()

        # a warning will be printed for deprecated keys
        default_cfg.merge_from_list(["QUANTIZATION.QAT.LOAD_PRETRAINED", True])
        # exception will raise for renamed keys
        self.assertRaises(
            KeyError,
            default_cfg.merge_from_list,
            ["QUANTIZATION.QAT.BACKEND", "fbgemm"],
        )


class TestConfigUtils(unittest.TestCase):
    """Test util functions in config/utils.py"""

    def test_flatten_config_dict(self):
        """Check flatten config dict to single layer dict"""
        d = {"c0": {"c1": {"c2": 3}}, "b0": {"b1": "b2"}, "a0": "a1"}

        # reorder=True
        fdict = flatten_config_dict(d, reorder=True)
        gt = {"a0": "a1", "b0.b1": "b2", "c0.c1.c2": 3}
        self.assertEqual(fdict, gt)
        self.assertEqual(list(fdict.keys()), list(gt.keys()))

        # reorder=False
        fdict = flatten_config_dict(d, reorder=False)
        gt = {"c0.c1.c2": 3, "b0.b1": "b2", "a0": "a1"}
        self.assertEqual(fdict, gt)
        self.assertEqual(list(fdict.keys()), list(gt.keys()))

    def test_config_dict_to_list_str(self):
        """Check convert config dict to str list"""
        d = {"a0": "a1", "b0": {"b1": "b2"}, "c0": {"c1": {"c2": 3}}}
        str_list = config_dict_to_list_str(d)
        gt = ["a0", "a1", "b0.b1", "b2", "c0.c1.c2", "3"]
        self.assertEqual(str_list, gt)

    def test_get_from_flattened_config_dict(self):
        d = {"MODEL": {"MIN_DIM_SIZE": 360}}
        self.assertEqual(
            get_from_flattened_config_dict(d, "MODEL.MIN_DIM_SIZE"), 360
        )  # exist
        self.assertEqual(
            get_from_flattened_config_dict(d, "MODEL.MODEL.INPUT_SIZE"), None
        )  # non-exist

    def test_get_cfg_diff_table(self):
        """Check compare two dicts"""
        d1 = {"a0": "a1", "b0": {"b1": "b2"}, "c0": {"c1": {"c2": 3}}}
        d2 = {"a0": "a1", "b0": {"b1": "b3"}, "c0": {"c1": {"c2": 4}}}
        table = get_cfg_diff_table(d1, d2)
        self.assertTrue("a0" not in table)  # a0 are the same
        self.assertTrue("b0.b1" in table)  # b0.b1 are different
        self.assertTrue("c0.c1.c2" in table)  # c0.c1.c2 are different


class TestAutoScaleWorldSize(unittest.TestCase):
    def test_8gpu_to_1gpu(self):
        """
        when scaling a 8-gpu config to 1-gpu one, the batch size will be reduced by 8x
        """
        cfg = GeneralizedRCNNRunner().get_default_cfg()
        self.assertEqual(cfg.SOLVER.REFERENCE_WORLD_SIZE, 8)
        batch_size_x8 = cfg.SOLVER.IMS_PER_BATCH
        assert batch_size_x8 % 8 == 0, "default batch size is not multiple of 8"
        auto_scale_world_size(cfg, new_world_size=1)
        self.assertEqual(cfg.SOLVER.REFERENCE_WORLD_SIZE, 1)
        self.assertEqual(cfg.SOLVER.IMS_PER_BATCH * 8, batch_size_x8)

    def test_not_scale_for_zero_world_size(self):
        """
        when reference world size is 0, no scaling should happen
        """
        cfg = GeneralizedRCNNRunner().get_default_cfg()
        self.assertEqual(cfg.SOLVER.REFERENCE_WORLD_SIZE, 8)
        cfg.SOLVER.REFERENCE_WORLD_SIZE = 0
        batch_size_x8 = cfg.SOLVER.IMS_PER_BATCH
        auto_scale_world_size(cfg, new_world_size=1)
        self.assertEqual(cfg.SOLVER.REFERENCE_WORLD_SIZE, 0)
        self.assertEqual(cfg.SOLVER.IMS_PER_BATCH, batch_size_x8)


if __name__ == "__main__":
    unittest.main()
