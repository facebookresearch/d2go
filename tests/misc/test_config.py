#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved


import glob
import logging
import os
import unittest

from d2go.config import (
    auto_scale_world_size,
    CfgNode,
    load_full_config_from_file,
    reroute_config_path,
    temp_new_allowed,
)
from d2go.config.utils import (
    config_dict_to_list_str,
    flatten_config_dict,
    get_cfg_diff_table,
    get_diff_cfg,
    get_from_flattened_config_dict,
)
from d2go.registry.builtin import CONFIG_UPDATER_REGISTRY
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
                GeneralizedRCNNRunner.get_default_cfg().merge_from_file(fn)

    def test_load_arch_defs(self):
        """Test arch def str-to-dict conversion compatible with merging"""
        default_cfg = GeneralizedRCNNRunner.get_default_cfg()
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
        default_cfg = GeneralizedRCNNRunner.get_default_cfg()

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

    def test_temp_new_allowed(self):
        default_cfg = GeneralizedRCNNRunner.get_default_cfg()

        def set_field(cfg):
            cfg.THIS_BETTER_BE_A_NEW_CONFIG = 4

        self.assertFalse("THIS_BETTER_BE_A_NEW_CONFIG" in default_cfg)
        with temp_new_allowed(default_cfg):
            set_field(default_cfg)
        self.assertTrue("THIS_BETTER_BE_A_NEW_CONFIG" in default_cfg)
        self.assertTrue(default_cfg.THIS_BETTER_BE_A_NEW_CONFIG == 4)

    def test_default_cfg_dump_and_load(self):
        default_cfg = GeneralizedRCNNRunner.get_default_cfg()

        cfg = default_cfg.clone()
        with make_temp_directory("detectron2go_tmp") as tmp_dir:
            file_name = os.path.join(tmp_dir, "config.yaml")
            # this is same as the one in fblearner_launch_utils_detectron2go.py
            with open(file_name, "w") as f:
                f.write(cfg.dump(default_flow_style=False))

            # check if the dumped config file can be merged
            cfg.merge_from_file(file_name)

    def test_default_cfg_deprecated_keys(self):
        default_cfg = GeneralizedRCNNRunner.get_default_cfg()

        # a warning will be printed for deprecated keys
        default_cfg.merge_from_list(["QUANTIZATION.QAT.LOAD_PRETRAINED", True])
        # exception will raise for renamed keys
        self.assertRaises(
            KeyError,
            default_cfg.merge_from_list,
            ["QUANTIZATION.QAT.BACKEND", "fbgemm"],
        )

    def test_merge_from_list_with_new_allowed(self):
        """
        YACS's merge_from_list doesn't take new_allowed into account, D2Go override its behavior, and this test covers it.
        """
        # new_allowed is not set
        cfg = CfgNode()
        cfg.A = CfgNode()
        cfg.A.X = 1
        self.assertRaises(Exception, cfg.merge_from_list, ["A.Y", "2"])

        # new_allowed is set for sub key
        cfg = CfgNode()
        cfg.A = CfgNode(new_allowed=True)
        cfg.A.X = 1
        cfg.merge_from_list(["A.Y", "2"])
        self.assertEqual(cfg.A.Y, 2)  # note that the string will be converted to number
        # however new_allowed is not set for root key
        self.assertRaises(Exception, cfg.merge_from_list, ["B", "3"])


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

    def test_get_diff_cfg(self):
        """check config that is diff from default config, no new keys"""
        # create base config
        cfg1 = CfgNode()
        cfg1.A = CfgNode()
        cfg1.A.Y = 2
        # case 1: new allowed not set, new config has only old keys
        cfg2 = cfg1.clone()
        cfg2.set_new_allowed(False)
        cfg2.A.Y = 3
        gt = CfgNode()
        gt.A = CfgNode()
        gt.A.Y = 3
        self.assertEqual(gt, get_diff_cfg(cfg1, cfg2))

    def test_diff_cfg_no_new_allowed(self):
        """check that if new_allowed is False, new keys cause key error"""
        # create base config
        cfg1 = CfgNode()
        cfg1.A = CfgNode()
        cfg1.A.set_new_allowed(False)
        cfg1.A.Y = 2
        # case 2: new allowed not set, new config has new keys
        cfg2 = cfg1.clone()
        cfg2.A.X = 2
        self.assertRaises(KeyError, get_diff_cfg, cfg1, cfg2)

    def test_diff_cfg_with_new_allowed(self):
        """diff config with new keys and new_allowed set to True"""
        # create base config
        cfg1 = CfgNode()
        cfg1.A = CfgNode()
        cfg1.A.set_new_allowed(True)
        cfg1.A.Y = 2
        # case 3: new allowed set, new config has new keys
        cfg2 = cfg1.clone()
        cfg2.A.X = 2
        gt = CfgNode()
        gt.A = CfgNode()
        gt.A.X = 2
        self.assertEqual(gt, get_diff_cfg(cfg1, cfg2))

    def test_get_cfg_diff_table(self):
        """Check compare two dicts"""
        d1 = {"a0": "a1", "b0": {"b1": "b2"}, "c0": {"c1": {"c2": 3}}}
        d2 = {"a0": "a1", "b0": {"b1": "b3"}, "c0": {"c1": {"c2": 4}}}
        table = get_cfg_diff_table(d1, d2)
        self.assertTrue("a0" not in table)  # a0 are the same
        self.assertTrue("b0.b1" in table)  # b0.b1 are different
        self.assertTrue("c0.c1.c2" in table)  # c0.c1.c2 are different

    def test_get_cfg_diff_table_mismatched_keys(self):
        """Check compare two dicts, the keys are mismatched"""
        d_orig = {"a0": "a1", "b0": {"b1": "b2"}, "c0": {"c1": {"c2": 3}}}
        d_new = {"a0": "a1", "b0": {"b1": "b3"}, "c0": {"c4": {"c2": 4}}}
        table = get_cfg_diff_table(d_new, d_orig)
        self.assertTrue("a0" not in table)  # a0 are the same
        self.assertTrue("b0.b1" in table)  # b0.b1 are different
        self.assertTrue("c0.c1.c2" in table)  # c0.c1.c2 key mismatched
        self.assertTrue("c0.c4.c2" in table)  # c0.c4.c2 key mismatched
        self.assertTrue("Key not exists" in table)  # has mismatched key


class TestAutoScaleWorldSize(unittest.TestCase):
    def test_8gpu_to_1gpu(self):
        """
        when scaling a 8-gpu config to 1-gpu one, the batch size will be reduced by 8x
        """
        cfg = GeneralizedRCNNRunner.get_default_cfg()
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
        cfg = GeneralizedRCNNRunner.get_default_cfg()
        self.assertEqual(cfg.SOLVER.REFERENCE_WORLD_SIZE, 8)
        cfg.SOLVER.REFERENCE_WORLD_SIZE = 0
        batch_size_x8 = cfg.SOLVER.IMS_PER_BATCH
        auto_scale_world_size(cfg, new_world_size=1)
        self.assertEqual(cfg.SOLVER.REFERENCE_WORLD_SIZE, 0)
        self.assertEqual(cfg.SOLVER.IMS_PER_BATCH, batch_size_x8)


class TestConfigDefaultsGen(unittest.TestCase):
    def test_case1(self):

        # register in local scope
        @CONFIG_UPDATER_REGISTRY.register()
        def _test1(cfg):
            cfg.TEST1 = CfgNode()
            cfg.TEST1.X = 1
            return cfg

        @CONFIG_UPDATER_REGISTRY.register()
        def _test2(cfg):
            cfg.TEST2 = CfgNode()
            cfg.TEST2.Y = 2
            return cfg

        filename = get_resource_path("defaults_gen_case1.yaml")
        cfg = load_full_config_from_file(filename)
        default_cfg = cfg.get_default_cfg()
        # default value is 1
        self.assertEqual(default_cfg.TEST1.X, 1)
        self.assertEqual(default_cfg.TEST2.Y, 2)
        # yaml file overwrites it to 3
        self.assertEqual(cfg.TEST1.X, 3)


if __name__ == "__main__":
    unittest.main()
