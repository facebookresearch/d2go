#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved


import glob
import logging
import os
import unittest

from d2go.config import auto_scale_world_size, reroute_config_path
from d2go.runner import GeneralizedRCNNRunner
from mobile_cv.common.misc.file_utils import make_temp_directory


logger = logging.getLogger(__name__)


class TestConfigs(unittest.TestCase):
    def test_configs_load(self):
        """ Make sure configs are loadable """

        for location in ["detectron2", "detectron2go"]:
            root_dir = os.path.abspath(reroute_config_path(f"{location}://."))
            files = glob.glob(os.path.join(root_dir, "**/*.yaml"), recursive=True)
            self.assertGreater(len(files), 0)
            for fn in sorted(files):
                logger.info("Loading {}...".format(fn))
                GeneralizedRCNNRunner().get_default_cfg().merge_from_file(fn)

    def test_arch_def_loads(self):
        """ Test arch def str-to-dict conversion compatible with merging """
        default_cfg = GeneralizedRCNNRunner().get_default_cfg()
        cfg = default_cfg.clone()
        cfg.merge_from_file(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                         "resources/arch_def_merging.yaml"))

        with make_temp_directory("detectron2go_tmp") as tmp_dir:
            # Dump out config with arch def
            file_name = os.path.join(tmp_dir, "test_archdef_config.yaml")
            with open(file_name, "w") as f:
                f.write(cfg.dump())

            # Attempt to reload the config
            another_cfg = default_cfg.clone()
            another_cfg.merge_from_file(file_name)

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
