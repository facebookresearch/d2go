#!/usr/bin/env python3

import os
import unittest
from dataclasses import dataclass, field
from typing import List

import d2go.setup as d2go_setup
from d2go.config import CfgNode
from d2go.runner.default_runner import GeneralizedRCNNRunner
from mobile_cv.common.misc.file_utils import make_temp_directory


@dataclass
class DummyArgs:
    runner: str = "d2go.runner.GeneralizedRCNNRunner"
    config_file: str = ""
    output_dir: str = "dummy_outputdir"
    opts: List = field(default_factory=list)
    datasets: List = field(default_factory=list)
    min_size: int = 0
    max_size: int = 0


class TestSetup(unittest.TestCase):
    def test_prepare_for_launch(self):
        """Check that prepare_for_launch returns cfg, output_dir, runner"""
        args = DummyArgs()
        cfg, output_dir, runner = d2go_setup.prepare_for_launch(args)
        self.assertTrue(isinstance(cfg, CfgNode))
        self.assertEqual(output_dir, args.output_dir)
        self.assertTrue(isinstance(runner, GeneralizedRCNNRunner))

    def test_prepare_for_launch_config_files(self):
        """Check different locations of config files"""
        args = DummyArgs()

        # create local config
        default_cfg = GeneralizedRCNNRunner().get_default_cfg()
        cfg = default_cfg.clone()
        with make_temp_directory("detectron2go_tmp") as tmp_dir:
            local_cfg_fname = os.path.join(tmp_dir, "config.yaml")
            with open(local_cfg_fname, "w") as f:
                f.write(cfg.dump(default_flow_style=False))

            # check different config_files
            for config_file in [
                "",
                local_cfg_fname,
                "manifold://mobile_vision_tests/tree/d2go/unittest_resources/configs/generalized_rcnn_default.yaml",
            ]:
                args.config_file = config_file
                cfg, output_dir, runner = d2go_setup.prepare_for_launch(args)
                self.assertTrue(isinstance(cfg, CfgNode))
