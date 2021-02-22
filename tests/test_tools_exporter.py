#!/usr/bin/env python3

import os
import unittest

from d2go.tools.exporter import get_parser, run_with_cmdline_args
import torch
from mobile_cv.common.misc.file_utils import make_temp_directory


ON_DEVSERVER = torch.cuda.device_count() > 0


class TestToolsExporter(unittest.TestCase):
    # @unittest.skipIf(not ON_DEVSERVER, "Test only on devserver")
    def test_tools_exporter(self):
        with make_temp_directory("detectron2go_tmp_export") as output_dir:
            cfg_file = "detectron2go://e2e_faster_rcnn_fbnet.yaml"
            args = [
                "--config-file",
                cfg_file,
                "--predictor-types",
                "torchscript",
                "torchscript_int8",
                "caffe2",
                # TODO: support logdb
                # "logdb",
                "--output-dir",
                output_dir,
                "D2GO_DATA.TEST.MAX_IMAGES",
                "1",
                "SOLVER.IMS_PER_BATCH",
                "1",
                "DATASETS.TRAIN",
                "('coco_2017_val_100',)",
            ]
            ret = run_with_cmdline_args(get_parser().parse_args(args))
            out_paths = ret["predictor_paths"]
            self.assertEqual(len(out_paths), 3)
            self.assertSetEqual(
                set(out_paths.keys()),
                {"torchscript", "torchscript_int8", "caffe2"},
            )
            for _, path in out_paths.items():
                self.assertTrue(os.path.exists(path))

    def test_tools_exporter_qat(self):
        with make_temp_directory("detectron2go_tmp_export") as output_dir:
            cfg_file = "detectron2go://e2e_faster_rcnn_fbnet.yaml"
            args = [
                "--config-file",
                cfg_file,
                "--predictor-types",
                "torchscript",
                "torchscript_int8",
                "--output-dir",
                output_dir,
                "D2GO_DATA.TEST.MAX_IMAGES",
                "1",
                "SOLVER.IMS_PER_BATCH",
                "1",
                "DATASETS.TRAIN",
                "('coco_2017_val_100',)",
                "QUANTIZATION.QAT.ENABLED",
                "True",
            ]
            ret = run_with_cmdline_args(get_parser().parse_args(args))
            out_paths = ret["predictor_paths"]
            self.assertEqual(len(out_paths), 2)
            self.assertSetEqual(
                set(out_paths.keys()),
                {"torchscript", "torchscript_int8"},
            )
            for _, path in out_paths.items():
                self.assertTrue(os.path.exists(path))

    # @unittest.skipIf(not ON_DEVSERVER, "Test only on devserver")
    def test_tools_exporter_resnet(self):
        with make_temp_directory("detectron2go_tmp_export") as output_dir:
            cfg_file = "detectron2://COCO-Detection/faster_rcnn_R_50_C4_1x.yaml"
            args = [
                "--config-file",
                cfg_file,
                "--output-dir",
                output_dir,
                "--predictor-types",
                "torchscript",
                "torchscript_int8",
                "caffe2",
                "--skip-if-fail",
                "D2GO_DATA.TEST.MAX_IMAGES",
                "1",
                "SOLVER.IMS_PER_BATCH",
                "1",
                "DATASETS.TRAIN",
                "('coco_2017_val_100',)",
                "MODEL.WEIGHTS",
                "''",
            ]
            ret = run_with_cmdline_args(get_parser().parse_args(args))
            out_paths = ret["predictor_paths"]
            # int8 conversion will fail so only two exported predictors
            self.assertEqual(len(out_paths), 2)
            self.assertSetEqual(set(out_paths.keys()), {"torchscript", "caffe2"})
            for _, path in out_paths.items():
                self.assertTrue(os.path.exists(path))
