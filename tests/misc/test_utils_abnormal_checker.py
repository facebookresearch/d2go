#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved


import glob
import tempfile
import unittest

import d2go.utils.abnormal_checker as ac
import torch


class Model(torch.nn.Module):
    def forward(self, x):
        return {"loss": x}


class TestUtilsAbnormalChecker(unittest.TestCase):
    def test_utils_abnormal_checker(self):
        counter = 0

        def _writer(all_data):
            nonlocal counter
            counter += 1

        checker = ac.AbnormalLossChecker(-1, writers=[_writer])
        losses = [5, 4, 3, 10, 9, 2, 5, 4]

        for loss in losses:
            checker.check_step({"loss": loss})

        self.assertEqual(counter, 2)

    def test_utils_abnormal_checker_wrapper(self):
        model = Model()

        with tempfile.TemporaryDirectory() as tmp_dir:
            checker = ac.AbnormalLossChecker(-1, writers=[ac.FileWriter(tmp_dir)])
            cmodel = ac.AbnormalLossCheckerWrapper(model, checker)

            losses = [5, 4, 3, 10, 9, 2, 5, 4]
            for loss in losses:
                cur = cmodel(loss)
                cur_gt = model(loss)
                self.assertEqual(cur, cur_gt)

            log_files = glob.glob(f"{tmp_dir}/*.pth")
            self.assertEqual(len(log_files), 2)

            GT_INVALID_INDICES = [3, 6]
            logged_indices = []
            for cur_log_file in log_files:
                cur_log = torch.load(cur_log_file, map_location="cpu")
                self.assertIsInstance(cur_log, dict)
                self.assertIn("data", cur_log)
                logged_indices.append(cur_log["step"])
            self.assertSetEqual(set(logged_indices), set(GT_INVALID_INDICES))
