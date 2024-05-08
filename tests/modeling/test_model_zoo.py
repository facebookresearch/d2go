#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import unittest

import torch.nn as nn
from d2go.model_zoo import model_zoo
from mobile_cv.common.misc.oss_utils import is_oss


class TestD2GoModelZoo(unittest.TestCase):
    @unittest.skipIf(is_oss(), "Model Zoo is not available in OSS")
    def test_model_zoo_pretrained(self):
        configs = list(model_zoo._ModelZooUrls.CONFIG_PATH_TO_URL_SUFFIX.keys())
        for cfgfile in configs:
            model = model_zoo.get(cfgfile, trained=True)
            self.assertTrue(isinstance(model, nn.Module))


if __name__ == "__main__":
    unittest.main()
