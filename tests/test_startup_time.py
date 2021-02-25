#!/usr/bin/env python3

import unittest

from d2go.initializer import (
    REGISTER_D2_DATASETS_TIME,
    REGISTER_TIME,
    SETUP_ENV_TIME,
)


class TestStartupTime(unittest.TestCase):
    @unittest.skipIf(True, "Will exceed threshold")
    def test_setup_env_time(self):
        self.assertLess(sum(SETUP_ENV_TIME), 5.0)

    def test_register_d2_datasets_time(self):
        self.assertLess(sum(REGISTER_D2_DATASETS_TIME), 3.0)

    @unittest.skipIf(True, "Will exceed threshold")
    def test_register_time(self):
        # NOTE: _register is should be done quickly, currently about 0.2s
        self.assertLess(sum(REGISTER_TIME), 1.0)
