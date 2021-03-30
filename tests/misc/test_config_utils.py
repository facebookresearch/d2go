#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved


import copy
import unittest

from d2go.config.utils import (
    config_dict_to_list_str,
    flatten_config_dict,
    str_wrap_fbnet_arch_def,
)


class TestConfigUtils(unittest.TestCase):
    def test_str_wrap_fbnet_arch_def(self):
        """Check that fbnet modeldef converted to str"""
        d = {"MODEL": {"FBNET_V2": {"ARCH_DEF": {"key0": "val0"}}}}
        new_dict = str_wrap_fbnet_arch_def(d)
        gt = {"MODEL": {"FBNET_V2": {"ARCH_DEF": """'{"key0": "val0"}'"""}}}
        self.assertEqual(new_dict, gt)
        self.assertNotEqual(d, new_dict)

        # check only fbnet arch is changed
        d = {"a0": "a1", "b0": {"b1": "b2"}}
        gt = copy.deepcopy(d)
        new_dict = str_wrap_fbnet_arch_def(d)
        self.assertEqual(new_dict, gt)

    def test_flatten_config_dict(self):
        """Check flatten config dict to single layer dict"""
        d = {"a0": "a1", "b0": {"b1": "b2"}, "c0": {"c1": {"c2": 3}}}
        fdict = flatten_config_dict(d)
        gt = {"a0": "a1", "b0.b1": "b2", "c0.c1.c2": 3}
        self.assertEqual(fdict, gt)

    def test_config_dict_to_list_str(self):
        """Check convert config dict to str list"""
        d = {"a0": "a1", "b0": {"b1": "b2"}, "c0": {"c1": {"c2": 3}}}
        str_list = config_dict_to_list_str(d)
        gt = ["a0", "a1", "b0.b1", "b2", "c0.c1.c2", "3"]
        self.assertEqual(str_list, gt)
