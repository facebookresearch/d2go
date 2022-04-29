#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import glob
import os
import unittest

import pkg_resources


class TestSubPackageInitFile(unittest.TestCase):
    def test_has_init_files(self):
        """We require every subpackage has an __init__.py file"""
        root = pkg_resources.resource_filename("d2go", "")

        all_py_files = glob.glob(f"{root}/**/*.py", recursive=True)
        all_package_dirs = [os.path.dirname(f) for f in all_py_files]
        all_package_dirs = sorted(set(all_package_dirs))  # dedup

        init_files = [
            os.path.join(os.path.relpath(d, root), "__init__.py")
            for d in all_package_dirs
        ]
        print("Checking following files ...\n{}".format("\n".join(init_files)))
        missing_init_files = [
            f for f in init_files if not os.path.isfile(os.path.join(root, f))
        ]
        self.assertTrue(
            len(missing_init_files) == 0,
            "Missing following __init__.py files:\n{}".format(
                "\n".join(missing_init_files)
            ),
        )
