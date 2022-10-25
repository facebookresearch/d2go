#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import importlib
import sys
import unittest

from d2go.initializer import initialize_all
from d2go.registry import bootstrap

# manual initialize without bootstrap
initialize_all(boostrap_registries=False)


def _unimport(package_name):
    # remove sub modules from sys
    modules = [
        key
        for key in sys.modules
        if (
            (key == package_name or key.startswith(package_name + "."))
            # previent the parent package of this file being removed
            and not __name__.startswith(key)
        )
    ]
    for key in sorted(modules, reverse=True):
        sys.modules.pop(key)

    # invalidate the cache of removed sub modules
    importlib.invalidate_caches()


class TestRegistryBootstrap(unittest.TestCase):
    def setUp(self):
        # NOTE: reload this file since the imported modules (eg. `d2go.registry.bootstrap`)
        # might be "unimported" during `tearDown`.
        importlib.reload(sys.modules[__name__])

    def tearDown(self):
        # NOTE: "unimport" bootstrapped libraries, so that each test runs like starting
        # a new python program.

        # TODO: match list with the bootstrapped packages
        _unimport("d2go.registry")
        _unimport("mobile_cv")
        _unimport("detectron2")

    def test_bootstrap_core_lib(self):
        self.assertFalse(bootstrap._IS_BOOTSTRAPPED)
        bootstrap.bootstrap_registries(enable_cache=False, catch_exception=False)
        self.assertTrue(bootstrap._IS_BOOTSTRAPPED)

    def test_bootstrap_with_cache(self):
        self.assertFalse(bootstrap._IS_BOOTSTRAPPED)
        bootstrap.bootstrap_registries(enable_cache=True, catch_exception=False)
        self.assertTrue(bootstrap._IS_BOOTSTRAPPED)
