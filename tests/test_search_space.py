#!/usr/bin/env python3

# Unittests to check using the D2GO specific chamnet registries
# Should be able to build search space, predefined arch, filter funcs
# by name
#
# Registers are global so we reregister unittest values with a different name
# in each test

import unittest

import chamnet.search_space as ss
from d2go.search_spaces.build import (
    D2GO_PREDEFINED_ARCH_REGISTRY,
    D2GO_SEARCH_SPACE_FILTER_REGISTRY,
    D2GO_SEARCH_SPACE_REGISTRY,
    build_search_space,
)


TEST_SEARCH_SPACE = {
    "MODEL": {
        "FBNET_V2": {
            # pyre-fixme
            "ARCH_DEF": [ss.SR(1, 101, 1, 1), ss.SR(1, 101, 1, 1), ss.SR(1, 101, 1, 1)]
        }
    }
}

TEST_PREDEFINED_ARCHS = [{"MODEL": {"FBNET_V2": {"ARCH_DEF": [1, 1, 1]}}}]


class TestSearchSpace(unittest.TestCase):
    def test_add_search_space(self):
        """Check registration of a search space"""
        fname = "test_add_search_space"
        D2GO_SEARCH_SPACE_REGISTRY.register(fname, TEST_SEARCH_SPACE)
        self.assertEqual(D2GO_SEARCH_SPACE_REGISTRY.get(fname), TEST_SEARCH_SPACE)

    def test_build_search_space(self):
        """Check that search space can be built"""
        fname = "test_build_search_space"
        D2GO_SEARCH_SPACE_REGISTRY.register(fname, TEST_SEARCH_SPACE)
        search_space = build_search_space(fname)
        self.assertEqual(search_space.search_space, TEST_SEARCH_SPACE)
        self.assertEqual(search_space.get_combination_count(), 10 ** 6)

    def test_add_predefined_arch(self):
        """Check registration of a predefined arch"""
        fname = "test_add_predefined_arch"
        D2GO_PREDEFINED_ARCH_REGISTRY.register(fname, TEST_PREDEFINED_ARCHS)
        self.assertEqual(
            D2GO_PREDEFINED_ARCH_REGISTRY.get(fname), TEST_PREDEFINED_ARCHS
        )

    def test_build_predefined_arch(self):
        """Check build the search space with predefined arch"""
        fname = "test_build_predefined_arch"
        D2GO_SEARCH_SPACE_REGISTRY.register(fname, TEST_SEARCH_SPACE)
        D2GO_PREDEFINED_ARCH_REGISTRY.register(fname, TEST_PREDEFINED_ARCHS)
        search_space = build_search_space(fname, predefined_arch_name=fname)
        self.assertEqual(search_space.predefined_archs, TEST_PREDEFINED_ARCHS)

    def test_add_filter(self):
        """Check that filter function can be registered"""

        @D2GO_SEARCH_SPACE_FILTER_REGISTRY.register()
        def _filter_test_add_filter():
            return True

        self.assertEqual(
            D2GO_SEARCH_SPACE_FILTER_REGISTRY.get("_filter_test_add_filter"),
            _filter_test_add_filter,
        )

    def test_build_filter(self):
        """Check building search space with filter function"""
        fname = "test_build_filter"
        D2GO_SEARCH_SPACE_REGISTRY.register(fname, TEST_SEARCH_SPACE)
        D2GO_PREDEFINED_ARCH_REGISTRY.register(fname, TEST_PREDEFINED_ARCHS)

        @D2GO_SEARCH_SPACE_FILTER_REGISTRY.register()
        def _filter_test_build_filter():
            return True

        search_space = build_search_space(
            fname, predefined_arch_name=fname, filter_name="_filter_test_build_filter"
        )
        self.assertEqual(search_space.filter_func, _filter_test_build_filter)
        self.assertTrue(search_space.filter_func())
