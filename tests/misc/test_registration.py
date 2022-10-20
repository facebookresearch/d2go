#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import importlib
import logging
import pkgutil
import unittest

# NOTE: don't import anything related to D2/D2Go so that the test of one registry is
# isolated from others


logger = logging.getLogger(__name__)


# copied from https://stackoverflow.com/questions/3365740/how-to-import-all-submodules
def import_submodules(package, recursive=True):
    """Import all submodules of a module, recursively, including subpackages

    :param package: package (name or actual module)
    :type package: str | module
    :rtype: dict[str, types.ModuleType]
    """
    if isinstance(package, str):
        package = importlib.import_module(package)
    results = {}
    for _loader, name, is_pkg in pkgutil.walk_packages(package.__path__):
        full_name = package.__name__ + "." + name
        results[full_name] = importlib.import_module(full_name)
        if recursive and is_pkg:
            results.update(import_submodules(full_name))
    return results


class BaseRegistryPopulationTests(object):
    """
    Test D2Go's core registries are populated once top-level module is imported.
    """

    def get_registered_items(self):
        """return a list of registered items"""
        raise NotImplementedError()

    def import_all_modules(self):
        """import all related modules"""
        raise NotImplementedError()

    def test_is_polulated(self):
        registered_before_import_all = self.get_registered_items()
        self.import_all_modules()
        registered_after_import_all = self.get_registered_items()
        self.assertEqual(registered_before_import_all, registered_after_import_all)


class TestMetaArchRegistryPopulation(unittest.TestCase, BaseRegistryPopulationTests):
    def get_registered_items(self):
        from d2go.registry.builtin import META_ARCH_REGISTRY

        return [k for k, v in META_ARCH_REGISTRY]

    def import_all_modules(self):
        import d2go.modeling

        import_submodules(d2go.modeling)


class TestDataMapperRegistryPopulation(unittest.TestCase, BaseRegistryPopulationTests):
    def get_registered_items(self):
        from d2go.data.dataset_mappers import D2GO_DATA_MAPPER_REGISTRY

        return [k for k, v in D2GO_DATA_MAPPER_REGISTRY]

    def import_all_modules(self):
        import d2go.data.dataset_mappers

        import_submodules(d2go.data.dataset_mappers)
