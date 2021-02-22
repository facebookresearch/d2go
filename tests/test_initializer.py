#!/usr/bin/env python3

import unittest

import d2go.initializer  # noqa
from fvcore.common.file_io import PathHandler, PathManager


class TestInitializer(unittest.TestCase):
    def test_env_patch_path_manager(self):
        self.assertTrue(hasattr(PathManager, "register_handler_original"))
        self.assertNotEqual(
            PathManager.register_handler_original, PathManager.register_handler
        )

        HANDLE_NAME = "_test_handle_://"

        class FakeHandler(PathHandler):
            def __init__(self, name):
                self.name = name

            def _get_supported_prefixes(self):
                return [HANDLE_NAME]

        handler = FakeHandler("h1")
        PathManager.register_handler(handler, allow_override=True)
        self.assertIn(HANDLE_NAME, PathManager._path_handlers.keys())
        self.assertEqual(handler, PathManager._path_handlers[HANDLE_NAME])

        handler1 = FakeHandler("h2")
        PathManager.register_handler(handler1, allow_override=True)
        self.assertEqual(handler, PathManager._path_handlers[HANDLE_NAME])
        self.assertNotEqual(handler1, PathManager._path_handlers[HANDLE_NAME])
