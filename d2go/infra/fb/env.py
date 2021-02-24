#!/usr/bin/env python3

import logging
from collections import OrderedDict

from fvcore.common.file_io import PathHandler, PathManager
from detectron2.utils.file_io import PathManager as d2_PathManager
from mobile_cv.common.fb.everstore import EverstorePathHandler
from ...data.portal_ent_pathhandler import PortalEntPathHandler

_ENV_SETUP_DONE = False


def setup_environment():
    global _ENV_SETUP_DONE
    if _ENV_SETUP_DONE:
        return
    _ENV_SETUP_DONE = True

    from detectron2.fb.env import setup_environment

    setup_environment()
    _setup_d2go_environment()


def _setup_d2go_environment():
    # register for both the global pm, and d2's pm.
    # TODO avoid conflicts and don't use global pm.
    # See comments in patch_path_manager as well.
    for pm in [PathManager, d2_PathManager]:
        pm.register_handler(
            EverstorePathHandler(
                default_context="d2go/everstore",
                memcache_key_prefix="d2go/everstore_memcache",
                memcache_log_freq=10000,
            ),
            allow_override=True,
        )
        pm.register_handler(
            PortalEntPathHandler(
                memcache_key_prefix="d2go/portal_fbid_memcache",
                memcache_log_freq=10000,
            ),
            allow_override=True,
        )
    patch_path_manager()


def patch_path_manager():
    """
    This is a temporary fix to make sure the pre-registered handlers (currently
    ManifoldHandler) will not be overridden by other imported code.
    The reason for doing so is that other code registering ManifoldHandler may not
    be specific the `memcache_prefix` that causes memcache to be disabled, or
    specify a different `memcache_prefix` that causes other projects' memcache
    could not be reused.
    Using a predefined `memcache_prefix` and not allow others to override it
    seems like a reasonable way to address this issue in the short term as the
    path are finally concatenated with memcahce_prefix to to a unique path and
    manifold path itself is unique.

    This code should be removed because PathManager can be non-global after D21238755.
    """

    def _register_handler_once(
        handler: PathHandler, allow_override: bool = True
    ) -> None:
        """ Skip register handlers if it has been registered
            allow_override is not used
        """
        assert hasattr(PathManager, "_path_handlers")
        assert isinstance(PathManager._path_handlers, OrderedDict)

        assert isinstance(handler, PathHandler), handler
        for prefix in handler._get_supported_prefixes():
            if prefix not in PathManager._path_handlers:
                PathManager._path_handlers[prefix] = handler
            else:
                logger = logging.getLogger(__name__)
                logger.warning(f"Handler {handler} for {prefix} has existed. Skipped.")

        # Sort path handlers in reverse order so longer prefixes take priority,
        # eg: http://foo/bar before http://foo
        PathManager._path_handlers = OrderedDict(
            sorted(PathManager._path_handlers.items(), key=lambda t: t[0], reverse=True)
        )

    PathManager.register_handler_original = PathManager.register_handler
    PathManager.register_handler = _register_handler_once
