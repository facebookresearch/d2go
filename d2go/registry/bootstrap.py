#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import ast
import builtins
import contextlib
import glob
import logging
import os
import time
import traceback

import pkg_resources
from mobile_cv.common.misc.py import MoreMagicMock
from mobile_cv.common.misc.registry import (
    CLASS_OR_FUNCTION_TYPES,
    LazyRegisterable,
    Registry,
)

logger = logging.getLogger(__name__)


orig_import = builtins.__import__
orig_open = builtins.open
orig__register = Registry._register
_INSIDE_BOOTSTRAP = False
_IS_BOOTSTRAPPED = False

_BOOTSTRAP_PACKAGE = "d2go.registry._bootstrap"


def _log(lvl, msg):
    _VERBOSE_LEVEL = 0

    if _VERBOSE_LEVEL >= lvl:
        print(msg)


class _catchtime:
    def __enter__(self):
        self.time = time.perf_counter()
        return self

    def __exit__(self, type, value, traceback):
        self.time = time.perf_counter() - self.time


def _match(name, module_full_name, match_submodule=False):
    if name == module_full_name:
        return True

    if match_submodule:
        if name.startswith(module_full_name + "."):
            return True

    return False


def _match_any(name, module_full_names, match_submodule=False):
    return any(
        _match(name, module_full_name, match_submodule=match_submodule)
        for module_full_name in module_full_names
    )


def _import_mock(name, globals=None, locals=None, fromlist=(), level=0):

    use_orig_import = False

    # enable some first-party packages
    if _match_any(
        name,
        [
            # allow using pdb during patch
            "pdb",
            "readline",
            "linecache",
            "reprlib",
            "io",
            # allow using builtins.__import__
            "builtins",
        ],
    ):
        use_orig_import = True

    # enable some known third-party packages, these pacakges might have been imported
    if _match_any(
        name,
        [
            # "torch",
            # "numpy",
            # "mobile_cv.arch.fbnet_v2.modeldef_utils",
        ],
    ):
        use_orig_import = True

    # enable modules under d2go.registry
    if _match(name, "d2go.registry", match_submodule=True):
        use_orig_import = True

    if use_orig_import:
        # import as normal
        return orig_import(name, globals, locals, fromlist=fromlist, level=level)
    else:
        # return a Mock instead of making a real import
        _log(2, f"mock import: {name}; fromlist={fromlist}; level={level}")
        m = MoreMagicMock()
        return m


def _open_mock(*args, **kwargs):
    return MoreMagicMock()


def _register_mock(self, name, obj):
    """Convert `obj` to LazyRegisterable"""

    # Instead of register the (possibly mocked) object which is created under the
    # "fake" package _BOOTSTRAP_PACKAGE, register a lazy-object (i.e. a string) pointing
    # to its original (possibly un-imported) module.
    def _resolve_real_module(module_in_bootstrap_package):
        assert module_in_bootstrap_package.startswith(_BOOTSTRAP_PACKAGE + ".")
        orig_module = module_in_bootstrap_package[len(_BOOTSTRAP_PACKAGE + ".") :]
        return orig_module

    if isinstance(obj, MoreMagicMock):
        assert obj.mocked_obj_info is not None, obj
        if name is None:
            name = obj.mocked_obj_info["__name__"]
        obj = LazyRegisterable(
            module=_resolve_real_module(obj.mocked_obj_info["__module__"]),
            name=obj.mocked_obj_info["__qualname__"],
        )
    elif isinstance(obj, LazyRegisterable):
        pass
    else:
        assert isinstance(obj, CLASS_OR_FUNCTION_TYPES), obj
        if name is None:
            name = obj.__name__
        obj = LazyRegisterable(
            module=_resolve_real_module(obj.__module__), name=obj.__qualname__
        )

    return orig__register(self, name, obj)


@contextlib.contextmanager
def _bootstrap_patch():
    global _INSIDE_BOOTSTRAP

    builtins.__import__ = _import_mock
    builtins.open = _open_mock
    Registry._register = _register_mock
    _INSIDE_BOOTSTRAP = True

    try:
        yield
    finally:
        builtins.__import__ = orig_import
        builtins.open = orig_open
        Registry._register = orig__register
        _INSIDE_BOOTSTRAP = False


def _bootstrap_file(filename):
    # convert absolute path to full module name
    # eg. ".../d2go/a/b/c.py" -> "d2go.a.b.c"
    # eg. ".../d2go/a/b/__init__.py" -> "d2go.a.b"
    package_root = os.path.dirname(pkg_resources.resource_filename("d2go", ""))
    assert filename.startswith(package_root), (filename, package_root)
    rel_path = os.path.relpath(filename, package_root)
    assert rel_path.endswith(".py")
    module = rel_path[: -len(".py")]
    if module.endswith("/__init__"):
        module = module[: -len("/__init__")]
    module = module.replace("/", ".")

    exec_globals = {
        "__file__": filename,
        # execute in a "fake" package to minimize potential side effect
        "__name__": "{}.{}".format(_BOOTSTRAP_PACKAGE, module),
    }

    with _catchtime() as t:
        with open(filename) as f:
            content = f.read()
        tree = ast.parse(content)

        # HACK: convert multiple inheritance to single inheritance, this is needed
        # because current implementation of MoreMagicMock can't handle this well.
        # eg. `class MyClass(MyMixin, nn.Module)` -> `class MyClass(MyMixin)`
        for stmt in tree.body:
            if isinstance(stmt, ast.ClassDef):
                if len(stmt.bases) > 1:
                    stmt.bases = stmt.bases[:1]
                stmt.keywords.clear()

    _log(2, f"Parsing AST takes {t.time} sec")

    with _catchtime() as t:
        with _bootstrap_patch():
            exec(compile(tree, filename, "exec"), exec_globals)  # noqa
    _log(2, f"Execute file takes {t.time} sec")


class _BootstrapBreakException(Exception):
    pass


def break_bootstrap():
    """
    In case the file can't be perfectly executed by `_bootstrap_file`, users can call
    this function to break the process. Because the remaining content in the file will
    be skipped, avoid using registration statement after calling this function.
    """

    if _INSIDE_BOOTSTRAP:
        # raise a special exception which will be catched later
        raise _BootstrapBreakException()

    # non-op outside of bootstrap
    return


def bootstrap_registries(catch_exception=True):
    """
    Bootstrap all registries so that all objects are effectively registered.

    This function will "import" all the files from certain locations (eg. d2go package)
    and look for a set of known registries (eg. d2go's builtin registries). The "import"
    should not have any side effect, which is achieved by mocking builtin.__import__.
    """

    global _IS_BOOTSTRAPPED
    if _IS_BOOTSTRAPPED:
        logger.warning("Registries are already bootstrapped, skipped!")
        return

    if _INSIDE_BOOTSTRAP:
        _log(1, "calling bootstrap_registries() inside bootstrap process, skip ...")
        return

    start = time.perf_counter()

    # locate all the files under d2go package
    # NOTE: we may extend to support user-defined locations if necessary
    d2go_root = pkg_resources.resource_filename("d2go", "")
    logger.info(f"Start bootstrapping for d2go_root: {d2go_root} ...")
    all_files = glob.glob(f"{d2go_root}/**/*.py", recursive=True)

    skip_files = []
    exception_files = []
    time_per_file = {}
    for filename in all_files:
        _log(
            1,
            f"bootstrap for file under d2go_root: {os.path.relpath(filename, d2go_root)}",
        )

        with _catchtime() as t:
            try:
                _bootstrap_file(filename)
            except _BootstrapBreakException:
                # the bootstrap process is manually skipped
                skip_files.append(filename)
                continue
            except Exception as e:
                if catch_exception:
                    _log(
                        1,
                        "Encountered the following error during bootstrap:"
                        + "".join(
                            traceback.format_exception(type(e), e, e.__traceback__)
                        ),
                    )
                    exception_files.append(filename)
                else:
                    raise e
        time_per_file[filename] = t.time

    end = time.perf_counter()
    duration = end - start
    logger.info(
        f"Finished bootstrapping for {len(all_files)} files ({len(skip_files)} break-ed)"
        f" in {duration:.2f} seconds."
    )
    if len(exception_files) > 0:
        logger.warning(
            "Encountered error bootstrapping following {} files,"
            " registration inside those files might not work!\n{}".format(
                len(exception_files),
                "\n".join(exception_files),
            )
        )

    # Log slowest Top-N files
    TOP_N = 100
    _log(2, f"Top-{TOP_N} slowest files during bootstrap:")
    all_time = [(os.path.relpath(k, d2go_root), v) for k, v in time_per_file.items()]
    for x in sorted(all_time, key=lambda x: x[1])[-TOP_N:]:
        _log(2, x)

    _IS_BOOTSTRAPPED = True
