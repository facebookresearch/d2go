#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import ast
import builtins
import contextlib
import glob
import hashlib
import logging
import os
import tempfile
import time
import traceback
from collections import defaultdict
from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple

import pkg_resources
import yaml
from mobile_cv.common.misc.py import dynamic_import, MoreMagicMock
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
_BOOTSTRAP_CACHE_FILENAME = "registry_bootstrap.v1.yaml"


def _log(lvl: int, msg: str):
    _VERBOSE_LEVEL = 0

    if _VERBOSE_LEVEL >= lvl:
        print(msg)


# Simple version copied from fvcore/iopath
def _get_cache_dir() -> str:
    cache_dir = os.path.expanduser("~/.torch/d2go_cache")
    try:
        os.makedirs(cache_dir, exist_ok=True)
        assert os.access(cache_dir, os.R_OK | os.W_OK | os.X_OK)
    except (OSError, AssertionError):
        tmp_dir = os.path.join(tempfile.gettempdir(), "d2go_cache")
        logger.warning(f"{cache_dir} is not accessible! Using {tmp_dir} instead!")
        os.makedirs(tmp_dir, exist_ok=True)
        cache_dir = tmp_dir
    return cache_dir


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


def _register_mock(self, name: Optional[str], obj: Any) -> None:
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

    assert isinstance(obj, LazyRegisterable)
    # During bootstrap, it's possible that the object is already registered
    # (as non-lazy), because importing a library first and then bootstramp it. Simply
    # skip the lazy-registration.
    if name in self and not isinstance(self[name], LazyRegisterable):
        if self[name].__module__ == obj.module and (
            obj.name is None or self[name].__name__ == obj.name
        ):
            _log(2, f"{obj} has already registered as {self[name]}, skip...")
            return

    orig__register(self, name, obj)


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


def _get_registered_names() -> Dict[str, List[str]]:
    """Return the currently registered names for each registry"""
    # NOTE: currently only support D2Go's builtin registry module, which can be extended
    # in future.
    import d2go.registry.builtin

    modules = [
        d2go.registry.builtin,
    ]

    registered = {}
    for module in modules:
        registered_in_module = {
            f"{module.__name__}.{name}": obj.get_names()
            for name, obj in module.__dict__.items()
            if isinstance(obj, Registry)
        }
        registered.update(registered_in_module)

    return registered


class BootstrapStatus(Enum):
    CACHED = 0
    FULLY_IMPORTED = 1
    PARTIALLY_IMPORTED = 2
    FAILED = 3


@dataclass
class CachedResult:
    sha1: str
    registered: Dict[str, str]
    status: str  # string representation of BootstrapStatus


def _bootstrap_file(
    rel_path: str,
    catch_exception: bool,
    cached_result: Optional[CachedResult] = None,
) -> Tuple[CachedResult, BootstrapStatus]:
    # convert relative path to full module name
    # eg. ".../d2go/a/b/c.py" -> "d2go.a.b.c"
    # eg. ".../d2go/a/b/__init__.py" -> "d2go.a.b"
    package_root = os.path.dirname(pkg_resources.resource_filename("d2go", ""))
    filename = os.path.join(package_root, rel_path)
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

        file_hash = hashlib.sha1(content.encode("utf-8")).hexdigest()
        if cached_result is not None and file_hash == cached_result.sha1:
            _log(
                2,
                f"Hash {file_hash} matches, lazy registering cached registerables ...",
            )
            registerables = cached_result.registered
            for registry_module_dot_name, names_to_register in registerables.items():
                registry = dynamic_import(registry_module_dot_name)
                for name in names_to_register:
                    # we only store the registered name in the cache, here we know the
                    # module of bootstrapped file, which should be sufficient.
                    registry.register(name, LazyRegisterable(module=module))
            return cached_result, BootstrapStatus.CACHED

        tree = ast.parse(content)

        # HACK: convert multiple inheritance to single inheritance, this is needed
        # because current implementation of MoreMagicMock can't handle this well.
        # eg. `class MyClass(MyMixin, nn.Module)` -> `class MyClass(MyMixin)`
        def _truncate_multiple_inheritance(ast_tree):
            for stmt in ast_tree.body:
                if isinstance(stmt, ast.ClassDef):
                    if len(stmt.bases) > 1:
                        stmt.bases = stmt.bases[:1]
                    stmt.keywords.clear()
                    _truncate_multiple_inheritance(stmt)

        _truncate_multiple_inheritance(tree)

    _log(2, f"Parsing AST takes {t.time} sec")

    prev_registered = _get_registered_names()
    with _catchtime() as t:
        try:
            with _bootstrap_patch():
                exec(compile(tree, filename, "exec"), exec_globals)  # noqa
            status = BootstrapStatus.FULLY_IMPORTED
        except _BootstrapBreakException:
            status = BootstrapStatus.PARTIALLY_IMPORTED
        except Exception as e:
            if catch_exception:
                _log(
                    1,
                    "Encountered the following error during bootstrap:"
                    + "".join(traceback.format_exception(type(e), e, e.__traceback__)),
                )
            else:
                raise e
            status = BootstrapStatus.FAILED
    _log(2, f"Execute file takes {t.time} sec")

    # compare and get the newly registered
    cur_registered = _get_registered_names()
    assert set(cur_registered.keys()) == set(prev_registered.keys())
    newly_registered = {
        k: sorted(set(cur_registered[k]) - set(prev_registered[k]))
        for k in sorted(cur_registered.keys())
    }
    newly_registered = {k: v for k, v in newly_registered.items() if len(v) > 0}
    result = CachedResult(
        sha1=file_hash,
        registered=newly_registered,
        status=status.name,
    )
    return result, status


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


def lazy_on_bootstrap(f: Callable) -> Callable:
    """
    A decorator to mark a function as "lazy" during bootstrap, such that the decorated
    function will skip the execution and immediately return a MagicMock object during
    the bootstrap (the decorator is a non-op outside of bootstrap). This can be used to
    hide un-executable code (usually related to import-time computation) during the
    bootstrap.

    For registration related import-time computation, please consider using the
    `LazyRegisterable` since it will also save time for the normal import.
    """

    def wrapped(*args, **kwargs):
        if _INSIDE_BOOTSTRAP:
            return MoreMagicMock()
        else:
            return f(*args, **kwargs)

    return wrapped


def _load_cached_results(filename: str) -> Dict[str, CachedResult]:
    with open(filename) as f:
        content = f.read()
        loaded = yaml.safe_load(content)
    assert isinstance(loaded, dict), f"Wrong format: {content}"
    results = {
        filename: CachedResult(**result_dic) for filename, result_dic in loaded.items()
    }
    return results


def _dump_cached_results(cached_results: Dict[str, CachedResult], filename: str):
    results_dict = {
        filename: asdict(result_dic) for filename, result_dic in cached_results.items()
    }
    dumped = yaml.safe_dump(results_dict)
    with open(filename, "w") as f:
        f.write(dumped)


def bootstrap_registries(enable_cache: bool = True, catch_exception: bool = True):
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

    # load cached bootstrap results if exist
    cached_bootstrap_results: Dict[str, CachedResult] = {}
    if enable_cache:
        filename = os.path.join(_get_cache_dir(), _BOOTSTRAP_CACHE_FILENAME)
        if os.path.isfile(filename):
            logger.info(f"Loading bootstrap cache at {filename} ...")
            cached_bootstrap_results = _load_cached_results(filename)
        else:
            logger.info(
                f"Can't find the bootstrap cache at {filename}, start from scratch"
            )

    # locate all the files under d2go package
    # NOTE: we may extend to support user-defined locations if necessary
    d2go_root = pkg_resources.resource_filename("d2go", "")
    logger.info(f"Start bootstrapping for d2go_root: {d2go_root} ...")
    all_files = glob.glob(f"{d2go_root}/**/*.py", recursive=True)
    all_files = [os.path.relpath(x, os.path.dirname(d2go_root)) for x in all_files]

    new_bootstrap_results: Dict[str, CachedResult] = {}
    files_per_status = defaultdict(list)
    time_per_file = {}
    for filename in all_files:
        _log(1, f"bootstrap for file: {filename}")

        cached_result = cached_bootstrap_results.get(filename, None)
        with _catchtime() as t:
            result, status = _bootstrap_file(filename, catch_exception, cached_result)
        new_bootstrap_results[filename] = result
        files_per_status[status].append(filename)
        time_per_file[filename] = t.time

    end = time.perf_counter()
    duration = end - start
    status_breakdown = ", ".join(
        [f"{len(files_per_status[status])} {status.name}" for status in BootstrapStatus]
    )
    logger.info(
        f"Finished bootstrapping for {len(all_files)} files ({status_breakdown})"
        f" in {duration:.2f} seconds."
    )
    exception_files = [
        filename
        for filename, result in new_bootstrap_results.items()
        if result.status == BootstrapStatus.FAILED.name
    ]
    if len(exception_files) > 0:
        logger.warning(
            "Found exception for the following {} files (either during this bootstrap"
            " run or from previous cached result), registration inside those files"
            " might not work!\n{}".format(
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

    if enable_cache:
        filename = os.path.join(_get_cache_dir(), _BOOTSTRAP_CACHE_FILENAME)
        logger.info(f"Writing updated bootstrap results to {filename} ...")
        _dump_cached_results(new_bootstrap_results, filename)

    _IS_BOOTSTRAPPED = True
