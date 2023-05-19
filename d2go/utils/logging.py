# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import builtins
import logging
import sys
import time
import uuid
from functools import wraps
from typing import Any, Callable, Optional, TypeVar

from mobile_cv.common.misc.oss_utils import fb_overwritable


# Saving the builtin print to wrap it up later.
BUILTIN_PRINT = builtins.print

_T = TypeVar("_T")


@fb_overwritable()
def initialize_logging(logging_level: int) -> None:
    root_logger = logging.getLogger()
    root_logger.setLevel(logging_level)


def replace_print_with_logging() -> None:
    builtins.print = _print_to_logging


def _print_to_logging(
    *objects: Any,
    sep: Optional[str] = " ",
    end: Optional[str] = "\n",
    file: Optional[Any] = None,
    flush: bool = False,
) -> None:
    """Wraps built-in print to replace it with using the logging module. Only
    writing to stdout and stderr are replaced, printing to a file will be
    executed unmodified.

    This function is on the module level because otherwise numba breaks.
    """
    # Mimicking the behavior of Python's built-in print function.
    if sep is None:
        sep = " "
    if end is None:
        end = "\n"

    # Don't replace prints to files.
    if file is not None and file != sys.stdout and file != sys.stderr:
        BUILTIN_PRINT(*objects, sep=sep, end=end, file=file, flush=flush)
        return

    logging.info(sep.join(map(str, objects)), stacklevel=3)


@fb_overwritable()
def _log_enter(category: str, name: str, unique_id: str) -> None:
    logging.info(f"Entering logging context, {category=}, {name=}, {unique_id=}")


@fb_overwritable()
def _log_exit(category: str, name: str, unique_id: str, duration: float) -> None:
    logging.info(
        f"Exiting logging context, {category=}, {name=}, {unique_id=}, {duration=}"
    )


def log_interval(
    category: Optional[str] = None, name: Optional[str] = None
) -> Callable[[Callable[..., _T]], Callable[..., _T]]:

    _unique_id = uuid.uuid1().int >> 97
    _overwrite_category = category
    _overwrite_name = name

    def log_interval_deco(func: Callable[..., _T]) -> Callable[..., _T]:

        _category = _overwrite_category or func.__qualname__.split(".")[0]
        _name = _overwrite_name or func.__name__

        @wraps(func)
        def wrapper(*args, **kwargs) -> _T:
            _log_enter(_category, _name, _unique_id)
            _start = time.perf_counter()
            ret = func(*args, **kwargs)
            _log_exit(_category, _name, _unique_id, time.perf_counter() - _start)
            return ret

        return wrapper

    return log_interval_deco
