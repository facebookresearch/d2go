# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import builtins
import logging
import sys
from typing import Any, Optional

from mobile_cv.common.misc.oss_utils import fb_overwritable


# Saving the builtin print to wrap it up later.
BUILTIN_PRINT = builtins.print


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
