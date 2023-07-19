#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import logging
from typing import Callable, TypeVar

from torch.distributed.elastic.multiprocessing.errors import (
    _NOT_AVAILABLE,
    ChildFailedError,
    get_error_handler,
)

logger = logging.getLogger(__name__)

_RT = TypeVar("_RT")


def mast_error_handler(func: Callable[..., _RT]) -> Callable[..., _RT]:
    def wrapper(*args, **kwargs) -> _RT:
        logger.info("Starting main")
        error_handler = get_error_handler()
        logger.debug(f"Error handler is: {type(error_handler)=}, {error_handler=}")
        error_handler.initialize()
        logger.debug("Error handler has been initialized")
        try:
            logger.debug("Entered main for d2go")
            return func(*args, **kwargs)
        except ChildFailedError as e:
            logger.info(f"Got a ChildFailedError: {e=}")
            rank, failure = e.get_first_failure()
            if failure.error_file != _NOT_AVAILABLE:
                error_handler.dump_error_file(failure.error_file, failure.exitcode)
            else:
                logger.info(
                    (
                        f"local_rank {rank} FAILED with no error file."
                        f" Decorate your entrypoint fn with @record for traceback info."
                        f" See: https://pytorch.org/docs/stable/elastic/errors.html"
                    )
                )
                raise
        except Exception as e:
            logger.info(f"Caught a generic exception: {e=}")
            error_handler.record_exception(e)
            raise

    return wrapper


def gather_mast_errors(func: Callable[..., _RT]) -> Callable[..., _RT]:
    def wrapper(*args, **kwargs) -> _RT:
        logger.info("Starting CLI application")
        try:
            return func(*args, **kwargs)
        finally:
            logging.info("Entering final reply file generation step")
            import glob
            import os
            import shutil

            torchx_reply_files = glob.glob("/tmp/torchx_*/**/*.json", recursive=True)
            logger.info(
                f"Found the following reply files on this host: {torchx_reply_files}"
            )
            first_reply_file = None
            first_reply_file_st = float("Inf")
            for f in torchx_reply_files:
                if (mtime := os.stat(f).st_mtime) < first_reply_file_st:
                    first_reply_file = f
                    first_reply_file_st = mtime
            if first_reply_file and os.environ.get("MAST_HPC_TASK_FAILURE_REPLY_FILE"):
                logger.info(
                    f'Copying {first_reply_file=} to {os.environ["MAST_HPC_TASK_FAILURE_REPLY_FILE"]}'
                )
                shutil.copyfile(
                    first_reply_file, os.environ["MAST_HPC_TASK_FAILURE_REPLY_FILE"]
                )

    return wrapper
