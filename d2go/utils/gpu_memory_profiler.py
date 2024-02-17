import logging
import os
import pickle

import torch
from d2go.config import CfgNode as CN
from detectron2.utils.file_io import PathManager
from mobile_cv.torch.utils_pytorch import comm
from torch.cuda._memory_viz import segment_plot, trace_plot

logger: logging.Logger = logging.getLogger(__name__)


def add_memory_profiler_configs(_C: CN):
    _C.MEMORY_PROFILER = CN()
    _C.MEMORY_PROFILER.ENABLED = False
    # max number of trace entries in memory snapshot
    _C.MEMORY_PROFILER.TRACE_MAX_ENTRIES = 1000000
    # Configs to be used by d2go.utils.gpu_memory_profiler.D2GoGpuMemorySnapshot
    # determine the number of iterations to log memory snapshots for
    _C.MEMORY_PROFILER.LOG_N_STEPS = 3
    # determine at what iteration to start recording gpu memory
    _C.MEMORY_PROFILER.LOG_DURING_TRAIN_AT = 550


def add_zoomer_default_config(_C: CN):
    _C.ZOOMER = CN()
    _C.ZOOMER.ENABLE_STACK_TRACING = (
        False  # Do not enable by default, since it may cause performance regression
    )
    _C.ZOOMER.ENABLE_MEMORY_PROFILING = False


def oom_logger_wrapper(output_dir):
    def oom_logger(
        device: int, alloc: int, device_alloc: int, device_free: int
    ) -> None:
        """
        Log memory snapshot in the event of CUDA OOM.
        """
        logger.info(
            f"Saving memory snapshot device: {device}, alloc: {alloc}, device_alloc: {device_alloc}, device_free: {device_free}"
        )
        try:
            log_memory_snapshot(output_dir, file_prefix="oom")
        except Exception as e:
            logger.error(f"Failed to log memory snapshot during OOM {e}")

    return oom_logger


def log_memory_snapshot(output_dir: str, file_prefix: str = "") -> None:
    """
    Log memory snapshots to output_dir
    """
    if not torch.cuda.is_available():
        logger.info("CUDA unavailable. Not logging snapshot")
        return

    try:
        rank = comm.get_rank()
        save_dir = os.path.join(
            output_dir, "memory_snapshot", f"{file_prefix}_rank{rank}"
        )
        logger.info(f"Logging memory snapshot to {save_dir}")
        snapshot = torch.cuda.memory._snapshot()
        dump_snapshot(save_dir, snapshot)
    except Exception as e:
        logger.error(f"Failed to log memory snapshot to {save_dir}: {e}")


def dump_snapshot(save_dir: str, snapshot):
    """
    Dump memory snapshot and useful plots to save_dir.
    This is a rewrite of torch.cuda.memory._dump_snapshot() with PathManager.
    """
    if not PathManager.exists(save_dir):
        PathManager.mkdirs(save_dir)
    with PathManager.open(os.path.join(save_dir, "snapshot.pickle"), "wb") as f:
        pickle.dump(snapshot, f)
    with PathManager.open(os.path.join(save_dir, "trace_plot.html"), "w") as f:
        f.write(trace_plot(snapshot))
    with PathManager.open(os.path.join(save_dir, "segment_plot.html"), "w") as f:
        f.write(segment_plot(snapshot))
    logger.info(f"Saved memory snapshot to {save_dir}")


def record_memory_history(trace_max_entries=1000000) -> None:
    """
    Start recording memory history and stack traces.
    """
    if not torch.cuda.is_available():
        logger.info("CUDA unavailable. Not recording memory history")
        return

    torch.cuda.memory._record_memory_history(
        enabled="all", max_entries=trace_max_entries
    )
    logger.info("Started recording memory history")


def attach_oom_logger(output_dir, trace_max_entries=1000000) -> None:
    """
    Start recording memory history and attach the OOM logger.
    """
    if not torch.cuda.is_available():
        logger.info("CUDA unavailable. Not attaching OOM logger")
        return

    record_memory_history(trace_max_entries)
    torch._C._cuda_attach_out_of_memory_observer(oom_logger_wrapper(output_dir))
    logger.info("Attached GPU OOM logger")
