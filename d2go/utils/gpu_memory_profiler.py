import logging

from d2go.config import CfgNode as CN

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
