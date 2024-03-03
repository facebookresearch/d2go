#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import atexit
import logging
import pickle
import shutil
import uuid

import numpy as np
from detectron2.utils import comm
from detectron2.utils.logger import log_every_n_seconds

logger = logging.getLogger(__name__)

# NOTE: Use unique ROOT_CACHE_DIR for each run, during the run, each instance of data
# loader will create a `cache_dir` under ROOT_CACHE_DIR. When the DL instance is GC-ed,
# the `cache_dir` will be removed by __del__; when the run is finished or interrupted,
# atexit.register will be triggered to remove the ROOT_CACHE_DIR to make sure there's no
# leftovers. Regarding DDP, although each GPU process has their own random value for
# ROOT_CACHE_DIR, but each GPU process uses the same `cache_dir` broadcasted from local
# master rank, which is then inherited by each data loader worker, this makes sure that
# `cache_dir` is in-sync between all GPUs and DL works on the same node.
ROOT_CACHE_DIR = "/tmp/DatasetFromList_cache_" + uuid.uuid4().hex[:8]


def _local_master_gather(func, check_equal=False):
    if comm.get_local_rank() == 0:
        x = func()
        assert x is not None
    else:
        x = None
    x_all = comm.all_gather(x)
    x_local_master = [x for x in x_all if x is not None]

    if check_equal:
        master = x_local_master[0]
        assert all(x == master for x in x_local_master), x_local_master

    return x_local_master


class DiskCachedList(object):
    """
    Wrap a list, the underlying storage is off-loaded to disk to save RAM usage.
    """

    def __init__(self, lst, strategy="batched_static"):
        """
        Args:
            lst (list): a list which contains elements to produce.
            strategy (str): strategy of using diskcache, supported strategies:
                - native: saving each item individually.
                - batched_static: group N items together, where N is calculated from
                    the average item size.
        """
        self._lst = lst
        self._diskcache_strategy = strategy

        def _serialize(data):
            buffer = pickle.dumps(data, protocol=-1)
            return np.frombuffer(buffer, dtype=np.uint8)

        logger.info(
            "Serializing {} elements to byte tensors ...".format(len(self._lst))
        )
        self._lst = [_serialize(x) for x in self._lst]
        total_size = sum(len(x) for x in self._lst)
        # TODO: only enabling DiskCachedDataset for large enough dataset
        logger.info("Serialized dataset takes {:.2f} MiB".format(total_size / 1024**2))
        self._initialize_diskcache()

    def _initialize_diskcache(self):
        from mobile_cv.common.misc.local_cache import LocalCache

        cache_dir = "{}/{}".format(ROOT_CACHE_DIR, uuid.uuid4().hex[:8])
        cache_dir = comm.all_gather(cache_dir)[0]  # use same cache_dir
        logger.info("Creating diskcache database in: {}".format(cache_dir))
        self._cache = LocalCache(cache_dir=cache_dir, num_shards=8)
        # self._cache.cache.clear(retry=True)  # seems faster if index exists

        if comm.get_local_rank() == 0:

            if self._diskcache_strategy == "naive":
                for i, item in enumerate(self._lst):
                    ret = self._write_to_local_db((i, item))
                    assert ret, "Error writing index {} to local db".format(i)
                    pct = 100.0 * i / len(self._lst)
                    self._log_progress(pct)

            # NOTE: each item might be small in size (hundreds of bytes),
            # writing million of them can take a pretty long time (hours)
            # because of frequent disk access. One solution is grouping a batch
            # of items into larger blob.
            elif self._diskcache_strategy == "batched_static":
                TARGET_BYTES = 50 * 1024
                average_bytes = np.average(
                    [
                        self._lst[int(x)].size
                        for x in np.linspace(0, len(self._lst) - 1, 1000)
                    ]
                )
                self._chuck_size = max(1, int(TARGET_BYTES / average_bytes))
                logger.info(
                    "Average data size: {} bytes; target chuck data size {} KiB;"
                    " {} items per chuck; {} chucks in total".format(
                        average_bytes,
                        TARGET_BYTES / 1024,
                        self._chuck_size,
                        int(len(self._lst) / self._chuck_size),
                    )
                )
                for i in range(0, len(self._lst), self._chuck_size):
                    chunk = self._lst[i : i + self._chuck_size]
                    chunk_i = int(i / self._chuck_size)
                    ret = self._write_to_local_db((chunk_i, chunk))
                    assert ret, "Error writing index {} to local db".format(chunk_i)
                    pct = 100.0 * i / len(self._lst)
                    self._log_progress(pct)

            # NOTE: instead of using fixed chuck size, items can be grouped dynamically
            elif self._diskcache_strategy == "batched_dynamic":
                raise NotImplementedError()

            else:
                raise NotImplementedError(self._diskcache_strategy)

        comm.synchronize()
        logger.info(
            "Finished writing to local disk, db size: {:.2f} MiB".format(
                self._cache.cache.volume() / 1024**2
            )
        )
        # Optional sync for some strategies
        if self._diskcache_strategy == "batched_static":
            # propagate chuck size and make sure all local rank 0 uses the same value
            self._chuck_size = _local_master_gather(
                lambda: self._chuck_size, check_equal=True
            )[0]
            logger.info("Gathered chuck size: {}".format(self._chuck_size))

        # free the memory of self._lst
        self._size = _local_master_gather(lambda: len(self._lst), check_equal=True)[0]
        logger.info("Gathered list size: {}".format(self._size))
        del self._lst

    def _write_to_local_db(self, task):
        index, record = task
        db_path = str(index)
        # suc = self._cache.load(lambda path, x: x, db_path, record)
        # record = BytesIO(np.random.bytes(np.random.randint(70000, 90000)))
        suc = self._cache.cache.set(db_path, record, retry=True)
        return suc

    def _log_progress(self, percentage):
        log_every_n_seconds(
            logging.INFO,
            "({:.2f}%) Wrote {} elements to local disk cache, db size: {:.2f} MiB".format(
                percentage,
                len(self._cache.cache),
                self._cache.cache.volume() / 1024**2,
            ),
            n=10,
        )

    def __len__(self):
        if self._diskcache_strategy == "batched_static":
            return self._size
        else:
            raise NotImplementedError()

    def __getitem__(self, idx):
        if self._diskcache_strategy == "naive":
            bytes = memoryview(self._cache.cache[str(idx)])
            return pickle.loads(bytes)

        elif self._diskcache_strategy == "batched_static":
            chunk_i, residual = divmod(idx, self._chuck_size)
            chunk = self._cache.cache[str(chunk_i)]
            bytes = memoryview(chunk[residual])
            return pickle.loads(bytes)

        else:
            raise NotImplementedError()

    @property
    def cache_dir(self):
        """return the current cache dirs of DiskCachedDatasetFromList instance"""
        return self._cache.cache_dir

    @staticmethod
    @atexit.register
    def _clean_up_root_cache_dir():
        # in case the program exists unexpectly, clean all the cache dirs created by
        # this session.
        if comm.get_local_rank() == 0:
            _clean_up_cache_dir(ROOT_CACHE_DIR)

    def __del__(self):
        # when data loader goes are GC-ed, remove the cache dir. This is needed to not
        # waste disk space in case that multiple data loaders are used, eg. running
        # evaluations on multiple datasets during training.
        if comm.get_local_rank() == 0:
            _clean_up_cache_dir(self.cache_dir)


def _clean_up_cache_dir(cache_dir):
    print("Cleaning up cache dir: {}".format(cache_dir))
    shutil.rmtree(
        cache_dir,
        onerror=lambda func, path, ex: print(
            "Catch error when removing {}; func: {}; exc_info: {}".format(
                path, func, ex
            )
        ),
    )
