# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
import json
import os
from contextlib import nullcontext
from typing import Callable, cast, IO

import detectron2.utils.comm as comm
import torch

from d2go.checkpoint.checkpoint_instrumentation import instrument_checkpoint
from d2go.checkpoint.utils import (
    gather_ema_state_dict,
    gather_optimizer_state_dict,
    scatter_ema_state_dict,
    scatter_optimizer_state_dict,
)
from d2go.quantization.modeling import QATCheckpointer
from d2go.trainer.fsdp import FSDPWrapper
from d2go.utils.misc import _log_api_usage_on_main_process

from mobile_cv.torch.utils_pytorch.distributed_helper import interleave_by_rank

from torch.distributed.fsdp.fully_sharded_data_parallel import StateDictType


LOG_API_IDENTIFIER = "checkpointing.FSDPCheckpointer"


def get_max_checkpoint_concurrency() -> int:
    return comm.get_world_size()


# TODO: replace FSDPCheckpointer with central D2GoCheckpointer
class FSDPCheckpointer(QATCheckpointer):
    """
    Extend the Checkpointer to support saving/loading FSDP models
    """

    def __init__(
        self,
        *args,
        concurrency_limit_fetcher: Callable[[], int] = get_max_checkpoint_concurrency,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self._concurrency_limit_fetcher: Callable[[], int] = concurrency_limit_fetcher

    def is_distributed(self) -> bool:
        return True

    @instrument_checkpoint("load")
    def load(self, path: str, checkpointables=None):
        """
        Add support for loading sharded optimizer states in FSDP.

        .. note:: Loading optimizer states from regular checkpoints into FSDP models is currently not supported.
            In general users should not resume non-FSDP training with FSDP.
        """
        if isinstance(self.model, FSDPWrapper):
            load_path = path
            if path:
                # loading path is a directory: local or sharded state dict is used
                if self.path_manager.isdir(path):
                    # Get state dict type from metadata file
                    metadata = self._load_metadata(path)
                    state_dict_type = (
                        metadata["state_dict_type"] if metadata else "LOCAL_STATE_DICT"
                    )

                    assert state_dict_type in ["LOCAL_STATE_DICT", "SHARDED_STATE_DICT"]
                    self.logger.info(
                        f"[FSDPCheckpointer] Loading from {state_dict_type} checkpoint ..."
                    )
                    self.model.load_state_dict_type = StateDictType[state_dict_type]
                    load_path = os.path.join(path, f"rank{comm.get_rank()}.pth")
                # loading path is a file: full global state dict is used
                else:
                    self.logger.info(
                        "[FSDPCheckpointer] Loading from FULL_STATE_DICT checkpoint ..."
                    )
                    self.model.load_state_dict_type = StateDictType.FULL_STATE_DICT
                    # since checkpoints are the same across ranks, we can download from
                    # rank0 and shared the local file.
                    load_path = self._get_local_path_per_host(load_path)

            _log_api_usage_on_main_process(
                f"{LOG_API_IDENTIFIER}.load.fsdp.{self.model.load_state_dict_type.name}"  # pyre-ignore
            )
            # Convert local ckpt to global ckpt when we load from a local ckpt but want to save to global ckpt
            convert_local_ckpt_to_global = (
                path
                and self.model.load_state_dict_type == StateDictType.LOCAL_STATE_DICT
                and self.model.state_dict_type == StateDictType.FULL_STATE_DICT
            )

            # Load all checkpointables from local ckpt if we want to convert to global ckpt
            checkpointables_iter = (
                self.checkpointables.keys()
                if checkpointables is None or convert_local_ckpt_to_global
                else checkpointables
            )
            checkpointables_filtered = [
                name
                for name in checkpointables_iter
                if name not in ["optimizer", "ema_state"]
            ]

            checkpoint = super().load(
                load_path, checkpointables=checkpointables_filtered
            )
            if "optimizer" in checkpointables_iter:
                self.logger.info(
                    f"[FSDPCheckpointer] Loading optimizer from {load_path} ..."
                )
                optimizer = self.checkpointables["optimizer"]
                osd = checkpoint.pop("optimizer")
                scatter_optimizer_state_dict(optimizer, osd, self.model)
            if "ema_state" in checkpointables_iter:
                self.logger.info(
                    f"[FSDPCheckpointer] Loading ema_state from {load_path} ..."
                )
                ema_state = checkpoint.pop("ema_state")
                scatter_ema_state_dict(ema_state, self.model)
            # Convert local ckpt by resaving the current state
            if convert_local_ckpt_to_global:
                self.logger.info(
                    "[FSDPCheckpointer] Converting local FSDP checkpoint to global checkpoint ..."
                )
                self.save(os.path.basename(path), tag_last_ckpt=False, **checkpoint)
                self.logger.info(
                    "[FSDPCheckpointer] Local-to-global checkpoint conversion finishes"
                )

            # return all remaining checkpoints
            return checkpoint
        else:
            _log_api_usage_on_main_process(f"{LOG_API_IDENTIFIER}.load.ddp")
            return super().load(path, checkpointables=checkpointables)

    @instrument_checkpoint("save")
    def save(self, name: str, tag_last_ckpt=True, **kwargs) -> None:
        """
        Add support for saving sharding models and optimizers.
        The rest of the code is copied from implementation in the superclass
        """
        # checkpoint_type is used to annotate preemption checkpoints for internal checkpointer. Ignore it here
        kwargs.pop("checkpoint_type", None)
        # If no sharding, only the main process enters the saving codepath;
        # otherwise, all processes need to call state_dict() to enable state broadcasting among ranks
        if not isinstance(self.model, FSDPWrapper):
            _log_api_usage_on_main_process(f"{LOG_API_IDENTIFIER}.save.ddp")
            if comm.is_main_process():
                return super().save(name, **kwargs)
            return

        _log_api_usage_on_main_process(
            f"{LOG_API_IDENTIFIER}.save.fsdp.{self.model.state_dict_type.name}"
        )

        data = {}
        # FSDP: model.state_dict() needs to be called by all ranks before saving
        data["model"] = self.model.state_dict()
        for key, obj in self.checkpointables.items():
            if key == "optimizer":
                data[key] = gather_optimizer_state_dict(obj, self.model)
            elif key == "ema_state":
                data[key] = gather_ema_state_dict(obj, self.model)
            else:
                data[key] = obj.state_dict()
        data.update(kwargs)

        # If using full state dict, only the main process does checkpoint saving; Otherwise, all processes do
        if self.model.state_dict_type != StateDictType.FULL_STATE_DICT:
            # Main process creates directory for local saves
            new_save_dir = os.path.join(self.save_dir, name)
            if comm.is_main_process():
                if not self.path_manager.exists(new_save_dir):
                    self.path_manager.mkdirs(new_save_dir)
            comm.synchronize()
            # Saving checkpoints
            basename = "rank{}.pth".format(comm.get_rank())
            save_file = os.path.join(new_save_dir, basename)
            assert os.path.basename(save_file) == basename, basename
            # Limit the write concurrency to avoid QPS overload
            with interleave_by_rank(
                concurrency_limit=self._concurrency_limit_fetcher()
            ):
                self._save_file(data, save_file)
            # Main process tags last checkpoint if no errors in all processes
            comm.synchronize()
            if comm.is_main_process():
                self._save_metadata(new_save_dir)
                if tag_last_ckpt:
                    self.tag_last_checkpoint(name)
        else:
            if comm.is_main_process():
                basename = "{}.pth".format(name)
                save_file = os.path.join(self.save_dir, basename)
                assert os.path.basename(save_file) == basename, basename

                self.logger.info(
                    f"[FSDPCheckpointer] Rank {comm.get_rank()} is checkpointing {save_file}."
                )
                self._save_file(data, save_file)
                if tag_last_ckpt:
                    self.tag_last_checkpoint(basename)
            else:
                self.logger.info(
                    f"[FSDPCheckpointer] Rank {comm.get_rank()} is deferring checkpointing to the main process."
                )

            comm.synchronize()

    def _save_file(self, data, filename):
        self.logger.info("Saving checkpoint to {}".format(filename))

        with self.path_manager.open(filename, "wb") as f:
            torch.save(data, cast(IO[bytes], f))

        self.logger.info("Finished saving checkpoint to {}".format(filename))

    def _load_file(self, f: str):
        if isinstance(self.model, FSDPWrapper):
            with (
                interleave_by_rank(concurrency_limit=self._concurrency_limit_fetcher())
                if self.model.state_dict_type != StateDictType.FULL_STATE_DICT
                else nullcontext()  # FULL_STATE_DICT doesn't need interleaving
            ):
                # use mmap for FSDP checkpoints
                return torch.load(f, map_location=torch.device("cpu"), mmap=True)
        else:
            return super()._load_file(f)

    def _save_metadata(self, path):
        metadata_file = os.path.join(path, "metadata.json")
        obj = {"state_dict_type": self.model.state_dict_type.name}
        with self.path_manager.open(metadata_file, "w") as f:
            json.dump(obj, f)

    def _load_metadata(self, path):
        metadata_file = os.path.join(path, "metadata.json")
        if self.path_manager.exists(metadata_file):
            with self.path_manager.open(metadata_file, "r") as f:
                return json.load(f)
        else:
            return None

    def _get_local_path_per_host(self, path: str) -> str:
        """Download file only on local master, return the downloaded path for all ranks"""

        from torchtnt.utils.distributed import get_local_rank, get_local_world_size

        self.logger.info("Start getting local path per host ...")
        # check if paths are the same on the same node
        all_paths = comm.all_gather(path)
        local_master = (
            comm.get_rank() // get_local_world_size() * get_local_world_size()
        )
        if path != all_paths[local_master]:
            raise ValueError(
                f"All paths must be the same on the same node, got {path} vs {all_paths[local_master]}"
            )
        # local master downloads the file, while non-master skips
        if get_local_rank() == 0:
            self.logger.info(f"Start downloading {path} to local file ...")
            local_path = self.path_manager.get_local_path(path)
            self.logger.info(f"Finished downloading {path} to local file: {local_path}")
        else:
            local_path = None
            self.logger.info("Waiting for local master to finish downloading ...")
        # broadcast the local path to all other ranks
        local_paths = comm.all_gather(local_path)
        local_path = local_paths[local_master]
        assert local_path is not None, f"Local path is None, {local_paths=}"
        self.logger.info("Finished getting local path per host")
        return local_path
