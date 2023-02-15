# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
import os
from typing import cast, IO

import detectron2.utils.comm as comm
import torch
from d2go.modeling.ema import EMAState

from d2go.quantization.modeling import QATCheckpointer
from d2go.trainer.fsdp import FSDPWrapper

from torch.distributed.fsdp.fully_sharded_data_parallel import (
    FullyShardedDataParallel as FSDP,
)


# TODO: replace FSDPCheckpointer with central D2GoCheckpointer
class FSDPCheckpointer(QATCheckpointer):
    """
    Extend the Checkpointer to support saving/loading FSDP models
    """

    def is_distributed(self) -> bool:
        return True

    def load(self, path: str, checkpointables=None):
        """
        Add support for loading sharded optimizer states in FSDP.

        .. note:: Loading optimizer states from regular checkpoints into FSDP models is currently not supported.
            In general users should not resume non-FSDP training with FSDP.
        """
        if isinstance(self.model, FSDPWrapper):
            load_path = path
            if path:
                # loading path is a directory: sharded local state dict is used
                if self.path_manager.isdir(path):
                    self.logger.info(
                        "[FSDPCheckpointer] Loading from local checkpoint ..."
                    )
                    self.model.load_local_state_dict = True
                    load_path = os.path.join(path, f"rank{comm.get_rank()}.pth")
                # loading path is a file: full global state dict is used
                else:
                    self.logger.info(
                        "[FSDPCheckpointer] Loading from global checkpoint ..."
                    )
                    self.model.load_local_state_dict = False

            # Convert local ckpt to global ckpt when we load from a local ckpt but want to save to global ckpt
            convert_local_ckpt_to_global = (
                path
                and self.model.load_local_state_dict
                and not self.model.use_local_state_dict
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
            return super().load(path, checkpointables=checkpointables)

    def save(self, name: str, tag_last_ckpt=True, **kwargs) -> None:
        """
        Add support for saving sharding models and optimizers.
        The rest of the code is copied from implementation in the superclass
        """
        # If no sharding, only the main process enters the saving codepath;
        # otherwise, all processes need to call state_dict() to enable state broadcasting among ranks
        if not isinstance(self.model, FSDPWrapper):
            if comm.is_main_process():
                return super().save(name, **kwargs)
            return

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
        if self.model.use_local_state_dict:
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
            self._save_file(data, save_file)
            # Main process tags last checkpoint if no errors in all processes
            comm.synchronize()
            if comm.is_main_process() and tag_last_ckpt:
                self.tag_last_checkpoint(name)
        elif comm.is_main_process():
            basename = "{}.pth".format(name)
            save_file = os.path.join(self.save_dir, basename)
            assert os.path.basename(save_file) == basename, basename
            self._save_file(data, save_file)
            if tag_last_ckpt:
                self.tag_last_checkpoint(basename)

    def _save_file(self, data, filename):
        self.logger.info("Saving checkpoint to {}".format(filename))
        with self.path_manager.open(filename, "wb") as f:
            torch.save(data, cast(IO[bytes], f))


def gather_optimizer_state_dict(optimizer, model: FSDPWrapper):
    """
    Get full/local optimizer state dict from an FSDP model.
    """
    # FSDP: full_optim_state_dict() needs to be called by all ranks
    if not model.use_local_state_dict:
        return FSDP.full_optim_state_dict(model, optimizer, rank0_only=model.rank0_only)
    return optimizer.state_dict()


def scatter_optimizer_state_dict(optimizer, optim_state_dict, model: FSDPWrapper):
    """
    Load a full/local optimizer state dict to a FSDP model.
    If using full state dict, shard and scatter the optimizer state dict before loading
    """
    if not model.load_local_state_dict:
        optim_state_dict = FSDP.shard_full_optim_state_dict(optim_state_dict, model)
    optimizer.load_state_dict(optim_state_dict)


def gather_ema_state_dict(ema_state, model: FSDPWrapper):
    """
    Get full/local EMA state dict from an FSDP model.
    If using full state dict, gather local sharded EMA states from all FSDP processes and aggregate them into a full EMA state dict
    """
    if not model.use_local_state_dict:
        # Apply local ema states to the model and unshard them
        with ema_state.apply_and_restore(model):
            with FSDP.summon_full_params(
                model,
                writeback=False,
                offload_to_cpu=model.offload_to_cpu,
                rank0_only=model.rank0_only,
            ):
                state = EMAState.FromModel(model)
            return state.state
    return ema_state.state_dict()


def scatter_ema_state_dict(ema_state_dict, model: FSDPWrapper):
    """
    Load a full/local EMA state dict to a FSDP model.
    If loading full state dict, ema_state_dict needs to be properly sharded for each FSDP process to store locally
    """
    if not model.load_local_state_dict:
        # Store the current model state.
        old_local_state = EMAState.FromModel(model)

        # Apply ema_state as a FULL state dict to the model so it can be properly sharded
        # Currently only [offload_to_cpu=False, rank0_only=False] is supported
        with FSDP.summon_full_params(
            model,
            writeback=True,
            offload_to_cpu=False,
            rank0_only=False,
        ):
            ema_state = EMAState()
            ema_state.load_state_dict(ema_state_dict)
            ema_state.apply_to(model)

        # Load ema_state from model
        model.ema_state.save_from(model)
        # Restore the old model state
        old_local_state.apply_to(model)
    else:
        model.ema_state.load_state_dict(ema_state_dict)
