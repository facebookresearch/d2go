# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
import os

import detectron2.utils.comm as comm
import torch
from d2go.modeling.model_ema import EMAState

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

    def load(self, path: str, checkpointables=None):
        """
        Add support for loading sharded optimizer states in FSDP.

        .. note:: Loading optimizer states from regular checkpoints into FSDP models is currently not supported.
            In general users should not resume regular training with FSDP.
        """
        if isinstance(self.model, FSDPWrapper):
            checkpointables_iter = (
                self.checkpointables.keys()
                if checkpointables is None
                else checkpointables
            )
            checkpointables_filtered = [
                name
                for name in checkpointables_iter
                if name not in ["optimizer", "ema_state"]
            ]

            checkpoint = super().load(path, checkpointables=checkpointables_filtered)
            if "optimizer" in checkpointables_iter:
                self.logger.info("Loading optimizer from {} ...".format(path))
                osd = checkpoint.pop("optimizer")
                sharded_osd = FSDP.shard_full_optim_state_dict(osd, self.model)
                self.checkpointables["optimizer"].load_state_dict(sharded_osd)
            if "ema_state" in checkpointables_iter:
                self.logger.info("Loading ema_state from {} ...".format(path))
                ema_state = checkpoint.pop("ema_state")
                scatter_ema_state_dict(ema_state, self.model)
            # return all remaining checkpoints
            return checkpoint
        else:
            return super().load(path, checkpointables=checkpointables)

    def save(self, name: str, **kwargs) -> None:
        """
        Add support for saving sharding models and optimizers.
        The rest of the code is copied from implementation in the superclass
        """
        # If no sharding, only the main process enters the saving codepath;
        # otherwise, all processes need to call state_dict() to enable state broadcasting among ranks
        if not isinstance(self.model, FSDP) and not comm.is_main_process():
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

        # Only the main process does checkpoint saving; code copied from vision/fair/fvcore/fvcore/common/checkpoint.py
        if comm.is_main_process():
            basename = "{}.pth".format(name)
            save_file = os.path.join(self.save_dir, basename)
            assert os.path.basename(save_file) == basename, basename
            self.logger.info("Saving checkpoint to {}".format(save_file))
            with self.path_manager.open(save_file, "wb") as f:
                # pyre-fixme[6]: For 2nd param expected `Union[PathLike[typing.Any],
                #  IO[bytes], str, BinaryIO]` but got `Union[IO[bytes], IO[str]]`.
                torch.save(data, f)
            self.tag_last_checkpoint(basename)


def gather_optimizer_state_dict(optimizer, model=None):
    # FSDP: full_optim_state_dict() needs to be called by all ranks
    if isinstance(model, FSDPWrapper):
        return FSDP.full_optim_state_dict(model, optimizer, rank0_only=model.rank0_only)
    return optimizer.state_dict()


def gather_ema_state_dict(ema_state, model):
    """
    Get EMA state dict.
    For FSDP, gather local sharded EMA states from all FSDP processes and aggregate them into a FULL GLOBAL state dict
    """
    if isinstance(model, FSDPWrapper):
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
    else:
        return ema_state.state_dict()


def scatter_ema_state_dict(ema_state_dict, model):
    """
    Load an EMA state dict to the model.
    EMA state represents a FULL GLOBAL state dict and needs to be properly sharded for each FSDP process to store locally
    """
    if isinstance(model, FSDPWrapper):
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
