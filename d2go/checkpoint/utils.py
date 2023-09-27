import copy

import logging

from d2go.modeling.ema import EMAState
from d2go.trainer.fsdp import FSDPWrapper
from torch.distributed.fsdp.fully_sharded_data_parallel import (
    FullyShardedDataParallel as FSDP,
    StateDictType,
)

logger = logging.getLogger(__name__)


def gather_optimizer_state_dict(optimizer, model: FSDPWrapper):
    """
    Get full/local optimizer state dict from an FSDP model.
    """
    # FSDP: full_optim_state_dict() needs to be called by all ranks
    if model.state_dict_type == StateDictType.FULL_STATE_DICT:
        return FSDP.full_optim_state_dict(
            model, optim=optimizer, rank0_only=model.rank0_only
        )
    elif model.state_dict_type == StateDictType.SHARDED_STATE_DICT:
        return FSDP.sharded_optim_state_dict(model, optim=optimizer)
    return optimizer.state_dict()


def scatter_optimizer_state_dict(optimizer, optim_state_dict, model: FSDPWrapper):
    """
    Load a full/local optimizer state dict to a FSDP model.
    If using full state dict, shard and scatter the optimizer state dict before loading
    """
    if model.load_state_dict_type == StateDictType.FULL_STATE_DICT:
        optim_state_dict = FSDP.shard_full_optim_state_dict(
            optim_state_dict, model, optim=optimizer
        )
    elif model.load_state_dict_type == StateDictType.SHARDED_STATE_DICT:
        optim_state_dict = FSDP.flatten_sharded_optim_state_dict(
            optim_state_dict, model, optim=optimizer
        )
    optimizer.load_state_dict(optim_state_dict)


def gather_ema_state_dict(ema_state: EMAState, model: FSDPWrapper):
    """
    Get full/local EMA state dict from an FSDP model.
    If using full state dict, gather local sharded EMA states from all FSDP processes and aggregate them into a full EMA state dict
    """
    if model.state_dict_type in [
        StateDictType.FULL_STATE_DICT,
        StateDictType.SHARDED_STATE_DICT,
    ]:
        with ema_state.apply_and_restore(model):
            # must deepcopy the state dict, else we return a reference to the model state
            return dict(copy.deepcopy(model.state_dict()))
    else:
        return ema_state.state_dict()


def scatter_ema_state_dict(ema_state_dict, model: FSDPWrapper):
    """
    Load a full/sharded/local EMA state dict to a FSDP model.
    If loading full state dict, ema_state_dict needs to be properly sharded for each FSDP process to store locally
    Note that, at load-time, model.state_dict_type is automatically set to the type of the state dict being loaded
    by accessing metadata, so there's no possibility of a save-load mismatch
    """
    if model.load_state_dict_type in [
        StateDictType.FULL_STATE_DICT,
        StateDictType.SHARDED_STATE_DICT,
    ]:
        # Store current model state temporarily
        old_state = EMAState.FromModel(model)

        # Load the ema state dict into the model
        m, u = model.load_state_dict(ema_state_dict, strict=False)
        if len(m) > 0:
            logger.info("Missing keys while loading EMA:", m)
        if len(u) > 0:
            logger.info("Unexpected keys while loading EMA:", u)

        # Save ema state with correct FQNs via EMAState.save_from
        model.ema_state.save_from(model)

        # restore old model state
        old_state.apply_to(model)
    else:
        model.ema_state.load_state_dict(ema_state_dict)
