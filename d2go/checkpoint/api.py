# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

from fvcore.common.checkpoint import Checkpointer


def is_distributed_checkpoint(checkpointer: Checkpointer) -> bool:
    """
    Check if checkpointer supports distributed checkpointing,
    in which case all ops need to be invoked in every rank.
    """

    if hasattr(checkpointer, "is_distributed"):
        return checkpointer.is_distributed()
    return False
