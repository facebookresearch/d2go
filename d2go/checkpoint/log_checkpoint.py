# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import logging

from mobile_cv.common.misc.oss_utils import fb_overwritable


logger = logging.getLogger(__name__)


@fb_overwritable()
def log_checkpoint(checkpoint_type=str, unique_id=int, state=str) -> None:
    logger.info(f"Checkpoint:{unique_id} {checkpoint_type} {state} ")
