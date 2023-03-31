# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import logging

from mobile_cv.common.misc.oss_utils import fb_overwritable


@fb_overwritable()
def initialize_logging(logging_level: int) -> None:
    root_logger = logging.getLogger()
    root_logger.setLevel(logging_level)
