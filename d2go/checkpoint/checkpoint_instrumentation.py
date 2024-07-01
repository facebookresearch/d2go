# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import logging
import uuid
from contextlib import ContextDecorator
from typing import Optional

from d2go.checkpoint.log_checkpoint import log_checkpoint


logger = logging.getLogger(__name__)


class instrument_checkpoint(ContextDecorator):
    def __init__(
        self,
        checkpoint_type: str,
    ) -> None:
        super().__init__()
        self.unique_id: Optional[int] = None
        self.checkpoint_type = checkpoint_type

    def __enter__(self) -> "instrument_checkpoint":
        self.unique_id = uuid.uuid1().int >> 97
        log_checkpoint(
            checkpoint_type=self.checkpoint_type,
            unique_id=self.unique_id,
            state="begin",
        )
        return self

    def __exit__(self, exc_type, exc_value, tb) -> bool:
        log_checkpoint(
            checkpoint_type=self.checkpoint_type,
            unique_id=self.unique_id,
            state="end",
        )

        if exc_value is not None:
            # Re-raising the exception, otherwise it will be swallowed
            raise exc_value

        return True
