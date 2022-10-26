#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import os

from d2go.initializer import initialize_all

# NOTE: by default a list of initializations will run whenever D2Go is first imported,
# so that users don't need to do any manual iniitialization other than importing `d2go`.

# Environment variable can be used to skip initialization for special cases like unit test
skip_initialization = os.environ.get("D2GO_IMPORT_SKIP_INITIALIZATION", "0") == "1"

if not skip_initialization:
    initialize_all()
