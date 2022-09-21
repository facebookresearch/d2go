#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved


from d2go.initializer import initialize_all

# NOTE: by default a list of initializations will run whenever D2Go is first imported,
# so that users don't need to do any manual iniitialization other than importing `d2go`.
initialize_all()
