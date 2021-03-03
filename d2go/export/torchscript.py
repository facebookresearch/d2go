#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved


import logging
import os
from typing import Tuple, Optional, Dict

import torch
from fvcore.common.file_io import PathManager
from torch import nn


logger = logging.getLogger(__name__)


def trace_and_save_torchscript(
    model: nn.Module,
    inputs: Tuple[torch.Tensor],
    output_path: str,
    _extra_files: Optional[Dict[str, bytes]] = None,
):
    logger.info("Tracing and saving TorchScript to {} ...".format(output_path))

    # TODO: patch_builtin_len depends on D2, we should either copy the function or
    # dynamically registering the D2's version.
    from detectron2.export.torchscript_patch import patch_builtin_len
    with torch.no_grad(), patch_builtin_len():
        script_model = torch.jit.trace(model, inputs)

    if _extra_files is None:
        _extra_files = {}
    model_file = os.path.join(output_path, "model.jit")

    PathManager.mkdirs(output_path)
    with PathManager.open(model_file, "wb") as f:
        torch.jit.save(script_model, f, _extra_files=_extra_files)

    data_file = os.path.join(output_path, "data.pth")
    with PathManager.open(data_file, "wb") as f:
        torch.save(inputs, f)

    # NOTE: new API doesn't require return
    return model_file
