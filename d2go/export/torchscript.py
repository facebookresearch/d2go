#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved


import contextlib
import logging
import os
from typing import Tuple, Optional, Dict, NamedTuple, List, AnyStr, Set

import torch
from detectron2.utils.file_io import PathManager
from mobile_cv.common.misc.file_utils import make_temp_directory
from torch import nn
from torch._C import MobileOptimizerType
from torch.utils.bundled_inputs import augment_model_with_bundled_inputs
from torch.utils.mobile_optimizer import optimize_for_mobile


logger = logging.getLogger(__name__)


class MobileOptimizationConfig(NamedTuple):
    # optimize_for_mobile
    optimization_blocklist: Set[MobileOptimizerType] = None
    preserved_methods: List[AnyStr] = None
    backend: str = "CPU"
    methods_to_optimize: List[AnyStr] = None


def trace_and_save_torchscript(
    model: nn.Module,
    inputs: Tuple[torch.Tensor],
    output_path: str,
    mobile_optimization: Optional[MobileOptimizationConfig] = None,
    _extra_files: Optional[Dict[str, bytes]] = None,
):
    logger.info("Tracing and saving TorchScript to {} ...".format(output_path))
    PathManager.mkdirs(output_path)
    if _extra_files is None:
        _extra_files = {}

    # TODO: patch_builtin_len depends on D2, we should either copy the function or
    # dynamically registering the D2's version.
    from detectron2.export.torchscript_patch import patch_builtin_len

    with torch.no_grad(), patch_builtin_len():
        script_model = torch.jit.trace(model, inputs)

    with make_temp_directory("trace_and_save_torchscript") as tmp_dir:

        @contextlib.contextmanager
        def _synced_local_file(rel_path):
            remote_file = os.path.join(output_path, rel_path)
            local_file = os.path.join(tmp_dir, rel_path)
            yield local_file
            PathManager.copy_from_local(local_file, remote_file, overwrite=True)

        with _synced_local_file("model.jit") as model_file:
            torch.jit.save(script_model, model_file, _extra_files=_extra_files)

        with _synced_local_file("data.pth") as data_file:
            torch.save(inputs, data_file)

        if mobile_optimization is not None:
            logger.info("Applying optimize_for_mobile ...")
            liteopt_model = optimize_for_mobile(
                script_model,
                optimization_blocklist=mobile_optimization.optimization_blocklist,
                preserved_methods=mobile_optimization.preserved_methods,
                backend=mobile_optimization.backend,
                methods_to_optimize=mobile_optimization.methods_to_optimize,
            )
            with _synced_local_file("mobile_optimized.ptl") as lite_path:
                liteopt_model._save_for_lite_interpreter(lite_path)
            # liteopt_model(*inputs)  # sanity check
            op_names = torch.jit.export_opnames(liteopt_model)
            logger.info(
                "Operator names from lite interpreter:\n{}".format("\n".join(op_names))
            )

            logger.info("Applying augment_model_with_bundled_inputs ...")
            augment_model_with_bundled_inputs(liteopt_model, [inputs])
            liteopt_model.run_on_bundled_input(0)  # sanity check
            with _synced_local_file("mobile_optimized_bundled.ptl") as lite_path:
                liteopt_model._save_for_lite_interpreter(lite_path)
