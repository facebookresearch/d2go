#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved


import contextlib
import logging
import os
from typing import Tuple, Optional, Dict, NamedTuple, List, AnyStr, Set

import torch
from d2go.export.api import ModelExportMethodRegistry, ModelExportMethod
from detectron2.config.instantiate import dump_dataclass, instantiate
from detectron2.export.flatten import TracingAdapter, flatten_to_tuple
from detectron2.export.torchscript_patch import patch_builtin_len
from detectron2.utils.file_io import PathManager
from mobile_cv.common.misc.file_utils import make_temp_directory
from mobile_cv.common.misc.iter_utils import recursive_iterate
from torch import nn
from torch._C import MobileOptimizerType
from torch.utils.bundled_inputs import augment_model_with_bundled_inputs
from torch.utils.mobile_optimizer import optimize_for_mobile


logger = logging.getLogger(__name__)

TORCHSCRIPT_FILENAME_KEY: str = "torchscript_filename"


class MobileOptimizationConfig(NamedTuple):
    # optimize_for_mobile
    optimization_blocklist: Set[MobileOptimizerType] = None
    preserved_methods: List[AnyStr] = None
    backend: str = "CPU"
    torchscript_filename: str = "mobile_optimized.ptl"


def trace_and_save_torchscript(
    model: nn.Module,
    inputs: Tuple[torch.Tensor],
    output_path: str,
    torchscript_filename: str = "model.jit",
    mobile_optimization: Optional[MobileOptimizationConfig] = None,
    _extra_files: Optional[Dict[str, bytes]] = None,
):
    logger.info("Tracing and saving TorchScript to {} ...".format(output_path))
    PathManager.mkdirs(output_path)
    if _extra_files is None:
        _extra_files = {}

    with torch.no_grad():
        script_model = torch.jit.trace(model, inputs)

    with make_temp_directory("trace_and_save_torchscript") as tmp_dir:

        @contextlib.contextmanager
        def _synced_local_file(rel_path):
            remote_file = os.path.join(output_path, rel_path)
            local_file = os.path.join(tmp_dir, rel_path)
            yield local_file
            PathManager.copy_from_local(local_file, remote_file, overwrite=True)

        with _synced_local_file(torchscript_filename) as model_file:
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
            )
            torchscript_filename = mobile_optimization.torchscript_filename
            with _synced_local_file(torchscript_filename) as lite_path:
                liteopt_model._save_for_lite_interpreter(
                    lite_path, _extra_files=_extra_files
                )
            # liteopt_model(*inputs)  # sanity check
            op_names = torch.jit.export_opnames(liteopt_model)
            logger.info(
                "Operator names from lite interpreter:\n{}".format("\n".join(op_names))
            )

            logger.info("Applying augment_model_with_bundled_inputs ...")
            # make all tensors zero-like to save storage
            iters = recursive_iterate(inputs)
            for x in iters:
                if isinstance(x, torch.Tensor):
                    iters.send(torch.zeros_like(x).contiguous())
            inputs = iters.value
            augment_model_with_bundled_inputs(liteopt_model, [inputs])
            liteopt_model(*liteopt_model.get_all_bundled_inputs()[0])  # sanity check
            name, ext = os.path.splitext(torchscript_filename)
            with _synced_local_file(name + "_bundled" + ext) as lite_path:
                liteopt_model._save_for_lite_interpreter(lite_path)

        return torchscript_filename


class TorchscriptWrapper(nn.Module):
    """ """

    def __init__(self, module, int8_backend=None):
        super().__init__()
        self.module = module
        self.int8_backend = int8_backend

    def forward(self, *args, **kwargs):
        # TODO: set int8 backend accordingly if needed
        return self.module(*args, **kwargs)


def load_torchscript(model_path):
    extra_files = {}
    # NOTE: may support loading extra_file specified by model_info
    # extra_files["predictor_info.json"] = ""

    with PathManager.open(model_path, "rb") as f:
        ts = torch.jit.load(f, _extra_files=extra_files)

    return TorchscriptWrapper(ts)


def tracing_adapter_wrap_export(old_f):
    def new_f(cls, model, input_args, *args, **kwargs):
        adapter = TracingAdapter(model, input_args)
        load_kwargs = old_f(cls, adapter, adapter.flattened_inputs, *args, **kwargs)
        inputs_schema = dump_dataclass(adapter.inputs_schema)
        outputs_schema = dump_dataclass(adapter.outputs_schema)
        assert "inputs_schema" not in load_kwargs
        assert "outputs_schema" not in load_kwargs
        load_kwargs.update(
            {"inputs_schema": inputs_schema, "outputs_schema": outputs_schema}
        )
        return load_kwargs

    return new_f


class TracingAdapterModelWrapper(nn.Module):
    def __init__(self, traced_model, inputs_schema, outputs_schema):
        super().__init__()
        self.traced_model = traced_model
        self.inputs_schema = inputs_schema
        self.outputs_schema = outputs_schema

    def forward(self, *input_args):
        flattened_inputs, _ = flatten_to_tuple(input_args)
        flattened_outputs = self.traced_model(*flattened_inputs)
        return self.outputs_schema(flattened_outputs)


def tracing_adapter_wrap_load(old_f):
    def new_f(cls, save_path, **load_kwargs):
        assert "inputs_schema" in load_kwargs, load_kwargs.keys()
        assert "outputs_schema" in load_kwargs, load_kwargs.keys()
        inputs_schema = instantiate(load_kwargs.pop("inputs_schema"))
        outputs_schema = instantiate(load_kwargs.pop("outputs_schema"))
        traced_model = old_f(cls, save_path, **load_kwargs)

        return TracingAdapterModelWrapper(traced_model, inputs_schema, outputs_schema)

    return new_f


@ModelExportMethodRegistry.register("torchscript")
@ModelExportMethodRegistry.register("torchscript_int8")
@ModelExportMethodRegistry.register("torchscript_mobile")
@ModelExportMethodRegistry.register("torchscript_mobile_int8")
class DefaultTorchscriptExport(ModelExportMethod):
    @classmethod
    def export(
        cls,
        model: nn.Module,
        input_args: Tuple[Tuple[torch.Tensor]],
        save_path: str,
        export_method: Optional[str],
        **export_kwargs
    ):
        if export_method is not None:
            # update export_kwargs based on export_method
            assert isinstance(export_method, str)
            if "_mobile" in export_method:
                if "mobile_optimization" in export_kwargs:
                    logger.warning(
                        "`mobile_optimization` is already specified, keep using it"
                    )
                else:
                    export_kwargs["mobile_optimization"] = MobileOptimizationConfig()

        torchscript_filename = trace_and_save_torchscript(
            model, input_args, save_path, **export_kwargs
        )
        return {TORCHSCRIPT_FILENAME_KEY: torchscript_filename}

    @classmethod
    def load(cls, save_path, *, torchscript_filename="model.jit"):
        model_path = os.path.join(save_path, torchscript_filename)
        return load_torchscript(model_path)


@ModelExportMethodRegistry.register("torchscript@tracing")
@ModelExportMethodRegistry.register("torchscript_int8@tracing")
@ModelExportMethodRegistry.register("torchscript_mobile@tracing")
@ModelExportMethodRegistry.register("torchscript_mobile_int8@tracing")
class D2TorchscriptTracingExport(DefaultTorchscriptExport):
    @classmethod
    @tracing_adapter_wrap_export
    def export(cls, model, input_args, save_path, export_method, **export_kwargs):
        with patch_builtin_len():
            return super().export(
                model, input_args, save_path, export_method, **export_kwargs
            )

    @classmethod
    @tracing_adapter_wrap_load
    def load(cls, save_path, **load_kwargs):
        return super().load(save_path, **load_kwargs)
