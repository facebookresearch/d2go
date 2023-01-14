#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved


import contextlib
import logging
import os
from typing import Any, AnyStr, Dict, List, NamedTuple, Optional, Set, Tuple

import torch
from d2go.export.api import ModelExportMethod, ModelExportMethodRegistry
from detectron2.config.instantiate import dump_dataclass, instantiate
from detectron2.export import dump_torchscript_IR
from detectron2.export.flatten import flatten_to_tuple, TracingAdapter
from detectron2.export.torchscript_patch import patch_builtin_len
from detectron2.utils.file_io import PathManager
from mobile_cv.common.misc.file_utils import make_temp_directory
from mobile_cv.common.misc.iter_utils import recursive_iterate
from torch import nn
from torch.utils.bundled_inputs import augment_model_with_bundled_inputs
from torch.utils.mobile_optimizer import MobileOptimizerType, optimize_for_mobile


logger = logging.getLogger(__name__)

TORCHSCRIPT_FILENAME_KEY: str = "torchscript_filename"
DEFAULT_JIT_MODE = "trace"


class MobileOptimizationConfig(NamedTuple):
    # optimize_for_mobile
    optimization_blocklist: Set[MobileOptimizerType] = None
    preserved_methods: List[AnyStr] = None
    backend: str = "CPU"
    torchscript_filename: str = "mobile_optimized.ptl"


def export_optimize_and_save_torchscript(
    model: nn.Module,
    inputs: Optional[Tuple[Any]],
    output_path: str,
    *,
    jit_mode: Optional[str] = DEFAULT_JIT_MODE,
    torchscript_filename: str = "model.jit",
    mobile_optimization: Optional[MobileOptimizationConfig] = None,
    _extra_files: Optional[Dict[str, bytes]] = None,
) -> str:
    """
    The primary function for exporting PyTorch model to TorchScript.

    Args:
        model (nn.Module): the model to export. When given a ScriptModule, skip the export
            and only optimize and save model.
        inputs (tuple or None): input arguments of model, can be called as model(*inputs).
            Will not be used when scripting the model.
        output_path (str): directory that the model will be saved.
        jit_mode (str): trace/script or None if the model is already a ScriptModule.
        torchscript_filename (str): the filename of non-mobile-optimized model.
        mobile_optimization (MobileOptimizationConfig): when provided, the mobile optimization
            will be applied.
        _extra_files (Dict[str, bytes]): when provided, extra files will be saved.

    Returns:
        (str): filename of the final model no matter optmized or not.
    """

    logger.info("Export, optimize and saving TorchScript to {} ...".format(output_path))
    PathManager.mkdirs(output_path)
    if _extra_files is None:
        _extra_files = {}

    if isinstance(model, torch.jit.ScriptModule):
        if jit_mode is not None:
            logger.info("The input model is already a ScriptModule, skip the jit step")
    elif jit_mode == "trace":
        logger.info("Tracing the model ...")
        with torch.no_grad():
            script_model = torch.jit.trace(model, inputs)
    elif jit_mode == "script":
        logger.info("Scripting the model ...")
        script_model = torch.jit.script(model)
    else:
        raise ValueError("Unsupported jit_mode: {}".format(jit_mode))

    with make_temp_directory("export_optimize_and_save_torchscript") as tmp_dir:

        @contextlib.contextmanager
        def _synced_local_file(rel_path):
            remote_file = os.path.join(output_path, rel_path)
            local_file = os.path.join(tmp_dir, rel_path)
            yield local_file
            PathManager.copy_from_local(local_file, remote_file, overwrite=True)

        with _synced_local_file(torchscript_filename) as model_file:
            logger.info(f"Saving torchscript model to: {torchscript_filename}")
            torch.jit.save(script_model, model_file, _extra_files=_extra_files)
        dump_torchscript_IR(script_model, os.path.join(output_path, "torchscript_IR"))

        data_filename = "data.pth"
        with _synced_local_file(data_filename) as data_file:
            logger.info(f"Saving example data to: {data_filename}")
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
                logger.info(f"Saving mobile optimized model to: {torchscript_filename}")
                liteopt_model._save_for_lite_interpreter(
                    lite_path, _extra_files=_extra_files
                )

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

            # For non-cpu backends (e.g. Metal, Vulkan) the bundled inputs need to be
            # converted with `torch.to(<myDevice>)` in order to predict successfully
            # This is a temporary bypass until PT Edge supports automatic backend
            # conversion in the bundled inputs interface, or we can auto-add a input tensor
            # conversion op to Metal and Vulkan models.
            target_backend = mobile_optimization.backend.lower()
            if target_backend == "cpu":
                # Sanity check by running
                logger.info("Running sanity check for the mobile optimized model ...")
                liteopt_model(*liteopt_model.get_all_bundled_inputs()[0])
            name, ext = os.path.splitext(torchscript_filename)
            input_bundled_path = name + "_bundled" + ext
            with _synced_local_file(input_bundled_path) as lite_path:
                logger.info(f"Saving input bundled model to: {input_bundled_path}")
                liteopt_model._save_for_lite_interpreter(lite_path)

        return torchscript_filename


# For backward compatibility, TODO: remove this function.
def trace_and_save_torchscript(
    model: nn.Module,
    inputs: Optional[Tuple[Any]],
    output_path: str,
    torchscript_filename: str = "model.jit",
    mobile_optimization: Optional[MobileOptimizationConfig] = None,
    _extra_files: Optional[Dict[str, bytes]] = None,
):
    return export_optimize_and_save_torchscript(
        model,
        inputs,
        output_path,
        jit_mode="trace",
        torchscript_filename=torchscript_filename,
        mobile_optimization=mobile_optimization,
        _extra_files=_extra_files,
    )


class TorchscriptWrapper(nn.Module):
    """ """

    def __init__(self, module, int8_backend=None):
        super().__init__()
        self.module = module
        self.int8_backend = int8_backend

    def forward(self, *args, **kwargs):
        # TODO: set int8 backend accordingly if needed
        return self.module(*args, **kwargs)

    def get_wrapped_models(self):
        return self.module


def load_torchscript(model_path):
    extra_files = {}
    # NOTE: may support loading extra_file specified by model_info
    # extra_files["predictor_info.json"] = ""

    with PathManager.open(model_path, "rb") as f:
        ts = torch.jit.load(f, _extra_files=extra_files)

    return TorchscriptWrapper(ts)


def _is_data_flattened_tensors(data):
    if isinstance(data, torch.Tensor):
        return True

    if isinstance(data, (tuple, list)):
        if all(isinstance(x, torch.Tensor) for x in data):
            return True

    return False


def tracing_adapter_wrap_export(old_f):
    def new_f(cls, model, input_args, save_path, export_method, **export_kwargs):
        force_disable_tracing_adapter = export_kwargs.pop(
            "force_disable_tracing_adapter", False
        )
        is_trace_mode = export_kwargs.get("jit_mode", "trace") == "trace"
        if force_disable_tracing_adapter or not is_trace_mode:
            logger.info("Not trace mode, export normally")
            return old_f(
                cls, model, input_args, save_path, export_method, **export_kwargs
            )

        if _is_data_flattened_tensors(input_args):
            logger.info("Dry run the model to check if TracingAdapter is needed ...")
            outputs = model(*input_args)
            if _is_data_flattened_tensors(outputs):
                logger.info(
                    "Both inputs and outputs are flattened tensors, export the model as is."
                )
                load_kwargs = old_f(
                    cls, model, input_args, save_path, export_method, **export_kwargs
                )
                assert "tracing_adapted" not in load_kwargs
                load_kwargs.update({"tracing_adapted": False})
                return load_kwargs
            else:
                logger.info(
                    "The outputs are not flattened tensors, can't trace normally."
                )
        else:
            logger.info("The inputs are not flattened tensors, can't trace normally.")

        logger.warning(
            "Wrap the model with TracingAdapter to handle non-flattened inputs/outputs,"
            " please be aware that the exported model will have different input/output data structure."
        )
        adapter = TracingAdapter(model, input_args)
        load_kwargs = old_f(
            cls,
            adapter,
            adapter.flattened_inputs,
            save_path,
            export_method,
            **export_kwargs,
        )
        inputs_schema = dump_dataclass(adapter.inputs_schema)
        outputs_schema = dump_dataclass(adapter.outputs_schema)
        assert "tracing_adapted" not in load_kwargs
        assert "inputs_schema" not in load_kwargs
        assert "outputs_schema" not in load_kwargs
        load_kwargs.update(
            {
                "tracing_adapted": True,
                "inputs_schema": inputs_schema,
                "outputs_schema": outputs_schema,
            }
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

    def get_wrapped_models(self):
        return self.traced_model


def tracing_adapter_wrap_load(old_f):
    def new_f(cls, save_path, **load_kwargs):
        tracing_adapted = load_kwargs.pop("tracing_adapted", False)
        if not tracing_adapted:
            logger.info("The model is not tracing adapted, load it normally.")
            return old_f(cls, save_path, **load_kwargs)

        logger.info(
            "The model is tracing adapted, load the schema and wrap the model for inference."
        )
        assert "inputs_schema" in load_kwargs, load_kwargs.keys()
        assert "outputs_schema" in load_kwargs, load_kwargs.keys()
        inputs_schema = instantiate(load_kwargs.pop("inputs_schema"))
        outputs_schema = instantiate(load_kwargs.pop("outputs_schema"))
        traced_model = old_f(cls, save_path, **load_kwargs)

        return TracingAdapterModelWrapper(traced_model, inputs_schema, outputs_schema)

    return new_f


def update_export_kwargs_from_export_method(old_f):
    """
    Provide some convenient way of updating export_kwargs by adding trigger words in
    `export_method`. For example, instead of setting `mobile_optimization` in the
    model_export_kwargs of the PredictorExportConfig, user can simply put the `_mobile`
    trigger word in the --predictor-type (which will then be forwarded as `export_method`
    in most cases) to enable mobile optimizaiton.

    Please note that there's a finite set of allowed "export_method" values,
    and an error will be raised if the string cannot be fully parsed.
    The recognized values generally follow a pattern of:
        "torchscript[_mobile][_int8][-vulkan | -metal][@scripting | @tracing]"

    Some examples (not comprehensive because flag words' order can be swapped):
        "torchscript"
        "torchscript_mobile"
        "torchscript_mobile-metal"
        "torchscript_mobile-vulkan"
        "torchscript_mobile_int8"
        "torchscript@scripting"
        "torchscript_int8@scripting"
        "torchscript_mobile@scripting"
        "torchscript_mobile-metal@scripting"
        "torchscript_mobile-vulkan@scripting"
        "torchscript_mobile_int8@scripting"
        "torchscript@tracing"
        "torchscript_int8@tracing"
        "torchscript_mobile@tracing"
        "torchscript_mobile-metal@tracing"
        "torchscript_mobile-vulkan@tracing"
        "torchscript_mobile_int8@tracing"
    """

    def new_f(cls, model, input_args, save_path, export_method, **export_kwargs):
        if export_method is not None:
            assert isinstance(export_method, str)
            original_export_method = export_method

            if "_mobile" in export_method:
                if "mobile_optimization" in export_kwargs:
                    logger.warning(
                        "`mobile_optimization` is already specified, keep using it"
                    )
                else:
                    # Infer a MobileOptimizationConfig if none was provided
                    # "CPU" backend default. If found appropriate suffix, update the backend
                    if "-metal" in export_method:
                        mobile_opt_config = MobileOptimizationConfig(backend="metal")
                        export_method = export_method.replace("-metal", "", 1)
                    elif "-vulkan" in export_method:
                        mobile_opt_config = MobileOptimizationConfig(backend="vulkan")
                        export_method = export_method.replace("-vulkan", "", 1)
                    else:
                        mobile_opt_config = MobileOptimizationConfig()
                    export_kwargs["mobile_optimization"] = mobile_opt_config

                export_method = export_method.replace("_mobile", "", 1)

            if "@scripting" in export_method:
                jit_mode = export_kwargs.get("jit_mode", None)
                if jit_mode and jit_mode != "script":
                    logger.warning(
                        "`jit_mode` is already specified as {}, overwrite it to `script`"
                        " since @scripting appears in export_method".format(jit_mode)
                    )
                export_kwargs["jit_mode"] = "script"
                export_method = export_method.replace("@scripting", "", 1)

            if "@tracing" in export_method:
                jit_mode = export_kwargs.get("jit_mode", None)
                if jit_mode and jit_mode != "trace":
                    logger.warning(
                        "`jit_mode` is already specified as {}, overwrite it to `trace`"
                        " since @tracing appears in export_method".format(jit_mode)
                    )
                export_kwargs["jit_mode"] = "trace"
                export_method = export_method.replace("@tracing", "", 1)

            if "_int8" in export_method:
                export_method = export_method.replace("_int8", "", 1)

            if export_method != "torchscript":
                logger.warning(
                    "Suspcious export_method after removing triggering words,"
                    " original export_method: {}, remaining: {}".format(
                        original_export_method, export_method
                    )
                )

        return old_f(cls, model, input_args, save_path, export_method, **export_kwargs)

    return new_f


class DefaultTorchscriptExport(ModelExportMethod):
    @classmethod
    @update_export_kwargs_from_export_method
    def export(
        cls,
        model: nn.Module,
        input_args: Tuple[Tuple[torch.Tensor]],
        save_path: str,
        export_method: Optional[str],
        **export_kwargs,
    ):
        expected_arguments = {
            "jit_mode",
            "torchscript_filename",
            "mobile_optimization",
            "_extra_files",
        }
        filtered_kwargs = {
            k: v for k, v in export_kwargs.items() if k in expected_arguments
        }

        torchscript_filename = export_optimize_and_save_torchscript(
            model, input_args, save_path, **filtered_kwargs
        )
        return {TORCHSCRIPT_FILENAME_KEY: torchscript_filename}

    @classmethod
    def load(cls, save_path, *, torchscript_filename="model.jit"):
        model_path = os.path.join(save_path, torchscript_filename)
        return load_torchscript(model_path)


@ModelExportMethodRegistry.register("torchscript")
@ModelExportMethodRegistry.register("torchscript_int8")
@ModelExportMethodRegistry.register("torchscript_mobile")
@ModelExportMethodRegistry.register("torchscript_mobile-metal")
@ModelExportMethodRegistry.register("torchscript_mobile-vulkan")
@ModelExportMethodRegistry.register("torchscript_mobile_int8")
@ModelExportMethodRegistry.register("torchscript@scripting")
@ModelExportMethodRegistry.register("torchscript_int8@scripting")
@ModelExportMethodRegistry.register("torchscript_mobile@scripting")
@ModelExportMethodRegistry.register("torchscript_mobile-metal@scripting")
@ModelExportMethodRegistry.register("torchscript_mobile-vulkan@scripting")
@ModelExportMethodRegistry.register("torchscript_mobile_int8@scripting")
@ModelExportMethodRegistry.register("torchscript@tracing")
@ModelExportMethodRegistry.register("torchscript_int8@tracing")
@ModelExportMethodRegistry.register("torchscript_mobile@tracing")
@ModelExportMethodRegistry.register("torchscript_mobile-metal@tracing")
@ModelExportMethodRegistry.register("torchscript_mobile-vulkan@tracing")
@ModelExportMethodRegistry.register("torchscript_mobile_int8@tracing")
class TracingAdaptedTorchscriptExport(DefaultTorchscriptExport):
    @classmethod
    @update_export_kwargs_from_export_method
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
