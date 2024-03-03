# pyre-ignore-all-errors
import functools
from abc import ABC
from copy import deepcopy
from dataclasses import dataclass
from types import MethodType
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union

import torch
from d2go.config import CfgNode
from d2go.quantization.modeling import prepare_fake_quant_model
from d2go.utils.misc import mode
from mobile_cv.arch.quantization.observer import update_stat as observer_update_stat
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.utilities import rank_zero_info
from torch.ao.quantization.qconfig import (
    get_default_qat_qconfig,
    get_default_qconfig,
    QConfig,
    QConfigDynamic,
)
from torch.ao.quantization.quant_type import QuantType
from torch.ao.quantization.quantize_fx import convert_fx, prepare_fx, prepare_qat_fx
from torch.ao.quantization.utils import get_fqn_to_example_inputs, get_quant_type


QConfigDicts = Dict[str, Dict[str, Union[QConfig, QConfigDynamic]]]
PREPARED = "_prepared"


def rsetattr(obj: Any, attr: str, val: Any) -> None:
    """Same as setattr but supports deeply nested objects."""
    pre, _, post = attr.rpartition(".")
    return setattr(rgetattr(obj, pre) if pre else obj, post, val)


def rgetattr(obj: Any, attr: str, *args) -> Any:
    """Same as getattr but supports deeply nested objects."""

    def _getattr(obj, attr):
        return getattr(obj, attr, *args)

    return functools.reduce(_getattr, [obj] + attr.split("."))


def rhasattr(obj: Any, attr: str, *args) -> bool:
    """Same as hasattr but supports deeply nested objects."""

    try:
        _ = rgetattr(obj, attr, *args)
    except AttributeError:
        return False
    return True


def _quantized_forward(self, *args, **kwargs):
    """Forward method for a quantized module."""
    if not self.training and hasattr(self, "_quantized"):
        return self._quantized(*args, **kwargs)
    return self._prepared(*args, **kwargs)


def _requires_calibration(config_dicts: QConfigDicts) -> bool:
    """Returns whether the given config_dicts for quantization requires calibration.

    A config_dicts requires calibration if at least one of the configs in the
    dictioary is a QConfig with an activation observer.

    Args:
        config: The config dictionary to check.

    Returns:
        Boolean as described.
    """
    for qconfig_dict in config_dicts.values():
        for qconfig in qconfig_dict.values():
            qtype = get_quant_type(qconfig)
            if qtype == QuantType.STATIC:
                return True
    return False


def checkpoint_has_prepared(checkpoint: Dict[str, Any]) -> bool:
    return any(k.startswith(PREPARED) for k in checkpoint["state_dict"].keys())


def maybe_prepare_for_quantization(model: LightningModule, checkpoint: Dict[str, Any]):
    if checkpoint_has_prepared(checkpoint) and not hasattr(model, PREPARED):
        # model has been prepared for QAT before saving into checkpoint
        copied = deepcopy(model)
        prepared = prepare_fake_quant_model(copied.cfg, copied.model, is_qat=True)
        copied.model = prepared
        setattr(model, PREPARED, copied)


class QuantizationMixin(ABC):
    """Mixin defining an overrideable API for quantization customization.

    For example, suppose our model contains traceable and non-traceable modules:

    >>> class MyNonTraceableModel(LightningModule):
    ...     def __init__(self):
    ...         self.traceable = ...
    ...         self.non_traceable = ...
    ...
    ...     def forward(self, x):
    ...         x = self.traceable(x)
    ...         return self.non_traceable(x)

    Then using FX-mode quantization, we can only quantize the traceable pieces.
    As such, we could do something like the below, shown here for QAT.

    >>> class MyQuantizationCallback(QuantizedAwareTraining):
    ...     def prepare(self, model, config, attrs):
    ...         model.traceable = prepare_qat_fx(model.traceable, config)
    ...         return model
    ...
    ...     def convert(self, model, attr):
    ...         model.traceable = convert_fx(model.traceable)
    ...         return model

    We can then use this callback as with any other.:

    Example::
        >>> model = MyNonTraceableModel(...)
        >>> quantization = MyQuantizationCallback()
        >>> trainer = Trainer(
        ...    callbacks=[quantization],
        ... )
        >>> trainer.fit(model)


    """

    def prepare(
        self, root: LightningModule, configs: QConfigDicts, attrs: Set[str]
    ) -> torch.nn.Module:
        """Prepares the root user modules for quantization.

        By default, this tries to prepare the entire LightningModule. If this is
        not possible (eg, due to traceability, etc.), the recommended method to
        use is to override the `prepare` method to prepare the root as
        appropriate, and also override the `quantize` method to only quantize
        the prepared pieces of the root.

        Args:
            root: The LightningModule as given to the lightning Trainer in train mode.
            configs: Specification to be used when preparing the model, as provided by the user.
                It is guaranteed that no key is a suffix of another.
            attrs: The list of attributes to maintain for the module.

        Returns:
            The prepared Module to be used for quantized aware training.
        """
        is_qat = isinstance(self, QuantizationAwareTraining)
        self._convert_fx_callback = None
        if hasattr(root.model, "custom_prepare_fx"):
            prepared, convert_fx_callback = root.model.custom_prepare_fx(
                root.cfg, is_qat
            )
            self._convert_fx_callback = convert_fx_callback
            root.model = prepared
            return root
        prep_fn = prepare_qat_fx if is_qat else prepare_fx
        old_attrs = {
            attr: rgetattr(root, attr) for attr in attrs if rhasattr(root, attr)
        }
        prepared = root
        if "" in configs:
            prepared = prep_fn(root, configs[""], root.example_input_array)
        else:
            fqn_to_example_inputs = get_fqn_to_example_inputs(
                root, root.example_input_array
            )

            for name, config in configs.items():
                submodule = rgetattr(root, name)
                rsetattr(
                    root, name, prep_fn(submodule, config, fqn_to_example_inputs[name])
                )

        for attr, value in old_attrs.items():
            rsetattr(prepared, attr, value)
        return prepared

    def convert(
        self, root: torch.nn.Module, submodules: Set[str], attrs: Set[str]
    ) -> torch.nn.Module:
        """Quantizes a previously prepared module (as returned by `prepare`).

        By default, this simply quantizes the entire root. If the `prepare`
        method was customized, this will need to be changed as well.

        Args:
            root: The prepared model as returned by `prepare`, after training.
            submodules: An iterator of fully qualified submodules names that require
            converting.
            attrs: The list of attributes to maintain for the module across this call.

        Returns:
            The quantized model.
        """
        if self._convert_fx_callback is not None:
            return self._convert_fx_callback(root)
        old_attrs = {
            attr: rgetattr(root, attr) for attr in attrs if rhasattr(root, attr)
        }
        converted = root
        if "" in submodules:
            converted = convert_fx(root)
        else:
            for name in submodules:
                prepared = rgetattr(root, name)
                rsetattr(root, name, convert_fx(prepared))
        for attr, value in old_attrs.items():
            rsetattr(converted, attr, value)
            rsetattr(root, attr, value)
        return converted


@dataclass(frozen=True)
class ModelTransform:
    """Defines a step or interval at which fn should be .apply(fn)'ed and a message to log.

    Properties:
        fn: The function to apply. Must be passable to torch.nn.Module.apply(fn).
        step: Only one of `step` or `interval` must be defined. If step is defined,
             `fn` will be applied exactly once right before `step` step begins.
        interval: Only one of `step` or `interval` must be defined. If `interval`
            is defined, the transform will be applied periodically every
            `interval` steps.
        message: A short non-punctuated message to log in the master worker when
        this transform is triggered.
    """

    fn: Callable[[torch.nn.Module], None]
    message: str
    step: Optional[int] = None
    interval: Optional[int] = None

    def __post_init__(self) -> None:
        """Validate a few properties for early failure."""
        if (self.step is None and self.interval is None) or (
            self.step is not None and self.interval is not None
        ):
            raise TypeError("Exactly one of step or interval must be defined.")
        if self.step is not None and self.step < 0:
            raise ValueError("step must be non-negative.")
        if self.interval is not None and self.interval <= 0:
            raise ValueError("interval must be positive.")


class QuantizationAwareTraining(Callback, QuantizationMixin):
    """Enable QAT of a model using the STL Trainer.

    Node that this callback makes changes during training in order to properly
    quantize the provided LightningModule.

    Example::
        >>> from stl.lightning.callbacks.quantization import QuantizationAwareTraining
        >>> from pytorch_lightning import Trainer
        >>> from stl.lightning.utilities.model import mode

        ...

        # MyLightningModule must define val_dataloader() which is used both for
        # validation as well as calibration of the quantized model.
        >>> model = MyLightningModule(...)
        >>> qat = QuantizationAwareTraining()
        >>> trainer = Trainer(
        ...    callbacks=[qat],
        ... )

        # This will convert the model into one that is quantizeable, train it,
        # and then quantize it after training is done.
        >>> trainer.fit(model)

        # You can use the model directly.
        >>> input = ...
        >>> with mode(model, training=False) as m:
        ...     quantized_out = m(input)

    If you only wish to quantize parts of your model, please see QuantizationMixin
    for an example of how to do this.

    Properties:
        transforms: A list of ModelTransform's applied to the model exactly once
            as specified during training. Example transforms are enabling/disabling
            observers/quants, which are added to this list based on the init
            parameters to this callback. Users can further augment the list
            with more custom modules.
        prepared: If set, this is the prepared model. Only available
            after .fit() starts.
        qconfig_dicts:
            This is a map from the `module_qualified_name` to the corresponding QConfigDict
            to apply to that module. For example, suppose your LightningModule contains
            two submodules module.scriptable and module.not_scriptable. You'd provide
            a qconfig_dicts like:
                {
                    "scriptable": ...
                }
            This will quantize just module.scriptable using the provided QConfigDict,
            or a default one. If you wish to quantize the entire LightningModule,
            simply use "" as the qualified name. The name should match the names
            returned by module.named_modules().
        quantized: If set, this is the fully quantized model. Only available
            after .fit() finishes.
    """

    def __init__(
        self,
        start_step: int = 0,
        enable_observer: Tuple[int, Optional[int]] = (0, None),
        freeze_bn_step: Optional[int] = None,
        qconfig_dicts: Optional[
            Dict[str, Optional[Dict[str, Union[QConfig, QConfigDynamic]]]]
        ] = None,
        preserved_attrs: Optional[List[str]] = None,
        skip_conversion: bool = False,
    ) -> None:
        """
        Args:
            start_step: The training step at which QAT is enabled. The model is
                always mutated with the appropriate stubs, but they are disabled
                until the start of this training step.
                See FakeQuantizeBase.fake_quant_enabled
            enable_observer: The half-open interval [a, b) in steps during which the
                observers are enabled. See FakeQuantizeBase.observer_enabled. If
                b is None, the observer is never disabled once enabled.
            freeze_bn_step: If specified, the step at which we apply freeze the
                collection of batch normalization layer statistics for QAT.
            qconfig_dicts: If given, used for quantization of the model during training.
            preserved_attrs: If provided, a list of attributes to preserve across
                quantized modules. These are preserved only if they already exists.
        """
        if start_step < 0:
            raise ValueError(
                f"The starting step of QAT must be non-negative. Got {start_step}."
            )
        start_observer, end_observer = enable_observer
        if start_observer < 0:
            raise ValueError(
                f"The starting step for the observer must be non-negative. Got {start_observer}."
            )
        if end_observer and end_observer <= start_observer:
            raise ValueError(
                f"The observation interval must contain at least one step. Got [{start_step}, {end_observer})."
            )
        if freeze_bn_step and freeze_bn_step < 0:
            raise ValueError(
                f"The step at which batch norm layers are frozen must be non-negative. Got {freeze_bn_step}."
            )
        self.transforms: List[ModelTransform] = []
        if start_step > 0:
            self.transforms.extend(
                [
                    # Enabled by default, so the assumption for > 0 is that the
                    # user wants it disabled then enabled.
                    ModelTransform(
                        fn=torch.ao.quantization.disable_fake_quant,
                        step=0,
                        message="Disable fake quant",
                    ),
                    ModelTransform(
                        fn=torch.ao.quantization.enable_fake_quant,
                        step=start_step,
                        message="Enable fake quant to start QAT",
                    ),
                ]
            )
        if start_observer > 0:
            self.transforms.extend(
                # See comment for start_step above.
                [
                    ModelTransform(
                        fn=torch.ao.quantization.disable_observer,
                        step=0,
                        message="Disable observer",
                    ),
                    ModelTransform(
                        fn=torch.ao.quantization.enable_observer,
                        step=start_observer,
                        message="Start observer",
                    ),
                ]
            )
        if end_observer is not None:
            self.transforms.append(
                ModelTransform(
                    fn=torch.ao.quantization.disable_observer,
                    step=end_observer,
                    message="End observer",
                )
            )
        if freeze_bn_step is not None:
            self.transforms.append(
                ModelTransform(
                    fn=torch.nn.intrinsic.qat.freeze_bn_stats,
                    step=freeze_bn_step,
                    message="Freeze BN",
                )
            )

        self.prepared: Optional[torch.nn.Module] = None
        self.preserved_attrs = set([] if preserved_attrs is None else preserved_attrs)
        if not qconfig_dicts:
            self.qconfig_dicts: QConfigDicts = {"": {"": get_default_qat_qconfig()}}
        else:
            self.qconfig_dicts: QConfigDicts = {
                key: value if value else {"": get_default_qat_qconfig()}
                for key, value in qconfig_dicts.items()
            }
        self.quantized: Optional[torch.nn.Module] = None
        self.skip_conversion = skip_conversion

    @classmethod
    def from_config(cls, cfg: CfgNode):
        qat = cfg.QUANTIZATION.QAT
        callback = cls(
            qconfig_dicts=(
                {submodule: None for submodule in cfg.QUANTIZATION.MODULES}
                if cfg.QUANTIZATION.MODULES
                else None
            ),
            # We explicitly pass this to maintain properties for now.
            preserved_attrs=["model.backbone.size_divisibility"],
            start_step=qat.START_ITER,
            enable_observer=(qat.ENABLE_OBSERVER_ITER, qat.DISABLE_OBSERVER_ITER),
            freeze_bn_step=qat.FREEZE_BN_ITER,
            skip_conversion=True,  # convert_fx will be handled by D2Go exporter
        )
        if qat.UPDATE_OBSERVER_STATS_PERIODICALLY:
            callback.transforms.append(
                ModelTransform(
                    interval=qat.UPDATE_OBSERVER_STATS_PERIOD,
                    fn=observer_update_stat,
                    message="Updating observers.",
                )
            )
        return callback

    def setup(self, trainer: Trainer, pl_module: LightningModule, stage: str) -> None:
        """Override the model with a quantized-aware version on setup.

        This is the earliest place we can override this model which allows for
        appropriate behavior when restoring from checkpoints, as well as connecting
        to accelerators, etc.

        The model is only prepared once.
        """
        # Only prepare the model once.
        if hasattr(pl_module, "_prepared"):
            return

        with mode(pl_module, training=True) as train:
            prepared = self.prepare(
                deepcopy(train),
                configs=self.qconfig_dicts,
                attrs=self.preserved_attrs,
            )
            # freeze the original model since only the prepared model will
            # participate in forward.
            for x in train.parameters():
                x.requires_grad = False
            pl_module._prepared = prepared
        pl_module.forward = MethodType(_quantized_forward, pl_module)
        self.prepared = pl_module._prepared

    def on_train_batch_start(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        batch: Any,
        batch_idx: int,
    ) -> None:
        """Applies model transforms at as specified during training."""
        apply_only_once = []
        current_step = trainer.global_step
        for i, transform in enumerate(self.transforms):
            if (transform.step is not None and transform.step <= current_step) or (
                transform.interval is not None
                and current_step % transform.interval == 0
            ):
                self.prepared.apply(transform.fn)
                rank_zero_info(
                    f"[QAT] {transform.message} at step={trainer.global_step}."
                )
            if transform.step is not None and transform.step <= current_step:
                apply_only_once.append(i)

        if apply_only_once:
            self.transforms = [
                transform
                for i, transform in enumerate(self.transforms)
                if i not in set(apply_only_once)
            ]

    def on_fit_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Quantize the weights since training has finalized."""
        if hasattr(pl_module, "_quantized") or self.skip_conversion:
            return
        pl_module._quantized = self.convert(
            pl_module._prepared, self.qconfig_dicts.keys(), attrs=self.preserved_attrs
        )
        self.quantized = pl_module._quantized

    def on_test_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Make sure we have a quantized version.

        This handles the edge case where a user does .test() without .fit() first.
        """
        if hasattr(pl_module, "_quantized"):
            return
        pl_module._quantized = self.convert(
            pl_module._prepared, self.qconfig_dicts.keys(), attrs=self.preserved_attrs
        )
        self.quantized = pl_module._quantized


class PostTrainingQuantization(Callback, QuantizationMixin):
    """Enable post-training quantization, such as dynamic, static, and weight-only.

    This is an idempotent callback (to contrast with QuantizationAwareTraining).

    If calibration is required, we will use the validation data set provided to
    the STL Trainer, and this occurs on each validation run.

    The quantized model is made available as a property of the callback.

    Example::
        >>> from stl.lightning.callbacks.quantization import PostTrainingQuantization
        >>> from pytorch_lightning import Trainer
        >>> from stl.lightning.utilities.model import mode

        ...

        # MyLightningModule must define val_dataloader() which is used both for
        # validation as well as calibration of the quantized model.
        >>> model = MyLightningModule(...)
        >>> post_training_quant = PostTrainingQuantization()
        >>> trainer = Trainer(
        ...    callbacks=[post_training_quant],
        ... )

        # This will both train the model + create a *separate* quantized version.
        # The original model is left unchaged.
        >>> trainer.fit(model)

        # You can access the quantized version of the model directly.
        >>> input = ...
        >>> with mode(post_training_quant.quantized, training=False) as m:
        ...     quantized_out = m(input)

    If you only wish to quantize parts of your model, please see QuantizationMixin
    for an example of how to do this.

    Properties:
        prepared: If set, this is the prepared model which can be used for
            calibration. Only available after validation start.
        qconfig_dicts: See `QuantizedAwareTraining` for full description.
        quantized: If set, this is the fully quantized model calibrated using
            the validation data. Only available after validation has ended.
    """

    def __init__(
        self,
        qconfig_dicts: Optional[QConfigDicts] = None,
        preserved_attrs: Optional[List[str]] = None,
    ) -> None:
        """Initialize the callback."""
        self.qconfig_dicts = qconfig_dicts or {"": {"": get_default_qconfig()}}
        self.preserved_attrs = set([] if preserved_attrs is None else preserved_attrs)
        self.prepared: Optional[torch.nn.Module] = None
        self.quantized: Optional[torch.nn.Module] = None
        self.should_calibrate = _requires_calibration(self.qconfig_dicts)

    @classmethod
    def from_config(cls, cfg: CfgNode):
        return cls(
            qconfig_dicts=(
                {submodule: None for submodule in cfg.QUANTIZATION.MODULES}
                if cfg.QUANTIZATION.MODULES
                else None
            ),
            # We explicitly pass this to maintain properties for now.
            preserved_attrs=["model.backbone.size_divisibility"],
        )

    def on_validation_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """
        On validation start, prepare a module for quantization by adding
        observers and loading weights from current model.
        """
        # Pass a copy to quantization APIs.
        self.prepared = self.prepare(
            deepcopy(pl_module).eval(),
            configs=self.qconfig_dicts,
            attrs=self.preserved_attrs,
        )

    def on_validation_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Convert the calibrated model to its finalized quantized version."""
        self.quantized = self.convert(
            self.prepared, self.qconfig_dicts.keys(), attrs=self.preserved_attrs
        )

    def on_validation_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: Any,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int,
    ) -> None:
        """Also run the validation batch through the quantized model for calibration."""
        if self.should_calibrate:
            with torch.no_grad():
                self.prepared(batch)
