#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved


import copy
import logging
import math
from typing import Any, Dict, Optional, Tuple

import detectron2.utils.comm as comm
import torch
from d2go.quantization import learnable_qat
from d2go.quantization.fx import get_convert_fn, get_prepare_fn, get_prepare_fx_fn
from d2go.quantization.qconfig import (
    set_backend_and_create_qconfig,
    smart_decode_backend,
)
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.engine.train_loop import HookBase, SimpleTrainer
from detectron2.utils.file_io import PathManager
from mobile_cv.arch.quantization.observer import update_stat as observer_update_stat
from mobile_cv.arch.utils import fuse_utils
from mobile_cv.common.misc.iter_utils import recursive_iterate
from torch import nn
from torch.ao.quantization.quantize_pt2e import (
    convert_pt2e,
    prepare_pt2e,
    prepare_qat_pt2e,
)
from torch.ao.quantization.quantizer.xnnpack_quantizer import (
    get_symmetric_quantization_config,
    XNNPACKQuantizer,
)

TORCH_VERSION: Tuple[int, ...] = tuple(int(x) for x in torch.__version__.split(".")[:2])
# some tests still import prepare/convert from below. So don't remove these.
if TORCH_VERSION > (1, 10):
    from torch.ao.quantization.quantize import convert
    from torch.ao.quantization.quantize_fx import convert_fx, prepare_fx, prepare_qat_fx
else:
    pass

logger = logging.getLogger(__name__)

_CONVERT_FX_CALLBACK_ATTRIBUTE = "_convert_fx_callback"
_CONVERT_PT2E_CALLBACK_ATTRIBUTE = "_convert_pt2e_callback"
_STATE_DICT_KEY = "state_dict"
_OLD_STATE_DICT_KEY = "model"
_OLD_EMA_KEY = "ema_state"


def _is_observer_key(state_dict_key):
    observer_keys = ["activation_post_process", "weight_fake_quant"]
    return any(x in state_dict_key for x in observer_keys)


# TODO: replace QATCheckpointer with central D2GoCheckpointer which supports customize
# state_dict re-mapping (which includes QAT re-mapping).
class QATCheckpointer(DetectionCheckpointer):
    """
    Extend the Checkpointer to support loading (QAT / non-QAT) weight into
    (QAT / non-QAT) model.
    """

    def __init__(
        self,
        model,
        save_dir="",
        *,
        load_ckpt_to_gpu=False,
        save_to_disk=None,
        **checkpointables,
    ):
        super().__init__(
            model,
            save_dir,
            save_to_disk=save_to_disk,
            **checkpointables,
        )
        self.load_ckpt_to_gpu = load_ckpt_to_gpu

    @classmethod
    def _is_q_state_dict(cls, state_dict):
        return any(_is_observer_key(k) for k in state_dict)

    # HACK: temporarily put it here, move to centrail D2GoCheckpointer later on
    def _load_file(self, filename):
        # support loading lightning checkpointer
        if filename.endswith(".ckpt"):
            # assume file is from lightning; no one else seems to use the ".ckpt" extension
            with PathManager.open(filename, "rb") as f:
                data = self._torch_load(f)

            _convert_to_d2(data)
            return data

        return super()._load_file(filename)

    def _torch_load(self, f):
        device = (
            "cuda:{}".format(torch.cuda.current_device())
            if self.load_ckpt_to_gpu
            else "cpu"
        )
        return torch.load(f, map_location=torch.device(device))

    def _load_model(self, checkpoint):
        model_is_qat = self._is_q_state_dict(self.model.state_dict())
        checkpoint_is_qat = self._is_q_state_dict(checkpoint["model"])

        if model_is_qat and not checkpoint_is_qat:
            logger.info("Loading QAT model with non-QAT checkpoint, ignore observers!")
            mapping = getattr(self.model, "_non_qat_to_qat_state_dict_map", {})

            # map the key from non-QAT model to QAT model if possible
            checkpoint_state_dict = {
                mapping.get(k, k): v for k, v in checkpoint["model"].items()
            }
            checkpoint["model"] = checkpoint_state_dict
            incompatible = super()._load_model(checkpoint)

            # suppress the missing observer keys warning
            # NOTE: for some reason incompatible.missing_keys can have duplicated keys,
            # here we replace the entire list rather than calling .remove()
            missing_non_qat_keys = [
                k for k in incompatible.missing_keys if not _is_observer_key(k)
            ]
            incompatible.missing_keys[:] = missing_non_qat_keys

            return incompatible

        elif not model_is_qat and checkpoint_is_qat:
            raise NotImplementedError()
        elif model_is_qat and checkpoint_is_qat:
            # TODO: maybe suppress shape mismatch
            # For models trained with QAT and per-channel quant, the inital size of the
            # buffers in fake_quant and observer modules does not reflect the size in
            # state_dict, which is updated only when convert is called.
            return super()._load_model(checkpoint)
        else:
            return super()._load_model(checkpoint)


def add_quantization_default_configs(_C):
    CfgNode = type(_C)
    _C.QUANTIZATION = CfgNode()
    # Note: EAGER_MODE == False currently represents FX graph mode quantization
    _C.QUANTIZATION.EAGER_MODE = True
    # Available backends include PyTorch's natively supported backends (i.e. fbgemm and
    # qnnpack), plus D2Go-defined backends such as "qnnpack@symmetric".
    _C.QUANTIZATION.BACKEND = "fbgemm"

    # used to enable metarch set_custom_qscheme (need to implement)
    # this is a limited implementation where only str is provided to change options
    _C.QUANTIZATION.CUSTOM_QSCHEME = ""
    _C.QUANTIZATION.MODULES = []
    # Lightning quantization callback name
    _C.QUANTIZATION.NAME = ""
    _C.QUANTIZATION.ACT_BITS = 8
    _C.QUANTIZATION.WEIGHT_BITS = 8

    # quantization-aware training
    _C.QUANTIZATION.QAT = CfgNode()
    _C.QUANTIZATION.QAT.ENABLED = False
    # Methods for QAT training, could be "default" or "learnable"
    _C.QUANTIZATION.QAT.FAKE_QUANT_METHOD = "default"
    # QAT will use more GPU memory, user can change this factor to reduce the batch size
    # after fake quant is enabled. Setting it to 0.5 should guarantee no memory increase
    # compared with QAT is disabled.
    _C.QUANTIZATION.QAT.BATCH_SIZE_FACTOR = 1.0
    # the iteration number to start QAT, (i.e. enable fake quant). The default value of
    # SOLVER.MAX_ITER is 40k and SOLVER.STEPS is (30k,), here we turn on QAT at 35k, so
    # the last 5k iterations will run with QAT with decreased learning rate.
    _C.QUANTIZATION.QAT.START_ITER = 35000
    # the iteration number to enable observer, it's usually set to be the same as
    # QUANTIZATION.QAT.START_ITER.
    _C.QUANTIZATION.QAT.ENABLE_OBSERVER_ITER = 35000
    # the iteration number to enable learnable observer, only used when METHOD == "learnable"
    _C.QUANTIZATION.QAT.ENABLE_LEARNABLE_OBSERVER_ITER = 36000
    # the iteration number to disable observer, here it's 3k after enabling the fake
    # quant, 3k roughly corresponds to 7 out of 90 epochs in classification.
    _C.QUANTIZATION.QAT.DISABLE_OBSERVER_ITER = 35000 + 3000
    # the iteration number to freeze BN, here it's 3k after enabling the fake quant, 2k
    # roughly corresponds to 5 out of 90 epochs for classification.
    _C.QUANTIZATION.QAT.FREEZE_BN_ITER = 35000 + 2000
    # qat hook will run observers update_stat if it exists
    # after update_observer_stats_period iters
    _C.QUANTIZATION.QAT.UPDATE_OBSERVER_STATS_PERIODICALLY = False
    _C.QUANTIZATION.QAT.UPDATE_OBSERVER_STATS_PERIOD = 1
    _C.QUANTIZATION.WEIGHT_OBSERVERS = None
    _C.QUANTIZATION.ACTIVATION_OBSERVERS = None

    # post-training quantization
    _C.QUANTIZATION.PTQ = CfgNode()
    _C.QUANTIZATION.PTQ.CALIBRATION_NUM_IMAGES = 16  # NOTE: this is actually iterations
    _C.QUANTIZATION.PTQ.CALIBRATION_FORCE_ON_GPU = False

    _C.QUANTIZATION.PT2E = False
    _C.QUANTIZATION.RECIPE = None

    # register deprecated and renamed keys
    _C.register_deprecated_key("QUANTIZATION.QAT.LOAD_PRETRAINED")
    _C.register_renamed_key("QUANTIZATION.QAT.BACKEND", "QUANTIZATION.BACKEND")
    _C.register_deprecated_key("QUANTIZATION.ENABLE_CUSTOM_QSCHEME")
    _C.register_deprecated_key("QUANTIZATION.SILICON_QAT")
    _C.register_deprecated_key("QUANTIZATION.SILICON_QAT.ENABLED")


# TODO: model.to(device) might not work for detection meta-arch, this function is the
# workaround, in general, we might need a meta-arch API for this if needed.
def _cast_model_to_device(model, device):
    if hasattr(
        model, "_cast_model_to_device"
    ):  # we can make this formal by removing "_"
        return model._cast_model_to_device(device)
    else:
        logger.warning(
            "model.to(device) doesn't guarentee moving the entire model, "
            "if customization is needed, please implement _cast_model_to_device "
            "for the MetaArch"
        )
        return model.to(device)


def add_d2_quant_mapping(mappings):
    """HACK: Add d2 specific module mapping for eager model quantization"""
    import torch.ao.quantization.quantization_mappings as qm

    for k, v in mappings.items():
        if k not in qm.get_default_static_quant_module_mappings():
            qm.DEFAULT_STATIC_QUANT_MODULE_MAPPINGS[k] = v
        if k not in qm.get_default_qat_module_mappings():
            qm.DEFAULT_QAT_MODULE_MAPPINGS[k] = v


# The `mock_quantization_type` decorate may not be needed anymore to unify
# detectron2.layers modules and torch.nn modules since Pytorch 1.5. See comments on D23790034.
def mock_quantization_type(quant_func):
    import builtins
    import functools
    from unittest import mock

    import detectron2.layers as d2l

    type_mapping = {d2l.Linear: torch.nn.Linear}
    from d2go.utils.misc import check_version

    if check_version(torch, "1.7.2", warning_only=True):
        add_d2_quant_mapping(type_mapping)

    real_type = builtins.type

    def _new_type(obj):
        rtype = real_type(obj)
        return type_mapping.get(rtype, rtype)

    @functools.wraps(quant_func)
    def wrapper(cfg, model, *args, **kwargs):
        if d2l.Linear == torch.nn.Linear:
            # we do not need the moc after when the type is expected, consider
            # remving those related code
            logger.warning(
                "`detectron2.layers.Linear` is in expected type (torch.nn.Linear),"
                "consider removing this code `mock_quantization_type`."
            )
            return quant_func(cfg, model, *args, **kwargs)

        if not cfg.QUANTIZATION.EAGER_MODE:
            return quant_func(cfg, model, *args, **kwargs)

        # `from_float()` in `torch.nn.quantized.modules.linear.Linear` and
        # `torch.nn.qat.modules.linear` checkes if the type of `mod` is torch.Linear,
        # hack it to return the expected value
        with mock.patch("torch.nn.quantized.modules.linear.type") as mock_type:
            with mock.patch("torch.nn.qat.modules.linear.type") as mock_type2:
                mock_type.side_effect = _new_type
                mock_type2.side_effect = _new_type
                return quant_func(cfg, model, *args, **kwargs)

    return wrapper


def default_prepare_for_quant(cfg, model):
    """
    Default implementation of preparing a model for quantization. This function will
    be called to before training if QAT is enabled, or before calibration during PTQ if
    the model is not already quantized.

    NOTE:
        - This is the simplest implementation, most meta-arch needs its own version.
        - For eager model, user should make sure the returned model has Quant/DeQuant
            insert. This can be done by wrapping the model or defining the model with
            quant stubs.
        - QAT/PTQ can be determined by model.training.
        - Currently the input model can be changed inplace since we won't re-use the
            input model.
        - Currently this API doesn't include the final torch.ao.quantization.prepare(_qat)
            call since existing usecases don't have further steps after it.

    Args:
        model (nn.Module): a non-quantized model.
        cfg (CfgNode): config

    Return:
        nn.Module: a ready model for QAT training or PTQ calibration
    """
    assert cfg.QUANTIZATION.EAGER_MODE
    qconfig = set_backend_and_create_qconfig(cfg, is_train=model.training)
    model = fuse_utils.fuse_model(
        model,
        is_qat=cfg.QUANTIZATION.QAT.ENABLED,
        inplace=True,
    )
    model.qconfig = qconfig
    # TODO(future diff): move the torch.ao.quantization.prepare(...) call
    # here, to be consistent with the FX branch

    logger.info("Setup the model with qconfig:\n{}".format(qconfig))
    return model


def default_custom_prepare_fx(cfg, model, is_qat, example_input=None):
    """
    Similar to default_prepare_for_quant, but for FX graph mode.

    Args:
        example_input (Optional[Any]): optional example_input for model,
        if it is not provided we'll use `model.example_input` when example_input
        is required, Note: d2go assumes we always have a single example_input
    """

    assert not cfg.QUANTIZATION.EAGER_MODE
    qconfig = set_backend_and_create_qconfig(cfg, is_train=is_qat)
    qconfig_dict = {"": qconfig}
    if example_input is None:
        raise NotImplementedError(
            "prepare FX requires `example_input`, user should implement this for"
            " their own MetaArch."
        )

    prepare_fn = get_prepare_fx_fn(cfg, is_qat)
    model = prepare_fn(
        model,
        qconfig_mapping=qconfig_dict,
        example_inputs=(example_input,),
    )
    convert_fn = get_convert_fn(cfg, (example_input,))
    return model, convert_fn


def _get_symmetric_xnnpack_quantizer() -> XNNPACKQuantizer:
    quantizer = XNNPACKQuantizer()
    operator_config = get_symmetric_quantization_config(is_per_channel=False)
    quantizer.set_global(operator_config)
    return quantizer


def prepare_fake_quant_model(cfg, model, is_qat, example_input=None):
    """
    Centralized function to prepare fp32 model (D2Go's MetaArch) to fake quant model.
    """
    if cfg.QUANTIZATION.PT2E:  # pt2e quantization
        if hasattr(model, "custom_prepare_pt2e"):
            model, convert_pt2e_callback = model.custom_prepare_pt2e(
                cfg, is_qat, example_input
            )
        else:
            logger.info("Using default pt2e quantization APIs with XNNPACKQuantizer")
            if TORCH_VERSION >= (2, 5, 0):
                captured_model = torch.export.export_for_training(
                    model, example_input
                ).module()
            else:
                captured_model = torch._export.capture_pre_autograd_graph(
                    model, example_input
                ).module()
            quantizer = _get_symmetric_xnnpack_quantizer()
            if is_qat:
                model = prepare_qat_pt2e(captured_model, quantizer)
            else:
                model = prepare_pt2e(captured_model, quantizer)
            convert_pt2e_callback = convert_pt2e
        setattr(model, _CONVERT_PT2E_CALLBACK_ATTRIBUTE, convert_pt2e_callback)
    else:  # pt1.x/legacy quantization recipe
        # TODO: create a warning for the direct use of `torch.ao.quantization.get_default_qconfig`
        # or `torch.ao.quantization.get_default_qat_qconfig` without calling D2Go's high-level
        # `set_backend_and_create_qconfig` API.
        if cfg.QUANTIZATION.EAGER_MODE:
            if hasattr(model, "prepare_for_quant"):
                model = model.prepare_for_quant(cfg)
            else:
                logger.info(
                    "Using default implementation for prepare_for_quant (eager mode)"
                )
                model = default_prepare_for_quant(cfg, model)
            # NOTE: eager model needs to call prepare after `prepare_for_quant`
            prepare_fn = get_prepare_fn(cfg, is_qat)
            prepare_fn(model, inplace=True)

        else:
            # FX graph mode requires the model to be symbolically traceable, swap common
            # modules like SyncBN to FX-friendly version.
            if not is_qat:
                # NOTE: we only do this for PTQ, because we want to keep using unmodified
                # model during QAT.
                model = fuse_utils.swap_modules(model)

            if hasattr(model, "custom_prepare_fx"):
                ret = model.custom_prepare_fx(cfg, is_qat, example_input)
                if not (isinstance(ret, tuple) and len(ret) == 2):
                    raise ValueError(
                        "`custom_prepare_fx` requires return model and convert_callback"
                    )
                model, convert_fx_callback = ret
            else:
                logger.info(
                    "Using default implementation for custom_prepare_fx (FX graph mode)"
                )
                model, convert_fx_callback = default_custom_prepare_fx(
                    cfg, model, is_qat, example_input
                )

            # HACK: store the convert_callback function as model attribute, which can be
            # later accessed to convert fake quant model to quantized model. We'll find a
            # better place to store this.
            if hasattr(model, _CONVERT_FX_CALLBACK_ATTRIBUTE):
                raise AttributeError(
                    f"{_CONVERT_FX_CALLBACK_ATTRIBUTE} is already set in model: {model}"
                )
            setattr(model, _CONVERT_FX_CALLBACK_ATTRIBUTE, convert_fx_callback)

    return model


def convert_to_quantized_model(cfg, fp32_model):
    """
    Contralized function to convert fake quant model (fp32 operators) to "real"
    quantized model (int8 operators).
    """
    if cfg.QUANTIZATION.PT2E:  # pt2e quantization
        logger.info("Using pt2e convert")
        convert_pt2e_callback = getattr(fp32_model, _CONVERT_PT2E_CALLBACK_ATTRIBUTE)
        quantized_model = convert_pt2e_callback(fp32_model)
    else:
        if cfg.QUANTIZATION.EAGER_MODE:
            convert_fn = get_convert_fn(cfg)
            quantized_model = convert_fn(fp32_model, inplace=False)
        else:
            # FX graph mode quantization
            if not hasattr(fp32_model, _CONVERT_FX_CALLBACK_ATTRIBUTE):
                raise AttributeError(
                    f"Can't find {_CONVERT_FX_CALLBACK_ATTRIBUTE} in model, please check "
                    f"`prepare_fake_quant_model` has been called: {fp32_model}"
                )

            convert_fx_callback = getattr(fp32_model, _CONVERT_FX_CALLBACK_ATTRIBUTE)
            quantized_model = convert_fx_callback(fp32_model)
        logger.info(f"Quantization backend: {cfg.QUANTIZATION.BACKEND}")

    return quantized_model


@mock_quantization_type
def post_training_quantize(cfg, model, data_loader):
    """Calibrate a model, convert it to a quantized pytorch model"""
    model = copy.deepcopy(model)
    model.eval()
    # TODO: check why some parameters will have gradient
    for param in model.parameters():
        param.requires_grad = False

    example_input = next(iter(data_loader))
    model = prepare_fake_quant_model(cfg, model, False, example_input)
    logger.info("Prepared the PTQ model for calibration:\n{}".format(model))

    # Option for forcing running calibration on GPU, works only when the model supports
    # casting both model and inputs.
    calibration_force_on_gpu = (
        cfg.QUANTIZATION.PTQ.CALIBRATION_FORCE_ON_GPU and torch.cuda.is_available()
    )
    if calibration_force_on_gpu:
        # NOTE: model.to(device) may not handle cases such as normalizer, FPN, only
        # do move to GPU if specified.
        _cast_model_to_device(model, "cuda")

    calibration_iters = cfg.QUANTIZATION.PTQ.CALIBRATION_NUM_IMAGES
    for idx, inputs in enumerate(data_loader):
        # Setting CALIBRATION_NUM_IMAGES to 0 allows skipping calibration
        if idx == calibration_iters:
            break
        logger.info("Running calibration iter: {}/{}".format(idx, calibration_iters))

        if calibration_force_on_gpu:
            iters = recursive_iterate(inputs)
            for x in iters:
                if isinstance(x, torch.Tensor):
                    iters.send(x.to("cuda"))
            inputs = iters.value

        with torch.no_grad():
            model(inputs)
    else:
        logger.warning("Can't run enough calibration iterations")

    # cast model back to the original device
    if calibration_force_on_gpu:
        _cast_model_to_device(model, cfg.MODEL.DEVICE)

    return model


@mock_quantization_type
def setup_qat_model(
    cfg,
    model_fp32,
    enable_fake_quant: bool = False,
    enable_observer: bool = False,
    enable_learnable_observer: bool = False,
    example_input: Optional[Any] = None,
):
    assert cfg.QUANTIZATION.QAT.FAKE_QUANT_METHOD in [
        "default",
        "learnable",
        "learnable_act",
    ]

    if hasattr(model_fp32, "_non_qat_to_qat_state_dict_map"):
        raise RuntimeError("The model is already setup to be QAT, cannot setup again!")

    device = model_fp32.device
    # FIXME: seems that we can remove this
    torch.backends.quantized.engine = smart_decode_backend(cfg.QUANTIZATION.BACKEND)
    qat_method = cfg.QUANTIZATION.QAT.FAKE_QUANT_METHOD

    # prepare for qat may modify the fp32 model directly so we create a copy
    model_fp32_state_dict = model_fp32.state_dict()

    # prepare model for qat
    model = prepare_fake_quant_model(cfg, model_fp32, True, example_input=example_input)

    # make sure the proper qconfig are used in the model
    learnable_qat.check_for_learnable_fake_quant_ops(qat_method, model)

    # Move newly added observers to the original device
    model.to(device)

    if not enable_fake_quant:
        logger.info("Disabling fake quant ...")
        model.apply(torch.ao.quantization.disable_fake_quant)
        model.apply(learnable_qat.disable_lqat_fake_quant)
    if not enable_observer:
        logger.info("Disabling static observer ...")
        model.apply(torch.ao.quantization.disable_observer)
        model.apply(learnable_qat.disable_lqat_static_observer)
    if not enable_learnable_observer and qat_method.startswith("learnable"):
        logger.info("Disabling learnable observer ...")
        model.apply(learnable_qat.disable_lqat_learnable_observer)

    # qat state dict mapper
    if not getattr(model, "_non_qat_to_qat_state_dict_map", None):
        qat_state_dict_keys_to_ignore = getattr(
            model, "qat_model_state_dict_keys_to_ignore", ()
        )
        model = _setup_non_qat_to_qat_state_dict_map(
            model_fp32_state_dict,
            model,
            cfg.QUANTIZATION.EAGER_MODE,
            qat_state_dict_keys_to_ignore,
        )

    # qat optimizer group for learnable qat
    model = learnable_qat.setup_qat_get_optimizer_param_groups(model, qat_method)

    return model


def _setup_non_qat_to_qat_state_dict_map(
    model_fp32_state_dict: Dict,
    model_qat: nn.Module,
    is_eager_mode: bool,
    qat_state_dict_keys_to_ignore: Tuple[str] = (),  # pyre-ignore
):
    """
    Args:
        model_fp32_state_dict: state dict of the orignal fp32 pytorch model
        model_qat: prepared qat model
        is_eager_mode: whether the model is eager mode
        qat_state_dict_keys_to_ignore: QAT model obtained by fuse_model_qat_fx() (https://fburl.com/code/t70qv2aq) may contain new state dict keys that are not present in the original model state dict (https://fburl.com/code/f8vk47w3). Such keys need to be ignored when we build a mapping from original model state dict keys to new qat model state dict keys below
    """
    original_state_dict_shapes = {k: v.shape for k, v in model_fp32_state_dict.items()}
    # fuse_model and prepare_qat may change the state_dict of model, keep a map from the
    # orginal model to the key QAT in order to load weight from non-QAT model.
    new_state_dict_shapes = {k: v.shape for k, v in model_qat.state_dict().items()}
    new_state_dict_non_observer_keys = [
        k for k in new_state_dict_shapes if not _is_observer_key(k)
    ]
    new_state_dict_non_observer_keys_not_ignored = list(
        set(new_state_dict_non_observer_keys).difference(
            set(qat_state_dict_keys_to_ignore)
        )
    )

    if not len(new_state_dict_non_observer_keys_not_ignored) == len(
        original_state_dict_shapes
    ):
        a = set(new_state_dict_non_observer_keys_not_ignored)
        b = set(original_state_dict_shapes.keys())
        a_diff_b = a.difference(b)
        b_diff_a = b.difference(a)
        logger.info("unique keys in qat model state dict")
        for key in a_diff_b:
            logger.info(f"{key}")
        logger.info("unique keys in original model state dict")
        for key in b_diff_a:
            logger.info(f"{key}")

        raise RuntimeError(
            f"an inconsistent number of keys in state dict of new qat and original model: {len(a)} vs {len(b)}"
        )

    if is_eager_mode:
        for n_k, o_k in zip(
            new_state_dict_non_observer_keys, original_state_dict_shapes
        ):
            assert (
                new_state_dict_shapes[n_k] == original_state_dict_shapes[o_k]
            ), f"QAT model shapes is inconsistent. FP32.{o_k}={original_state_dict_shapes[o_k]} , QAT.{n_k}={new_state_dict_shapes[n_k]}"
        # _q_state_dict_map will store
        model_qat._non_qat_to_qat_state_dict_map = dict(
            zip(original_state_dict_shapes, new_state_dict_non_observer_keys)
        )
    else:
        # in FX, the order of where modules appear in the state_dict may change,
        # so we need to match by key

        def get_new_bn_key(old_bn_key):
            # tries to adjust the key for conv-bn fusion, where
            # root
            #   - conv
            #   - bn
            #
            # becomes
            #
            # root
            #   - conv
            #     - bn
            return old_bn_key.replace(".bn.", ".conv.bn.")

        model_qat._non_qat_to_qat_state_dict_map = {}
        for key in original_state_dict_shapes.keys():
            if key in new_state_dict_non_observer_keys:
                model_qat._non_qat_to_qat_state_dict_map[key] = key
            else:
                maybe_new_bn_key = get_new_bn_key(key)
                if maybe_new_bn_key in new_state_dict_non_observer_keys:
                    model_qat._non_qat_to_qat_state_dict_map[key] = maybe_new_bn_key
    return model_qat


class QATHook(HookBase):
    def __init__(self, cfg, build_data_loader_func=None):
        self.cfg = cfg
        self.build_data_loader_func = build_data_loader_func
        self._applied = {
            "enable_fake_quant": False,
            "enable_observer": False,
            "enable_learnable_observer": False,
            "disable_observer": False,
            "freeze_bn_stats": False,
        }

        assert (
            cfg.QUANTIZATION.QAT.ENABLE_OBSERVER_ITER
            <= cfg.QUANTIZATION.QAT.DISABLE_OBSERVER_ITER
        ), "Can't diable observer before enabling it"

    def before_step(self):
        cur_iter = self.trainer.iter
        model = self.trainer.model
        cfg = self.cfg

        # if we load model in enable_fake_quant state, we need to disable fake quant again, if QAT.START_ITER > 0
        if cur_iter < cfg.QUANTIZATION.QAT.START_ITER and cur_iter == 0:
            logger.info(
                "[QAT] disable fake quant to start QAT, iter = {}".format(cur_iter)
            )
            model.apply(torch.ao.quantization.disable_fake_quant)
            model.apply(learnable_qat.disable_lqat_fake_quant)
            self._applied["enable_fake_quant"] = False

            _reset_qat_data_loader_if_needed(
                self.cfg, self.trainer, self.build_data_loader_func
            )
        if (
            not self._applied["enable_fake_quant"]
            and cur_iter >= cfg.QUANTIZATION.QAT.START_ITER
        ):
            logger.info(
                "[QAT] enable fake quant to start QAT, iter = {}".format(cur_iter)
            )
            model.apply(torch.ao.quantization.enable_fake_quant)
            model.apply(learnable_qat.enable_lqat_fake_quant)
            self._applied["enable_fake_quant"] = True

            _reset_qat_data_loader_if_needed(
                self.cfg, self.trainer, self.build_data_loader_func
            )

        if cur_iter < cfg.QUANTIZATION.QAT.ENABLE_OBSERVER_ITER and cur_iter == 0:
            logger.info("[QAT] disable static observer, iter = {}".format(cur_iter))
            model.apply(torch.ao.quantization.disable_observer)
            model.apply(learnable_qat.disable_lqat_static_observer)
            self._applied["disable_observer"] = False
        if (
            not self._applied["enable_observer"]
            and cur_iter >= cfg.QUANTIZATION.QAT.ENABLE_OBSERVER_ITER
            and cur_iter < cfg.QUANTIZATION.QAT.DISABLE_OBSERVER_ITER
        ):
            logger.info("[QAT] enable static observer, iter = {}".format(cur_iter))
            model.apply(torch.ao.quantization.enable_observer)
            model.apply(learnable_qat.enable_lqat_static_observer)
            self._applied["enable_observer"] = True

        if (
            cur_iter < cfg.QUANTIZATION.QAT.ENABLE_LEARNABLE_OBSERVER_ITER
            and cur_iter == 0
        ):
            logger.info(f"[QAT] disabling learnable observer, iter = {cur_iter}")
            model.apply(learnable_qat.disable_lqat_learnable_observer)
            self._applied["disable_learnable_observer"] = False
        if (
            not self._applied["enable_learnable_observer"]
            and cur_iter >= cfg.QUANTIZATION.QAT.ENABLE_LEARNABLE_OBSERVER_ITER
        ):
            logger.info(f"[QAT] enabling learnable observer, iter = {cur_iter}")
            model.apply(learnable_qat.enable_lqat_learnable_observer)
            self._applied["enable_learnable_observer"] = True

        if (
            not self._applied["disable_observer"]
            and cur_iter >= cfg.QUANTIZATION.QAT.DISABLE_OBSERVER_ITER
        ):
            logger.info(
                "[QAT] disabling observer for sub seq iters, iter = {}".format(cur_iter)
            )
            model.apply(torch.ao.quantization.disable_observer)
            model.apply(learnable_qat.disable_lqat_static_observer)
            model.apply(learnable_qat.disable_lqat_learnable_observer)
            self._applied["disable_observer"] = True

        if (
            not self._applied["freeze_bn_stats"]
            and cur_iter >= cfg.QUANTIZATION.QAT.FREEZE_BN_ITER
        ):
            logger.info(
                "[QAT] freezing BN for subseq iters, iter = {}".format(cur_iter)
            )
            model.apply(torch.nn.intrinsic.qat.freeze_bn_stats)
            self._applied["freeze_bn_stats"] = True

        if (
            self._applied["enable_fake_quant"]
            and not self._applied["disable_observer"]
            and cfg.QUANTIZATION.QAT.UPDATE_OBSERVER_STATS_PERIODICALLY
            and cur_iter % cfg.QUANTIZATION.QAT.UPDATE_OBSERVER_STATS_PERIOD == 0
        ):
            logger.info(f"[QAT] updating observers, iter = {cur_iter}")
            model.apply(observer_update_stat)


def _reset_qat_data_loader_if_needed(cfg, trainer, build_loader_func):
    if cfg.QUANTIZATION.QAT.BATCH_SIZE_FACTOR != 1.0:
        loader_cfg = cfg.clone()
        loader_cfg.defrost()
        num_gpus = comm.get_world_size()
        old_bs = cfg.SOLVER.IMS_PER_BATCH // num_gpus
        new_bs = math.ceil(old_bs * cfg.QUANTIZATION.QAT.BATCH_SIZE_FACTOR)
        loader_cfg.SOLVER.IMS_PER_BATCH = new_bs * num_gpus
        loader_cfg.freeze()

        logger.info(
            "[QAT] Rebuild data loader with batch size per GPU: {} -> {}".format(
                old_bs, new_bs
            )
        )

        assert isinstance(
            trainer, SimpleTrainer
        ), "Trainer needs to be a subclass of SimpleTrainer to support resetting the dataloader"

        trainer.reset_data_loader(lambda: build_loader_func(loader_cfg))


def forward_custom_prepare_fx(root, sub_module_name, orig_ret):
    """Helper function to forward return of `custom_prepare_fx` from sub module"""
    new_sub_module, callback = orig_ret
    setattr(root, sub_module_name, new_sub_module)

    def new_callback(m):
        setattr(m, sub_module_name, callback(getattr(m, sub_module_name)))
        return m

    return root, new_callback


def _convert_to_d2(lightning_checkpoint: Dict[str, Any]) -> None:
    prefix = "model"  # based on DefaultTask.model.
    old_keys = [x.lstrip("model.") for x in lightning_checkpoint[_STATE_DICT_KEY]]
    for key in old_keys:
        if f"{prefix}.{key}" in lightning_checkpoint[_STATE_DICT_KEY]:
            lightning_checkpoint[_STATE_DICT_KEY][key] = lightning_checkpoint[
                _STATE_DICT_KEY
            ][f"{prefix}.{key}"]
            del lightning_checkpoint[_STATE_DICT_KEY][f"{prefix}.{key}"]

    for old, new in zip(
        [_STATE_DICT_KEY, "global_step"], [_OLD_STATE_DICT_KEY, "iteration"]
    ):
        lightning_checkpoint[new] = lightning_checkpoint[old]
        del lightning_checkpoint[old]

    for old, new in zip(
        ["optimizer_states", "lr_schedulers"], ["optimizer", "scheduler"]
    ):
        if old not in lightning_checkpoint:
            continue
        lightning_checkpoint[new] = [lightning_checkpoint[old]]
        del lightning_checkpoint[old]

    for key in [
        "epoch",
        "pytorch-lightning_versio",
        "callbacks",
        "hparams_name",
        "hyper_parameters",
    ]:
        if key in lightning_checkpoint:
            del lightning_checkpoint[key]
