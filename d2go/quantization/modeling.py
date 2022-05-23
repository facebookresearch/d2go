#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved


import copy
import logging
import math
from typing import Tuple

import detectron2.utils.comm as comm
import torch
from d2go.quantization import learnable_qat
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.engine import HookBase, SimpleTrainer
from mobile_cv.arch.quantization.observer import update_stat as observer_update_stat
from mobile_cv.arch.utils import fuse_utils
from mobile_cv.common.misc.iter_utils import recursive_iterate

from .qconfig import set_backend_and_create_qconfig, smart_decode_backend

TORCH_VERSION: Tuple[int, ...] = tuple(int(x) for x in torch.__version__.split(".")[:2])
if TORCH_VERSION > (1, 10):
    from torch.ao.quantization.quantize_fx import convert_fx, prepare_fx, prepare_qat_fx
else:
    from torch.quantization.quantize_fx import convert_fx, prepare_fx, prepare_qat_fx

logger = logging.getLogger(__name__)


def _is_observer_key(state_dict_key):
    observer_keys = ["activation_post_process", "weight_fake_quant"]
    return any(x in state_dict_key for x in observer_keys)


class QATCheckpointer(DetectionCheckpointer):
    """
    Extend the Checkpointer to support loading (QAT / non-QAT) weight into
    (QAT / non-QAT) model.
    """

    @classmethod
    def _is_q_state_dict(cls, state_dict):
        return any(_is_observer_key(k) for k in state_dict)

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

    # post-training quantization
    _C.QUANTIZATION.PTQ = CfgNode()
    _C.QUANTIZATION.PTQ.CALIBRATION_NUM_IMAGES = 1
    _C.QUANTIZATION.PTQ.CALIBRATION_FORCE_ON_GPU = False

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

    import detectron2.layers as d2l
    import mock

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
    qconfig = set_backend_and_create_qconfig(cfg, is_train=model.training)

    if cfg.QUANTIZATION.EAGER_MODE:
        model = fuse_utils.fuse_model(
            model,
            is_qat=cfg.QUANTIZATION.QAT.ENABLED,
            inplace=True,
        )

        model.qconfig = qconfig
        # TODO(future diff): move the torch.ao.quantization.prepare(...) call
        # here, to be consistent with the FX branch
    else:  # FX graph mode quantization
        qconfig_dict = {"": qconfig}
        # TODO[quant-example-inputs]: needs follow up to change the api
        example_inputs = (torch.rand(1, 3, 3, 3),)
        if model.training:
            model = prepare_qat_fx(model, qconfig_dict, example_inputs)
        else:
            model = prepare_fx(model, qconfig_dict, example_inputs)

    logger.info("Setup the model with qconfig:\n{}".format(qconfig))

    return model


def default_prepare_for_quant_convert(cfg, model):
    return convert_fx(model)


def apply_prepare_for_quant(cfg, model):
    # TODO: create a warning for the direct use of `torch.ao.quantization.get_default_qconfig`
    # or `torch.ao.quantization.get_default_qat_qconfig` without calling D2Go's high-level
    # `set_backend_and_create_qconfig` API.

    if hasattr(model, "prepare_for_quant"):
        model = model.prepare_for_quant(cfg)
    else:
        logger.info("Using default implementation for prepare_for_quant")
        model = default_prepare_for_quant(cfg, model)

    return model


@mock_quantization_type
def post_training_quantize(cfg, model, data_loader):
    """Calibrate a model, convert it to a quantized pytorch model"""
    model = copy.deepcopy(model)
    model.eval()
    # TODO: check why some parameters will have gradient
    for param in model.parameters():
        param.requires_grad = False

    model = apply_prepare_for_quant(cfg, model)
    if cfg.QUANTIZATION.EAGER_MODE:
        torch.ao.quantization.prepare(model, inplace=True)
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
):
    assert cfg.QUANTIZATION.QAT.FAKE_QUANT_METHOD in ["default", "learnable"]

    if hasattr(model_fp32, "_non_qat_to_qat_state_dict_map"):
        raise RuntimeError("The model is already setup to be QAT, cannot setup again!")

    device = model_fp32.device
    # FIXME: seems that we can remove this
    torch.backends.quantized.engine = smart_decode_backend(cfg.QUANTIZATION.BACKEND)
    qat_method = cfg.QUANTIZATION.QAT.FAKE_QUANT_METHOD

    # prepare for qat may modify the fp32 model directly so we create a copy
    model_fp32_state_dict = model_fp32.state_dict()

    # prepare model for qat
    model = apply_prepare_for_quant(cfg, model_fp32)
    if cfg.QUANTIZATION.EAGER_MODE:
        torch.ao.quantization.prepare_qat(model, inplace=True)

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
    if not enable_learnable_observer and qat_method == "learnable":
        logger.info("Disabling learnable observer ...")
        model.apply(learnable_qat.disable_lqat_learnable_observer)

    # qat state dict mapper
    if not getattr(model, "_non_qat_to_qat_state_dict_map", None):
        model = _setup_non_qat_to_qat_state_dict_map(
            model_fp32_state_dict, model, is_eager_mode=cfg.QUANTIZATION.EAGER_MODE
        )

    # qat optimizer group for learnable qat
    model = learnable_qat.setup_qat_get_optimizer_param_groups(model, qat_method)

    return model


def _setup_non_qat_to_qat_state_dict_map(
    model_fp32_state_dict, model_qat, is_eager_mode
):
    original_state_dict_shapes = {k: v.shape for k, v in model_fp32_state_dict.items()}
    # fuse_model and prepare_qat may change the state_dict of model, keep a map from the
    # orginal model to the key QAT in order to load weight from non-QAT model.
    new_state_dict_shapes = {k: v.shape for k, v in model_qat.state_dict().items()}
    new_state_dict_non_observer_keys = [
        k for k in new_state_dict_shapes if not _is_observer_key(k)
    ]
    assert len(new_state_dict_non_observer_keys) == len(original_state_dict_shapes)

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
        # This method assumes the data loader can be replaced from trainer
        assert trainer.__class__ == SimpleTrainer
        del trainer._data_loader_iter
        del trainer.data_loader
        data_loader = build_loader_func(loader_cfg)
        trainer.data_loader = data_loader
        trainer._data_loader_iter = iter(data_loader)
