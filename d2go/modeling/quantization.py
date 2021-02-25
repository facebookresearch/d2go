#!/usr/bin/env python3

import contextlib
import copy
import inspect
import logging

import torch
import torch.quantization.quantize_fx
from detectron2.checkpoint import DetectionCheckpointer
from mobile_cv.arch.utils import fuse_utils
from mobile_cv.common.misc.iter_utils import recursive_iterate


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
    _C.QUANTIZATION.BACKEND = "fbgemm"

    # used to enable metarch set_custom_qscheme (need to implement)
    # this is a limited implementation where only str is provided to change options
    _C.QUANTIZATION.CUSTOM_QSCHEME = ""

    # quantization-aware training
    _C.QUANTIZATION.QAT = CfgNode()
    _C.QUANTIZATION.QAT.ENABLED = False
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

    # deprecated
    _C.QUANTIZATION.SILICON_QAT = CfgNode()
    _C.QUANTIZATION.SILICON_QAT.ENABLED = False

    # register deprecated and renamed keys
    _C.register_deprecated_key("QUANTIZATION.QAT.LOAD_PRETRAINED")
    _C.register_renamed_key("QUANTIZATION.QAT.BACKEND", "QUANTIZATION.BACKEND")
    _C.register_deprecated_key("QUANTIZATION.ENABLE_CUSTOM_QSCHEME")


@contextlib.contextmanager
def silicon_qat_build_model_context(cfg):
    mock_ctx_managers = []
    if cfg.QUANTIZATION.SILICON_QAT.ENABLED:
        from mobile_cv.silicon_pytorch_qat.replace_op import mock_quant_ops

        mock_ctx_managers.extend(
            [
                mock_quant_ops(quant_op="quant_add"),
                mock_quant_ops(quant_op="quant_fbb_convbnrelu"),
            ]
        )

    with contextlib.ExitStack() as stack:
        for mgr in mock_ctx_managers:
            stack.enter_context(mgr)
        yield


# TODO: model.to(device) might not work for detection meta-arch, this function is the
# workaround, in general, we might need a meta-arch API for this if needed.
def _cast_detection_model(model, device):
    # check model is an instance of one of the meta arch
    from detectron2.export.caffe2_modeling import Caffe2MetaArch
    from detectron2.modeling import META_ARCH_REGISTRY

    if isinstance(model, Caffe2MetaArch):
        model._wrapped_model = _cast_detection_model(model._wrapped_model, device)
        return model

    assert isinstance(model, tuple(META_ARCH_REGISTRY._obj_map.values()))
    model.to(device)
    # cast normalizer separately
    if hasattr(model, "normalizer") and not (
        hasattr(model, "pixel_mean") and hasattr(model, "pixel_std")
    ):
        pixel_mean = inspect.getclosurevars(model.normalizer).nonlocals["pixel_mean"]
        pixel_std = inspect.getclosurevars(model.normalizer).nonlocals["pixel_std"]
        pixel_mean = pixel_mean.to(device)
        pixel_std = pixel_std.to(device)
        model.normalizer = lambda x: (x - pixel_mean) / pixel_std
    return model


def add_d2_quant_mapping(mappings):
    """ HACK: Add d2 specific module mapping for eager model quantization
    """
    import torch.quantization.quantization_mappings as qm
    for k, v in mappings.items():
        if k not in qm.get_default_static_quant_module_mappings():
            qm.DEFAULT_STATIC_QUANT_MODULE_MAPPINGS[k] = v
        if k not in qm.get_default_qat_module_mappings():
            qm.DEFAULT_QAT_MODULE_MAPPINGS[k] = v

# The `mock_quantization_type` decorate may not be needed anymore to unify
# detectron2.layers modules and torch.nn modules since Pytorch 1.5. See comments on D23790034.
def mock_quantization_type(quant_func):
    import mock
    import builtins
    import functools
    import detectron2.layers as d2l

    type_mapping = {d2l.Linear: torch.nn.Linear}
    from d2go.utils.misc import check_version
    if check_version(torch, '1.7.2', warning_only=True):
        add_d2_quant_mapping(type_mapping)

    real_type = builtins.type

    def _new_type(obj):
        rtype = real_type(obj)
        return type_mapping.get(rtype, rtype)

    @functools.wraps(quant_func)
    def wrapper(cfg, model, *args, **kwargs):
        if type(d2l.Linear) == torch.nn.Linear:
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
        - Currently this API doesn't include the final torch.quantization.prepare(_qat)
            call since existing usecases don't have further steps after it.

    Args:
        model (nn.Module): a non-quantized model.
        cfg (CfgNode): config

    Return:
        nn.Module: a ready model for QAT training or PTQ calibration
    """
    qconfig = (
        torch.quantization.get_default_qat_qconfig(cfg.QUANTIZATION.BACKEND)
        if model.training
        else torch.quantization.get_default_qconfig(cfg.QUANTIZATION.BACKEND)
    )

    if cfg.QUANTIZATION.EAGER_MODE:
        model = fuse_utils.fuse_model(model, inplace=True)

        torch.backends.quantized.engine = cfg.QUANTIZATION.BACKEND
        model.qconfig = qconfig
        # TODO(future diff): move the torch.quantization.prepare(...) call
        # here, to be consistent with the FX branch
    else:  # FX graph mode quantization
        qconfig_dict = {"": qconfig}
        if model.training:
            model = torch.quantization.quantize_fx.prepare_qat_fx(model, qconfig_dict)
        else:
            model = torch.quantization.quantize_fx.prepare_fx(model, qconfig_dict)

    logger.info("Setup the model with qconfig:\n{}".format(qconfig))

    return model


@mock_quantization_type
def post_training_quantize(cfg, model, data_loader):
    """ Calibrate a model, convert it to a quantized pytorch model """
    model = copy.deepcopy(model)
    model.eval()
    # TODO: check why some parameters will have gradient
    for param in model.parameters():
        param.requires_grad = False

    if hasattr(model, "prepare_for_quant"):
        model = model.prepare_for_quant(cfg)
    else:
        logger.info("Using default implementation for prepare_for_quant")
        model = default_prepare_for_quant(cfg, model)

    if cfg.QUANTIZATION.EAGER_MODE:
        torch.quantization.prepare(model, inplace=True)
    logger.info("Prepared the PTQ model for calibration:\n{}".format(model))

    # Option for forcing running calibration on GPU, works only when the model supports
    # casting both model and inputs.
    calibration_force_on_gpu = (
        cfg.QUANTIZATION.PTQ.CALIBRATION_FORCE_ON_GPU and torch.cuda.is_available()
    )
    if calibration_force_on_gpu:
        # NOTE: model.to(device) may not handle cases such as normalizer, FPN, only
        # do move to GPU if specified.
        _cast_detection_model(model, "cuda")

    calibration_iters = cfg.QUANTIZATION.PTQ.CALIBRATION_NUM_IMAGES
    for idx, inputs in enumerate(data_loader):
        logger.info("Running calibration iter: {}/{}".format(idx, calibration_iters))

        if calibration_force_on_gpu:
            iters = recursive_iterate(inputs)
            for x in iters:
                if isinstance(x, torch.Tensor):
                    iters.send(x.to("cuda"))
            inputs = iters.value

        with torch.no_grad():
            model(inputs)
        if idx + 1 == calibration_iters:
            break
    else:
        logger.warning("Can't run enough calibration iterations")

    # cast model back to the original device
    if calibration_force_on_gpu:
        _cast_detection_model(model, cfg.MODEL.DEVICE)

    return model


@mock_quantization_type
def setup_qat_model(cfg, model, enable_fake_quant=False, enable_observer=False):
    if hasattr(model, "_non_qat_to_qat_state_dict_map"):
        raise RuntimeError("The model is already setup to be QAT, cannot setup again!")

    device = model.device
    torch.backends.quantized.engine = cfg.QUANTIZATION.BACKEND
    original_state_dict_shapes = {k: v.shape for k, v in model.state_dict().items()}

    if cfg.QUANTIZATION.EAGER_MODE:
        if hasattr(model, "prepare_for_quant"):
            model = model.prepare_for_quant(cfg)
        else:
            logger.info("Using default implementation for prepare_for_quant")
            model = default_prepare_for_quant(cfg, model)

        # TODO(future diff): move this into prepare_for_quant to match FX branch
        torch.quantization.prepare_qat(model, inplace=True)
    else:  # FX graph mode quantization
        if hasattr(model, "prepare_for_quant"):
            model = model.prepare_for_quant(cfg)
        else:
            logger.info("Using default implementation for prepare_for_quant")
            model = default_prepare_for_quant(cfg, model)

    # Move newly added observers to the original device
    model.to(device)

    if not enable_fake_quant:
        logger.info("Disabling fake quant ...")
        model.apply(torch.quantization.disable_fake_quant)
    if not enable_observer:
        logger.info("Disabling observer ...")
        model.apply(torch.quantization.disable_observer)

    # fuse_model and prepare_qat may change the state_dict of model, keep a map from the
    # orginal model to the key QAT in order to load weight from non-QAT model.
    new_state_dict_shapes = {k: v.shape for k, v in model.state_dict().items()}
    new_state_dict_non_observer_keys = [
        k for k in new_state_dict_shapes if not _is_observer_key(k)
    ]
    assert len(new_state_dict_non_observer_keys) == len(original_state_dict_shapes)

    if cfg.QUANTIZATION.EAGER_MODE:
        for n_k, o_k in zip(new_state_dict_non_observer_keys, original_state_dict_shapes):
            assert new_state_dict_shapes[n_k] == original_state_dict_shapes[o_k]
        # _q_state_dict_map will store
        model._non_qat_to_qat_state_dict_map = dict(
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

        model._non_qat_to_qat_state_dict_map = {}
        for key in original_state_dict_shapes.keys():
            if key in new_state_dict_non_observer_keys:
                model._non_qat_to_qat_state_dict_map[key] = key
            else:
                maybe_new_bn_key = get_new_bn_key(key)
                if maybe_new_bn_key in new_state_dict_non_observer_keys:
                    model._non_qat_to_qat_state_dict_map[key] = maybe_new_bn_key

    return model
