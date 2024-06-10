from typing import Tuple

import torch
from d2go.quantization.learnable_qat import convert_to_learnable_qconfig
from mobile_cv.common.misc.registry import Registry

TORCH_VERSION: Tuple[int, ...] = tuple(int(x) for x in torch.__version__.split(".")[:2])
if TORCH_VERSION > (1, 10):
    from torch.ao.quantization.quantize_fx import convert_fx, prepare_fx, prepare_qat_fx
else:
    from torch.quantization.quantize_fx import convert_fx, prepare_fx, prepare_qat_fx
from mobile_cv.common.misc.oss_utils import fb_overwritable


QCONFIG_CREATOR_REGISTRY = Registry("QCONFIG_CREATOR_REGISTRY")


def set_backend_and_create_qconfig(cfg, *, is_train):
    """
    Recommended function to create qconfig given D2Go's quantization config.
    """

    # In case we need different implmentation, we can add a new key called
    # QUANTIZATION.QCONFIG_CREATOR with "smart" as default value, and use this key
    # to toggle between registries.
    return QCONFIG_CREATOR_REGISTRY.get("smart")(cfg, is_train=is_train)


@fb_overwritable()
def holistic_get_qconfig(backend, is_qat, use_symmetric=False, cfg=None):
    """
    Utility to create the QConfig based on backend, is_qat, and use_symmetric.
    cfg (unused) is to customize QConfig based on the cfg.
    """

    if use_symmetric:
        if not backend == "qnnpack":
            raise ValueError(
                f"Only qnnpack supports Symmetric quantization, given: {backend}"
            )
        if is_qat:
            return torch.ao.quantization.default_symmetric_qnnpack_qat_qconfig
        else:
            return torch.ao.quantization.default_symmetric_qnnpack_qconfig
    else:
        if is_qat:
            return torch.ao.quantization.get_default_qat_qconfig(backend)
        else:
            return torch.ao.quantization.get_default_qconfig(backend)


@QCONFIG_CREATOR_REGISTRY.register("smart")
def _smart_set_backend_and_create_qconfig(cfg, *, is_train):
    """
    This is the default / "smart" way to create qconfig based on various of configs,
    supports:
        - learnable QAT
        - set symmetric quantization via backend.
    """

    backend, options = _smart_parse_extended_backend(cfg.QUANTIZATION.BACKEND)
    is_symmetric = options["is_symmetric"]

    # Set backend
    torch.backends.quantized.engine = backend

    qat_method = cfg.QUANTIZATION.QAT.FAKE_QUANT_METHOD
    assert qat_method in ["default", "learnable"]

    qconfig = holistic_get_qconfig(
        backend=backend, is_qat=is_train, use_symmetric=is_symmetric, cfg=cfg
    )
    if is_train and qat_method == "learnable":
        qconfig = convert_to_learnable_qconfig(qconfig)

    return qconfig


def validate_native_backend(backend):
    _PYTORCH_NATIVE_BACKENDS = ["fbgemm", "qnnpack"]
    if backend not in _PYTORCH_NATIVE_BACKENDS:
        raise ValueError(
            f"Unrecognized backend: {backend}, PyTorch"
            f" supported backends are: {_PYTORCH_NATIVE_BACKENDS}"
        )


@fb_overwritable()
def _smart_parse_extended_backend(extended_backend):
    """
    D2Go extends the definition of quantization "backend". In addition to PyTorch's
    native backends (i.e. qnnpack and fbgemm), we allow other type of backend so users
    can easily express different settings. Here are the supported cases:
        1. Symmetric quantization: "qnnpack@symmetric" refers to using QNNPACK with
            symmetric QConfig.
    """
    backend = extended_backend
    # default options
    options = {
        "is_symmetric": False,
    }

    if "@symmetric" in backend:
        options["is_symmetric"] = True
        backend = backend.replace("@symmetric", "", 1)

    validate_native_backend(backend)
    return backend, options


def smart_decode_backend(extended_backend):
    """
    Since we extend the definition of quantization backend, user shouldn't directly use
    cfg.QUANTIZATION.BACKEND under PyTorch's context, this is the translation function
    if direct use is necessary.
    """
    return _smart_parse_extended_backend(extended_backend)[0]
