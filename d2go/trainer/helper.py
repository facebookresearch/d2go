from functools import partial
from typing import Any, Callable, Iterable, List, Optional, Union

import torch

from detectron2.utils.registry import Registry
from torch.distributed.fsdp.wrap import (
    always_wrap_policy as _always_wrap_policy,
    ModuleWrapPolicy,
    size_based_auto_wrap_policy as _size_based_auto_wrap_policy,
)


D2GO_WRAP_POLICY_REGISTRY = Registry("D2GO_WRAP_POLICY_REGISTRY")


def parse_precision_from_string(
    precision: str, lightning=False
) -> Union[str, int, torch.dtype]:
    """
    Convert our string format for precision to what Detectron2 / lightning Trainer expects, controlled by the *lightning* flag
    """
    if precision == "float64":
        return torch.float64 if not lightning else 64
    if precision == "float32":
        return torch.float32 if not lightning else 32
    elif precision == "float16":
        return torch.float16 if not lightning else 16
    elif precision == "bfloat16":
        return torch.bfloat16 if not lightning else "bf16"
    else:
        raise ValueError(f"Invalid precision dtype {precision}")


def get_module_class_from_name(module, name):
    """
    Gets a class from a module by its name. Code borrowed from HuggingFace
    Args:
        module (`torch.nn.Module`): The module to get the class from.
        name (`str`): The name of the class.
    """
    modules_children = list(module.children())
    if module.__class__.__name__ == name:
        return module.__class__
    elif len(modules_children) == 0:
        return
    else:
        for child_module in modules_children:
            module_class = get_module_class_from_name(child_module, name)
            if module_class is not None:
                return module_class


def get_layer_cls_from_names(
    model: Any, layer_names: Iterable[str]
) -> List[torch.nn.Module]:
    """
    Get a list of layers from a model that match a list of layer names.
    """
    layer_cls = []
    for name in layer_names:
        closure = get_module_class_from_name(model, name)
        if closure is None:
            raise Exception(
                f"Could not find the layer class {name} to wrap in the model."
            )
        layer_cls.append(closure)

    return layer_cls


@D2GO_WRAP_POLICY_REGISTRY.register()
def never_wrap_policy(model, **kwargs) -> Optional[Callable]:
    """
    Don't wrap any child module, only wrap the root
    """

    def never_wrap(*args, **kwargs):
        return False

    return never_wrap


@D2GO_WRAP_POLICY_REGISTRY.register()
def always_wrap_policy(model, **kwargs) -> Optional[Callable]:
    """
    Wrapper for always_wrap_policy() from torch.distributed.fsdp.wrap
    """
    return _always_wrap_policy


@D2GO_WRAP_POLICY_REGISTRY.register()
def size_based_auto_wrap_policy(
    model, min_num_params=1e4, **kwargs
) -> Optional[Callable]:
    """
    Wrapper for size_based_auto_wrap_policy() from torch.distributed.fsdp.wrap
    """
    # Note: be careful when using auto wrap with shared parameters.
    # Errors will be thrown if shared parameters reside in different FSDP units
    return partial(
        _size_based_auto_wrap_policy,
        min_num_params=min_num_params,
    )


@D2GO_WRAP_POLICY_REGISTRY.register()
def layer_based_auto_wrap_policy(
    model, layer_names: Iterable[str], **kwargs
) -> Optional[Callable]:
    """
    Wrapper for ModuleWrapPolicy() from torch.distributed.fsdp.wrap
    Args:
        layer_names: a list of layer names
    """
    assert (
        len(layer_names) > 0
    ), "layer_names should be a nonempty list of layer names contained in the model"
    layer_cls = get_layer_cls_from_names(model, layer_names)
    return ModuleWrapPolicy(module_classes=layer_cls)
