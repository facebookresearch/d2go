from typing import Any, Iterable, List, Union

import torch


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
