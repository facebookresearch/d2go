from typing import Union

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
