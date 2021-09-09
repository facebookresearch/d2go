#!/usr/bin/env python3

import itertools
from typing import Any, Dict, Optional

import torch
import torch.nn as nn


class EMAState(object):
    """Stores Exponential Moving Average state for a model.

    Args:
        decay: EMA decay factor, should be in [0, 1]. A decay of 0 corresponds to
            always using the latest value (no EMA) and a decay of 1 corresponds to
            not updating weights after initialization. Default to 0.999.
        device: If not None, move model EMA state to device.
    """

    def __init__(self, decay: float = 0.999, device: Optional[str] = None):
        if decay < 0 or decay > 1.0:
            raise ValueError(f"Decay should be in [0, 1], {decay} was given.")
        self.decay: float = decay
        self.state: Dict[str, Any] = {}
        self.device: Optional[str] = device

    @classmethod
    def from_model(
        cls,
        model: nn.Module,
        decay: float = 0.999,
        device: Optional[str] = None,
    ) -> "EMAState":
        """Constructs model state from the model and move to device if given."""
        ret = cls(decay, device)
        ret.load_from(model)
        return ret

    def load_from(self, model: nn.Module) -> None:
        """Load state from the model."""
        self.state.clear()
        for name, val in self._get_model_state_iterator(model):
            val = val.detach().clone()
            self.state[name] = val.to(self.device) if self.device else val

    def has_inited(self) -> bool:
        return len(self.state) > 0

    def apply_to(self, model: nn.Module) -> None:
        """Apply EMA state to the model."""
        with torch.no_grad():
            for name, val in self._get_model_state_iterator(model):
                assert (
                    name in self.state
                ), f"Name {name} does not exist, available names are {self.state.keys()}"
                val.copy_(self.state[name])

    def state_dict(self) -> Dict[str, Any]:
        return self.state

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        self.state.clear()
        for name, val in state_dict.items():
            self.state[name] = val.to(self.device) if self.device else val

    def to(self, device: torch.device) -> None:
        """moves EMA state to device."""
        for name, val in self.state.items():
            self.state[name] = val.to(device)

    def _get_model_state_iterator(self, model: nn.Module):
        param_iter = model.named_parameters()
        # pyre-fixme[16]: `nn.Module` has no attribute `named_buffers`.
        buffer_iter = model.named_buffers()
        return itertools.chain(param_iter, buffer_iter)

    def update(self, model: nn.Module) -> None:
        with torch.no_grad():
            for name, val in self._get_model_state_iterator(model):
                ema_val = self.state[name]
                if self.device:
                    val = val.to(self.device)
                ema_val.copy_(ema_val * self.decay + val * (1.0 - self.decay))
