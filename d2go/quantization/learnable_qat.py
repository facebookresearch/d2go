#!/usr/bin/env python3
import logging
from functools import partial

import torch
import torch.distributed as dist
from d2go.utils.parse_module_params import iterate_module_named_parameters
from torch.ao.quantization._learnable_fake_quantize import _LearnableFakeQuantize

logger = logging.getLogger(__name__)


def mixin_with_subclass(module, mix_class):
    """Create a subclass of type(module) and mix_class while using all the data
    from the `module` object
    """
    ModuleType = type(module)

    class SubClass(mix_class, ModuleType):
        def __init__(self, module):
            assert isinstance(module, ModuleType)
            # initialize the parent by copying the dict directly
            self.__dict__ = module.__dict__.copy()

    ret = SubClass(module)
    return ret


def _has_module(model, module_type):
    for x in model.modules():
        if isinstance(x, module_type):
            return True
    return False


def check_for_learnable_fake_quant_ops(qat_method, model):
    """Make sure learnable observers are used if qat method is `learnable`"""
    if qat_method.startswith("learnable"):
        if not _has_module(model, _LearnableFakeQuantize):
            raise Exception(
                "No learnable fake quant is used for learnable quantzation, please use d2go.quantization.learnable_qat.get_learnable_qat_qconfig() to get proper qconfig"
            )


def convert_to_learnable_qconfig(qconfig):
    """
    Convert a QConfig to its learnable counterpart.
    """

    def _update_fused_moving_avg_obs_fake_quantize(keywords):
        # requires setting use_grad_scaling to True, all other parameters are the same
        # as default setting of FusedMovingAvgObsFakeQuantize (both qnnpack and fbgemm).
        assert "use_grad_scaling" not in keywords
        keywords["use_grad_scaling"] = True
        return keywords

    _OVERWRITE_PARAMS = {
        # map from supported FakeQuant type to the its new parameters in order to convert
        # it to a learnable FakeQuant
        torch.ao.quantization.fake_quantize.FusedMovingAvgObsFakeQuantize: _update_fused_moving_avg_obs_fake_quantize
    }

    def _update_to_learnable(wrapper):
        assert isinstance(
            wrapper, torch.ao.quantization.observer._PartialWrapper
        ), wrapper
        assert isinstance(wrapper.p, partial), wrapper

        keywords_updater = _OVERWRITE_PARAMS[wrapper.p.func]
        keywords = keywords_updater(wrapper.p.keywords)

        new_p = partial(_LearnableFakeQuantize, *wrapper.p.args, **keywords)
        wrapper.p = new_p
        return wrapper

    activation = _update_to_learnable(qconfig.activation)
    weight = _update_to_learnable(qconfig.weight)
    return torch.quantization.QConfig(activation=activation, weight=weight)


def get_world_size() -> int:
    if not dist.is_available():
        return 1
    if not dist.is_initialized():
        return 1
    return dist.get_world_size()


def sync_tensor(data):
    world_size = get_world_size()
    if world_size > 1:
        dist.all_reduce(data, op=dist.ReduceOp.SUM)
        data /= world_size


def toggle_lqat_fake_quant(mod, enable):
    """Toggle fake quantization for learnable qat"""
    if type(mod) == _LearnableFakeQuantize:
        mod.toggle_fake_quant(enable)


# enable/disable fake quantization for learnable qat
enable_lqat_fake_quant = partial(toggle_lqat_fake_quant, enable=True)
disable_lqat_fake_quant = partial(toggle_lqat_fake_quant, enable=False)


def toggle_lqat_static_observer(mod, enable):
    """Toggle static observers for learnable qat"""
    if type(mod) == _LearnableFakeQuantize:
        mod.toggle_observer_update(enable)


# enable/disable static observer for learnable qat
enable_lqat_static_observer = partial(toggle_lqat_static_observer, enable=True)
disable_lqat_static_observer = partial(toggle_lqat_static_observer, enable=False)


def enable_lqat_learnable_observer(mod):
    """Enable learning observers, will disable static observer updates"""
    if type(mod) == _LearnableFakeQuantize:
        sync_tensor(mod.scale.data)
        sync_tensor(mod.zero_point.data)
        mod.toggle_qparam_learning(enabled=True).toggle_observer_update(enabled=False)


def disable_lqat_learnable_observer(mod):
    """Disable learning observers"""
    if type(mod) == _LearnableFakeQuantize:
        mod.toggle_qparam_learning(enabled=False)


def get_optimizer_param_groups_learnable_qat(model, _):
    """Set the weight decay for scale/zero_point for learnable_fake_quant to 0"""
    params = []
    for (
        _module_name,
        module,
        module_param_name,
        value,
    ) in iterate_module_named_parameters(model, check_requires_grad=False):
        if isinstance(module, _LearnableFakeQuantize):
            if module_param_name in ("scale", "zero_point"):
                params += [
                    {
                        "params": [value],
                        "weight_decay": 0.0,
                    }
                ]

    return params


def _is_observer_key(state_dict_key):
    observer_keys = ["activation_post_process", "weight_fake_quant"]
    return any(x in state_dict_key for x in observer_keys)


def _is_q_state_dict(state_dict):
    return any(_is_observer_key(k) for k in state_dict)


class ModelGetOptimizerParamGroupLearnableQATMixin:
    def get_optimizer_param_groups(self, opts):
        ret = []
        if hasattr(super(), "get_optimizer_param_groups"):
            ret = super().get_optimizer_param_groups(opts)
        ret += get_optimizer_param_groups_learnable_qat(self, opts)
        return ret


def setup_qat_get_optimizer_param_groups(model, qat_method):
    """Add a function `get_optimizer_param_groups` to the model so that it could
    return proper weight decay for learnable qat
    """
    if not qat_method.startswith("learnable"):
        return model

    assert _is_q_state_dict(model.state_dict())

    model = mixin_with_subclass(model, ModelGetOptimizerParamGroupLearnableQATMixin)
    assert hasattr(model, "get_optimizer_param_groups")
    return model
