#!/usr/bin/env python3
# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
import contextlib
import logging
from enum import Enum
from functools import partial
from typing import Callable, Generator, Iterable, Optional

import detectron2.utils.comm as comm
import torch
import torch.nn as nn
from d2go.config import CfgNode as CN
from d2go.modeling import modeling_hook as mh
from d2go.registry.builtin import MODELING_HOOK_REGISTRY
from d2go.trainer.helper import parse_precision_from_string
from detectron2.utils.registry import Registry
from torch.cuda.amp import GradScaler
from torch.distributed.fsdp.fully_sharded_data_parallel import (
    BackwardPrefetch,
    CPUOffload,
    FullStateDictConfig,
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
    LocalStateDictConfig,
    ShardingStrategy,
    StateDictType,
)
from torch.distributed.fsdp.sharded_grad_scaler import ShardedGradScaler
from torch.distributed.fsdp.wrap import (
    always_wrap_policy as _always_wrap_policy,
    size_based_auto_wrap_policy as _size_based_auto_wrap_policy,
    transformer_auto_wrap_policy as _layer_based_auto_wrap_policy,
)


logger = logging.getLogger(__name__)

D2GO_FSDP_WRAP_POLICY_REGISTRY = Registry("D2GO_FSDP_WRAP_POLICY_REGISTRY")


def add_fsdp_configs(_C: CN):
    _C.FSDP = CN()
    _C.FSDP.ALGORITHM = ""  # 'grad_optim' or 'full'

    # Configs for fully sharded data parallel (fsdp)
    # Check out https://pytorch.org/docs/stable/fsdp.html
    # and docstring of torch.distributed.fsdp.fully_sharded_data_parallel
    # See docstring of CpuOffload and BackwardPrefetch in torch.distributed.fsdp.fully_sharded_data_parallel
    _C.FSDP.CPU_OFFLOAD = False
    _C.FSDP.BACKWARD_PREFETCH = True
    # Find autowrap policy at D2GO_FSDP_WRAP_POLICY_REGISTRY, or use '' to disable autowrap
    _C.FSDP.AUTO_WRAP_POLICY = "never_wrap_policy"
    _C.FSDP.AUTO_WRAP_MIN_PARAMS = int(1e4)
    # A list of layer cls names to wrap, case sensitive
    _C.FSDP.AUTO_WRAP_LAYER_CLS = []
    # Whether to use local state dict
    _C.FSDP.USE_LOCAL_STATE_DICT = False
    # Whether to offload state dict to cpu
    _C.FSDP.STATE_DICT_CPU_OFFLOAD = False
    # Whether to materialize state dict on rank 0
    _C.FSDP.STATE_DICT_RANK0_ONLY = True


class ShardingAlgorithm(str, Enum):
    """
    This enum specifies the sharding algorithm to be used by FullyShardedDataParallel (FSDP).
    It matches the strings used in D2Go config with the enum class :class:`ShardingStrategy` used by Pytorch FSDP module:
        "grad_optim" => ShardingAlgorithm.SHARD_GRAD_OP => ShardingStrategy.SHARD_GRAD_OP
        "full" => ShardingAlgorithm.FULL_SHARD => ShardingStrategy.FULL_SHARD
    """

    SHARD_GRAD_OP = "grad_optim"
    FULL_SHARD = "full"


def is_fsdp_enabled(cfg):
    return "FSDPModelingHook" in cfg.MODEL.MODELING_HOOKS


def get_grad_scaler(cfg):
    return ShardedGradScaler() if is_fsdp_enabled(cfg) else GradScaler()


class FSDPWrapper(FSDP):
    def __init__(
        self,
        model,
        use_local_state_dict=False,
        load_local_state_dict=False,
        state_dict_cpu_offload=True,
        state_dict_rank0_only=True,
        **fsdp_kwargs,
    ):
        self.use_local_state_dict = use_local_state_dict
        self.load_local_state_dict = load_local_state_dict
        self.offload_to_cpu = state_dict_cpu_offload
        self.rank0_only = state_dict_rank0_only
        self.precision = (
            fsdp_kwargs["mixed_precision"].param_dtype
            if fsdp_kwargs.get("mixed_precision", None)
            else None
        )
        super().__init__(model, **fsdp_kwargs)

    def forward(self, *args, **kwargs):
        # Wrap forward() in autocast if mixed precision is enabled
        # FIXME: turn off autocast if needed
        if not torch.is_autocast_enabled():
            from torch.cuda.amp import autocast

            with autocast(dtype=self.precision):
                return super().forward(*args, **kwargs)
        else:
            return super().forward(*args, **kwargs)

    @staticmethod
    @contextlib.contextmanager
    def state_dict_type_and_config(self, is_sharded: bool) -> Generator:
        if is_sharded:
            state_dict_type = StateDictType.LOCAL_STATE_DICT
            # only offload_to_cpu=False is supported for local state dict
            state_dict_config = LocalStateDictConfig(offload_to_cpu=False)
        else:
            state_dict_type = StateDictType.FULL_STATE_DICT
            state_dict_config = FullStateDictConfig(
                offload_to_cpu=self.offload_to_cpu, rank0_only=self.rank0_only
            )
        with FSDP.state_dict_type(self, state_dict_type, state_dict_config):
            yield

    def state_dict(self, *args, **kwargs):
        # NOTE: model.state_dict() needs to be called by all ranks because synchronization primitives are used
        with self.state_dict_type_and_config(self, self.use_local_state_dict):
            return super().state_dict(*args, **kwargs)

    def load_state_dict(
        self,
        state_dict,
        *args,
        **kwargs,
    ):
        with self.state_dict_type_and_config(self, self.load_local_state_dict):
            return super().load_state_dict(state_dict, *args, **kwargs)


def build_fsdp(
    model,
    *,
    sharding_algorithm: str = ShardingAlgorithm.FULL_SHARD,
    auto_wrap_policy_name: str = "",
    auto_wrap_policy_kwargs: Optional[dict] = None,
    use_cpu_offload: bool = False,
    use_backward_prefetch: bool = True,
    param_dtype: Optional[torch.dtype] = None,
    reduce_dtype: Optional[torch.dtype] = None,
    buffer_dtype: Optional[torch.dtype] = None,
    use_local_state_dict: bool = False,
    load_local_state_dict: bool = False,
    state_dict_cpu_offload: bool = True,
    state_dict_rank0_only: bool = True,
    device_id: Optional[int] = None,
):
    if sharding_algorithm == ShardingAlgorithm.SHARD_GRAD_OP:
        sharding_strategy = ShardingStrategy.SHARD_GRAD_OP
        logger.info("Optimizer + Gradient State Sharding (ZeRO-2) is used")
    elif sharding_algorithm == ShardingAlgorithm.FULL_SHARD:
        sharding_strategy = ShardingStrategy.FULL_SHARD
        logger.info("Optimizer + Gradient + Horizontal Model Sharding (ZeRO-3) is used")
    else:
        raise ValueError(
            f"Invalid sharding algorithm for FSDP. Can be either {ShardingAlgorithm.SHARD_GRAD_OP} or {ShardingAlgorithm.FULL_SHARD}."
        )

    auto_wrap_policy = (
        D2GO_FSDP_WRAP_POLICY_REGISTRY.get(auto_wrap_policy_name)(
            model, **auto_wrap_policy_kwargs
        )
        if auto_wrap_policy_name != ""
        else None
    )
    cpu_offload = CPUOffload(offload_params=use_cpu_offload)
    mixed_precision = MixedPrecision(
        param_dtype=param_dtype,
        reduce_dtype=reduce_dtype,
        buffer_dtype=buffer_dtype,
        keep_low_precision_grads=False,
    )
    backward_prefetch = (
        BackwardPrefetch.BACKWARD_PRE
        if use_backward_prefetch
        else BackwardPrefetch.BACKWARD_POST
    )
    fsdp_kwargs = {
        "sharding_strategy": sharding_strategy,
        "cpu_offload": cpu_offload,
        "mixed_precision": mixed_precision,
        "auto_wrap_policy": auto_wrap_policy,
        "backward_prefetch": backward_prefetch,
        "device_id": torch.cuda.current_device() if not device_id else device_id,
    }
    wrapper_kwargs = {
        "use_local_state_dict": use_local_state_dict,
        "load_local_state_dict": load_local_state_dict,
        "state_dict_cpu_offload": state_dict_cpu_offload,
        "state_dict_rank0_only": state_dict_rank0_only,
    }

    return FSDPWrapper(model, **wrapper_kwargs, **fsdp_kwargs)


@MODELING_HOOK_REGISTRY.register()
class FSDPModelingHook(mh.ModelingHook):
    """Modeling hook that wraps model in FSDP based on config"""

    def apply(self, model: nn.Module) -> FSDPWrapper:
        # SOLVER.AMP.ENABLED and SOLVER.AMP.PRECISION controls mixed precision for all parameters, buffers and reduce in FSDP
        # FSDP mixed precision dtype has to be different from its original dtype; otherwise mixed precision should be disabled
        precision_dtype = (
            parse_precision_from_string(self.cfg.SOLVER.AMP.PRECISION, lightning=False)
            if self.cfg.SOLVER.AMP.ENABLED and not self.cfg.SOLVER.AMP.CAST_ENTIRE_MODEL
            else None
        )
        wrapped_model = build_fsdp(
            model,
            sharding_algorithm=self.cfg.FSDP.ALGORITHM,
            auto_wrap_policy_name=self.cfg.FSDP.AUTO_WRAP_POLICY,
            auto_wrap_policy_kwargs={
                "min_num_params": self.cfg.FSDP.AUTO_WRAP_MIN_PARAMS,
                "layer_names": self.cfg.FSDP.AUTO_WRAP_LAYER_CLS,
            },
            use_cpu_offload=self.cfg.FSDP.CPU_OFFLOAD,
            use_backward_prefetch=self.cfg.FSDP.BACKWARD_PREFETCH,
            param_dtype=precision_dtype,
            reduce_dtype=precision_dtype,
            buffer_dtype=precision_dtype,
            use_local_state_dict=self.cfg.FSDP.USE_LOCAL_STATE_DICT,
            load_local_state_dict=self.cfg.FSDP.USE_LOCAL_STATE_DICT,
            state_dict_cpu_offload=self.cfg.FSDP.STATE_DICT_CPU_OFFLOAD,
            state_dict_rank0_only=self.cfg.FSDP.STATE_DICT_RANK0_ONLY,
            device_id=torch.cuda.current_device(),
        )
        return wrapped_model

    def unapply(self, model: FSDPWrapper) -> nn.Module:
        raise NotImplementedError(
            "FSDPModelingHook.unapply() not implemented: can't unwrap a FSDP module"
        )


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


@D2GO_FSDP_WRAP_POLICY_REGISTRY.register()
def never_wrap_policy(model, **kwargs) -> Optional[Callable]:
    """
    Don't wrap any child module, only wrap the root
    """

    def never_wrap(*args, **kwargs):
        return False

    return never_wrap


@D2GO_FSDP_WRAP_POLICY_REGISTRY.register()
def always_wrap_policy(model, **kwargs) -> Optional[Callable]:
    """
    Wrapper for always_wrap_policy() from torch.distributed.fsdp.wrap
    """
    return _always_wrap_policy


@D2GO_FSDP_WRAP_POLICY_REGISTRY.register()
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


@D2GO_FSDP_WRAP_POLICY_REGISTRY.register()
def layer_based_auto_wrap_policy(
    model, layer_names: Iterable[str], **kwargs
) -> Optional[Callable]:
    """
    Wrapper for transformer_auto_wrap_policy() from torch.distributed.fsdp.wrap
    Args:
        layer_names: a list of layer names
    """
    assert (
        len(layer_names) > 0
    ), "FSDP.AUTO_WRAP_LAYER_CLS should be a nonempty list of layer names contained in the model"
    layer_cls = []
    for name in layer_names:
        closure = get_module_class_from_name(model, name)
        if closure is None:
            raise Exception(
                f"Could not find the layer class {name} to wrap in the model."
            )
        layer_cls.append(closure)
    return partial(
        _layer_based_auto_wrap_policy,
        transformer_layer_cls=layer_cls,
    )
