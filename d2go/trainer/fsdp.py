#!/usr/bin/env python3
# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
import contextlib
import logging
from enum import Enum
from typing import Any, Dict, Generator, List, Optional, Set, Tuple

import torch
import torch.nn as nn
from d2go.config import CfgNode as CN
from d2go.modeling.modeling_hook import ModelingHook
from d2go.registry.builtin import MODELING_HOOK_REGISTRY
from d2go.trainer.helper import D2GO_WRAP_POLICY_REGISTRY, parse_precision_from_string
from torch.ao.pruning import fqn_to_module
from torch.cuda.amp import GradScaler
from torch.distributed.fsdp.fully_sharded_data_parallel import (
    BackwardPrefetch,
    CPUOffload,
    FullStateDictConfig,
    FullyShardedDataParallel as FSDP,
    LocalStateDictConfig,
    MixedPrecision,
    ShardedStateDictConfig,
    ShardingStrategy,
    StateDictType,
)
from torch.distributed.fsdp.sharded_grad_scaler import ShardedGradScaler
from torch.utils._mode_utils import no_dispatch
from torch.utils._python_dispatch import TorchDispatchMode


logger = logging.getLogger(__name__)


class CpuOverrideMode(TorchDispatchMode):
    def __torch_dispatch__(self, func, types, args, kwargs=None):
        if kwargs is not None:
            if "device" in kwargs:
                kwargs["device"] = torch.device("cpu")
        return func(*args, **kwargs)


def add_fsdp_configs(_C: CN):
    _C.FSDP = CN()
    _C.FSDP.ALGORITHM = "grad_optim"  # 'grad_optim', 'full', 'hybrid', 'hybrid_zero2'

    # Configs for fully sharded data parallel (fsdp)
    # Check out https://pytorch.org/docs/stable/fsdp.html
    # and docstring of torch.distributed.fsdp.fully_sharded_data_parallel
    _C.FSDP.CPU_OFFLOAD = False
    _C.FSDP.BACKWARD_PREFETCH = True
    _C.FSDP.USE_ORIG_PARAMS = False
    # Find autowrap policy at D2GO_WRAP_POLICY_REGISTRY, or use '' to disable autowrap
    _C.FSDP.AUTO_WRAP_POLICY = "never_wrap_policy"
    _C.FSDP.AUTO_WRAP_MIN_PARAMS = int(1e4)
    # A list of layer cls names to wrap, case sensitive
    _C.FSDP.AUTO_WRAP_LAYER_CLS = []
    # Whether to use local state dict -- superseded by STATE_DICT_TYPE
    _C.FSDP.USE_LOCAL_STATE_DICT = True
    # State dict type to use when calling FSDPWrapper.state_dict() (used when saving).
    # If None, defaults to checking the value of USE_LOCAL_STATE_DICT
    _C.FSDP.STATE_DICT_TYPE = "SHARDED_STATE_DICT"
    # Whether to offload state dict to cpu
    _C.FSDP.STATE_DICT_CPU_OFFLOAD = False
    # Whether to materialize state dict on rank 0
    _C.FSDP.STATE_DICT_RANK0_ONLY = True
    # The ignored modules, if any
    _C.FSDP.IGNORED_MODULES = None
    # Whether to prefetch in forward pass
    _C.FSDP.FORWARD_PREFETCH_OPTION = "no"
    # if False, this allows the CPU thread to schedule all-gathers without any extra synchronization
    _C.FSDP.LIMIT_ALL_GATHERS = False
    # flag for distributed FSDP model initialization
    _C.FSDP.DISTRIBUTED_INIT = False


class ShardingAlgorithm(str, Enum):
    """
    This enum specifies the sharding algorithm to be used by FullyShardedDataParallel (FSDP).
    It matches the strings used in D2Go config with the enum class :class:`ShardingStrategy` used by Pytorch FSDP module:
        "grad_optim" => ShardingAlgorithm.SHARD_GRAD_OP => ShardingStrategy.SHARD_GRAD_OP
        "full" => ShardingAlgorithm.FULL_SHARD => ShardingStrategy.FULL_SHARD
        "hybrid" => ShardingAlgorithm.HYBRID_SHARD => ShardingStrategy.HYBRID_SHARD
        "hybrid_zero2" => ShardingAlgorithm.HYBRID_SHARD_ZERO2 => ShardingStrategy._HYBRID_SHARD_ZERO2
    """

    SHARD_GRAD_OP = "grad_optim"
    FULL_SHARD = "full"
    HYBRID_SHARD = "hybrid"
    HYBRID_SHARD_ZERO2 = "hybrid_zero2"


class ForwardPrefetchOption(str, Enum):
    """
    This enum specifies the forward prefetch types to be used by FullyShardedDataParallel (FSDP).
        "auto" => Use the default forward prefetch mechanism in FSDP.
        "manual" => Use custom forward prefetch mechansim, implemented as training hooks.
        "no" => No forward prefetch.
    """

    AUTO = "auto"
    MANUAL = "manual"
    NO = "no"


def is_fsdp_enabled(cfg):
    return "FSDPModelingHook" in cfg.MODEL.MODELING_HOOKS


def get_grad_scaler(cfg):
    return ShardedGradScaler() if is_fsdp_enabled(cfg) else GradScaler()


def bottom_up_nested_fsdp(root_module, fsdp_kwargs: Dict[str, Any]):
    import torch.distributed as dist

    modules_to_fsdp: Tuple = tuple(fsdp_kwargs["auto_wrap_policy"]._module_classes)
    del fsdp_kwargs["auto_wrap_policy"]
    modules_not_to_fsdp: List = fsdp_kwargs["ignored_modules"]
    device_id = fsdp_kwargs["device_id"]
    cuda_device = torch.device(f"cuda:{device_id}")

    # postorder traversal (i.e. bottom-up)
    visited_modules: Set[nn.Module] = {root_module}

    def postorder_fsdp_wrap(
        module: nn.Module,
        module_name: str,
        fqn: str,
        parent_module: Optional[nn.Module],
        ignore_branch: bool,
    ):

        rank = dist.get_rank()
        # don't traverse branches of specified ignored modules
        if module in modules_not_to_fsdp:
            ignore_branch = True
        for child_name, child_module in module.named_children():
            if child_module not in visited_modules:
                visited_modules.add(child_module)
                postorder_fsdp_wrap(
                    child_module,
                    child_name,
                    f"{fqn}.{child_name}",
                    module,
                    ignore_branch,
                )

        logger.info(
            f"(Distributed FSDP init) Rank {rank} Beginning processing module: {fqn}"
        )
        # regardless of wrapping, we need to transfer all
        # module params and buffers to device, and if not rank 0,
        # need to retreive data from rank 0
        with torch.no_grad():
            if rank != 0:
                with no_dispatch():
                    for name, param in module.named_parameters(recurse=False):
                        setattr(
                            module,
                            name,
                            torch.nn.Parameter(
                                torch.empty_like(param, device=cuda_device),
                                requires_grad=param.requires_grad,
                            ),
                        )
                    for name, buffer in module.named_buffers(recurse=False):
                        setattr(
                            module,
                            name,
                            torch.empty_like(
                                buffer,
                                device=cuda_device,
                                requires_grad=buffer.requires_grad,
                            ),
                        )
            else:
                for _, param in module.named_parameters(recurse=False):
                    param.data = param.to(cuda_device)
                for _, buffer in module.named_buffers(recurse=False):
                    buffer.data = buffer.to(cuda_device)
            for _, param in module.named_parameters(recurse=False):
                dist.broadcast(param, 0)
            for _, buffer in module.named_buffers(recurse=False):
                dist.broadcast(buffer, 0)

            # if module is marked for FSDP, wrap it
            # AND if not in ignored branch
            if not ignore_branch and isinstance(module, modules_to_fsdp):
                logger.info(
                    f"(Distributed FSDP init) Rank {rank} FSDP Wrapping module: {fqn}"
                )
                setattr(parent_module, module_name, FSDP(module, **fsdp_kwargs))
        logger.info(
            f"(Distributed FSDP init) Rank {rank} Finished processing module: {fqn}"
        )

    postorder_fsdp_wrap(root_module, "root", "root", None, False)


class FSDPWrapper(FSDP):
    def __init__(
        self,
        model,
        state_dict_type: StateDictType,
        load_state_dict_type: StateDictType,
        amp_autocast_dtype: Optional[torch.dtype] = None,
        state_dict_cpu_offload: bool = True,
        state_dict_rank0_only: bool = True,
        distributed_init: bool = False,
        **fsdp_kwargs,
    ):
        self.precision = amp_autocast_dtype
        self.state_dict_type = state_dict_type
        self.load_state_dict_type = load_state_dict_type
        self.offload_to_cpu = state_dict_cpu_offload
        self.rank0_only = state_dict_rank0_only
        self.distributed_init = distributed_init

        if self.distributed_init:
            # NOTE traverse and apply all non-root level FSDP
            # and then wrap root level FSDP
            bottom_up_nested_fsdp(model, fsdp_kwargs)
        super().__init__(model, **fsdp_kwargs)
        logger.info(f"FSDP Wrapped model architecture: {self}")

    def forward(self, *args, **kwargs):
        # Wrap forward() in autocast if mixed precision is enabled
        if self.precision is not None and not torch.is_autocast_enabled():
            from torch.cuda.amp import autocast

            with autocast(dtype=self.precision):
                return super().forward(*args, **kwargs)
        else:
            return super().forward(*args, **kwargs)

    @contextlib.contextmanager
    def state_dict_type_and_config(self, state_dict_type: StateDictType) -> Generator:
        if state_dict_type == StateDictType.LOCAL_STATE_DICT:
            # only offload_to_cpu=False is supported for local state dict
            state_dict_config = LocalStateDictConfig(offload_to_cpu=False)
        elif state_dict_type == StateDictType.FULL_STATE_DICT:
            state_dict_config = FullStateDictConfig(
                offload_to_cpu=self.offload_to_cpu, rank0_only=self.rank0_only
            )
        else:
            state_dict_config = ShardedStateDictConfig(
                offload_to_cpu=self.offload_to_cpu
            )
        with FSDP.state_dict_type(self, state_dict_type, state_dict_config):
            yield

    def state_dict(self, *args, **kwargs):
        # NOTE: model.state_dict() needs to be called by all ranks because synchronization primitives are used
        with self.state_dict_type_and_config(self.state_dict_type):
            return super().state_dict(*args, **kwargs)

    def load_state_dict(
        self,
        state_dict,
        *args,
        **kwargs,
    ):
        with self.state_dict_type_and_config(self.load_state_dict_type):
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
    amp_autocast_dtype: Optional[torch.dtype] = None,
    # TODO: to remove after migration to state_dict_type completes
    use_local_state_dict: bool = False,
    load_local_state_dict: bool = False,
    state_dict_type: Optional[StateDictType] = None,
    state_dict_cpu_offload: bool = True,
    state_dict_rank0_only: bool = True,
    ignored_modules: Optional[nn.Module] = None,
    forward_prefetch: bool = False,
    use_orig_params: bool = False,
    device_id: Optional[int] = None,
    limit_all_gathers: bool = False,
    distributed_init: bool = False,
):
    if sharding_algorithm == ShardingAlgorithm.SHARD_GRAD_OP:
        sharding_strategy = ShardingStrategy.SHARD_GRAD_OP
        logger.info("Optimizer + Gradient State Sharding (ZeRO-2) is used")
    elif sharding_algorithm == ShardingAlgorithm.FULL_SHARD:
        sharding_strategy = ShardingStrategy.FULL_SHARD
        logger.info("Optimizer + Gradient + Horizontal Model Sharding (ZeRO-3) is used")
    elif sharding_algorithm == ShardingAlgorithm.HYBRID_SHARD:
        sharding_strategy = ShardingStrategy.HYBRID_SHARD
        logger.info(
            "Optimizer + Gradient + Horizontal Model Sharding (ZeRO-3) within a node is used"
        )
    elif sharding_algorithm == ShardingAlgorithm.HYBRID_SHARD_ZERO2:
        sharding_strategy = ShardingStrategy._HYBRID_SHARD_ZERO2
        logger.info(
            "Optimizer + Gradient State Sharding (ZeRO-2) within a node is used"
        )
    else:
        raise ValueError(
            f"Invalid sharding algorithm for FSDP. Can be {ShardingAlgorithm.SHARD_GRAD_OP}, "
            + f"{ShardingAlgorithm.FULL_SHARD}, {ShardingAlgorithm.HYBRID_SHARD}, or {ShardingAlgorithm.HYBRID_SHARD_ZERO2}."
        )

    auto_wrap_policy = (
        D2GO_WRAP_POLICY_REGISTRY.get(auto_wrap_policy_name)(
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
        "ignored_modules": ignored_modules,
        "forward_prefetch": forward_prefetch,
        "use_orig_params": use_orig_params,
        "device_id": torch.cuda.current_device() if not device_id else device_id,
        "limit_all_gathers": limit_all_gathers,
    }
    # default to using use_local_state_dict if state_dict_type is None
    if not state_dict_type:
        _state_dict_type = (
            StateDictType.LOCAL_STATE_DICT
            if use_local_state_dict
            else StateDictType.FULL_STATE_DICT
        )
    else:
        _state_dict_type = state_dict_type
    # load_state_dict_type defaults to load_local_state_dict
    _load_state_dict_type = (
        StateDictType.LOCAL_STATE_DICT
        if load_local_state_dict
        else StateDictType.FULL_STATE_DICT
    )
    wrapper_kwargs = {
        "amp_autocast_dtype": amp_autocast_dtype,
        "state_dict_type": _state_dict_type,
        "load_state_dict_type": _load_state_dict_type,
        "state_dict_cpu_offload": state_dict_cpu_offload,
        "state_dict_rank0_only": state_dict_rank0_only,
        "distributed_init": distributed_init,
    }

    return FSDPWrapper(model, **wrapper_kwargs, **fsdp_kwargs)


@MODELING_HOOK_REGISTRY.register()
class FSDPModelingHook(ModelingHook):
    """Modeling hook that wraps model in FSDP based on config"""

    def apply(self, model: nn.Module) -> FSDPWrapper:
        # SOLVER.AMP.ENABLED and SOLVER.AMP.PRECISION controls mixed precision for all parameters, buffers and reduce in FSDP
        precision_dtype = (
            parse_precision_from_string(self.cfg.SOLVER.AMP.PRECISION, lightning=False)
            if self.cfg.SOLVER.AMP.ENABLED
            else None
        )

        ignored_modules = None
        if isinstance(self.cfg.FSDP.IGNORED_MODULES, list):
            ignored_modules = []
            for mod_name in self.cfg.FSDP.IGNORED_MODULES:
                mod = fqn_to_module(model, mod_name)
                assert mod is not None, f"Module {mod_name} cannot be found in model."
                ignored_modules.append(mod)

        forward_prefetch = (
            self.cfg.FSDP.FORWARD_PREFETCH_OPTION == ForwardPrefetchOption.AUTO
        )
        _state_dict_type = (
            StateDictType[self.cfg.FSDP.STATE_DICT_TYPE]
            if self.cfg.FSDP.STATE_DICT_TYPE
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
            buffer_dtype=None,
            amp_autocast_dtype=precision_dtype,
            use_local_state_dict=self.cfg.FSDP.USE_LOCAL_STATE_DICT,
            load_local_state_dict=self.cfg.FSDP.USE_LOCAL_STATE_DICT,
            state_dict_type=_state_dict_type,
            state_dict_cpu_offload=self.cfg.FSDP.STATE_DICT_CPU_OFFLOAD,
            state_dict_rank0_only=self.cfg.FSDP.STATE_DICT_RANK0_ONLY,
            ignored_modules=ignored_modules,
            forward_prefetch=forward_prefetch,
            use_orig_params=self.cfg.FSDP.USE_ORIG_PARAMS,
            device_id=torch.cuda.current_device(),
            limit_all_gathers=self.cfg.FSDP.LIMIT_ALL_GATHERS,
            distributed_init=self.cfg.FSDP.DISTRIBUTED_INIT,
        )
        return wrapped_model

    def unapply(self, model: FSDPWrapper) -> nn.Module:
        raise NotImplementedError(
            "FSDPModelingHook.unapply() not implemented: can't unwrap a FSDP module"
        )
