"""Capture checkpoint-coordinate shards and their worker-local tensor bindings."""

import os
from collections import Counter
from collections.abc import Callable, Mapping, Set
from contextlib import ExitStack, contextmanager
from copy import deepcopy
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Protocol, TypeAlias

import torch

from shardfly.layout import StaticTensorLayout

if TYPE_CHECKING:
    from shardfly.integrations.loader_capture import LoaderCapture

TensorKey: TypeAlias = str | tuple[str, tuple[int, ...]]
TensorBinding: TypeAlias = torch.Tensor | StaticTensorLayout


@dataclass
class CapturedWeights:
    """Serializable shard records paired with local tensors from one capture."""

    shards: list[dict]
    tensors: dict[TensorKey, TensorBinding]
    materialized: set[TensorKey] = field(default_factory=set)


class CapturePlan(Protocol):
    """Worker-local API shared by training and inference capture sources."""

    def snapshot(self, names: Set[str] | None = None) -> CapturedWeights: ...


class TrainingCapture:
    """Read rank-local checkpoint chunks from a DCP-compatible model export."""

    def __init__(self, model, state_dict: Callable | None, stage: bool):
        self.model = model
        self.state_dict = state_dict
        self.stage = stage

    @torch.no_grad()
    def snapshot(self, names: Set[str] | None = None) -> CapturedWeights:
        from torch.distributed.checkpoint import DefaultSavePlanner
        from torch.distributed.checkpoint.state_dict import get_model_state_dict
        from torch.distributed.tensor import DTensor

        export = self.state_dict or get_model_state_dict
        state = export(self.model)
        if not isinstance(state, Mapping):
            raise TypeError("Training capture requires a DCP-compatible state dict")
        if names is not None:
            state = {name: value for name, value in state.items() if name in names}
        planner = DefaultSavePlanner()
        # Collect the local plan before DCP deduplicates replica writes.
        planner.set_up_planner(dict(state), is_coordinator=True)
        items = [
            item
            for item in planner.create_local_plan().items
            if item.tensor_data is not None and all(item.tensor_data.chunk.sizes)
        ]
        counts = Counter(item.index.fqn for item in items)
        native = set()
        if not self.stage:
            models = (
                self.model if isinstance(self.model, (list, tuple)) else [self.model]
            )
            for module in models:
                for parameter in module.parameters():
                    tensor = (
                        parameter.to_local()
                        if isinstance(parameter, DTensor)
                        else parameter
                    )
                    native.add((tensor.device, tensor.untyped_storage().data_ptr()))

        capture = CapturedWeights([], {})
        for item in items:
            name = item.index.fqn
            data = item.tensor_data
            offsets = tuple(data.chunk.offsets)
            shape = tuple(data.chunk.sizes)
            key = name if counts[name] == 1 else (name, offsets)
            tensor = planner.resolve_data(item)
            if key in capture.tensors:
                raise ValueError(f"Duplicate checkpoint region for {name} at {offsets}")
            if tuple(tensor.shape) != shape or tensor.dtype != data.properties.dtype:
                raise ValueError(
                    f"Training export returned a mismatched shard for {name}"
                )
            capture.shards.append(
                {
                    "name": name,
                    "global_shape": tuple(data.size),
                    "global_offset": offsets,
                    "shape": shape,
                    "dtype": str(tensor.dtype).removeprefix("torch."),
                    "binding_key": key,
                }
            )
            capture.tensors[key] = tensor.detach()
            if (tensor.device, tensor.untyped_storage().data_ptr()) not in native:
                capture.materialized.add(key)
        capture.shards.sort(key=lambda shard: (shard["name"], shard["global_offset"]))
        return capture


def capture_inference(model) -> "LoaderCapture":
    """Bind the layout recorded by the vLLM capture loader to its worker model."""
    from shardfly.integrations.loader_capture import LoaderCapture

    return LoaderCapture(model)


def capture_training(
    model, *, state_dict: Callable | None = None, stage=False
) -> TrainingCapture:
    """Capture a wrapped model on each training rank after the optimizer step.

    An optional state_dict(model) callback supplies canonical checkpoint names
    and DCP-compatible shard objects. Set stage=True if the framework replaces
    native parameter storage between publications.
    """
    return TrainingCapture(model, state_dict, stage)


def _megatron_views(task, parameter):
    """Interpret the resolved Bridge mapping as views of its rank-local parameter."""
    from megatron.bridge.models.conversion.param_mapping import (
        AutoMapping,
        ColumnParallelMapping,
        GatedMLPMapping,
        QKVMapping,
        ReplicatedMapping,
        RowParallelMapping,
    )
    from megatron.bridge.models.gpt_oss.gpt_oss_bridge import (
        GPTOSSMLPDownProjMapping,
        GPTOSSMLPGateUpProjMapping,
    )
    from megatron.bridge.utils.common_utils import extract_expert_number_from_param

    mapping = task.mapping
    weight = parameter.detach()
    module = task.megatron_module
    if isinstance(mapping, QKVMapping):
        config = mapping._get_config(module)
        head_dim = config.kv_channels
        query_rows = head_dim * (config.num_attention_heads // config.num_query_groups)
        group_rows = query_rows + 2 * head_dim
        groups, remainder = divmod(weight.shape[0], group_rows)
        if not groups or remainder or groups * mapping.tp_size != config.num_query_groups:
            raise ValueError(f"Unsupported local QKV layout for {task.global_param_name}")
        for key, start, rows in (("q", 0, query_rows), ("k", query_rows, head_dim),
                                 ("v", query_rows + head_dim, head_dim)):
            spans = tuple(weight.narrow(0, group * group_rows + start, rows) for group in range(groups))
            shape = (groups * rows, *weight.shape[1:])
            yield mapping.hf_param[key], StaticTensorLayout(shape, spans), 0, mapping
        return
    if isinstance(mapping, GatedMLPMapping):
        gate, up = weight.chunk(2, dim=0)
        yield mapping.hf_param["gate"], gate, 0, mapping
        yield mapping.hf_param["up"], up, 0, mapping
        return
    parallel = mapping
    if isinstance(parallel, AutoMapping):
        parallel = mapping._get_or_create_mapping(mapping._detect_parallelism_type(module))
    if isinstance(parallel, ColumnParallelMapping):
        axis = 0
    elif isinstance(parallel, RowParallelMapping):
        axis = 1 if weight.ndim > 1 else None
    elif isinstance(parallel, ReplicatedMapping):
        axis = None
    else:
        raise ValueError(
            f"Unsupported local Bridge mapping {type(mapping).__name__} for {task.global_param_name}"
        )
    if isinstance(mapping, (GPTOSSMLPGateUpProjMapping, GPTOSSMLPDownProjMapping)):
        from shardfly.integrations.loader_capture import gpt_oss_expert_keys

        expert = extract_expert_number_from_param(mapping.megatron_param)
        for name, _, projection, count in gpt_oss_expert_keys(mapping.hf_param, expert, 1):
            yield name, weight.chunk(count, dim=0)[projection], axis, parallel
        return
    permutation = getattr(mapping, "permute_dims", None)
    if permutation is not None:
        weight = weight.permute(permutation)
        if axis is not None:
            axis = permutation.index(axis)
    yield mapping.hf_param, weight, axis, parallel


class _MegatronCapture:
    """Compile checkpoint coordinates once and keep the Megatron storage live."""

    native = True

    def __init__(self, models, bridge):
        self.models = models
        self.bridge = bridge
        self.fsdp = []
        self._tasks = None
        self._snapshot = None
        self._snapshot_names = None
        self._views = {}
        self._pointers = {}
        for model in models:
            while model is not None:
                if (
                    hasattr(model, "all_gather_pipeline")
                    and getattr(model, "data_parallel_sharding_strategy", "no_shard") != "no_shard"
                ):
                    self.fsdp.append(model)
                    break
                model = getattr(model, "module", None)

    @property
    def static(self):
        return not self.fsdp

    def rank_coordinates(self):
        from megatron.core import parallel_state as mpu

        tp = mpu.get_tensor_model_parallel_rank()
        tp_size = mpu.get_tensor_model_parallel_world_size()
        dp = mpu.get_data_parallel_rank()
        cp_size = mpu.get_context_parallel_world_size()
        return dict(
            tp_rank=tp, tp_size=tp_size, attn_tp_rank=tp, attn_tp_size=tp_size,
            dp_rank=dp, dp_size=mpu.get_data_parallel_world_size(), attn_dp_rank=dp,
            pp_rank=mpu.get_pipeline_model_parallel_rank(),
            pp_size=mpu.get_pipeline_model_parallel_world_size(),
            ep_rank=mpu.get_expert_model_parallel_rank(),
            ep_size=mpu.get_expert_model_parallel_world_size(),
            ep_tp_rank=mpu.get_expert_tensor_parallel_rank(),
            ep_tp_size=mpu.get_expert_tensor_parallel_world_size(),
            cp_rank=mpu.get_context_parallel_rank(), cp_size=cp_size,
            cp_mode=os.environ.get("SHARDFLY_CP_MODE") or ("ring" if cp_size > 1 else "none"),
        )

    @contextmanager
    def publication(self):
        with ExitStack() as stack:
            for module in self.fsdp:
                stack.callback(module.synchronize_param_gather)
                module.start_param_sync(force_sync=True)
            yield

    @torch.no_grad()
    def snapshot(self, names=None):
        if self._snapshot is not None and self._snapshot_names == names:
            if self.static:
                return self._snapshot
            pointers = {key: tensor.data_ptr() for key, (tensor, _, _) in self._views.items()}
            if pointers != self._pointers:
                tensors = {}
                for key, (parameter, shape, spans) in self._views.items():
                    source = parameter.detach()
                    views = tuple(
                        source.as_strided(size, stride, source.storage_offset() + offset)
                        for size, stride, offset in spans
                    )
                    tensors[key] = StaticTensorLayout(shape, views) if shape else views[0]
                self._snapshot = CapturedWeights(self._snapshot.shards, tensors)
                self._pointers = pointers
            return self._snapshot
        if self._tasks is None:
            self._tasks = [
                task for task in self.bridge.get_conversion_tasks(self.models)
                if task.param_weight is not None and task.megatron_module is not None
            ]
        from torch.distributed.tensor import DTensor

        result = CapturedWeights([], {})
        self._views.clear()
        for task in self._tasks:
            parameter = task.param_weight
            original = getattr(parameter, "orig_param", None)
            if isinstance(original, torch.Tensor):
                parameter = original
            if isinstance(parameter, DTensor):
                parameter = parameter.to_local()
            for name, tensor, axis, mapping in _megatron_views(task, parameter):
                if names is not None and name not in names:
                    continue
                if name in result.tensors:
                    raise ValueError(f"Repeated local Bridge mapping for {name}")
                shape = tuple(tensor.shape)
                global_shape = list(shape)
                offset = [0] * len(shape)
                if axis is not None:
                    global_shape[axis] *= mapping.tp_size
                    offset[axis] = shape[axis] * mapping.tp_rank
                row = {
                    "name": name, "global_shape": tuple(global_shape),
                    "global_offset": tuple(offset), "shape": shape,
                    "dtype": str(parameter.dtype).removeprefix("torch."),
                }
                if isinstance(tensor, StaticTensorLayout):
                    row["span_numels"] = tensor.span_numels
                if self.fsdp:
                    views = tensor.spans if isinstance(tensor, StaticTensorLayout) else (tensor,)
                    self._views[name] = (
                        parameter,
                        shape if isinstance(tensor, StaticTensorLayout) else None,
                        tuple(
                            (span.shape, span.stride(), span.storage_offset() - parameter.storage_offset())
                            for span in views
                        ),
                    )
                result.shards.append(row)
                result.tensors[name] = tensor
        result.shards.sort(key=lambda shard: shard["name"])
        self._snapshot = result
        self._snapshot_names = None if names is None else set(names)
        self._pointers = {key: tensor.data_ptr() for key, (tensor, _, _) in self._views.items()}
        return result


def capture_megatron(models, bridge) -> CapturePlan:
    """Use Bridge checkpoint mappings to bind native Megatron parameter storage."""
    return _MegatronCapture(models, bridge)


def configure_megatron_checkpoint(provider, config):
    """Preserve the checkpoint's parameter layout before Bridge loads it."""
    if "Glm4MoeLiteForCausalLM" in (getattr(config, "architectures", None) or ()):
        provider.register_pre_wrap_hook(_preserve_router_fp32, prepend=True)
        stages, layers, draft = provider.pipeline_model_parallel_size, provider.num_layers, provider.mtp_num_layers or 0
        per_stage, remainder = divmod(layers + draft, stages)
        if (
            layers % stages and remainder == 0 and per_stage > draft
            and provider.num_layers_in_first_pipeline_stage is None
            and provider.num_layers_in_last_pipeline_stage is None
            and getattr(provider, "pipeline_model_parallel_layout", None) is None
        ):
            provider.num_layers_in_last_pipeline_stage = per_stage - draft
    if getattr(config, "model_type", None) != "deepseek_v3" or getattr(config, "q_lora_rank", False) is not None:
        return
    from megatron.core.transformer.mla_qk_norm_config import get_backend

    original_spec = provider.transformer_layer_spec

    def layer_spec(transformer_config, *args, **kwargs):
        block = deepcopy(original_spec(transformer_config, *args, **kwargs))
        for layer in block.layer_specs:
            attention = layer.submodules.self_attention

            class UncompressedQueryAttention(attention.module):
                def _resolve_qk_norm_config(self, submodules):
                    resolved = super()._resolve_qk_norm_config(submodules)
                    resolved["linear_q_proj"] = get_backend(
                        self.config.transformer_impl
                    ).column_parallel_linear()
                    return resolved

            attention.module = UncompressedQueryAttention
        return block

    provider.transformer_layer_spec = layer_spec


def _preserve_router_fp32(models):
    for model in models:
        for name, router in model.named_modules():
            if not name.endswith(".mlp.router"):
                continue
            kept = dict.fromkeys(getattr(router, "_keep_in_float32_parameter_names", ()))
            for name, parameter in router.named_parameters(recurse=False):
                if name in ("weight", "bias", "expert_bias"):
                    parameter.data = parameter.data.float()
                    kept[name] = None
            router._keep_in_float32_parameter_names = tuple(kept)
            expert_bias = dict(router.named_buffers(recurse=False)).get("expert_bias")
            if expert_bias is not None:
                expert_bias.data = expert_bias.data.float()
    return models
