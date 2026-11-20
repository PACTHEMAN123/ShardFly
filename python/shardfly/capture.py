"""Capture checkpoint-coordinate shards and their worker-local tensor bindings."""

from collections import Counter
from collections.abc import Callable, Mapping, Set
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Protocol, TypeAlias

import torch

if TYPE_CHECKING:
    from shardfly.integrations.loader_capture import LoaderCapture

TensorKey: TypeAlias = str | tuple[str, tuple[int, ...]]


@dataclass
class CapturedWeights:
    """Serializable shard records paired with local tensors from one capture."""

    shards: list[dict]
    tensors: dict[TensorKey, torch.Tensor]
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
