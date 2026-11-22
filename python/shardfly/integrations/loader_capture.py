"""Capture checkpoint slices written by an eager PyTorch model loader."""

import os
from bisect import bisect_left, bisect_right
from collections import Counter, defaultdict
from dataclasses import dataclass, field, replace
from functools import cache, partial
from math import prod
from operator import attrgetter
from typing import Callable, Iterable

import torch
from torch.multiprocessing.reductions import StorageWeakRef
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils._pytree import tree_leaves

from shardfly.capture import CapturedWeights, TensorKey

CAPTURE_ATTRIBUTE = "_shardfly_weight_load_plan"

# These operations only change the view of the original checkpoint storage.
_SOURCE_VIEWS = {
    "aten::alias",
    "aten::detach",
    "aten::expand",
    "aten::narrow",
    "aten::permute",
    "aten::select",
    "aten::slice",
    "aten::split",
    "aten::split_with_sizes",
    "aten::squeeze",
    "aten::t",
    "aten::transpose",
    "aten::unbind",
    "aten::unsqueeze",
    "aten::view",
}


@dataclass(frozen=True)
class _CheckpointTensor:
    name: str
    shape: tuple[int, ...]
    strides: tuple[int, ...]
    offset: int
    dtype: torch.dtype
    byte_start: int
    byte_end: int
    global_shape: tuple[int, ...]
    origin: tuple[int, ...]

    def region(self, view: torch.Tensor, allow_permutation: bool):
        remaining = view.storage_offset() - self.offset
        if (
            view.dtype != self.dtype
            or view.is_conj()
            or view.is_neg()
            or remaining < 0
        ):
            raise ValueError(f"Unsupported checkpoint slice for {self.name}")
        axes = {
            stride: axis for axis, (size, stride) in enumerate(zip(self.shape, self.strides))
            if size > 1
        }
        offsets = list(self.origin)
        sizes = [1] * len(self.shape)
        view_axes = [None] * len(self.shape)
        for stride, axis in sorted(axes.items(), reverse=True):
            offset, remaining = divmod(remaining, stride)
            offsets[axis] += offset
        last_axis = -1
        for view_axis, (size, stride) in enumerate(zip(view.shape, view.stride())):
            if size == 1:
                continue
            axis = axes.get(stride)
            if axis is None or view_axes[axis] is not None or (
                not allow_permutation and axis <= last_axis
            ):
                raise ValueError(f"Unsupported checkpoint slice for {self.name}")
            sizes[axis] = size
            view_axes[axis] = view_axis
            last_axis = axis
        if remaining or any(
            offset - origin + size > extent
            for offset, origin, size, extent in zip(offsets, self.origin, sizes, self.shape)
        ):
            raise ValueError(f"Unsupported checkpoint slice for {self.name}")
        return tuple(offsets), tuple(sizes), tuple(view_axes)


def _checkpoint_strides(tensor, shape, axes):
    strides = [1] * len(shape)
    next_stride = 1
    for index in reversed(range(len(shape))):
        axis = axes[index]
        strides[index] = tensor.stride(axis) if axis is not None else next_stride
        next_stride = strides[index] * shape[index]
    return tuple(strides)


@dataclass
class _CheckpointStorage:
    storage: StorageWeakRef
    tensors: list[_CheckpointTensor] = field(default_factory=list)

    def add(self, name: str, tensor: torch.Tensor, source=None, region=None):
        start = tensor.storage_offset() * tensor.element_size()
        end = start + tensor.numel() * tensor.element_size()
        index = bisect_left(self.tensors, start, key=attrgetter("byte_start"))
        if (index and self.tensors[index - 1].byte_end > start) or (
            index < len(self.tensors) and end > self.tensors[index].byte_start
        ):
            raise ValueError(f"Checkpoint storage overlaps another tensor for {name}")
        shape = tuple(tensor.shape) if region is None else region[1]
        self.tensors.insert(
            index,
            _CheckpointTensor(
                name,
                shape,
                tuple(tensor.stride()) if region is None else _checkpoint_strides(tensor, shape, region[2]),
                tensor.storage_offset(),
                tensor.dtype,
                start,
                end,
                shape if source is None else source.global_shape,
                (0,) * tensor.ndim if region is None else region[0],
            ),
        )

    def find(self, tensor: torch.Tensor) -> _CheckpointTensor | None:
        start = tensor.storage_offset() * tensor.element_size()
        index = bisect_right(self.tensors, start, key=attrgetter("byte_start")) - 1
        if index >= 0 and start < self.tensors[index].byte_end:
            return self.tensors[index]
        return None


@dataclass(frozen=True)
class _NativeParameter:
    name: str
    storage: StorageWeakRef
    shape: tuple[int, ...]
    strides: tuple[int, ...]
    offset: int
    dtype: torch.dtype

    @classmethod
    def from_tensor(cls, name, param):
        return cls(
            name, StorageWeakRef(param.untyped_storage()),
            tuple(param.shape), tuple(param.stride()), param.storage_offset(), param.dtype,
        )

    def check_binding(self, param: torch.Tensor):
        if (
            self.storage.expired()
            or _storage_key(param) != self.storage.cdata
            or tuple(param.shape) != self.shape
            or tuple(param.stride()) != self.strides
            or param.storage_offset() != self.offset
            or param.dtype != self.dtype
            or param.is_conj()
            or param.is_neg()
        ):
            raise ValueError(f"Loader changed the storage or layout of {self.name}")


@dataclass(frozen=True)
class _Copy:
    native_name: str
    native_offset: int
    global_shape: tuple[int, ...]
    shape: tuple[int, ...]
    strides: tuple[int, ...]


_EXPERT_PROJECTIONS = {
    "gate_up_proj": ("gate_proj", "up_proj"),
    "down_proj": ("down_proj",),
    "gate_up_proj_bias": ("gate_proj", "up_proj"),
    "down_proj_bias": ("down_proj",),
}


def gpt_oss_expert_keys(name, first, count):
    prefix, _, suffix = name.rpartition(".mlp.experts.")
    projections = _EXPERT_PROJECTIONS.get(suffix)
    if not prefix or projections is None:
        return
    kind = "bias" if suffix.endswith("_bias") else "weight"
    for expert in range(count):
        for row, projection in enumerate(projections):
            yield f"{prefix}.mlp.experts.{first + expert}.{projection}.{kind}", expert, row, len(projections)


def _gpt_oss_copies(name, offset, copy):
    keys = list(gpt_oss_expert_keys(name, offset[0], copy.shape[0]))
    if not keys:
        yield (name, offset), copy
    for key, expert, row, rows in keys:
        def project(values):
            if values[-1] % rows:
                raise ValueError(f"Unaligned expert projection in {name}")
            return (values[-1] // rows, *values[1:-1])

        yield (key, project(offset)), replace(
            copy,
            native_offset=copy.native_offset + expert * copy.strides[0] + row * copy.strides[-1],
            global_shape=project(copy.global_shape),
            shape=project(copy.shape),
            strides=(rows * copy.strides[-1], *copy.strides[1:-1]),
        )


@dataclass(frozen=True)
class WeightLoadPlan:
    """Rank-local checkpoint mapping. Bindings hold no tensor payloads."""

    _copies: dict[tuple[str, tuple[int, ...]], _Copy]
    _parameters: dict[str, _NativeParameter]
    _refresh: Callable | None = None

    def _checked_parameters(self, model: torch.nn.Module) -> dict[str, torch.Tensor]:
        native = {
            name: param for name, param in model.named_parameters() if param.numel()
        }
        if native.keys() != self._parameters.keys():
            raise ValueError("Loader changed the set of native parameters")
        for name, parameter in self._parameters.items():
            parameter.check_binding(native[name])
        return native

    def snapshot(self, model: torch.nn.Module, names=None) -> CapturedWeights:
        native = self._checked_parameters(model)
        counts = Counter(name for name, _ in self._copies)
        capture = CapturedWeights([], {})
        for (name, offset), copy in sorted(self._copies.items()):
            if names is not None and name not in names:
                continue
            key: TensorKey = name if counts[name] == 1 else (name, offset)
            binding = native[copy.native_name].detach()
            tensor = binding.as_strided(
                copy.shape, copy.strides, binding.storage_offset() + copy.native_offset
            )
            capture.shards.append({
                "name": name,
                "global_shape": copy.global_shape,
                "global_offset": offset,
                "shape": copy.shape,
                "dtype": str(tensor.dtype).removeprefix("torch."),
                "binding_key": key,
            })
            capture.tensors[key] = tensor
        return capture

    def refresh(self, model):
        if self._refresh is not None:
            self._refresh(model)

    def without_shared_parameters(self, model, owner):
        def layout(tensor):
            return (_storage_key(tensor), tensor.storage_offset(), tuple(tensor.shape), tensor.stride())

        owned = {layout(parameter) for parameter in owner.parameters() if parameter.numel()}
        shared = {
            name: _NativeParameter.from_tensor(name, parameter)
            for name, parameter in model.named_parameters()
            if parameter.numel() and layout(parameter) in owned
        }
        return replace(
            self,
            _parameters={**self._parameters, **shared},
            _copies={key: copy for key, copy in self._copies.items() if copy.native_name not in shared},
        )


@dataclass(frozen=True)
class _DraftLoadPlan:
    target: WeightLoadPlan
    draft: WeightLoadPlan

    def snapshot(self, model, names=None):
        target = self.target.snapshot(model.target, names)
        draft = self.draft.snapshot(model.draft, names)
        if target.tensors.keys() & draft.tensors.keys():
            raise ValueError("Target and MTP draft export the same checkpoint region")
        return CapturedWeights(target.shards + draft.shards, {**target.tensors, **draft.tensors})

    def _checked_parameters(self, model):
        self.target._checked_parameters(model.target)
        self.draft._checked_parameters(model.draft)

    def refresh(self, model):
        self.target.refresh(model.target)
        self.draft.refresh(model.draft)


class _DraftTransferModel(torch.nn.Module):
    def __init__(self, target, draft):
        super().__init__()
        self.target = target
        self.draft = draft
        self.config = target.config

    def named_parameters(self, prefix="", recurse=True, remove_duplicate=True):
        seen = set()
        for module, path in ((self.target, prefix), (self.draft, f"{prefix}.draft" if prefix else "draft")):
            for name, tensor in module.named_parameters(path, recurse, remove_duplicate):
                if remove_duplicate and id(tensor) in seen:
                    continue
                seen.add(id(tensor))
                yield name, tensor


def transfer_model(model_runner):
    model = model_runner.model
    plan = getattr(model, CAPTURE_ATTRIBUTE, None)
    if plan is None or model_runner.vllm_config.speculative_config is None:
        return model
    current = getattr(model_runner, "_shardfly_transfer_model", None)
    if current is not None:
        return current
    draft = getattr(getattr(model_runner, "speculator", None), "model", None)
    if draft is None:
        draft = getattr(getattr(model_runner, "drafter", None), "model", None)
    draft_plan = getattr(draft, CAPTURE_ATTRIBUTE, None)
    if draft_plan is None:
        raise ValueError("The MTP draft must also use the ShardFly capture loader")
    current = _DraftTransferModel(model, draft)
    setattr(current, CAPTURE_ATTRIBUTE, _DraftLoadPlan(plan, draft_plan.without_shared_parameters(draft, model)))
    model_runner._shardfly_transfer_model = current
    return current


class LoaderCapture:
    def __init__(self, model):
        self._model = model
        self._plan = getattr(model, CAPTURE_ATTRIBUTE, None)
        if self._plan is None:
            raise ValueError("The model was not loaded by the ShardFly capture loader")

    def snapshot(self, names=None) -> CapturedWeights:
        return self._plan.snapshot(self._model, names)

    def check_binding(self):
        self._plan._checked_parameters(self._model)

    def refresh(self):
        self._plan.refresh(self._model)


def _storage_key(tensor: torch.Tensor):
    if tensor.layout == torch.strided and tensor.device.type != "meta":
        return tensor.untyped_storage()._cdata
    return None


class WeightLoadCapture(TorchDispatchMode):
    """Observe real loader writes without keeping checkpoint tensors alive."""

    def __init__(self, *, stacked_experts=False):
        super().__init__()
        self._native: dict[int, _NativeParameter] = {}
        self._writes: dict[str, list[tuple[int, int]]] = defaultdict(list)
        self._sources: dict[int, _CheckpointStorage] = {}
        self._copies: dict[tuple[str, tuple[int, ...]], _Copy] = {}
        self._source_count = 0
        self._reading_checkpoint = False
        self._loading = False
        self._stacked_experts = stacked_experts

    def start(self, model: torch.nn.Module):
        if self._native:
            raise ValueError("Weight load capture can only track one model")
        for name, param in model.named_parameters():
            if not param.numel():
                continue
            key = _storage_key(param)
            if key is None or not param.is_contiguous() or key in self._native:
                raise ValueError(f"Unsupported native parameter storage for {name}")
            self._native[key] = _NativeParameter.from_tensor(name, param)
        self._loading = True

    def end(self):
        self._loading = False
        self._sources.clear()

    def weights(
        self, weights: Iterable[tuple[str, torch.Tensor]]
    ) -> Iterable[tuple[str, torch.Tensor]]:
        iterator = iter(weights)
        while True:
            self._reading_checkpoint = True
            try:
                name, tensor = next(iterator)
            except StopIteration:
                return
            finally:
                self._reading_checkpoint = False
            if tensor.numel():
                self._track_weight(name, tensor)
            yield name, tensor

    def _track_weight(self, name, tensor, source=None, region=None):
        key = _storage_key(tensor)
        if key is None or not tensor.is_contiguous() or key in self._native:
            raise ValueError(f"Unsupported checkpoint storage for {name}")
        self._source_count += 1
        if self._source_count % 256 == 0:
            self._sources = {
                key: source
                for key, source in self._sources.items()
                if not source.storage.expired()
            }
        if key not in self._sources:
            self._sources[key] = _CheckpointStorage(
                StorageWeakRef(tensor.untyped_storage())
            )
        self._sources[key].add(name, tensor, source, region)

    def _source(self, tensor):
        storage = self._sources.get(_storage_key(tensor))
        if storage is not None and tensor.numel():
            return storage.find(tensor)
        return None

    def finish(self, model: torch.nn.Module, prepare_refresh=None) -> WeightLoadPlan:
        if self._loading or not self._copies:
            raise ValueError("Model loader did not capture any checkpoint copies")
        live = {name: param for name, param in model.named_parameters() if param.numel()}
        checkpoint_targets = {copy.native_name for copy in self._copies.values()}
        parameters = {
            param.name: param
            for param in self._native.values()
            if param.name in live or param.name in checkpoint_targets
        }
        for parameter in parameters.values():
            cursor = 0
            for start, end in sorted(self._writes[parameter.name]):
                if start != cursor:
                    raise ValueError(
                        f"Incomplete or overlapping writes to {parameter.name}"
                    )
                cursor = end
            if cursor != prod(parameter.shape):
                raise ValueError(f"Incomplete writes to {parameter.name}")
        derived = live.keys() - parameters.keys()
        refresh = None
        if derived and prepare_refresh is not None:
            refresh = prepare_refresh(derived)
            parameters.update(
                (name, _NativeParameter.from_tensor(name, live[name])) for name in derived
            )
        plan = WeightLoadPlan(dict(self._copies), parameters, refresh)
        plan._checked_parameters(model)
        return plan

    def _record_write(self, schema, args, kwargs, native):
        target = args[0]
        name = schema.name
        if not target.numel():
            return
        if not self._loading:
            raise ValueError(f"Loader modified {native.name} after loading weights")
        if (
            not target.is_contiguous()
            or target.dtype != native.dtype
            or target.is_conj()
            or target.is_neg()
        ):
            raise ValueError(f"Unsupported native write layout for {native.name}")
        start = target.storage_offset() - native.offset
        end = start + target.numel()
        if start < 0 or end > prod(native.shape):
            raise ValueError(f"Write outside native parameter {native.name}")
        if name == "aten::copy_":
            source_view = args[1]
            source = self._source(source_view)
            if source is None or source_view.shape != target.shape:
                raise ValueError(
                    f"Expected a direct checkpoint slice for {native.name}"
                )
            if not source.shape:
                raise ValueError(f"Unsupported scalar checkpoint weight {source.name}")
            offset, shape, axes = source.region(source_view, self._stacked_experts)
            copy = _Copy(
                native.name, start, source.global_shape, shape,
                _checkpoint_strides(target, shape, axes),
            )
            copies = (
                _gpt_oss_copies(source.name, offset, copy)
                if self._stacked_experts else [((source.name, offset), copy)]
            )
            for key, region in copies:
                if key in self._copies:
                    raise ValueError(f"Repeated checkpoint region for {key[0]} at {key[1]}")
                self._copies[key] = region
        elif name == "aten::fill_" and schema.overload_name == "Scalar":
            value = args[1] if len(args) > 1 else kwargs["value"]
            if value != 0:
                raise ValueError(f"Expected zero padding for {native.name}")
        elif name != "aten::zero_":
            raise ValueError(f"Unsupported loader write {name} to {native.name}")
        self._writes[native.name].append((start, end))

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        if self._reading_checkpoint or not self._native:
            return func(*args, **kwargs)
        schema = func._schema
        if schema.is_mutable:
            writes = []
            for index, arg in enumerate(schema.arguments):
                if arg.alias_info is not None and arg.alias_info.is_write:
                    value = args[index] if index < len(args) else kwargs.get(arg.name)
                    writes.extend(
                        self._native[key]
                        for tensor in tree_leaves(value)
                        if isinstance(tensor, torch.Tensor)
                        and (key := _storage_key(tensor)) in self._native
                    )
            if writes:
                if len(writes) != 1 or schema.name not in {
                    "aten::copy_",
                    "aten::fill_",
                    "aten::zero_",
                }:
                    raise ValueError(
                        f"Unsupported loader mutation {schema.name} of {writes[0].name}"
                    )
                self._record_write(schema, args, kwargs, writes[0])
                return func(*args, **kwargs)
        if self._sources and schema.name not in _SOURCE_VIEWS:
            for tensor in tree_leaves((args, kwargs)):
                if isinstance(tensor, torch.Tensor):
                    source = self._source(tensor)
                    if source is not None:
                        if self._stacked_experts and schema.name == "aten::clone":
                            region = source.region(tensor, True)
                            result = func(*args, **kwargs)
                            self._track_weight(source.name, result, source, region)
                            return result
                        raise ValueError(
                            f"Unsupported checkpoint operation {schema.name} on {source.name}"
                        )
        return func(*args, **kwargs)


def _attention_refresh(names, *, model, dtype):
    from vllm.model_executor.model_loader import utils

    is_deferred = getattr(utils, "is_deferred_attention_layer", None)
    if is_deferred is None:
        attention = (utils.Attention, utils.MLAAttention, utils.MMEncoderAttention)

        def is_deferred(module):
            return isinstance(module, attention) and hasattr(module, "process_weights_after_loading")

    owners = sorted({name.rpartition(".")[0] for name in names})
    for name in owners:
        if not is_deferred(model.get_submodule(name)):
            raise ValueError(f"Loader added parameters outside deferred attention: {name}")

    def refresh(target):
        for name in owners:
            target.get_submodule(name).process_weights_after_loading(dtype)

    return refresh


@cache
def register_capture_loader():
    from vllm.model_executor import model_loader

    # Older vLLM versions can still use the standard ShardFly plugin.
    register = getattr(model_loader, "register_model_loader", None)
    if register is None:
        return

    @register("auto")
    class CaptureModelLoader(model_loader.DefaultModelLoader):
        def __init__(self, load_config):
            super().__init__(load_config)
            self._capture = None

        def load_model(self, vllm_config, model_config, *args, **kwargs):
            if model_config.quantization is not None:
                quantization = getattr(model_config.hf_config, "quantization_config", {})
                block = [
                    int(os.environ.get("SHARDFLY_FP8_BLOCK_ROWS", "128")),
                    int(os.environ.get("SHARDFLY_FP8_BLOCK_COLS", "128")),
                ]
                if (
                    model_config.quantization != "fp8"
                    or os.environ.get("SHARDFLY_FP8_BLOCKWISE") != "1"
                    or list((quantization or {}).get("weight_block_size") or ()) != block
                ):
                    raise ValueError(f"ShardFly capture requires native FP8 with block size {block}")
            speculative = vllm_config.speculative_config
            if speculative is not None:
                target = getattr(speculative, "target_model_config", model_config)
                architectures = getattr(target.hf_config, "architectures", None) or ()
                if "Glm4MoeLiteForCausalLM" not in architectures or not str(speculative.method).endswith("mtp"):
                    raise ValueError("ShardFly capture supports GLM-4.7-Flash MTP")
            hf_config = getattr(model_config, "hf_config", None)
            architectures = getattr(hf_config, "architectures", None) or ()
            with WeightLoadCapture(stacked_experts="GptOssForCausalLM" in architectures) as capture:
                self._capture = capture
                try:
                    model = super().load_model(
                        vllm_config, model_config, *args, **kwargs
                    )
                finally:
                    self._capture = None
            prepare_refresh = partial(_attention_refresh, model=model, dtype=getattr(model_config, "dtype", None))
            setattr(model, CAPTURE_ATTRIBUTE, capture.finish(model, prepare_refresh))
            return model

        def load_weights(self, model, model_config):
            if self._capture is None:
                return super().load_weights(model, model_config)
            self._capture.start(model)
            super().load_weights(model, model_config)
            self._capture.end()

        def get_all_weights(self, model_config, model):
            weights = super().get_all_weights(model_config, model)
            if self._capture is not None:
                return self._capture.weights(weights)
            return weights
