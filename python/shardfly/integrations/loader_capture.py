"""Capture checkpoint slices written by an eager PyTorch model loader."""

from bisect import bisect_left, bisect_right
from collections import Counter, defaultdict
from copy import copy as copy_config
from dataclasses import dataclass, field
from math import prod
from operator import attrgetter
from typing import Iterable

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

    def get_offset(self, view: torch.Tensor) -> tuple[int, ...]:
        remaining = view.storage_offset() - self.offset
        offsets = []
        if (
            view.ndim != len(self.shape)
            or view.dtype != self.dtype
            or view.is_conj()
            or view.is_neg()
            or remaining < 0
        ):
            raise ValueError(f"Unsupported checkpoint slice for {self.name}")
        for size, stride, full_size, full_stride in zip(
            view.shape, view.stride(), self.shape, self.strides
        ):
            offset, remaining = divmod(remaining, full_stride)
            if offset + size > full_size or (size > 1 and stride != full_stride):
                raise ValueError(f"Unsupported checkpoint slice for {self.name}")
            offsets.append(offset)
        if remaining:
            raise ValueError(f"Misaligned checkpoint slice for {self.name}")
        return tuple(offsets)


@dataclass
class _CheckpointStorage:
    storage: StorageWeakRef
    tensors: list[_CheckpointTensor] = field(default_factory=list)

    def add(self, name: str, tensor: torch.Tensor):
        start = tensor.storage_offset() * tensor.element_size()
        end = start + tensor.numel() * tensor.element_size()
        index = bisect_left(self.tensors, start, key=attrgetter("byte_start"))
        if (index and self.tensors[index - 1].byte_end > start) or (
            index < len(self.tensors) and end > self.tensors[index].byte_start
        ):
            raise ValueError(f"Checkpoint storage overlaps another tensor for {name}")
        self.tensors.insert(
            index,
            _CheckpointTensor(
                name,
                tuple(tensor.shape),
                tuple(tensor.stride()),
                tensor.storage_offset(),
                tensor.dtype,
                start,
                end,
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


@dataclass(frozen=True)
class WeightLoadPlan:
    """Rank-local checkpoint mapping. Bindings hold no tensor payloads."""

    _copies: dict[tuple[str, tuple[int, ...]], _Copy]
    _parameters: dict[str, _NativeParameter]

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
            tensor = (
                native[copy.native_name]
                .detach()
                .view(-1)
                .narrow(0, copy.native_offset, prod(copy.shape))
                .view(copy.shape)
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

    def bind(self, model: torch.nn.Module) -> dict[TensorKey, torch.Tensor]:
        return self.snapshot(model).tensors

    def metadata(self, model: torch.nn.Module) -> list[dict]:
        return self.snapshot(model).shards


class LoaderCapture:
    def __init__(self, model):
        self._model = model
        self._plan = getattr(model, CAPTURE_ATTRIBUTE, None)
        if self._plan is None:
            raise ValueError("Start vLLM with --load-format shardfly_capture")

    def snapshot(self, names=None) -> CapturedWeights:
        return self._plan.snapshot(self._model, names)

    def check_binding(self):
        self._plan._checked_parameters(self._model)


def _storage_key(tensor: torch.Tensor):
    if tensor.layout == torch.strided and tensor.device.type != "meta":
        return tensor.untyped_storage()._cdata
    return None


class WeightLoadCapture(TorchDispatchMode):
    """Observe real loader writes without keeping checkpoint tensors alive."""

    def __init__(self):
        super().__init__()
        self._native: dict[int, _NativeParameter] = {}
        self._writes: dict[str, list[tuple[int, int]]] = defaultdict(list)
        self._sources: dict[int, _CheckpointStorage] = {}
        self._copies: dict[tuple[str, tuple[int, ...]], _Copy] = {}
        self._source_count = 0
        self._reading_checkpoint = False
        self._loading = False

    def start(self, model: torch.nn.Module):
        if self._native:
            raise ValueError("Weight load capture can only track one model")
        for name, param in model.named_parameters():
            if not param.numel():
                continue
            key = _storage_key(param)
            if key is None or not param.is_contiguous() or key in self._native:
                raise ValueError(f"Unsupported native parameter storage for {name}")
            self._native[key] = _NativeParameter(
                name,
                StorageWeakRef(param.untyped_storage()),
                tuple(param.shape),
                tuple(param.stride()),
                param.storage_offset(),
                param.dtype,
            )
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

    def _track_weight(self, name, tensor):
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
        self._sources[key].add(name, tensor)

    def _source(self, tensor):
        storage = self._sources.get(_storage_key(tensor))
        if storage is not None and tensor.numel():
            return storage.find(tensor)
        return None

    def finish(self, model: torch.nn.Module) -> WeightLoadPlan:
        if self._loading or not self._copies:
            raise ValueError("Model loader did not capture any checkpoint copies")
        parameters = {param.name: param for param in self._native.values()}
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
        plan = WeightLoadPlan(dict(self._copies), parameters)
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
            key = (source.name, source.get_offset(source_view))
            if key in self._copies:
                raise ValueError(f"Repeated checkpoint region for {source.name} at {key[1]}")
            self._copies[key] = _Copy(
                native.name,
                start,
                source.shape,
                tuple(target.shape),
            )
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
                        raise ValueError(
                            f"Unsupported checkpoint operation {schema.name} on {source.name}"
                        )
        return func(*args, **kwargs)


def register_capture_loader():
    from vllm.model_executor import model_loader

    # Older vLLM versions can still use the standard ShardFly plugin.
    register = getattr(model_loader, "register_model_loader", None)
    if register is None:
        return

    @register("shardfly_capture")
    class CaptureModelLoader(model_loader.DefaultModelLoader):
        def __init__(self, load_config):
            load_config = copy_config(load_config)
            load_config.load_format = "auto"
            super().__init__(load_config)
            self._capture = None

        def load_model(self, vllm_config, model_config, *args, **kwargs):
            if model_config.quantization is not None:
                raise ValueError("shardfly_capture requires an unquantized vLLM model")
            if vllm_config.speculative_config is not None:
                raise ValueError("shardfly_capture does not support speculative models")
            with WeightLoadCapture() as capture:
                self._capture = capture
                try:
                    model = super().load_model(
                        vllm_config, model_config, *args, **kwargs
                    )
                finally:
                    self._capture = None
            setattr(model, CAPTURE_ATTRIBUTE, capture.finish(model))
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
