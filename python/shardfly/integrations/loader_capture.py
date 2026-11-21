"""Capture checkpoint slices written by an eager PyTorch model loader."""

from __future__ import annotations

from bisect import bisect_left, bisect_right
from collections import defaultdict
from dataclasses import dataclass, field, replace
from math import prod
from operator import attrgetter
from typing import Callable, Iterable

import torch
from torch.multiprocessing.reductions import StorageWeakRef
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils._pytree import tree_leaves

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

    def get_region(self, view: torch.Tensor) -> tuple[tuple[int, ...], tuple[int, ...]]:
        remaining = view.storage_offset() - self.offset
        offsets = []
        sizes = []
        if (
            view.dtype != self.dtype
            or view.is_conj()
            or view.is_neg()
            or remaining < 0
        ):
            raise ValueError(f"Unsupported checkpoint slice for {self.name}")
        varying_axes = iter(
            (size, stride) for size, stride in zip(view.shape, view.stride()) if size > 1
        )
        active = next(varying_axes, None)
        for full_size, full_stride in zip(self.shape, self.strides):
            offset, remaining = divmod(remaining, full_stride)
            size = 1
            if full_size > 1 and active is not None and active[1] == full_stride:
                size = active[0]
                active = next(varying_axes, None)
            if offset + size > full_size:
                raise ValueError(f"Unsupported checkpoint slice for {self.name}")
            offsets.append(offset)
            sizes.append(size)
        if active is not None or remaining:
            raise ValueError(f"Unsupported checkpoint slice for {self.name}")
        return tuple(offsets), tuple(sizes)


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

    @classmethod
    def from_tensor(cls, name: str, param: torch.Tensor) -> _NativeParameter:
        return cls(
            name,
            StorageWeakRef(param.untyped_storage()),
            tuple(param.shape),
            tuple(param.stride()),
            param.storage_offset(),
            param.dtype,
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
    global_offset: tuple[int, ...]
    shape: tuple[int, ...]


@dataclass(frozen=True)
class WeightLoadPlan:
    """Rank-local checkpoint mapping. Bindings hold no tensor payloads."""

    _copies: dict[str, _Copy]
    _parameters: dict[str, _NativeParameter]
    _refresh_derived: Callable[[torch.nn.Module], None] | None = None

    def with_checkpoint_alias(self, name: str, source: str) -> WeightLoadPlan:
        if name in self._copies or source not in self._copies:
            return self
        return replace(self, _copies={**self._copies, name: self._copies[source]})

    def _checked_parameters(self, model: torch.nn.Module) -> dict[str, torch.Tensor]:
        native = {
            name: param for name, param in model.named_parameters() if param.numel()
        }
        if native.keys() != self._parameters.keys():
            removed = sorted(self._parameters.keys() - native.keys())[:5]
            added = sorted(native.keys() - self._parameters.keys())[:5]
            raise ValueError(
                f"Loader changed the set of native parameters: removed={removed}, added={added}"
            )
        for name, parameter in self._parameters.items():
            parameter.check_binding(native[name])
        return native

    @torch.no_grad()
    def refresh(self, model: torch.nn.Module) -> None:
        if self._refresh_derived is not None:
            self._refresh_derived(model)
            self._checked_parameters(model)

    def bind(self, model: torch.nn.Module) -> dict[str, torch.Tensor]:
        """Return checkpoint-named views of the live inference parameters."""
        native = self._checked_parameters(model)
        return {
            name: (
                native[copy.native_name]
                .detach()
                .view(-1)
                .narrow(0, copy.native_offset, prod(copy.shape))
                .view(copy.shape)
            )
            for name, copy in self._copies.items()
        }

    def metadata(self, model: torch.nn.Module) -> list[dict]:
        self._checked_parameters(model)
        return [
            {
                "name": name,
                "global_shape": copy.global_shape,
                "global_offset": copy.global_offset,
                "shape": copy.shape,
                "dtype": str(self._parameters[copy.native_name].dtype).removeprefix(
                    "torch."
                ),
            }
            for name, copy in self._copies.items()
        ]


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
        self._copies: dict[str, _Copy] = {}
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

    def finish(
        self,
        model: torch.nn.Module,
        prepare_refresh: Callable[[set[str]], Callable[[torch.nn.Module], None]] | None = None,
    ) -> WeightLoadPlan:
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
        live = {name: param for name, param in model.named_parameters() if param.numel()}
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
            if source.name in self._copies or not source.shape:
                raise ValueError(
                    f"Unsupported multiple or scalar copies for {source.name}"
                )
            offset, shape = source.get_region(source_view)
            self._copies[source.name] = _Copy(
                native.name,
                start,
                source.shape,
                offset,
                shape,
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
