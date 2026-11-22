"""Capture RTP-LLM's checkpoint layout at load time and publish through ShardFly."""

from dataclasses import dataclass
from math import prod
from threading import Lock
from types import SimpleNamespace

import torch
import torch.distributed as dist
from torch.utils._python_dispatch import TorchDispatchMode, _disable_current_modes

from shardfly.capture import CapturedWeights
from shardfly.integrations.loader_capture import CAPTURE_ATTRIBUTE


class _LayoutOnly(TorchDispatchMode):
    # An integer probe may follow layout operations and zero padding. Reject
    # arithmetic and quantization rather than interpreting their output as a map.
    _operations = {
        "alias", "arange", "as_strided", "cat", "chunk", "clone", "concat",
        "concatenate", "constant_pad_nd", "contiguous", "detach", "empty", "expand",
        "flatten", "narrow", "new_zeros", "numpy_T", "pad", "permute", "reshape",
        "select", "slice", "split", "split_with_sizes", "squeeze", "stack",
        "t", "transpose", "unbind", "unflatten", "unsqueeze", "view", "_unsafe_view",
        "zero_", "zeros", "zeros_like", "_to_copy",
    }

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        name = func._schema.name.removeprefix("aten::")
        if name == "to":
            # Torch may expose the high-level conversion under inference mode.
            return args[0]
        if name not in self._operations:
            raise ValueError(f"RTP-LLM loader capture cannot replay {func}")
        if name in {"_to_copy", "zeros", "zeros_like", "new_zeros"}:
            kwargs = {**kwargs, "device": torch.device("cpu"), "dtype": torch.int64}
        return func(*args, **kwargs)


@dataclass(frozen=True)
class _CheckpointRegion:
    name: str
    shape: tuple[int, ...]
    first: int

    @property
    def end(self):
        return self.first + prod(self.shape)


class _CheckpointProbes:
    def __init__(self, database, checkpoint_names):
        self.database = database
        self.checkpoint_names = checkpoint_names
        self.regions = {}
        self.next_index = 1  # RTP-LLM uses zero for padding.

    def has_tensor(self, name):
        return name in self.checkpoint_names

    def get_database(self):
        return self.database

    def load_tensor(self, name, data_type=None):
        if not self.has_tensor(name):
            return []
        if name not in self.regions:
            with _disable_current_modes():
                get_shape = getattr(self.database, "get_tensor_shape", None)
                if get_shape is None:
                    tensors = self.database.load_tensor(name, torch.bfloat16)
                    if len(tensors) != 1:
                        raise ValueError(f"Expected one HF checkpoint tensor for {name}")
                    shape = tuple(tensors[0].shape)
                else:
                    shape = tuple(get_shape(name))
            if not shape or len(shape) > 2:
                raise ValueError(f"RTP-LLM capture requires a vector or matrix for {name}")
            region = _CheckpointRegion(name, shape, self.next_index)
            self.regions[name] = region
            self.next_index = region.end
        region = self.regions[name]
        return [torch.arange(region.first, region.end, dtype=torch.int64).view(region.shape)]


def _checkpoint_view(probe, target, region):
    mask = (probe >= region.first) & (probe < region.end)
    if not mask.any():
        return None
    bounds = []
    for dim in range(probe.ndim):
        other_dims = tuple(i for i in range(probe.ndim) if i != dim)
        present = mask.any(dim=other_dims) if other_dims else mask
        indices = present.nonzero().flatten()
        bounds.append(slice(int(indices[0]), int(indices[-1]) + 1))
    bounds = tuple(bounds)
    if not mask[bounds].all():
        raise ValueError(f"Non-rectangular RTP-LLM destination for {region.name}")
    markers = probe[bounds]
    if markers.ndim != len(region.shape):
        raise ValueError(f"RTP-LLM loader changes the number of axes for {region.name}")
    if markers.ndim == 2:
        width = region.shape[1]
        same_axes = (
            (markers.shape[0] == 1 or int(markers[1, 0] - markers[0, 0]) == width)
            and (markers.shape[1] == 1 or int(markers[0, 1] - markers[0, 0]) == 1)
        )
        if not same_axes:
            markers = markers.t()
            target = target[bounds].t()
        else:
            target = target[bounds]
        offset = divmod(int(markers[0, 0]) - region.first, width)
        base = markers[0, 0]
        if (
            not (markers[:, 0] == base + width * torch.arange(markers.shape[0])).all()
            or not (markers == markers[:, :1] + torch.arange(markers.shape[1])).all()
            or offset[1] + markers.shape[1] > width
        ):
            raise ValueError(f"Unsupported RTP-LLM permutation for {region.name}")
    else:
        target = target[bounds]
        offset = (int(markers[0]) - region.first,)
        if not (markers == markers[0] + torch.arange(markers.numel())).all():
            raise ValueError(f"Unsupported RTP-LLM slice for {region.name}")
    return offset, target


def _destination_views(probe, target, regions):
    if probe.ndim <= 2:
        slices = ((probe, target),)
    elif probe.ndim == 3:
        # Expert-packed weights store independent matrices along the first axis.
        slices = zip(probe.unbind(0), target.unbind(0))
    else:
        raise ValueError(f"Unsupported RTP-LLM destination shape {tuple(probe.shape)}")

    regions = tuple(regions)
    starts = torch.tensor([region.first for region in regions])
    for markers, destination in slices:
        count = int(torch.count_nonzero(markers))
        if not count:
            continue
        edges = markers if markers.ndim == 1 else torch.cat((markers[0], markers[:, 0]))
        candidates = torch.searchsorted(starts, edges.contiguous(), right=True).unique().tolist()
        candidates = {index - 1 for index in candidates if index > 0}
        covered = 0
        for indices in (sorted(candidates), (i for i in range(len(regions)) if i not in candidates)):
            for index in indices:
                region = regions[index]
                match = _checkpoint_view(markers, destination, region)
                if match is not None:
                    offset, view = match
                    yield region, offset, view
                    covered += view.numel()
            if covered == count:
                break
        if covered != count:
            raise ValueError(f"Incomplete RTP-LLM capture for destination {tuple(destination.shape)}")


class _RtpLlmLoadPlan:
    def __init__(self):
        self.capture = CapturedWeights([], {})
        self.copy_targets = []
        self.copy_sources = []

    @torch.no_grad()
    def bind(self, region, offset, target):
        key = (region.name, offset)
        existing = self.capture.tensors.get(key)
        if existing is not None:
            if existing.shape != target.shape or existing.dtype != target.dtype:
                raise ValueError(f"Conflicting RTP-LLM destinations for {key}")
            if existing.data_ptr() != target.data_ptr() or existing.stride() != target.stride():
                self.copy_targets.append(target)
                self.copy_sources.append(existing)
            return
        source = target
        if target.ndim > 1 and target.stride(-1) != 1:
            # RTP-LLM GEMM weights may be destination-contiguous in the other
            # axis order. Receive once and keep the native tensor address fixed.
            source = target.contiguous()
            self.copy_targets.append(target)
            self.copy_sources.append(source)
        self.capture.shards.append({
            "name": region.name,
            "global_shape": region.shape,
            "global_offset": offset,
            "shape": tuple(target.shape),
            "dtype": str(target.dtype).removeprefix("torch."),
            "binding_key": key,
        })
        self.capture.tensors[key] = source

    def snapshot(self, _weights, names=None):
        if names is None:
            return self.capture
        shards = [row for row in self.capture.shards if row["name"] in names]
        return CapturedWeights(shards, {
            row["binding_key"]: self.capture.tensors[row["binding_key"]] for row in shards
        })

    @torch.inference_mode()
    def refresh(self, _weights):
        if self.copy_targets:
            torch._foreach_copy_(self.copy_targets, self.copy_sources)


def load_weights_with_capture(loader, device, **kwargs):
    """Call from the RTP-LLM model loader before it releases checkpoint metadata."""
    config = loader.get_load_config()
    if config.is_ft_style_weight or config.merge_lora:
        raise ValueError("RTP-LLM capture requires the original HF checkpoint layout")
    weights = loader.load_weights(device, **kwargs)
    _, units = loader._generate_weight_info()
    database = config.database
    checkpoint_names = frozenset(database.get_pretrain_tensor_names())
    plan = _RtpLlmLoadPlan()
    bound = set()
    for weight, layer, _ in units:
        probes = _CheckpointProbes(database, checkpoint_names)
        with _LayoutOnly():
            outputs = weight.load(probes, layer, "cpu", config)
        live = weights.global_weights if layer is None else weights.weights[layer]
        for name, probe in outputs.items():
            target = live[name].detach()
            if probe.dtype != torch.int64 or probe.shape != target.shape:
                raise ValueError(
                    f"RTP-LLM loader replay changed the layout of {(layer, name)} "
                    f"from {tuple(target.shape)} to {tuple(probe.shape)} "
                    f"with marker dtype {probe.dtype}"
                )
            if target.dtype not in {torch.bfloat16, torch.float16, torch.float32}:
                raise ValueError(f"RTP-LLM capture does not support {target.dtype} at {(layer, name)}")
            for region, offset, view in _destination_views(probe, target, probes.regions.values()):
                plan.bind(region, offset, view)
            bound.add((layer, name))
    live_names = {(None, name) for name in weights.global_weights}
    live_names.update((layer, name) for layer, row in enumerate(weights.weights) for name in row)
    if live_names != bound or not plan.capture.shards:
        raise ValueError(f"RTP-LLM loader produced unbound weights: {live_names - bound}")
    setattr(weights, CAPTURE_ATTRIBUTE, plan)
    return weights


class RtpLlmWorker:
    """Bind a ShardFly worker RPC to a live RTP-LLM ModelWeights object."""

    def __init__(self, weights, rank_context, flush_cache):
        self.weights = weights
        device = torch.device(weights.device)
        index = torch.cuda.current_device() if device.index is None else device.index
        self.device = torch.device("cuda", index)
        self.rank_context = {
            **rank_context,
            "infer_engine_config": SimpleNamespace(comm_backend="transport"),
            "scheduler": SimpleNamespace(flush_cache=flush_cache, gpu_id=index),
        }

    def execute(self, task, **kwargs):
        with torch.cuda.device(self.device):
            if not dist.is_initialized():
                if self.rank_context["world_size"] != 1:
                    raise RuntimeError("Initialize the RTP-LLM worker group before publication")
                # Single-worker RTP-LLM can start without a Torch group. This
                # local group lets the shared reader register the NCCL transfer group.
                dist.init_process_group("gloo", store=dist.HashStore(), rank=0, world_size=1)
            return task(model=self.weights, model_context=self.rank_context, **kwargs)


class RtpLlmRollout:
    """Reuse the ShardFly reader and publication endpoints.

    The executor runs each task on every live worker. The quiesce context
    drains active inference before allowing the reader to overwrite weights.
    """

    def __init__(self, execute_on_workers, quiesce):
        self.execute_task_in_model_worker = execute_on_workers
        self.quiesce = quiesce
        self.reader = None
        self.lock = Lock()

    def initialize(self, config):
        from shardfly.integrations.reader import WeightsReader
        from shardfly.integrations.vllm import InferenceConfig

        with self.lock:
            if self.reader is not None:
                raise RuntimeError("RTP-LLM rollout already initialized")
            self.config = InferenceConfig.from_dict(config)
            self.engine_rank = self.config.engine_rank
            reader = WeightsReader(self)
            reader.initialize()
            self.reader = reader

    def update(self, step_id, **kwargs):
        with self.lock:
            if self.reader is None:
                raise RuntimeError("Initialize the RTP-LLM rollout before updating weights")
            with self.quiesce():
                self.reader.update_weights(step_id, **kwargs)


def register_rtp_llm_routes(app, rollout):
    """Expose the publication protocol on the rollout's control application."""
    @app.post("/shardfly_init")
    def initialize(config: dict):
        rollout.initialize(config)

    @app.post("/shardfly_update")
    def update(request: dict):
        rollout.update(request["step_id"], **request.get("kwargs", {}))
