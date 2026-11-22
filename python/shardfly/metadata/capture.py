"""Assemble either side's captured regions in checkpoint coordinates."""

from collections import defaultdict
from itertools import combinations
from math import prod

import torch

from shardfly._utils.common import _is_allowed_infer_only_alias, is_fused_fp8
from shardfly.metadata.parameter import (
    ParameterMeta,
    ParameterReplicaMeta,
    ParameterShardMeta,
    ShardingType,
)


def _validate_tiling(name, geometry, global_shape):
    if sum(prod(shape) for _, shape in geometry) != prod(global_shape):
        raise ValueError(f"Incomplete checkpoint coverage for {name}")
    axes = {
        i
        for _, shape in geometry
        for i, size in enumerate(shape)
        if size != global_shape[i]
    }
    if len(axes) <= 1 and axes:
        axis = next(iter(axes))
        end = 0
        for offset, shape in sorted(geometry):
            if offset[axis] != end:
                raise ValueError(
                    f"Overlapping or missing checkpoint regions for {name}"
                )
            end += shape[axis]
    elif len(axes) > 1:
        for (a, a_shape), (b, b_shape) in combinations(geometry, 2):
            if all(
                x < y + n and y < x + m for x, m, y, n in zip(a, a_shape, b, b_shape)
            ):
                raise ValueError(f"Overlapping checkpoint regions for {name}")
    return next(iter(axes)) if len(axes) == 1 else 0


def build_captured_parameters_meta(workers: list[dict]) -> list[ParameterMeta]:
    by_name = defaultdict(list)
    ranks = sorted(worker["rank_info"].global_rank for worker in workers)
    if any(worker["rank_info"].world_size != len(ranks) for worker in workers) or (
        ranks != list(range(len(ranks)))
    ):
        raise ValueError("Capture requires metadata from every participating rank")
    for worker in workers:
        for param in worker["params_meta"]:
            by_name[param["name"]].append((worker["rank_info"], param))
    if not by_name:
        raise ValueError("Capture did not export any checkpoint weights")

    parameters = []
    for name, entries in sorted(by_name.items()):
        global_shape = tuple(entries[0][1]["global_shape"])
        dtype = getattr(torch, entries[0][1]["dtype"])
        geometry = defaultdict(list)
        bindings = set()
        for rank, param in entries:
            shape = tuple(param["shape"])
            offset = tuple(param["global_offset"])
            key = param.get("binding_key", name)
            if isinstance(key, list):
                key = (key[0], tuple(key[1]))
            binding = rank.global_rank, key
            if (
                tuple(param["global_shape"]) != global_shape
                or getattr(torch, param["dtype"]) != dtype
                or binding in bindings
            ):
                raise ValueError(f"Inconsistent capture metadata for {name}")
            bindings.add(binding)
            if not len(shape) == len(offset) == len(global_shape) or any(
                start < 0 or size <= 0 or start + size > total
                for start, size, total in zip(offset, shape, global_shape)
            ):
                raise ValueError(f"Invalid checkpoint region for {name}")
            spans = tuple(param.get("span_numels", ()))
            if spans and (min(spans) < 1 or sum(spans) != prod(shape)):
                raise ValueError(f"Invalid captured spans for {name}")
            geometry[offset, shape].append((rank, key, spans))
        axis = _validate_tiling(name, geometry, global_shape)
        replica_count = len(next(iter(geometry.values())))
        replicas = [ParameterReplicaMeta([]) for _ in range(replica_count)]
        for (offset, shape), owners in sorted(geometry.items()):
            if (
                len(owners) != replica_count
                or len({rank.global_rank for rank, _, _ in owners}) != replica_count
            ):
                raise ValueError(f"Unevenly replicated checkpoint regions for {name}")
            for replica, (rank, key, spans) in zip(
                replicas, sorted(owners, key=lambda owner: owner[0].global_rank)
            ):
                replica.shards.append(
                    ParameterShardMeta(
                        name=name,
                        shape=shape,
                        numel=prod(shape),
                        dtype=dtype,
                        global_offset=offset,
                        sharding_type=(
                            ShardingType.TP_SHARDING
                            if len(geometry) > 1
                            else ShardingType.NO_SHARDING
                        ),
                        sharding_dim=axis,
                        num_shards=len(geometry),
                        tp_rank=rank.tp_rank,
                        attn_tp_rank=rank.attn_tp_rank,
                        pp_rank=rank.pp_rank,
                        ep_rank=rank.ep_rank,
                        ep_tp_rank=rank.ep_tp_rank,
                        global_rank=rank.global_rank,
                        world_size=rank.world_size,
                        engine_rank=rank.engine_rank,
                        dp_rank=rank.dp_rank,
                        dp_size=rank.dp_size,
                        cp_rank=rank.cp_rank,
                        cp_size=rank.cp_size,
                        cp_mode=rank.cp_mode,
                        binding_key=key if key != name else None,
                        span_numels=spans,
                    )
                )
        parameters.append(
            ParameterMeta(
                name=name,
                global_shape=global_shape,
                global_numel=prod(global_shape),
                dtype=dtype,
                replicas=replicas,
                shards=[shard for replica in replicas for shard in replica.shards],
            )
        )
    return parameters


def validate_captured_weights(training, inference):
    """Offsets can differ arbitrarily; checkpoint parameter identity must agree."""
    sources = {weight.name: (weight.global_shape, weight.dtype) for weight in training}
    targets = {weight.name: (weight.global_shape, weight.dtype) for weight in inference}
    missing = sorted(
        name for name in targets.keys() - sources.keys()
        if not _is_allowed_infer_only_alias(name, targets.keys(), sources.keys())
    )
    extra = sorted(sources.keys() - targets.keys())
    mismatched = []
    for name in sources.keys() & targets.keys():
        source_shape, source_dtype = sources[name]
        target_shape, target_dtype = targets[name]
        if source_shape != target_shape or (
            source_dtype != target_dtype and not is_fused_fp8(source_dtype, target_dtype)
        ):
            mismatched.append(name)
    if missing or extra or mismatched:
        raise ValueError(
            f"Incompatible captured weights: missing={missing}, extra={extra}, "
            f"shape/dtype mismatch={sorted(mismatched)}"
        )
