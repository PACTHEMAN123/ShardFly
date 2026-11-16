"""Assemble checkpoint-coordinate shards observed on inference workers."""

from collections import defaultdict
from math import prod

import torch

from shardfly.metadata.parameter import (
    ParameterMeta,
    ParameterReplicaMeta,
    ParameterShardMeta,
)
from shardfly.metadata.sharding import ShardingType


def build_captured_parameters_meta(workers: list[dict]) -> list[ParameterMeta]:
    by_name = defaultdict(list)
    ranks = sorted(worker["rank_info"].global_rank for worker in workers)
    if any(worker["rank_info"].world_size != len(ranks) for worker in workers) or (
        ranks != list(range(len(ranks)))
    ):
        raise ValueError("Loader capture requires metadata from every inference rank")
    for worker in workers:
        for param in worker["params_meta"]:
            by_name[param["name"]].append((worker["rank_info"], param))

    parameters = []
    for name, entries in sorted(by_name.items()):
        global_shape = tuple(entries[0][1]["global_shape"])
        dtype = getattr(torch, entries[0][1]["dtype"])
        geometry = defaultdict(list)
        split_axes = set()
        seen_ranks = set()
        for rank, param in entries:
            if (
                tuple(param["global_shape"]) != global_shape
                or getattr(torch, param["dtype"]) != dtype
                or rank.global_rank in seen_ranks
            ):
                raise ValueError(f"Inconsistent loader metadata for {name}")
            seen_ranks.add(rank.global_rank)
            shape = tuple(param["shape"])
            offset = tuple(param["global_offset"])
            if not len(shape) == len(offset) == len(global_shape) or any(
                start < 0 or size <= 0 or start + size > total
                for start, size, total in zip(offset, shape, global_shape)
            ):
                raise ValueError(f"Invalid checkpoint region for {name}")
            split_axes.update(
                axis
                for axis, (size, total) in enumerate(zip(shape, global_shape))
                if size != total
            )
            geometry[offset, shape].append(rank)
        if len(split_axes) > 1:
            raise ValueError(
                f"Loader capture requires a single sharding axis for {name}"
            )
        axis = next(iter(split_axes), 0)
        replica_count = len(next(iter(geometry.values())))
        replicas = [ParameterReplicaMeta([]) for _ in range(replica_count)]
        cursor = 0
        for (offset, shape), shard_ranks in sorted(geometry.items()):
            if offset[axis] != cursor or len(shard_ranks) != replica_count:
                raise ValueError(
                    f"Missing, overlapping, or unevenly replicated regions for {name}"
                )
            cursor += shape[axis]
            for replica, rank in zip(
                replicas, sorted(shard_ranks, key=lambda rank: rank.global_rank)
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
                            if split_axes
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
                    )
                )
        if cursor != global_shape[axis]:
            raise ValueError(f"Incomplete checkpoint coverage for {name}")
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
