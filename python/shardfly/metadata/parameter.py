# Copyright 2026 ShardFly contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Portions derived from the Awex project (https://github.com/inclusionAI/Awex),
# Copyright the Awex developers, licensed under the Apache License, Version 2.0.

from dataclasses import dataclass, field
from typing import List, Tuple

import torch

from shardfly import logging
from shardfly._utils.common import to_dict
from shardfly.capture import TensorKey
from shardfly.metadata.sharding import (
    ShardingType,
)

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class ParameterShardMeta:
    """
    Represents a single shard of a model parameter.
    Attributes:
        tp_rank (int): Tensor parallel rank of the shard.
        attn_tp_rank (int): Attention tensor parallel rank of the shard.
        pp_rank (int): Pipeline parallel rank of the shard.
        world_size (int): World size of the training world.
        engine_rank (int): Engine rank of the engine that the shard belongs to.
        shape (Tuple[int, ...]): Shape of the shard tensor.
        numel (int): Number of elements in the shard tensor.
        dtype (torch.dtype): Data type of the shard tensor.
        global_offset (Tuple[int, ...]): Offset of the shard in the global parameter tensor.
        sharding_type (ShardingType): The sharding type for this shard.
        num_shards (int): The number of shards for this parameter.
        sharding_dim (int): The dimension of the sharding.
    """

    tp_rank: int
    attn_tp_rank: int
    pp_rank: int
    ep_rank: int
    ep_tp_rank: int
    global_rank: int
    world_size: int
    engine_rank: int
    name: str
    shape: Tuple[int, ...]
    numel: int
    dtype: torch.dtype
    global_offset: Tuple[int, ...] = field(default_factory=tuple)
    sharding_type: ShardingType = ShardingType.NO_SHARDING
    num_shards: int = 1
    sharding_dim: int = 0
    cp_rank: int = 0
    cp_size: int = 1
    cp_mode: str = "none"
    dp_rank: int = 0
    dp_size: int = 1
    binding_key: TensorKey | None = None

    @property
    def tensor_key(self) -> TensorKey:
        return self.binding_key if self.binding_key is not None else self.name


@dataclass(slots=True)
class ParameterReplicaMeta:
    """
    Represents a replica of a parameter, which is a list of shards that together form a full logical copy.

    Attributes:
        shards (List[ParameterShardMeta]): List of shards in this replica.
    """

    shards: List[ParameterShardMeta]


@dataclass(slots=True)
class ParameterMeta:
    """
    Represents the metadata for a model parameter, including all its shards and replicas.

    Attributes:
        name (str): Name of the parameter.
        global_numel (int): Total number of elements in the global parameter.
        global_shape (Tuple[int, ...]): Shape of the global parameter.
        dtype (torch.dtype): Data type of the parameter.
        shards (List[ParameterShardMeta]): All shards for this parameter across all ranks.
        replicas (List[ParameterReplicaMeta]): List of logical replicas, each a list of shards.
    """

    name: str
    global_numel: int
    global_shape: Tuple[int, ...]
    dtype: torch.dtype
    shards: List[ParameterShardMeta]
    replicas: List[ParameterReplicaMeta]

    def fast_copy_with_engine_rank(self, engine_rank: int) -> "ParameterMeta":
        shards = []
        replicas = []
        for replica in self.replicas:
            replica_shards = []
            for shard in replica.shards:
                new_shard = ParameterShardMeta(
                    tp_rank=shard.tp_rank,
                    attn_tp_rank=shard.attn_tp_rank,
                    pp_rank=shard.pp_rank,
                    ep_rank=shard.ep_rank,
                    ep_tp_rank=shard.ep_tp_rank,
                    global_rank=shard.global_rank,
                    world_size=shard.world_size,
                    engine_rank=engine_rank,
                    cp_rank=shard.cp_rank,
                    cp_size=shard.cp_size,
                    cp_mode=shard.cp_mode,
                    dp_rank=shard.dp_rank,
                    dp_size=shard.dp_size,
                    name=shard.name,
                    shape=shard.shape,
                    numel=shard.numel,
                    dtype=shard.dtype,
                    global_offset=shard.global_offset,
                    sharding_type=shard.sharding_type,
                    num_shards=shard.num_shards,
                    sharding_dim=shard.sharding_dim,
                    binding_key=shard.binding_key,
                )
                replica_shards.append(new_shard)
            shards.extend(replica_shards)
            replicas.append(ParameterReplicaMeta(shards=replica_shards))
        return ParameterMeta(
            name=self.name,
            global_numel=self.global_numel,
            global_shape=self.global_shape,
            dtype=self.dtype,
            shards=shards,
            replicas=replicas,
        )

    def to_local_parameter_meta(self, global_rank: int) -> "ParameterMeta":
        engine_rank = self.shards[0].engine_rank
        new_meta = self.fast_copy_with_engine_rank(engine_rank)
        shards = [
            shard for shard in new_meta.shards if shard.global_rank == global_rank
        ]
        new_meta.shards = shards

        for replica in new_meta.replicas:
            replica.shards = [
                shard for shard in replica.shards if shard.global_rank == global_rank
            ]
        new_meta.replicas = [replica for replica in new_meta.replicas if replica.shards]

        return new_meta


def dump_parameters_meta(params_meta: List[ParameterMeta]):
    data = []
    for p in params_meta:
        data.append(
            to_dict(
                {
                    "name": p.name,
                    "global_numel": p.global_numel,
                    "global_shape": p.global_shape,
                    "dtype": p.dtype,
                    "num_shards": len(p.shards),
                    "num_replicas": len(p.replicas),
                    "replicas": p.replicas,
                }
            )
        )
    return data


def compute_total_model_size(params_meta: List[ParameterMeta]) -> int:
    """
    Compute the total size of the model in bytes.
    """
    return sum(param.global_numel * param.dtype.itemsize for param in params_meta)
