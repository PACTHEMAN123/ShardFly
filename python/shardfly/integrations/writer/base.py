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


import logging
import os
import threading
import time

import torch
import torch.distributed as dist

from shardfly._utils.common import compute_statistics
from shardfly.control.store import MetaServerClient
from shardfly.metadata.capture import (
    build_captured_parameters_meta,
    validate_captured_weights,
)
from shardfly.metadata.rank import RankInfo

logger = logging.getLogger(__name__)
_FLOAT_DTYPES = {torch.float16, torch.bfloat16, torch.float32}


class WeightsExchangeShardingWriter:
    def __init__(self, capture, config):
        self.capture = capture
        self.meta_server_addr = config["meta_server_addr"]
        self.meta_server_client = MetaServerClient(*self.meta_server_addr.split(":"))
        self.comm_backend = config.get("comm_backend", "transport")
        self.enable_debug_mode = config.get("enable_debug_mode", False)
        self.weights_validation_steps = config.get("weights_validation_steps", 0)
        self.validate_weights_every_n_steps = config.get(
            "validate_weights_every_n_steps", 1
        )
        self.dump_weights_list_for_validation = set(
            config.get("dump_weights_list_for_validation", ())
        )
        self.dump_weights_dir_for_validation = config.get(
            "dump_weights_dir_for_validation", os.getcwd()
        )
        self.timeout = 10000
        self.lock = threading.Lock()
        self.initialized = False
        self.already_initialized = False
        self.validated_steps = 0
        self.start_step = None
        self._history_write_weights_time = {}
        self._staging = {}

    def _initialize(self):
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        self.rank_info = RankInfo(
            tp_rank=0,
            tp_size=1,
            pp_rank=0,
            pp_size=1,
            dp_rank=rank,
            dp_size=world_size,
            ep_rank=0,
            ep_size=1,
            ep_tp_rank=0,
            ep_tp_size=1,
            attn_tp_rank=0,
            attn_tp_size=1,
            attn_dp_rank=rank,
            world_size=world_size,
            global_rank=rank,
            engine_rank=0,
            local_rank=int(os.environ.get("LOCAL_RANK", rank)),
            is_infer=False,
        )
        catalog = [None, None]
        if rank == 0:
            read = self.meta_server_client.get_object
            catalog = [
                read("infer_params_meta", timeout=self.timeout),
                read("num_infer_engines", timeout=self.timeout),
            ]
        dist.broadcast_object_list(catalog, src=0)
        self.infer_params_meta, self.num_infer_engines = catalog
        self.infer_instance_world_size = self.infer_params_meta[0].shards[0].world_size
        self.infer_world_size = self.infer_instance_world_size * self.num_infer_engines
        self.training_world_size = world_size
        self.transfer_world_size = self.infer_world_size + world_size
        self.transfer_rank = self.infer_world_size + rank
        self.target_dtypes = {
            parameter.name: parameter.dtype for parameter in self.infer_params_meta
        }

        error = None
        rows = []
        try:
            self._pending_snapshot = self.capture.snapshot(self.target_dtypes.keys())
            self._source_shards = self._pending_snapshot.shards
            for shard in self._source_shards:
                source = getattr(torch, shard["dtype"])
                target = self.target_dtypes[shard["name"]]
                if source != target and not {source, target} <= _FLOAT_DTYPES:
                    raise ValueError(
                        f"Unsupported capture dtype conversion for {shard['name']}: {source} to {target}"
                    )
                rows.append(dict(shard, dtype=str(target).removeprefix("torch.")))
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
        workers = [None] * world_size
        dist.all_gather_object(
            workers, {"rank_info": self.rank_info, "params_meta": rows, "error": error}
        )
        errors = [
            f"rank {i}: {worker['error']}"
            for i, worker in enumerate(workers)
            if worker["error"]
        ]
        if errors:
            raise ValueError("Training capture failed: " + "; ".join(errors))
        self.parameters_meta = build_captured_parameters_meta(workers)
        validate_captured_weights(self.parameters_meta, self.infer_params_meta)
        if rank == 0:
            self.meta_server_client.put_object(
                "training_params_meta", self.parameters_meta
            )

    def capture_snapshot(self):
        snapshot = self._pending_snapshot
        self._pending_snapshot = None
        if snapshot is None:
            snapshot = self.capture.snapshot(self.target_dtypes.keys())
        if snapshot.shards != self._source_shards:
            raise ValueError(
                "Training shard regions changed after planning; initialize a new publication"
            )
        return snapshot

    def bind_parameters(self, snapshot, required, device, previous):
        parameters = {}
        for key, name in required.items():
            source = snapshot.tensors[key]
            dtype = self.target_dtypes[name]
            if (
                key in snapshot.materialized
                or source.device != device
                or source.dtype != dtype
            ):
                if key not in self._staging:
                    self._staging[key] = torch.empty_like(
                        source,
                        device=device,
                        dtype=dtype,
                        memory_format=torch.contiguous_format,
                    )
                target = self._staging[key]
                target.copy_(source)
            else:
                target = source
            if key in previous:
                old = previous[key]
                if (
                    old.data_ptr() != target.data_ptr()
                    or old.stride() != target.stride()
                    or old.device != target.device
                    or old.dtype != target.dtype
                ):
                    raise ValueError(
                        f"Training storage changed for {name}; use capture_training(..., stage=True)"
                    )
                target = old
            parameters[key] = target
        return previous if parameters.keys() == previous.keys() else parameters

    @torch.no_grad()
    def write_weights(self, step_id, **kwargs):
        with self.lock:
            if not self.initialized:
                self._initialize()
                self.initialized = True
            start = time.perf_counter()
            self._write_weights(step_id, **kwargs)
            compute_statistics(
                self._history_write_weights_time,
                step_id,
                time.perf_counter() - start,
                "Write weights",
            )

    def _dump_weights(self, step_id, snapshot):
        if self.start_step is None:
            self.start_step = step_id
        if (
            self.validated_steps >= self.weights_validation_steps
            or (step_id - self.start_step) % self.validate_weights_every_n_steps
        ):
            return
        self.validated_steps += 1
        for key, tensor in snapshot.tensors.items():
            name = key if isinstance(key, str) else key[0]
            if name in self.dump_weights_list_for_validation:
                suffix = name if isinstance(key, str) else f"{name}.{key[1]}"
                path = os.path.join(
                    self.dump_weights_dir_for_validation,
                    f"writer_{os.getpid()}_{step_id}.{suffix}.pt",
                )
                torch.save(tensor.detach().cpu(), path)
