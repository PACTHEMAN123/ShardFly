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


"""Publish captured weights from training ranks."""

import os
import threading
import time
from contextlib import nullcontext
from dataclasses import replace

import torch
import torch.distributed as dist

from shardfly import logging
from shardfly._utils import device as device_util
from shardfly._utils.common import compute_statistics, is_fused_fp8
from shardfly._utils.process_group import (
    init_weights_update_group,
    setup_batch_isend_irecv,
)
from shardfly.capture import CapturePlan
from shardfly.control.store import MetaServerClient
from shardfly.layout import StaticTensorLayout
from shardfly.metadata.capture import (
    build_captured_parameters_meta,
    validate_captured_weights,
)
from shardfly.metadata.parameter import RankInfo
from shardfly.plan import (
    TransferPlanBuilder,
    compute_transfer_plan_hash,
    compute_transfer_plan_stats,
)
from shardfly.profiling.telemetry import emit_profile, profile_phase
from shardfly.transport import Transport

logger = logging.getLogger(__name__)

_FLOAT_DTYPES = {torch.float16, torch.bfloat16, torch.float32}


def _binding_signature(binding):
    spans = binding.spans if isinstance(binding, StaticTensorLayout) else (binding,)
    return tuple((span.data_ptr(), span.stride(), span.device, span.dtype) for span in spans)


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
        if hasattr(self.capture, "rank_coordinates"):
            self.rank_info = replace(self.rank_info, **self.capture.rank_coordinates())
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
            publication = getattr(self.capture, "publication", nullcontext)
            with publication():
                self._pending_snapshot = self.capture.snapshot(self.target_dtypes.keys())
            self._source_shards = self._pending_snapshot.shards
            if getattr(self.capture, "native", False) and not self.capture.static:
                # Bind again after the framework has opened the send's borrow.
                self._pending_snapshot = None
            for shard in self._source_shards:
                source = getattr(torch, shard["dtype"])
                target = self.target_dtypes[shard["name"]]
                if is_fused_fp8(source, target):
                    rows.append(shard)
                    continue
                if source != target and (
                    getattr(self.capture, "native", False) or not {source, target} <= _FLOAT_DTYPES
                ):
                    raise ValueError(
                        f"Unsupported capture dtype conversion for {shard['name']}: {source} to {target}"
                    )
                rows.append(dict(shard, dtype=str(target).removeprefix("torch.")))
        except Exception as exc:
            logger.exception("Local training capture failed on rank %s", rank)
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
        unchanged = True
        native = getattr(self.capture, "native", False)
        for key, name in required.items():
            source = snapshot.tensors[key]
            first = source.spans[0] if isinstance(source, StaticTensorLayout) else source
            dtype = self.target_dtypes[name]
            if is_fused_fp8(first.dtype, dtype):
                dtype = first.dtype
            if (
                key in snapshot.materialized
                or first.device != device
                or first.dtype != dtype
            ):
                if native:
                    raise ValueError(f"Checkpoint binding for {name} would require staging")
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
                if _binding_signature(old) == _binding_signature(target):
                    target = old
                elif native:
                    unchanged = False
                else:
                    raise ValueError(
                        f"Training storage changed for {name}; use capture_training(..., stage=True)"
                    )
            parameters[key] = target
        return previous if unchanged and parameters.keys() == previous.keys() else parameters

    @torch.no_grad()
    def write_weights(self, step_id, **kwargs):
        publication = getattr(self.capture, "publication", nullcontext)
        with self.lock:
            if not self.initialized:
                self._initialize()
                self.initialized = True
            start = time.perf_counter()
            with publication():
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
                if isinstance(tensor, StaticTensorLayout):
                    tensor = torch.cat([span.flatten() for span in tensor.spans]).view(tensor.shape)
                torch.save(tensor.detach().cpu(), path)


class TransportWriter(WeightsExchangeShardingWriter):
    def prepare_membership(self, specification: dict) -> dict:
        from shardfly.integrations.membership import prepare_model_membership

        with self.lock:
            if not self.initialized:
                raise RuntimeError("Publish the initial model snapshot before joining")
            return prepare_model_membership(self, specification, sender=True)

    def _initialize(self):
        super()._initialize()
        self.device_transport = None
        logger.info(
            f"Start to initialize NCCL weights writer for rank {self.transfer_rank}"
        )
        logger.info(f"Start to build transfer plan for rank {self.transfer_rank}")
        self.transfer_plan = TransferPlanBuilder(
            self.infer_world_size,
            self.training_world_size,
            self.num_infer_engines,
            self.enable_debug_mode,
        ).build_local_transfer_plan(
            self.infer_params_meta, self.parameters_meta, self.transfer_rank
        )
        inter_hash = compute_transfer_plan_hash(self.transfer_plan)
        logger.info(
            "Writer rank %s inter plan hash: %s", self.transfer_rank, inter_hash
        )
        logger.info(
            "Writer rank %s transfer plan stats: %s",
            self.transfer_rank,
            compute_transfer_plan_stats(self.transfer_plan),
        )
        self.recv_ranks = list(self.transfer_plan.operations.keys())
        self.required_parameters = self.required_shards(self.transfer_plan)
        self.device_parameters = {}
        self._prepared_parameters = None
        self._bound_snapshot = None
        logger.info(
            f"Writer rank {self.transfer_rank}: Built transfer plan to send to ranks: {self.recv_ranks}"
        )
        logger.info(
            "Writer rank %s: required captured shards for this plan: %s",
            self.transfer_rank,
            len(self.required_parameters),
        )
        logger.info(
            f"Writer rank {self.transfer_rank}: Operations per rank: {[(rank, len(ops)) for (rank, ops) in self.transfer_plan.operations.items()]}"
        )
        self.recv_ranks_sample = (
            self.recv_ranks[:8] + ["..."] + self.recv_ranks[-8:]
            if len(self.recv_ranks) > 16
            else self.recv_ranks
        )
        self.num_to_sends = sum(
            (len(operations) for operations in self.transfer_plan.operations.values())
        )
        logger.info(f"Finished building transfer plan for rank {self.transfer_rank}")
        logger.info(
            f"Start to get master info from meta server for rank {self.transfer_rank}"
        )
        master_info = self.meta_server_client.get_object(
            "master_info", timeout=self.timeout
        )
        (self.master_address, self.master_port) = master_info
        logger.info(
            f"Get master info from meta server for rank {self.transfer_rank}: {master_info}"
        )
        self._set_device()
        self._init_weights_exchange_process_group()
        self._shake_hands_with_reader()
        self.device_transport = Transport(
            self.weights_update_group,
            self.transfer_rank,
            self.transfer_world_size,
            infer_instance_world_size=self.infer_instance_world_size,
            num_infer_engines=self.num_infer_engines,
        )
        self.device_transport.resolve_rollout_topology()
        logger.info(
            f"Finished initializing NCCL weights writer for rank {self.transfer_rank}"
        )

    @staticmethod
    def required_shards(plan):
        return {
            op.send_shard_meta.tensor_key: op.send_shard_meta.name
            for operations in plan.operations.values()
            for op in operations
        }

    def bind_source(self, snapshot, required):
        device = torch.device("cuda", device_util.current_device())
        return self.bind_parameters(snapshot, required, device, self.device_parameters)

    def close(self) -> None:
        transport = getattr(self, "device_transport", None)
        if transport is not None:
            transport.close()
            self.device_transport = None

    def _shake_hands_with_reader(self):
        if self.transfer_rank == self.transfer_world_size - 1:
            logger.info(
                f"Start to test NCCL ready for rank {self.transfer_rank}, world size {self.transfer_world_size}"
            )
            dist.send(
                torch.tensor(1, device=device_util.get_torch_device()),
                dst=0,
                group=self.weights_update_group,
            )
            logger.info(
                f"NCCL ready: send tensor to rank {self.transfer_world_size - 1} from rank {self.transfer_rank}"
            )
        setup_batch_isend_irecv(
            self.weights_update_group, self.transfer_rank, self.transfer_world_size
        )

    def _set_device(self):
        device = device_util.current_device()
        count = device_util.device_count() or 1
        gpu_id = int(os.environ.get("DEVICE", device)) % count
        logger.info(
            f"[TransportWriter] Set device to {gpu_id} for rank {self.transfer_rank}, device env is {os.environ.get('DEVICE')}, previous device is {device}, device_count is {count}, {'/'.join(device_util.visible_devices_env_names())} env is {device_util.visible_devices_env_value() or '(unset)'}"
        )
        device_util.set_device(gpu_id)

    def _init_weights_exchange_process_group(self):
        if self.already_initialized:
            return
        self.weights_update_group = init_weights_update_group(
            master_address=self.master_address,
            master_port=self.master_port,
            rank=self.transfer_rank,
            world_size=self.transfer_world_size,
            group_name="weights_exchange",
            backend="nccl",
            role="train",
        )
        logger.info(f"Initialized NCCL weights writer for rank {self.transfer_rank}")
        dist.barrier(
            group=self.weights_update_group, device_ids=[device_util.current_device()]
        )
        logger.info(f"Barrier passed for weights writer with rank {self.transfer_rank}")
        self.already_initialized = True

    @torch.no_grad()
    def _write_weights(self, step_id, **kwargs):
        self._init_weights_exchange_process_group()
        start = time.perf_counter()
        parameters = self._prepared_parameters
        if parameters is None or not getattr(self.capture, "static", False):
            snapshot = self.capture_snapshot()
            self._dump_weights(step_id, snapshot)
            if snapshot is not self._bound_snapshot:
                parameters = self.bind_source(snapshot, self.required_parameters)
                if parameters is not self._prepared_parameters:
                    self.device_transport.prepare_send(parameters, self.transfer_plan, allow_staging=False)
                    self._prepared_parameters = parameters
                self.device_parameters = parameters
                if getattr(self.capture, "native", False):
                    self._bound_snapshot = snapshot
        bind_time_ms = (time.perf_counter() - start) * 1000
        if os.environ.get("SHARDFLY_PROFILE_SYNC_START", "0") == "1":
            dist.barrier(
                group=self.weights_update_group,
                device_ids=[device_util.current_device()],
            )
        backend_start = time.perf_counter()
        metrics = self.device_transport.send(parameters, self.transfer_plan, step_id)
        device_util.synchronize(device_id=device_util.current_device())
        metrics["backend_execute_time_ms"] = (
            time.perf_counter() - backend_start
        ) * 1000
        barrier_start = time.perf_counter()
        dist.barrier(
            group=self.weights_update_group, device_ids=[device_util.current_device()]
        )
        emit_profile(
            logger,
            event="weight_transfer",
            role="writer",
            backend=self.comm_backend,
            phase=profile_phase(step_id),
            step_id=int(step_id),
            rank=int(self.transfer_rank),
            capture_bind_time_ms=bind_time_ms,
            completion_barrier_time_ms=(time.perf_counter() - barrier_start) * 1000,
            total_transfer_time_ms=(time.perf_counter() - start) * 1000,
            **metrics,
        )


class CapturedTrainingEngine:
    def __init__(self, capture: CapturePlan, *, meta_server_addr: str, **config):
        self.global_step = -1
        self.weights_exchange_writer = TransportWriter(
            capture, dict(config, meta_server_addr=meta_server_addr)
        )

    def set_global_step(self, global_step: int):
        self.global_step = global_step

    def write_weights(self, **kwargs):
        if self.global_step < 0:
            raise ValueError("Set the training step before publishing")
        self.weights_exchange_writer.write_weights(self.global_step, **kwargs)

    def close(self):
        self.weights_exchange_writer.close()
