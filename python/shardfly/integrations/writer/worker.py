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


import os
import time

import torch
import torch.distributed as dist

from shardfly import logging
from shardfly._utils import device as device_util
from shardfly._utils.process_group import (
    init_weights_update_group,
    setup_batch_isend_irecv,
)
from shardfly._utils.profile import emit_profile, profile_phase
from shardfly.integrations.writer.base import WeightsExchangeShardingWriter
from shardfly.plan import (
    TransferPlanBuilder,
    compute_transfer_plan_hash,
    compute_transfer_plan_stats,
)
from shardfly.transport import Transport

logger = logging.getLogger(__name__)


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
            strict_param_key_match=True,
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
        transport_type = Transport
        self.device_transport = transport_type(
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
        snapshot = self.capture_snapshot()
        self._dump_weights(step_id, snapshot)
        parameters = self.bind_source(snapshot, self.required_parameters)
        if parameters is not self._prepared_parameters:
            self.device_transport.prepare_send(
                parameters, self.transfer_plan, allow_staging=False
            )
            self._prepared_parameters = parameters
        self.device_parameters = parameters
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
