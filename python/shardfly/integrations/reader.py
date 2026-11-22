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


"""Collect rollout captures, initialize the reader, and receive publication updates."""

import os
import pickle
import threading
import time
from typing import Any, Dict, List, Optional

import torch
import torch.distributed as dist

from shardfly import logging
from shardfly._utils import device as device_util
from shardfly._utils.common import (
    compute_statistics,
    get_free_port,
    get_ip_address,
    stripped_env_vars,
)
from shardfly._utils.tensor import (
    check_and_log_nan_values,
    compare_and_log_tensor_differences,
)
from shardfly.capture import capture_inference
from shardfly.control.store import MetaServerClient
from shardfly.metadata.capture import (
    build_captured_parameters_meta,
    validate_captured_weights,
)
from shardfly.metadata.parameter import ParameterMeta, RankInfo
from shardfly.plan import (
    TransferPlanBuilder,
    compute_transfer_plan_hash,
    compute_transfer_plan_stats,
)
from shardfly.profiling.telemetry import emit_profile, profile_phase

logger = logging.getLogger(__name__)


def get_vllm_rank_info(model_context, engine_rank) -> RankInfo:
    tp_size = model_context["tp_size"]
    tp_rank = model_context["tp_rank"]
    dp_size = model_context["dp_size"]
    dp_rank = model_context["dp_rank"]
    ep_size = model_context.get("ep_size", 1)
    ep_rank = model_context.get("ep_rank", 0)
    # vLLM expert parallel groups are initialized independently of TP.
    # Use explicit ep_tp_size/ep_tp_rank when provided; otherwise treat EP as a
    # standalone dimension (ep_tp_size=1) to avoid tp_size < ep_size issues.
    ep_tp_size = model_context.get("ep_tp_size", 1)
    ep_tp_rank = model_context.get("ep_tp_rank", 0)
    if ep_size <= 1:
        ep_rank = 0
        ep_tp_size = 1
        ep_tp_rank = 0
    cp_size = int(model_context.get("cp_size", 1) or 1)
    cp_rank = int(model_context.get("cp_rank", 0) or 0)
    cp_mode = model_context.get("cp_mode")
    if not cp_mode:
        cp_mode = "ring" if cp_size > 1 else "none"
    return RankInfo(
        tp_rank=tp_rank,
        tp_size=tp_size,
        pp_rank=model_context["pp_rank"],
        pp_size=model_context["pp_size"],
        dp_size=dp_size,
        dp_rank=dp_rank,
        ep_rank=ep_rank,
        ep_size=ep_size,
        ep_tp_rank=ep_tp_rank,
        ep_tp_size=ep_tp_size,
        attn_tp_rank=model_context["attn_tp_rank"],
        attn_tp_size=model_context["attn_tp_size"],
        attn_dp_rank=model_context["attn_dp_rank"],
        world_size=model_context["world_size"],
        global_rank=model_context["global_rank"],
        engine_rank=engine_rank,
        local_rank=model_context["local_rank"],
        is_infer=True,
        cp_rank=cp_rank,
        cp_size=cp_size,
        cp_mode=cp_mode,
    )


class InferParamMetaResolver:
    def __init__(self, inference_engine, engine_rank=0):
        workers = inference_engine.execute_task_in_model_worker(
            self._get_model_param_info, engine_rank=engine_rank
        )
        for worker in workers:
            if isinstance(worker["rank_info"], dict):
                worker["rank_info"] = RankInfo(**worker["rank_info"])
        self._params_meta = build_captured_parameters_meta(workers)

    def get_parameters_meta(self):
        return self._params_meta

    @staticmethod
    def _get_model_param_info(engine_rank=0, **kwargs):
        model = kwargs["model"]
        return {
            "rank_info": get_vllm_rank_info(kwargs["model_context"], engine_rank),
            "params_meta": capture_inference(model).snapshot().shards,
        }


class WeightsReader:
    parameters_meta: List[ParameterMeta]

    def __init__(self, inference_engine, meta_resolver: InferParamMetaResolver = None):
        self.inference_engine = inference_engine
        self.infer_engine_config = inference_engine.config
        self.weights_comm_nccl_group_size = self.infer_engine_config.weights_comm_nccl_group_size
        if meta_resolver is None:
            meta_resolver = InferParamMetaResolver(
                inference_engine,
                engine_rank=inference_engine.engine_rank,
            )
        self.meta_resolver = meta_resolver
        self.parameters_meta = []
        config = inference_engine.config
        self.meta_server_addr = config.meta_server_addr
        logger.info(f"Meta server address: {self.meta_server_addr}")
        self.meta_server_client = MetaServerClient(*self.meta_server_addr.split(":"))
        self.num_engines = config.num_engines
        logger.info("Put number of inference engines to meta server")
        self.engine_rank = self.inference_engine.engine_rank
        self.infer_instance_world_size = meta_resolver.get_parameters_meta()[0].shards[0].world_size
        self.infer_world_size = self.num_engines * self.infer_instance_world_size
        self.validated_steps = 0
        self.start_step = -1
        self.weights_validation_steps = config.weights_validation_steps
        self.validate_weights_every_n_steps = config.validate_weights_every_n_steps
        self.dump_weights_list_for_validation = config.dump_weights_list_for_validation
        self.dump_weights_dir_for_validation = (
            config.dump_weights_dir_for_validation or os.getcwd()
        )
        self.debug_mode_config = config.debug_mode_config or {}
        self.raise_on_validation_fail = self.debug_mode_config.get(
            "raise_on_validation_fail", False
        )
        self.timeout = 10000
        self.lock = threading.Lock()
        self.initialized = False

    def initialize(self, **kwargs):
        self.meta_server_client.put_object("num_infer_engines", self.num_engines)
        self.meta_server_client.add_object_to_set(
            "num_inited_inference_engines", self.engine_rank
        )

    def _initialize(self):
        logger.info(
            f"Initializing weights exchange reader for engine rank {self.engine_rank}"
        )
        self.parameters_meta = self.meta_resolver.get_parameters_meta()
        logger.info(
            "Finished querying and building parameters meta from all tp workers"
        )
        if self.engine_rank == 0:
            self.meta_server_client.put_object(
                "infer_params_meta", self.parameters_meta
            )
            logger.info("Put inference parameters meta to meta server")
        else:
            reference = self.meta_server_client.get_object(
                "infer_params_meta", timeout=self.timeout
            )
            if self.parameters_meta != [
                parameter.fast_copy_with_engine_rank(self.engine_rank)
                for parameter in reference
            ]:
                raise ValueError("Captured inference shard layouts differ across engines")
        logger.info(
            f"Start to get training parameters meta from meta server for engine rank {self.engine_rank}"
        )
        self.training_params_meta = self.meta_server_client.get_object(
            "training_params_meta", timeout=self.timeout
        )
        logger.info("Finished getting training parameters meta from meta server")
        self.training_world_size = self.training_params_meta[0].shards[0].world_size
        config = self.inference_engine.config
        validate_captured_weights(self.training_params_meta, self.parameters_meta)
        logger.info("Start to send parameters meta to tp workers")
        infer_parameters_meta_bytes = pickle.dumps(self.parameters_meta)
        train_parameters_meta_bytes = pickle.dumps(self.training_params_meta)
        self._preparation_results = self.inference_engine.execute_task_in_model_worker(
            self._init_in_tp_worker,
            parameters_meta_bytes=infer_parameters_meta_bytes,
            training_params_meta_bytes=train_parameters_meta_bytes,
            engine_rank=self.engine_rank,
            num_engines=self.num_engines,
            meta_server_addr=self.meta_server_addr,
            weights_comm_backend=config.comm_backend,
            enable_debug_mode=config.enable_debug_mode,
            debug_mode_config=config.debug_mode_config,
            disable_pipeline=config.disable_weights_exchange_pipeline,
            weights_comm_nccl_group_size=self.weights_comm_nccl_group_size,
            membership_specification=getattr(self, "_pending_membership", None),
        )
        logger.info(
            f"Finished full initialization of weights reader for engine rank {self.engine_rank}"
        )

    @staticmethod
    def _init_in_tp_worker(
        parameters_meta_bytes: bytes,
        training_params_meta_bytes: bytes,
        engine_rank: int,
        num_engines: int,
        meta_server_addr: str,
        weights_comm_backend: str,
        debug_mode_config: Dict[str, Any],
        disable_pipeline: bool,
        **kwargs,
    ):
        """Cache meta to avoid send to to worker everytime when update weights"""
        model = kwargs["model"]
        model_context = kwargs["model_context"]
        scheduler = model_context["scheduler"]
        parameters_meta = pickle.loads(parameters_meta_bytes)
        training_params_meta = pickle.loads(training_params_meta_bytes)
        infer_engine_config = model_context.get("infer_engine_config")
        if infer_engine_config is not None:
            if isinstance(infer_engine_config, dict):
                infer_engine_config["comm_backend"] = weights_comm_backend
            else:
                infer_engine_config.comm_backend = weights_comm_backend
        scheduler.shardfly_weights_reader = TransportWorkerReader(
            model=model,
            model_context=model_context,
            engine_rank=engine_rank,
            num_engines=num_engines,
            meta_server_addr=meta_server_addr,
            parameters_meta=parameters_meta,
            training_params_meta=training_params_meta,
            enable_debug_mode=kwargs.get("enable_debug_mode", False),
            debug_mode_config=debug_mode_config,
            disable_pipeline=disable_pipeline,
            weights_comm_nccl_group_size=kwargs.get("weights_comm_nccl_group_size"),
        )
        specification = kwargs.get("membership_specification")
        if specification is None:
            scheduler.shardfly_weights_reader.initialize()
        else:
            from shardfly.integrations.membership import prepare_model_membership

            WorkerWeightsReader.initialize(scheduler.shardfly_weights_reader)
            scheduler.shardfly_weights_reader._set_device()
            return prepare_model_membership(
                scheduler.shardfly_weights_reader, specification, sender=False
            )

    def prepare_membership(self, specification: dict):
        with self.lock:
            if getattr(self, "_retired", False):
                raise RuntimeError("Restart a retired reader before rejoining")
            engines = specification["num_engines"]
            if (
                not 1 <= engines
                or self.engine_rank >= engines
                or (self.initialized and engines == self.num_engines)
            ):
                raise ValueError("Model readers require a changed surviving cohort")
            self.num_engines = engines
            self.infer_engine_config.num_engines = engines
            self.infer_world_size = engines * self.infer_instance_world_size
            if not self.initialized:
                self._pending_membership = specification
                self._initialize()
                self.initialized = True
                return self._preparation_results
            return self.inference_engine.execute_task_in_model_worker(
                self._prepare_membership_in_tp_worker, specification=specification
            )

    def retire_membership(self, specification: dict):
        with self.lock:
            if not self.initialized or self.engine_rank < specification["num_engines"]:
                raise ValueError("Only initialized leaving readers can retire")
            self._retired = True
            results = self.inference_engine.execute_task_in_model_worker(
                self._retire_membership_in_tp_worker, specification=specification
            )
            return results

    @staticmethod
    def _retire_membership_in_tp_worker(specification: dict, **kwargs):
        from shardfly.integrations.membership import retire_model_membership

        worker = kwargs["model_context"]["scheduler"].shardfly_weights_reader
        return retire_model_membership(worker, specification)

    @staticmethod
    def _prepare_membership_in_tp_worker(specification: dict, **kwargs):
        from shardfly.integrations.membership import prepare_model_membership

        worker = kwargs["model_context"]["scheduler"].shardfly_weights_reader
        return prepare_model_membership(worker, specification, sender=False)

    def update_weights(self, step_id, **kwargs):
        with self.lock:
            if getattr(self, "_retired", False):
                raise RuntimeError("A retired reader cannot receive publications")
            if not self.initialized:
                logger.info(
                    f"Start to initialize weights exchange reader for engine rank {self.engine_rank}"
                )
                self._initialize()
                self.initialized = True
                logger.info(
                    f"Finished initializing weights exchange reader for engine rank {self.engine_rank}"
                )
            self._pre_validate_weights(step_id, **kwargs)
            start_time = time.time()
            logger.info(
                f"Start to update weights for step {step_id} for engine rank {self.engine_rank}"
            )
            self.inference_engine.execute_task_in_model_worker(
                self._update_parameters_in_tp_worker, step_id=step_id
            )
            duration = time.time() - start_time
            logger.info(
                f"Finished updating weights for step {step_id} for engine rank {self.engine_rank}, took {duration} seconds"
            )
            self._validate_weights(
                step_id,
                dump_weights_list_for_validation=self.dump_weights_list_for_validation,
                dump_weights_dir_for_validation=self.dump_weights_dir_for_validation,
                **kwargs,
            )

    def _pre_validate_weights(self, step_id, **kwargs):
        if self.validated_steps == 0:
            self.start_step = step_id
        if self.validated_steps >= self.weights_validation_steps:
            return
        if (step_id - self.start_step) % self.validate_weights_every_n_steps != 0:
            return
        self.inference_engine.execute_task_in_model_worker(
            self._pre_validate_weights_on_tp_worker, step_id=step_id
        )

    @classmethod
    def _pre_validate_weights_on_tp_worker(cls, step_id, **kwargs):
        model = kwargs["model"]
        scheduler = kwargs["model_context"]["scheduler"]
        logger.info(
            f"Start to copy parameters for step {step_id} for consistency check"
        )
        device_util.synchronize()
        scheduler._asystem_copied_parameters = {}
        for name, param in model.named_parameters():
            scheduler._asystem_copied_parameters[name] = (
                param.detach().to(device="cpu", copy=True).contiguous()
            )
        logger.info(
            f"Finished copying parameters for step {step_id} for consistency check"
        )
        with torch.inference_mode():
            for tensor in scheduler.shardfly_weights_reader.parameters.values():
                tensor.zero_()
        device_util.synchronize()

    def _validate_weights(
        self,
        step_id,
        dump_weights_list_for_validation: Optional[List[str]] = None,
        dump_weights_dir_for_validation: str = ".",
        **kwargs,
    ):
        if self.validated_steps == 0:
            self.start_step = step_id
        if self.validated_steps >= self.weights_validation_steps:
            return
        if (step_id - self.start_step) % self.validate_weights_every_n_steps != 0:
            return
        self.validated_steps += 1
        logger.info(f"Start to validate weights for step {step_id}")
        verify_results = self.inference_engine.execute_task_in_model_worker(
            self._verify_weights_on_tp_worker,
            step_id=step_id,
            dump_weights_list_for_validation=dump_weights_list_for_validation,
            dump_weights_dir_for_validation=dump_weights_dir_for_validation,
        )
        all_ok = True
        total_bad = 0
        for worker_rank, tp_results in enumerate(verify_results):
            if not all(tp_results.values()):
                not_consistent_weights = [
                    name for (name, result) in tp_results.items() if not result
                ]
                total_bad += len(not_consistent_weights)
                all_ok = False
                logger.error(
                    f"Weights for step {step_id} is not consistent for worker rank {worker_rank}: {not_consistent_weights}, total {len(tp_results)} weights"
                )
            else:
                logger.info(
                    f"Weights for step {step_id} is consistent for worker rank {worker_rank}, total {len(tp_results)} weights"
                )
        if all_ok:
            logger.info(
                f"[Validation] step {step_id} PASSED across {len(verify_results)} worker ranks"
            )
        else:
            logger.error(
                f"[Validation] step {step_id} FAILED: {total_bad} inconsistent weights"
            )
            if self.raise_on_validation_fail:
                raise RuntimeError(
                    f"ShardFly validation failed for step {step_id} with {total_bad} inconsistent weights"
                )

    @classmethod
    def _verify_weights_on_tp_worker(
        cls,
        step_id,
        dump_weights_list_for_validation=None,
        dump_weights_dir_for_validation=".",
        **kwargs,
    ):
        model = kwargs["model"]
        scheduler = kwargs["model_context"]["scheduler"]
        logger.info("Start to verify parameters")
        device_util.synchronize()
        results = {}
        dump_weights_list_for_validation = set(dump_weights_list_for_validation or [])
        for name, tensor_from_hg in scheduler._asystem_copied_parameters.items():
            if name in dump_weights_list_for_validation:
                abs_path = os.path.abspath(
                    os.path.join(
                        dump_weights_dir_for_validation,
                        f"reader_{os.getpid()}_from_hg_{step_id}.{name}.pt",
                    )
                )
                torch.save(tensor_from_hg, abs_path)
                logger.info(
                    f"[Reader] Saved parameter {name} loaded from hg to {abs_path}"
                )
        for name, tensor_from_commu in model.named_parameters():
            if name in dump_weights_list_for_validation:
                abs_path = os.path.abspath(
                    os.path.join(
                        dump_weights_dir_for_validation,
                        f"reader_{os.getpid()}_from_commu_{step_id}.{name}.pt",
                    )
                )
                torch.save(tensor_from_commu.detach().cpu().contiguous(), abs_path)
                logger.info(
                    f"[Reader] Saved parameter {name} loaded from communication to {abs_path}"
                )
        for name, param in model.named_parameters():
            copied_param = scheduler._asystem_copied_parameters.pop(name, None)
            if copied_param is None:
                logger.error(f"Parameter {name} not found in copied parameters")
                results[name] = False
                continue
            hg_param = copied_param
            param = param.detach().cpu()
            param_has_nan = check_and_log_nan_values(param, name)
            copied_param_has_nan = check_and_log_nan_values(
                hg_param, f"copied param {name}"
            )
            if param_has_nan or copied_param_has_nan:
                results[name] = False
                continue
            if not compare_and_log_tensor_differences(
                param, hg_param, name, exact_match=True
            ):
                results[name] = False
            else:
                logger.debug(f"Weights for {name} is consistent")
                results[name] = True
        scheduler._asystem_copied_parameters.clear()
        if all(results.values()):
            logger.info(f"Weights for step {step_id} is consistent")
        else:
            not_consistent_weights = [
                name for (name, result) in results.items() if not result
            ]
            logger.error(
                f"Weights for step {step_id} is not consistent: {not_consistent_weights}, total {len(results)} weights"
            )
        logger.info("Finished verifying parameters")
        return results

    @classmethod
    def _update_parameters_in_tp_worker(cls, **kwargs):
        model_context = kwargs["model_context"]
        scheduler = model_context["scheduler"]
        weights_reader = scheduler.shardfly_weights_reader
        weights_reader.update_weights(**kwargs)


class WorkerWeightsReader:
    def __init__(
        self,
        model,
        model_context,
        engine_rank,
        num_engines,
        meta_server_addr: str,
        parameters_meta: List[ParameterMeta],
        training_params_meta: List[ParameterMeta],
        enable_debug_mode: bool = False,
        debug_mode_config: Dict[str, Any] = None,
        disable_pipeline: bool = False,
        weights_comm_nccl_group_size: int = None,
    ):
        self.model = model
        self.model_context = model_context
        self.scheduler = model_context["scheduler"]
        self.infer_engine_config = model_context["infer_engine_config"]
        self.comm_backend = getattr(self.infer_engine_config, "comm_backend", "nccl")
        self.engine_rank = engine_rank
        self.num_engines = num_engines
        self.enable_debug_mode = enable_debug_mode
        self.debug_mode_config = debug_mode_config or {}
        self.raise_on_validation_fail = self.debug_mode_config.get(
            "raise_on_validation_fail", False
        )
        self.disable_pipeline = disable_pipeline
        self.weights_comm_nccl_group_size = weights_comm_nccl_group_size
        logger.info(f"Disable pipeline for weights reader: {self.disable_pipeline}")
        self.parameters_meta = parameters_meta
        self.training_params_meta = training_params_meta
        self.training_world_size = training_params_meta[0].shards[0].world_size
        self.infer_instance_world_size = parameters_meta[0].shards[0].world_size
        self.infer_world_size = num_engines * self.infer_instance_world_size
        self.transfer_world_size = self.training_world_size + self.infer_world_size
        self.rank_info = get_vllm_rank_info(model_context, engine_rank)
        logger.info(f"Reader rank info: {self.rank_info}")
        self.transfer_rank = (
            +self.engine_rank * self.infer_instance_world_size
            + self.rank_info.global_rank
        )
        self.meta_server_addr = meta_server_addr
        self.meta_server_client = MetaServerClient(*self.meta_server_addr.split(":"))
        self.capture = capture_inference(model)
        self.current_worker_parameters_meta = [
            p.to_local_parameter_meta(self.rank_info.global_rank)
            for p in self.parameters_meta
        ]
        self.total_local_num_elements = sum(
            (
                shard.numel
                for p in self.current_worker_parameters_meta
                for shard in p.shards
            )
        )
        self.total_local_param_size = sum(
            (
                shard.numel * shard.dtype.itemsize
                for p in self.current_worker_parameters_meta
                for shard in p.shards
            )
        )
        logger.info(
            f"[Reader {self.transfer_rank}] Total local number of elements: {self.total_local_num_elements}, total local parameter size: {self.total_local_param_size}"
        )
        self.timeout = 10000
        self._history_update_weights_time = {}
        self.already_initialized = False
        self.parameters = {}
        logger.info(f"Env varabbles for weights reader: {stripped_env_vars()}")
        logger.info(
            f"Created weights reader for rank {self.rank_info.global_rank}, engine rank {self.engine_rank}"
        )

    def initialize(self):
        self.parameters = self.capture.snapshot().tensors

    def update_weights(self, step_id, **kwargs):
        start_time = time.perf_counter()
        pre_update_sync_start = time.perf_counter()
        device_util.synchronize()
        pre_update_device_sync_time_ms = (
            time.perf_counter() - pre_update_sync_start
        ) * 1000.0
        update_body_start = time.perf_counter()
        self._update_weights(step_id, **kwargs)
        update_body_time_ms = (time.perf_counter() - update_body_start) * 1000.0
        logger.info(
            f"Start to flush cache for step {step_id} for rank {self.transfer_rank}"
        )
        flush_cache_start = time.perf_counter()
        flush_cache_call_start = time.perf_counter()
        flash_cache_success = self.scheduler.flush_cache()
        flush_cache_call_time_ms = (
            time.perf_counter() - flush_cache_call_start
        ) * 1000.0
        assert flash_cache_success, "Cache flush failed after updating weights"
        logger.info(
            f"Finished flushing cache for step {step_id} for rank {self.transfer_rank}"
        )
        post_flush_sync_start = time.perf_counter()
        device_util.synchronize()
        post_flush_device_sync_time_ms = (
            time.perf_counter() - post_flush_sync_start
        ) * 1000.0
        flush_cache_time_ms = (time.perf_counter() - flush_cache_start) * 1000.0
        duration = time.perf_counter() - start_time
        compute_statistics(
            self._history_update_weights_time, step_id, duration, "Update weights"
        )
        emit_profile(
            logger,
            event="reader_worker_update",
            role="reader_worker",
            backend=self.comm_backend,
            phase=profile_phase(step_id),
            step_id=int(step_id),
            rank=int(self.transfer_rank),
            pre_update_device_sync_time_ms=pre_update_device_sync_time_ms,
            update_body_time_ms=update_body_time_ms,
            flush_cache_call_time_ms=flush_cache_call_time_ms,
            post_flush_device_sync_time_ms=post_flush_device_sync_time_ms,
            flush_cache_time_ms=flush_cache_time_ms,
            worker_update_time_ms=duration * 1000.0,
        )

    def _update_weights(self, step_id, **kwargs):
        raise NotImplementedError



class TransportWorkerReader(WorkerWeightsReader):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.transfer_plan = None
        self.weights_update_group = None
        self.send_ranks = None
        self.send_ranks_sample = None
        self.num_to_recvs = None
        self.rank_coordinate = None
        self.device_transport = None

    def initialize(self):
        super().initialize()
        plan_builder = TransferPlanBuilder(
            self.infer_world_size,
            self.training_world_size,
            self.num_engines,
            self.enable_debug_mode,
        )
        self.transfer_plan = plan_builder.build_local_transfer_plan(
            self.parameters_meta, self.training_params_meta, self.transfer_rank
        )
        inter_hash = compute_transfer_plan_hash(self.transfer_plan)
        logger.info(
            "Reader rank %s inter plan hash: %s", self.transfer_rank, inter_hash
        )
        logger.info(
            "Reader rank %s transfer plan stats: %s",
            self.transfer_rank,
            compute_transfer_plan_stats(self.transfer_plan),
        )
        if self.transfer_rank == 0:
            master_address = get_ip_address()
            master_port = get_free_port()
            master_info = (master_address, master_port)
            self.meta_server_client.put_object("master_info", master_info)
            logger.info(
                f"Put master info to meta server for rank {self.transfer_rank}: {master_info}"
            )
        else:
            master_info = self.meta_server_client.get_object(
                "master_info", timeout=self.timeout
            )
            (master_address, master_port) = master_info
            logger.info(
                f"Get master info from meta server for rank {self.transfer_rank}: {master_info}"
            )
        logger.info(
            f"Start to initialize NCCL weights writer for rank {self.transfer_rank}"
        )
        self.master_address = master_address
        self.master_port = master_port
        self.world_size = self.transfer_world_size
        self._set_device()
        self._init_weights_exchange_process_group()
        self._shake_hands_with_writer()
        from shardfly.transport import Transport

        self.device_transport = Transport(
            self.weights_update_group,
            self.transfer_rank,
            self.world_size,
            infer_instance_world_size=self.infer_instance_world_size,
            num_infer_engines=self.num_engines,
        )
        self.device_transport.resolve_rollout_topology()
        self.device_transport.prepare_recv(self.parameters, self.transfer_plan)
        self.send_ranks = list(self.transfer_plan.operations.keys())
        self.send_ranks_sample = (
            self.send_ranks[:8] + ["..."] + self.send_ranks[-8:]
            if len(self.send_ranks) > 16
            else self.send_ranks
        )
        self.num_to_recvs = sum(
            (len(operations) for operations in self.transfer_plan.operations.values())
        )
        self.rank_coordinate = (
            f"{self.engine_rank}-{self.rank_info.global_rank}-{self.transfer_rank}"
        )
        logger.info(
            f"Created NCCL weights reader for rank {self.rank_info.global_rank}, engine rank {self.engine_rank}"
        )

    def _shake_hands_with_writer(self):
        from shardfly._utils.process_group import setup_batch_isend_irecv

        if self.transfer_rank == 0:
            logger.info(
                f"Start to test NCCL ready for rank {self.transfer_rank}, world size {self.transfer_world_size}"
            )
            dist.recv(
                self.ready_tensor,
                src=self.world_size - 1,
                group=self.weights_update_group,
            )
            logger.info(
                f"NCCL ready: recv tensor from rank 0 for rank {self.transfer_rank}"
            )
        setup_batch_isend_irecv(
            self.weights_update_group, self.transfer_rank, self.world_size
        )

    def _set_device(self):
        gpu_id = getattr(self.scheduler, "gpu_id", None) or getattr(
            self.scheduler, "local_rank", None
        )
        if gpu_id is None:
            gpu_id = int(os.environ.get("LOCAL_RANK", 0))
        device_type = device_util.get_device_type()
        device_count = device_util.device_count() or 1
        if device_type == "cuda":
            prev_device = torch.cuda.current_device()
            logger.info(
                f"[NCCLWeightsReader] Set device to {gpu_id} for rank {self.transfer_rank}, device env is {os.environ.get('DEVICE')}, previous device is {prev_device}, device_count is {device_count}, CUDA_VISIBLE_DEVICES env is {os.environ.get('CUDA_VISIBLE_DEVICES')}"
            )
            torch.cuda.set_device(gpu_id)
            self.barrier_device = torch.cuda.current_device()
            self.backend = "nccl"
            self.ready_tensor = torch.tensor(1).cuda()
        else:
            logger.info(
                f"[NCCLWeightsReader] Set device to {gpu_id} for rank {self.transfer_rank}, device env is {os.environ.get('DEVICE')}, previous device is {device_util.current_device()}, device_count is {device_count}, {'/'.join(device_util.visible_devices_env_names())} env is {device_util.visible_devices_env_value() or '(unset)'}"
            )
            device_util.set_device(gpu_id)
            self.barrier_device = device_util.current_device()
            self.backend = self.comm_backend
            self.ready_tensor = torch.tensor(1, device=device_util.get_torch_device())

    def _init_weights_exchange_process_group(self):
        if self.already_initialized:
            return
        from shardfly._utils.process_group import init_weights_update_group

        self.weights_update_group = init_weights_update_group(
            master_address=self.master_address,
            master_port=self.master_port,
            rank=self.transfer_rank,
            world_size=self.world_size,
            group_name="weights_exchange",
            backend="nccl",
            role="inference",
        )
        logger.info(
            f"Initialized NCCL weights reader for rank {self.transfer_rank}, engine rank {self.engine_rank}"
        )
        dist.barrier(group=self.weights_update_group, device_ids=[self.barrier_device])
        logger.info(f"Barrier passed for weights reader with rank {self.transfer_rank}")
        self.already_initialized = True

    @torch.inference_mode()
    def _update_weights(self, step_id, **kwargs):
        """
        Asynchronously receive weights from training ranks using torch.distributed.irecv.

        This method implements a pipelined approach where:
        1. For each sender rank, we maintain a queue of operations to receive
        2. We start irecv operations in parallel from all sender ranks
        3. When a receive completes, we immediately start the next receive from that rank
        4. We continue until all operations from all sender ranks are completed

        Args:
            step_id: The training step ID
            **kwargs: Additional keyword arguments (unused)
        """
        logger.info(
            f"Start to update weights using NCCL for step {step_id} from {len(self.transfer_plan.operations)} ranks({self.send_ranks_sample}) for rank {self.rank_coordinate}."
        )
        self._init_weights_exchange_process_group()
        start_time = time.perf_counter()
        profile_metrics = {
            "build_batch_time_ms": 0.0,
            "metadata_upload_time_ms": 0.0,
            "kernel_transfer_time_ms": 0.0,
            "reader_wait_time_ms": 0.0,
            "reader_copyback_time_ms": 0.0,
            "payload_bytes": 0.0,
        }
        logger.info("Reader: submitting device task batch")
        sync_start_barrier_time_ms = 0.0
        if os.environ.get("SHARDFLY_PROFILE_SYNC_START", "0") == "1":
            sync_start = time.perf_counter()
            dist.barrier(
                group=self.weights_update_group,
                device_ids=[device_util.current_device()],
            )
            sync_start_barrier_time_ms = (time.perf_counter() - sync_start) * 1000.0
        backend_execute_start = time.perf_counter()
        profile_metrics.update(
            self.device_transport.recv(self.parameters, self.transfer_plan, step_id)
        )
        self.capture.refresh()
        device_util.synchronize(device_id=device_util.current_device())
        profile_metrics["backend_execute_time_ms"] = (
            time.perf_counter() - backend_execute_start
        ) * 1000.0
        duration = time.perf_counter() - start_time
        logger.info(
            f"Finished receiving weights for step {step_id} using NCCL from {len(self.transfer_plan.operations)} ranks({self.send_ranks_sample}) to rank {self.rank_coordinate} with {self.num_to_recvs} receives, took {duration:.4f} seconds"
        )
        compute_statistics(
            self._history_update_weights_time,
            step_id,
            duration,
            "Receive weights using NCCL",
        )
        completion_barrier_start = time.perf_counter()
        dist.barrier(
            group=self.weights_update_group, device_ids=[device_util.current_device()]
        )
        completion_barrier_time_ms = (
            time.perf_counter() - completion_barrier_start
        ) * 1000.0
        logger.info(
            f"Barrier passed for reader step {step_id} with rank {self.transfer_rank}"
        )
        kernel_transfer_time_ms = profile_metrics.get("kernel_transfer_time_ms", 0.0)
        effective_gbps = 0.0
        if kernel_transfer_time_ms > 0:
            effective_gbps = profile_metrics.get("payload_bytes", 0.0) / (
                kernel_transfer_time_ms * 1000000.0
            )
        backend_execute_time_ms = profile_metrics.get("backend_execute_time_ms", 0.0)
        backend_effective_gbps = 0.0
        if backend_execute_time_ms > 0:
            backend_effective_gbps = profile_metrics.get("payload_bytes", 0.0) / (
                backend_execute_time_ms * 1000000.0
            )
        self.last_transfer_metrics = dict(profile_metrics)
        emit_profile(
            logger,
            event="weight_transfer",
            role="reader",
            backend=self.comm_backend,
            phase=profile_phase(step_id),
            step_id=int(step_id),
            rank=int(self.transfer_rank),
            sync_start_barrier_time_ms=sync_start_barrier_time_ms,
            completion_barrier_time_ms=completion_barrier_time_ms,
            total_transfer_time_ms=duration * 1000.0,
            effective_gbps=effective_gbps,
            backend_effective_gbps=backend_effective_gbps,
            **profile_metrics,
        )
        emit_profile(
            logger,
            event="weight_transfer_cleanup",
            role="reader",
            backend=self.comm_backend,
            phase=profile_phase(step_id),
            step_id=int(step_id),
            rank=int(self.transfer_rank),
            resource_cleanup_time_ms=0.0,
            gc_collect_time_ms=0.0,
            gc_collect_skipped=True,
        )
