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


"""vLLM server configuration, RPC adapters, and optional CLI entrypoint."""

import asyncio
import inspect
import os
import sys
from dataclasses import dataclass, field
from functools import cache
from typing import Any, Dict, List, Optional

from shardfly import logging

logger = logging.getLogger(__name__)


class InferenceConfigValidationError(ValueError):
    """Raised when InferenceConfig contains invalid or inconsistent settings."""

    pass


_VALID_COMM_BACKENDS: frozenset = frozenset({"transport"})


@dataclass
class InferenceConfig:
    """
    Configuration for inference.
    """

    model_path: Optional[str] = None
    tp_size: Optional[int] = None
    pp_size: Optional[int] = None
    dp_size: Optional[int] = None
    ep_size: Optional[int] = None
    nnodes: Optional[int] = None
    node_rank: Optional[int] = None
    local_rank: Optional[int] = None
    num_engines: int = 1
    engine_rank: int = 0
    meta_server_addr: Optional[str] = None
    comm_backend: str = "transport"
    weights_validation_steps: int = 0
    validate_weights_every_n_steps: int = 1
    dump_weights_list_for_validation: List[str] = field(default_factory=list)
    dump_weights_dir_for_validation: Optional[str] = None
    disable_weights_exchange_pipeline: bool = False
    enable_debug_mode: bool = False
    debug_mode_config: Dict = field(
        default_factory=dict, metadata={"help": "Debug mode configuration"}
    )
    weights_comm_nccl_group_size: int = 1

    def validate(self) -> None:
        """Validate configuration fields for consistency and correctness.

        Raises:
            InferenceConfigValidationError: if any field is invalid or fields are inconsistent.
        """
        errors: List[str] = []
        if self.comm_backend not in _VALID_COMM_BACKENDS:
            errors.append(
                f"comm_backend must be one of {sorted(_VALID_COMM_BACKENDS)}, got {self.comm_backend!r}"
            )
        if self.num_engines < 1:
            errors.append(f"num_engines must be >= 1, got {self.num_engines}")
        elif not 0 <= self.engine_rank < self.num_engines:
            errors.append(
                f"engine_rank must be in [0, num_engines), got engine_rank={self.engine_rank} with num_engines={self.num_engines}"
            )
        for name, val in [
            ("tp_size", self.tp_size),
            ("pp_size", self.pp_size),
            ("dp_size", self.dp_size),
            ("ep_size", self.ep_size),
            ("weights_comm_nccl_group_size", self.weights_comm_nccl_group_size),
        ]:
            if val is not None and val < 1:
                errors.append(f"{name} must be >= 1 when set, got {val}")
        if self.nnodes is not None:
            if self.nnodes < 1:
                errors.append(f"nnodes must be >= 1 when set, got {self.nnodes}")
            elif self.node_rank is not None and (not 0 <= self.node_rank < self.nnodes):
                errors.append(
                    f"node_rank must be in [0, nnodes), got node_rank={self.node_rank} with nnodes={self.nnodes}"
                )
        if self.validate_weights_every_n_steps < 1:
            errors.append(
                f"validate_weights_every_n_steps must be >= 1, got {self.validate_weights_every_n_steps}"
            )
        if self.weights_validation_steps < 0:
            errors.append(
                f"weights_validation_steps must be >= 0, got {self.weights_validation_steps}"
            )
        if self.dump_weights_list_for_validation and (
            not self.dump_weights_dir_for_validation
        ):
            errors.append(
                "dump_weights_dir_for_validation must be set when dump_weights_list_for_validation is non-empty"
            )
        if self.num_engines > 1 and (not self.meta_server_addr):
            errors.append(
                f"meta_server_addr must be set when num_engines > 1 and comm_backend is {self.comm_backend!r}"
            )
        if errors:
            msg = "InferenceConfig validation failed:\n" + "\n".join(
                (f"  - {e}" for e in errors)
            )
            raise InferenceConfigValidationError(msg)

    def validated(self) -> "InferenceConfig":
        """Call validate() and return self for easy chaining.

        Example::

            config = InferenceConfig.from_dict(d).validated()
        """
        self.validate()
        return self

    @staticmethod
    def from_dict(
        config_dict: Dict[str, Any], validate: bool = True
    ) -> "InferenceConfig":
        config_dict = {
            k: v
            for (k, v) in config_dict.items()
            if k in InferenceConfig.__dataclass_fields__
        }
        cfg = InferenceConfig(**config_dict)
        if validate:
            cfg.validate()
        return cfg


class ShardFlyVLLMServerAdapter:
    """Server-side ShardFly adapter that uses vLLM EngineCoreClient for execution."""

    def __init__(
        self,
        engine_client,
        meta_server_addr: str,
        engine_rank: int = 0,
        num_engines: int = 1,
        comm_backend: str = "transport",
        enable_debug_mode: bool = False,
        debug_mode_config: Optional[Dict[str, Any]] = None,
        disable_weights_exchange_pipeline: bool = False,
        weights_comm_nccl_group_size: int = 1,
        nnodes: Optional[int] = None,
        node_rank: Optional[int] = None,
        weights_validation_steps: int = 0,
        validate_weights_every_n_steps: int = 1,
        dump_weights_list_for_validation: Optional[list[str]] = None,
        dump_weights_dir_for_validation: Optional[str] = None,
        loop: Optional[asyncio.AbstractEventLoop] = None,
    ):
        self._engine_client = engine_client
        self._engine_core = engine_client.engine_core
        self._initialized = False
        self._initializing = False
        self._loop = loop
        vllm_config = engine_client.vllm_config
        parallel_config = vllm_config.parallel_config
        self.hf_config = engine_client.model_config.hf_config
        self.engine_name = "vllm"
        pc_nnodes = getattr(parallel_config, "nnodes", None)
        pc_node_rank = getattr(parallel_config, "node_rank", None)
        resolved_nnodes = nnodes if nnodes is not None else pc_nnodes or 1
        resolved_node_rank = node_rank if node_rank is not None else pc_node_rank or 0
        enable_expert_parallel = bool(
            getattr(parallel_config, "enable_expert_parallel", False)
        )
        pcp_size = int(
            getattr(parallel_config, "prefill_context_parallel_size", 1) or 1
        )
        inferred_ep_size = (
            parallel_config.tensor_parallel_size
            * parallel_config.data_parallel_size
            * pcp_size
            if enable_expert_parallel
            else 1
        )
        self._config = InferenceConfig(
            tp_size=parallel_config.tensor_parallel_size,
            pp_size=parallel_config.pipeline_parallel_size,
            dp_size=parallel_config.data_parallel_size,
            ep_size=inferred_ep_size,
            nnodes=resolved_nnodes,
            node_rank=resolved_node_rank,
            num_engines=num_engines,
            engine_rank=engine_rank,
            meta_server_addr=meta_server_addr,
            comm_backend=comm_backend,
            enable_debug_mode=enable_debug_mode,
            debug_mode_config=debug_mode_config or {},
            disable_weights_exchange_pipeline=disable_weights_exchange_pipeline,
            weights_comm_nccl_group_size=weights_comm_nccl_group_size,
            weights_validation_steps=weights_validation_steps,
            validate_weights_every_n_steps=validate_weights_every_n_steps,
            dump_weights_list_for_validation=dump_weights_list_for_validation or [],
            dump_weights_dir_for_validation=dump_weights_dir_for_validation,
        )
        self.weights_exchange_reader = None

    @property
    def config(self):
        return self._config

    @property
    def num_engines(self):
        return self._config.num_engines

    @property
    def engine_rank(self):
        return self._config.engine_rank

    def initialize(self) -> None:
        if self._initialized:
            return
        if self._initializing:
            return
        self._initializing = True
        logger.info("Initializing ShardFly weights reader in vLLM server process.")
        from shardfly.integrations.reader import WeightsReader

        self.weights_exchange_reader = WeightsReader(self)
        self.weights_exchange_reader.initialize()
        self._initialized = True
        self._initializing = False

    def update_weights(self, step_id: int, **kwargs):
        if not self._initialized:
            raise RuntimeError("ShardFly adapter not initialized.")
        self.weights_exchange_reader.update_weights(step_id=step_id, **kwargs)
        # Scheduler caches live outside model workers on vLLM V1.
        if self._call_engine_async("reset_prefix_cache") is False:
            raise RuntimeError(
                "vLLM prefix cache reset failed after weight publication"
            )
        if hasattr(self._engine_client, "reset_encoder_cache"):
            self._call_engine_async("reset_encoder_cache")

    def release_memory_occupation(self, tags=None) -> None:
        logger.info("Release memory occupation via vLLM sleep.")
        self._call_engine_async("sleep", 1)

    def resume_memory_occupation(self, tags=None) -> None:
        logger.info("Resume memory occupation via vLLM wake_up.")
        self._call_engine_async("wake_up", tags)

    def _call_engine_async(self, method: str, *args):
        fn = getattr(self._engine_client, method, None)
        if fn is None:
            raise RuntimeError(f"Engine client has no method {method}")
        if inspect.iscoroutinefunction(fn):
            return self._run_on_loop(fn(*args))
        return fn(*args)

    def _collective_rpc(
        self,
        method: str,
        timeout: Optional[float] = None,
        args: tuple = (),
        kwargs: Optional[dict] = None,
    ):
        if hasattr(self._engine_client, "collective_rpc"):
            fn = self._engine_client.collective_rpc
            if inspect.iscoroutinefunction(fn):
                return self._run_on_loop(fn(method, timeout, args, kwargs))
            return fn(method, timeout, args, kwargs)
        return self._engine_core.collective_rpc(
            method, timeout=timeout, args=args, kwargs=kwargs
        )

    def _collective_rpc_all_dp_cores(
        self,
        method: str,
        timeout: Optional[float] = None,
        args: tuple = (),
        kwargs: Optional[dict] = None,
    ):
        rpc_client = self._get_dp_rpc_client()
        if not (
            rpc_client is not None
            and hasattr(rpc_client, "core_engines")
            and hasattr(rpc_client, "_call_utility_async")
        ):
            return [
                self._collective_rpc(method, timeout=timeout, args=args, kwargs=kwargs)
            ]
        core_engines = list(getattr(rpc_client, "core_engines", []))
        if not core_engines:
            return [
                self._collective_rpc(method, timeout=timeout, args=args, kwargs=kwargs)
            ]
        call_utility = rpc_client._call_utility_async
        if inspect.iscoroutinefunction(call_utility):

            async def _gather_all():
                return await asyncio.gather(
                    *[
                        call_utility(
                            "collective_rpc",
                            method,
                            timeout,
                            args,
                            kwargs,
                            engine=engine,
                        )
                        for engine in core_engines
                    ]
                )

            return self._run_on_loop(_gather_all())
        return [
            call_utility("collective_rpc", method, timeout, args, kwargs, engine=engine)
            for engine in core_engines
        ]

    def _get_dp_rpc_client(self):
        if hasattr(self._engine_client, "core_engines") and hasattr(
            self._engine_client, "_call_utility_async"
        ):
            return self._engine_client
        engine_core = getattr(self._engine_client, "engine_core", None)
        if (
            engine_core is not None
            and hasattr(engine_core, "core_engines")
            and hasattr(engine_core, "_call_utility_async")
        ):
            return engine_core
        return None

    def _run_on_loop(self, coro):
        if self._loop is None:
            raise RuntimeError("No event loop available for async engine calls.")
        return asyncio.run_coroutine_threadsafe(coro, self._loop).result()

    @staticmethod
    def _is_meta_collection_call(method: str, payload: dict) -> bool:
        if method == "_get_model_param_info":
            return True
        if method != "shardfly_execute":
            return False
        task_qualname = payload.get("task_qualname", "")
        return isinstance(task_qualname, str) and task_qualname.endswith(
            "_get_model_param_info"
        )

    def execute_task_in_model_worker(self, fn, **kwargs):
        if not self._initialized and (not self._initializing):
            raise RuntimeError("ShardFly adapter not initialized.")
        if isinstance(fn, str):
            method = fn
            payload = kwargs
        else:
            method = "shardfly_execute"
            infer_engine_config = kwargs.get("infer_engine_config")
            if isinstance(infer_engine_config, InferenceConfig):
                kwargs = dict(kwargs)
                kwargs["infer_engine_config"] = infer_engine_config.__dict__
            payload = {
                "task_module": fn.__module__,
                "task_qualname": fn.__qualname__,
                "task_kwargs": kwargs,
            }
        # vLLM's DP load balancer executes on every core but returns only the
        # first core's result. Collect each core explicitly so a validation
        # failure on any expert/DP rank reaches the publication driver.
        core_results = self._collective_rpc_all_dp_cores(
            method, args=(), kwargs=payload
        )
        merged_results = []
        for result in core_results:
            if isinstance(result, list):
                merged_results.extend(result)
            else:
                merged_results.append(result)
        if self._is_meta_collection_call(method, payload):
            logger.info(
                "Collected model param meta from %s DP core(s), merged %s entries.",
                len(core_results),
                len(merged_results),
            )
        return merged_results


class PublicationVLLMServerAdapter:
    """Thin API-server adapter for mechanism-specific vLLM worker state."""

    def __init__(self, engine_client, loop, timeout_seconds: int):
        self._engine_client = engine_client
        self._engine_core = engine_client.engine_core
        self._loop = loop
        self.timeout_seconds = timeout_seconds
        self.mechanism = None
        # Reuse the tested vLLM RPC plumbing without initializing any
        # ShardFly transport, metadata, communicator, or weights reader.
        self._rpc = ShardFlyVLLMServerAdapter.__new__(ShardFlyVLLMServerAdapter)
        self._rpc._engine_client = engine_client
        self._rpc._engine_core = self._engine_core
        self._rpc._loop = loop

    def initialize(self, mechanism: str, config: dict):
        self.mechanism = mechanism
        return self._collective_rpc(
            "publication_init",
            timeout=self.timeout_seconds,
            args=(mechanism, config),
        )

    def update(self, step_id: int):
        if self.mechanism is None:
            raise RuntimeError("Publication adapter is not initialized")
        result = self._collective_rpc(
            "publication_update",
            timeout=self.timeout_seconds,
            args=(step_id,),
        )
        if self._rpc._call_engine_async("reset_prefix_cache") is False:
            raise RuntimeError("vLLM prefix cache reset failed after publication")
        if hasattr(self._engine_client, "reset_encoder_cache"):
            self._rpc._call_engine_async("reset_encoder_cache")
        return result

    def configure(self, config: dict):
        return self._collective_rpc(
            "publication_configure", timeout=self.timeout_seconds, args=(config,)
        )

    def execute_task_in_model_worker(self, task, **kwargs):
        return self._collective_rpc(
            "shardfly_execute", timeout=self.timeout_seconds,
            args=(task.__module__, task.__qualname__, kwargs),
        )

    def close(self):
        if self.mechanism is None:
            return []
        try:
            return self._collective_rpc(
                "publication_close",
                timeout=self.timeout_seconds,
            )
        finally:
            self.mechanism = None

    def _collective_rpc(self, method, timeout=None, args=(), kwargs=None):
        cores = self._rpc._collective_rpc_all_dp_cores(
            method, timeout=timeout, args=args, kwargs=kwargs
        )
        return [item for result in cores
                for item in (result if isinstance(result, list) else [result])]


def register_shardfly_plugin():
    main_spec = getattr(sys.modules.get("__main__"), "__spec__", None)
    if getattr(main_spec, "name", None) == "vllm.model_executor.models.registry":
        return

    from shardfly.integrations.loader_capture import register_capture_loader
    from shardfly.integrations.vllm_plugin import register_shardfly_plugin as register

    register_capture_loader()
    register()


@cache
def _load_worker_extension():
    # Spawned workers resolve the extension before constructing the model.
    if os.environ.get("VERL_VLLM_FP8_QUANT_ENABLED") == "1":
        from verl.utils.vllm.vllm_quant_utils import apply_vllm_quant_patches

        apply_vllm_quant_patches()

    from shardfly.integrations.vllm_plugin import _patch_shardfly_worker

    _patch_shardfly_worker()
    return _WorkerExtension


class _WorkerExtension:
    """Marker for vLLM's worker-extension loader."""


def __getattr__(name):
    if name == "ShardFlyWorkerExtension":
        return _load_worker_extension()
    raise AttributeError(name)

def main():
    register_shardfly_plugin()

    import uvloop
    from vllm.entrypoints.openai.api_server import run_server
    from vllm.entrypoints.openai.cli_args import (
        make_arg_parser,
        validate_parsed_serve_args,
    )

    try:
        from vllm.entrypoints.utils import cli_env_setup
    except ImportError:
        pass
    else:
        cli_env_setup()

    try:
        from vllm.utils import FlexibleArgumentParser
    except ImportError:
        from vllm.utils.argparse_utils import FlexibleArgumentParser

    parser = FlexibleArgumentParser(
        description="vLLM OpenAI-Compatible RESTful API server (ShardFly)."
    )
    args = make_arg_parser(parser).parse_args()
    if not args.worker_extension_cls:
        args.worker_extension_cls = "shardfly.integrations.vllm.ShardFlyWorkerExtension"
    validate_parsed_serve_args(args)

    logger.info("Starting vLLM server with ShardFly routes enabled")
    uvloop.run(run_server(args))


if __name__ == "__main__":
    main()
