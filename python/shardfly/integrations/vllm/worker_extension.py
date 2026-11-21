"""Register ShardFly inside every vLLM worker, including spawned DP workers.

An API-server monkey patch is inherited by forked TP workers, but not by fresh
DP processes. vLLM resolves this extension in each worker before constructing
its model worker. The methods remain on WorkerBase so existing integrations
and forked workers use the same implementation.
"""

import os

from shardfly.integrations.vllm.loader import register_capture_loader

if os.environ.get("VERL_VLLM_FP8_QUANT_ENABLED") == "1":
    # veRL records checkpoint layouts during the initial vLLM model load.
    from verl.utils.vllm.vllm_quant_utils import apply_vllm_quant_patches

    apply_vllm_quant_patches()

from shardfly.integrations.vllm.plugin import _patch_shardfly_worker

register_capture_loader()
_patch_shardfly_worker()


class ShardFlyWorkerExtension:
    """Import hook for vLLM's worker-extension loader."""

