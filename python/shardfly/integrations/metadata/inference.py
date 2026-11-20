"""Collect checkpoint-coordinate shards from the captured vLLM workers."""

from shardfly.capture import capture_inference
from shardfly.integrations.sharding.vllm import get_vllm_rank_info
from shardfly.metadata.capture import build_captured_parameters_meta
from shardfly.metadata.rank import RankInfo


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
