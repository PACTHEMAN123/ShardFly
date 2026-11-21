"""Exercise repeated Megatron-to-vLLM publications through shardfly_capture."""

import argparse
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import timedelta
from urllib.parse import urlsplit

import requests
import torch
from torch import distributed as dist

from shardfly.control.store import start_meta_server, stop_meta_server
from shardfly.integrations.engine.megatron import MegatronEngine
from shardfly.integrations.publication.transport import ShardFlyPublicationMechanism


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--rollout", action="append", required=True, metavar="HTTP_URL")
    parser.add_argument("--metadata-host", required=True)
    parser.add_argument("--tp", type=int, default=1)
    parser.add_argument("--pp", type=int, default=1)
    parser.add_argument("--cp", type=int, default=1)
    parser.add_argument("--ep", type=int, default=1)
    parser.add_argument("--expert-tp", type=int, default=1)
    parser.add_argument(
        "--mfsdp", choices=("optim", "optim_grads", "optim_grads_params")
    )
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--updates", type=int, default=3)
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    if args.warmup < 0 or args.updates < 1:
        parser.error("provide zero or more warmups and at least one measured update")
    for url in args.rollout:
        parts = urlsplit(url)
        if parts.scheme != "http" or not parts.hostname or not parts.port or parts.path:
            parser.error(f"expected http://host:port, got {url!r}")
    return args


def load_training_model(args):
    from megatron.bridge import AutoBridge
    from megatron.core import parallel_state
    from megatron.core.distributed import DistributedDataParallelConfig
    from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
    from transformers import AutoConfig

    parallelism = {
        "tensor_model_parallel_size": args.tp,
        "pipeline_model_parallel_size": args.pp,
        "context_parallel_size": args.cp,
        "expert_model_parallel_size": args.ep,
        "expert_tensor_parallel_size": args.expert_tp,
    }
    parallel_state.initialize_model_parallel(**parallelism, create_gloo_process_groups=False)
    model_parallel_cuda_manual_seed(42)
    hf_config = AutoConfig.from_pretrained(args.model, trust_remote_code=True)
    bridge = AutoBridge.from_hf_pretrained(args.model, trust_remote_code=True)
    provider = bridge.to_megatron_provider(load_weights=False)
    for name, size in parallelism.items():
        setattr(provider, name, size)
    provider.sequence_parallel = args.tp > 1
    provider.finalize()
    ddp_config = None
    if args.mfsdp:
        ddp_config = DistributedDataParallelConfig(
            use_distributed_optimizer=True,
            data_parallel_sharding_strategy=args.mfsdp,
            overlap_grad_reduce=True,
            overlap_param_gather=True,
        )
    models = provider.provide_distributed_model(
        ddp_config=ddp_config,
        wrap_with_ddp=ddp_config is not None,
        use_megatron_fsdp=ddp_config is not None,
        bf16=provider.bf16,
        fp16=provider.fp16,
    )
    bridge.load_hf_weights(models, args.model)
    return models, hf_config


class Exchange:
    comm_backend = "transport"
    validate = False

    def __init__(self, args, meta_server_addr):
        self.args = args
        self.is_driver = dist.get_rank() == 0
        self.meta_server_addr = meta_server_addr
        self.inference_config = {
            "num_engines": len(args.rollout),
            "comm_backend": self.comm_backend,
            "enable_debug_mode": False,
        }
        models, hf_config = load_training_model(args)
        self.megatron_engine = MegatronEngine(
            {"meta_server_addr": meta_server_addr, "comm_backend": self.comm_backend},
            hf_config,
            models,
        )
        self.megatron_engine.initialize()
        self.publication = ShardFlyPublicationMechanism(self)

    def publication_endpoints(self):
        parts = [urlsplit(url) for url in self.args.rollout]
        return [(rank, url.hostname, url.port) for rank, url in enumerate(parts)]

    def _training_barrier(self):
        dist.barrier(device_ids=[torch.cuda.current_device()])

    def profile(self, operation):
        def call(url):
            response = requests.post(
                f"{url}/shardfly_model_profile",
                # With a scale factor of one, the untouched checkpoint is the oracle.
                json={"operation": operation, "version": -2, "marker_mode": "scale"},
                timeout=300,
            )
            response.raise_for_status()
            result = response.json()
            if not result["success"]:
                raise RuntimeError(result)
            return result["ranks"]

        with ThreadPoolExecutor(max_workers=len(self.args.rollout)) as pool:
            return [rank for ranks in pool.map(call, self.args.rollout) for rank in ranks]

    def run(self):
        if self.is_driver:
            self.publication.initialize_driver()
        self._training_barrier()
        for step in range(self.args.warmup + self.args.updates):
            if self.args.verify and self.is_driver:
                self.profile("clear")
            self._training_barrier()
            self.megatron_engine.set_global_step(step)
            start = time.perf_counter()
            self.publication.publish()
            self._training_barrier()
            seconds = time.perf_counter() - start
            if self.is_driver:
                verified_ranks = 0
                if self.args.verify:
                    results = self.profile("verify")
                    if not results or not all(rank["verified"] for rank in results):
                        raise RuntimeError("Missing successful verification from rollout")
                    verified_ranks = len(results)
                print(
                    json.dumps(
                        {
                            "step": step,
                            "warmup": step < self.args.warmup,
                            "publication_seconds": seconds,
                            "verified_ranks": verified_ranks,
                            "mfsdp": self.args.mfsdp,
                        }
                    ),
                    flush=True,
                )


def main():
    args = parse_args()
    if int(os.environ["RANK"]) == 0:
        for url in args.rollout:
            requests.get(f"{url}/health", timeout=10).raise_for_status()
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("nccl", timeout=timedelta(minutes=10))
    is_driver = dist.get_rank() == 0
    exchange = None
    try:
        address = [None]
        if is_driver:
            host, port = start_meta_server(args.metadata_host)
            address[0] = f"{host}:{port}"
        dist.broadcast_object_list(address, src=0)
        exchange = Exchange(args, address[0])
        exchange.run()
    finally:
        if exchange is not None:
            exchange.megatron_engine.close()
        if is_driver:
            stop_meta_server()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
