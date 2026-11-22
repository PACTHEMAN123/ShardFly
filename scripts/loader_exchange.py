"""Verify repeated checkpoint-based weight publication."""

import argparse
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import timedelta
from types import SimpleNamespace
from urllib.parse import urlsplit

import requests
import torch
from torch import distributed as dist
from transformers import AutoConfig, AutoModelForCausalLM

from shardfly.capture import (
    capture_megatron,
    capture_training,
    configure_megatron_checkpoint,
)
from shardfly.control.store import start_meta_server, stop_meta_server
from shardfly.integrations.engine import CapturedTrainingEngine
from shardfly.integrations.publication import ShardFlyPublicationMechanism


def verify(url, operation):
    response = requests.post(
        f"{url}/shardfly_model_profile",
        json={"operation": operation, "version": -2, "marker_mode": "scale", "marker_period": 1},
        timeout=180,
    )
    response.raise_for_status()
    rows = response.json()["ranks"]
    if not rows or (operation == "verify" and not all(row["verified"] for row in rows)):
        raise RuntimeError(response.text)
    return len(rows)


def megatron_capture(args):
    from megatron.bridge import AutoBridge
    from megatron.core import parallel_state
    from megatron.core.distributed import DistributedDataParallelConfig
    from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed

    parallelism = dict(
        tensor_model_parallel_size=args.tp,
        pipeline_model_parallel_size=args.pp,
        context_parallel_size=args.cp,
        expert_model_parallel_size=args.ep,
        expert_tensor_parallel_size=args.expert_tp,
    )
    parallel_state.initialize_model_parallel(**parallelism, create_gloo_process_groups=False)
    model_parallel_cuda_manual_seed(42)
    bridge = AutoBridge.from_hf_pretrained(args.model, trust_remote_code=True)
    provider = bridge.to_megatron_provider(load_weights=True)
    for name, size in parallelism.items():
        setattr(provider, name, size)
    provider.sequence_parallel = args.tp > 1
    hf_config = AutoConfig.from_pretrained(args.model, trust_remote_code=True)
    configure_megatron_checkpoint(provider, hf_config)
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
    return capture_megatron(models, bridge)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--rollout", required=True, action="append")
    parser.add_argument("--metadata-host", required=True)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--fsdp2", action="store_true")
    mode.add_argument("--megatron", action="store_true")
    mode.add_argument("--mfsdp", choices=("optim", "optim_grads", "optim_grads_params"))
    parser.add_argument("--updates", type=int, default=2)
    for dimension in ("tp", "pp", "cp", "ep", "expert-tp"):
        parser.add_argument(f"--{dimension}", type=int, default=1)
    args = parser.parse_args()
    endpoints = [urlsplit(url) for url in args.rollout]
    for endpoint in endpoints:
        if endpoint.scheme != "http" or not endpoint.hostname or not endpoint.port or endpoint.path:
            parser.error("--rollout requires http://host:port")
    if args.updates < 1:
        parser.error("--updates requires a positive count")

    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl", timeout=timedelta(minutes=5))
    driver = dist.get_rank() == 0
    if driver:
        for url in args.rollout:
            requests.get(f"{url}/health", timeout=15).raise_for_status()
    if args.mfsdp or args.megatron:
        capture = megatron_capture(args)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            args.model, dtype=torch.bfloat16, device_map={"": local_rank}, trust_remote_code=True,
        )
        if args.fsdp2:
            from torch.distributed.fsdp import fully_shard

            fully_shard(model, mesh=dist.init_device_mesh("cuda", (dist.get_world_size(),)))
        capture = capture_training(model)

    address = [None]
    if driver:
        host, port = start_meta_server(args.metadata_host)
        address[0] = f"{host}:{port}"
    dist.broadcast_object_list(address, src=0)
    engine = CapturedTrainingEngine(capture, meta_server_addr=address[0])

    def barrier():
        dist.barrier(device_ids=[local_rank])

    harness = SimpleNamespace(
        comm_backend="transport",
        validate=False,
        is_driver=driver,
        meta_server_addr=address[0],
        inference_config={
            "num_engines": len(endpoints),
            "comm_backend": "transport",
            "enable_debug_mode": False,
        },
        training_engine=engine,
        publication_endpoints=lambda: [
            (rank, endpoint.hostname, endpoint.port) for rank, endpoint in enumerate(endpoints)
        ],
        _training_barrier=barrier,
    )
    publication = ShardFlyPublicationMechanism(harness)
    def check_rollouts(operation):
        with ThreadPoolExecutor(max_workers=len(endpoints)) as pool:
            return sum(pool.map(lambda url: verify(url, operation), args.rollout))

    try:
        if driver:
            publication.initialize_driver()
        barrier()
        for step in range(args.updates):
            if driver:
                check_rollouts("clear")
            barrier()
            engine.set_global_step(step)
            start = time.perf_counter()
            publication.publish()
            elapsed = time.perf_counter() - start
            if driver:
                ranks = check_rollouts("verify")
                print(json.dumps({"step": step, "seconds": elapsed, "verified_ranks": ranks}), flush=True)
            barrier()
    finally:
        engine.close()
        if driver:
            stop_meta_server()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
