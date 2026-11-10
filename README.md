# ShardFly

ShardFly is a CUDA transport for **cross-tensor in-flight conversion** of
sharded model weights. It publishes updated weights from training ranks to
disaggregated inference engines in one fused pass: layout conversion, dtype
conversion (BF16 and blockwise FP8), and the network transfer overlap instead of
materializing a full converted copy on either side.

The transport is plan-driven. A `TransferPlan` fixes the source and destination
shards once; every subsequent weight publication reuses the prepared batches,
device descriptors and ring routes. Membership epochs allow rollout groups to
grow or shrink without rebuilding the whole plan path.

## Layout

- `include/shardfly/`: public host API, CUDA work descriptors, lowering and
  topology helpers.
- `src/`: CUDA kernels and launch entry points, independent of Python/PyTorch.
- `bindings/torch/`: optional PyTorch tensor and stream binding.
- `python/shardfly/`: plans, metadata, membership, logical replicas and the
  `Transport` orchestration layer.
- `python/shardfly/integrations/`: vLLM and veRL adapters, model weight
  converters, sharding strategies, readers and writers.
- `python/shardfly/profiling/`: opt-in NVTX, memory and RDMA telemetry.
- `tests/`: CPU plan/conversion regressions plus transport protocol checks.

## Requirements

- Linux with an NVIDIA GPU (`sm_90` is the validated target; set
  `CMAKE_CUDA_ARCHITECTURES` for other architectures).
- CUDA 13 toolkit.
- NCCL with Device API headers (2.30.4 or newer; 2.31.2 is the pinned default).
- PyTorch only when building the Python binding or the vLLM/veRL adapters.

The pinned NCCL wheel can be installed into an isolated prefix:

```sh
python -m pip install --no-deps --target .deps/nccl-cu13-2.31.2 \
  -r requirements/nccl-cu13.txt
export SHARDFLY_NCCL_ROOT="$PWD/.deps/nccl-cu13-2.31.2/nvidia/nccl"
source scripts/nccl-default.sh
```

`scripts/nccl-default.sh` checks that headers and runtime match, then exports
`SHARDFLY_NCCL_LIB`, `LD_PRELOAD` and `LD_LIBRARY_PATH` so the same library is
used by CMake, the Python extension and every spawned worker.

## Build

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_ARCHITECTURES=90
cmake --build build -j
cmake --install build --prefix /path/to/install
```

CMake consumers can use the installed package without Python:

```cmake
find_package(ShardFly 0.1 CONFIG REQUIRED)
target_link_libraries(your_target PRIVATE ShardFly::shardfly)
```

The public host API is `<shardfly/runtime.h>`. The caller owns tensor memory and
CUDA streams.

To build and install the Python package with the current environment's PyTorch:

```sh
pip install --no-build-isolation . \
  -Ccmake.define.SHARDFLY_NCCL_ROOT="$SHARDFLY_NCCL_ROOT"
```

Importing the transport never JIT-compiles an extension; the CUDA library must be
built explicitly.

## Python transport

`shardfly.Transport` binds a `TransferPlan` to one NCCL communicator. Training
ranks call `send`, inference workers call `recv`, and both sides may call
`prepare_send` / `prepare_recv` ahead of the first publication. Plans and
prepared descriptors are reused across publications:

```python
from shardfly import Transport

transport = Transport(group, rank=rank, world_size=world_size,
                      num_infer_engines=num_engines)
transport.prepare_send(train_parameters, plan)
metrics = transport.send(train_parameters, plan, step_id)
```

See `python/shardfly/plan.py` and `tests/test_transfer_plan_cpu_sim.py` for plan
construction, and `python/shardfly/membership.py` for dynamic rollout groups.

## vLLM adapter

Installing the package registers a `vllm.general_plugins` entry point. A vLLM
server started with the package on `PYTHONPATH` exposes the publication API:

- `POST /shardfly_init`
- `POST /shardfly_update`
- `POST /shardfly_prepare_membership`
- `POST /shardfly_retire_membership`
- `POST /publication_init`, `/publication_update`, `/publication_close`
- `GET /publication_telemetry`

Model converters live in `python/shardfly/integrations/models/`. Workers are
patched by `python/shardfly/integrations/vllm/plugin.py`, including spawned DP
workers through `worker_extension.py`.

## veRL adapter

`python/shardfly/integrations/verl/` provides a checkpoint-engine adapter and
locality-aware Ray placement. The native NCCL publication path in
`integrations/publication/verl_native.py` uses the same transport and supports
FP8 checkpoints.

## Tests

CPU plan, conversion, routing and membership regressions need neither a GPU nor
a CUDA build:

```sh
PYTHONPATH=python python -m pytest -q
```

CUDA transport checks require the built extension and multiple GPUs; they are
selected by the transport tests in `tests/`.

## Citation

```bibtex
@misc{shardfly2026,
  title  = {When Shards Fly: Cross-Tensor In-Flight Conversion for Low-Latency,
            Low-Memory Weight Updates in Disaggregated RL},
  author = {Xiao, Jiayu and Wang, Ye and Zhang, Zhi and Guo, Jiarui and
            Liu, Kan and Lan, Tao and Yu, Minchen},
  year   = {2026},
  note   = {arXiv preprint}
}
```

## License

Apache License 2.0. See [LICENSE](LICENSE) and [NOTICE](NOTICE) for third-party
attribution.
