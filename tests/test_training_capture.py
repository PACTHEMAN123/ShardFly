from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import FullyShardedDataParallel, fully_shard
from torch.distributed.tensor import Shard, distribute_tensor

from shardfly.capture import capture_training
from shardfly.integrations import engine as writer_module
from shardfly.integrations.loader_capture import WeightLoadCapture
from shardfly.integrations.reader import get_vllm_rank_info
from shardfly.metadata.capture import build_captured_parameters_meta
from shardfly.plan import TransferPlanBuilder


def _inference_layout(checkpoint):
    workers = []
    targets = []
    for rank in range(2):
        model = torch.nn.Linear(
            2, checkpoint.shape[0], bias=False, dtype=torch.bfloat16
        )
        with torch.no_grad(), WeightLoadCapture() as capture:
            capture.start(model)
            for _, weight in capture.weights([("weight", checkpoint)]):
                model.weight.copy_(weight.narrow(1, rank * 2, 2))
            capture.end()
        snapshot = capture.finish(model).snapshot(model)
        rank_info = get_vllm_rank_info(
            dict(
                global_rank=rank,
                local_rank=rank,
                world_size=2,
                tp_rank=rank,
                tp_size=2,
                pp_rank=0,
                pp_size=1,
                dp_rank=0,
                dp_size=1,
                attn_tp_rank=rank,
                attn_tp_size=2,
                attn_dp_rank=0,
            ),
            0,
        )
        workers.append({"rank_info": rank_info, "params_meta": snapshot.shards})
        targets.append(snapshot.tensors["weight"])
    return build_captured_parameters_meta(workers), targets


def _capture_run(rank, world_size, rendezvous, strategy):
    dist.init_process_group(
        "gloo", init_method=rendezvous, rank=rank, world_size=world_size
    )
    try:
        checkpoint = torch.arange(20, dtype=torch.float32).reshape(5, 4)
        model = torch.nn.Linear(4, 5, bias=False)
        with torch.no_grad():
            model.weight.copy_(checkpoint)
        if strategy == "fsdp1":
            model = FullyShardedDataParallel(model, device_id=torch.device("cpu"))
        elif strategy == "hsdp":
            fully_shard(
                model,
                mesh=init_device_mesh(
                    "cpu", (2, 2), mesh_dim_names=("replicate", "shard")
                ),
            )
        elif strategy == "multi_axis":
            mesh = init_device_mesh("cpu", (2, 2))
            model.weight = torch.nn.Parameter(
                distribute_tensor(model.weight, mesh, (Shard(0), Shard(1)))
            )
        else:
            fully_shard(model, mesh=init_device_mesh("cpu", (2,)))

        capture = capture_training(model)
        infer_meta, targets = _inference_layout(checkpoint)
        assert all(target.dtype == torch.bfloat16 for target in targets)
        objects = {"infer_params_meta": infer_meta, "num_infer_engines": 1}
        writer_module.MetaServerClient = lambda *_: SimpleNamespace(
            get_object=lambda key, **_: objects[key], put_object=objects.__setitem__
        )
        writer = writer_module.WeightsExchangeShardingWriter(
            capture, {"meta_server_addr": "test:1234"}
        )
        writer._initialize()
        assert len(writer.parameters_meta[0].replicas) == (
            2 if strategy == "hsdp" else 1
        )
        transfer = TransferPlanBuilder(2, world_size, group_replicated_inference=False)
        plans = [
            transfer.build_local_transfer_plan(
                infer_meta, writer.parameters_meta, receiver
            )
            for receiver in range(2)
        ]
        original = writer.capture_snapshot()
        assert bool(original.materialized) == (strategy == "fsdp1")
        bound = writer.bind_parameters(
            original, {"weight": "weight"}, torch.device("cpu"), {}
        )
        optimizer = torch.optim.SGD(model.parameters(), lr=100)
        for step in range(1, 3):
            for parameter in model.parameters():
                parameter.grad = -torch.ones_like(parameter)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            snapshot = writer.capture_snapshot()
            assert (
                writer.bind_parameters(
                    snapshot, {"weight": "weight"}, torch.device("cpu"), bound
                )
                is bound
            )
            gathered = [None] * world_size
            dist.all_gather_object(gathered, bound["weight"].tolist())
            for receiver, (target, plan) in enumerate(zip(targets, plans)):
                for operations in plan.operations.values():
                    for op in operations:
                        source = torch.tensor(
                            gathered[op.send_shard_meta.global_rank], dtype=target.dtype
                        )
                        target[op.inf_slices].copy_(source[op.train_slices])
                expected = (
                    (checkpoint + 100 * step)
                    .to(target.dtype)
                    .narrow(1, receiver * 2, 2)
                )
                torch.testing.assert_close(target, expected, rtol=0, atol=0)
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(
    not dist.is_gloo_available(), reason="Capture simulation uses CPU Gloo"
)
@pytest.mark.parametrize(
    "strategy,world_size", [("fsdp1", 2), ("fsdp2", 2), ("hsdp", 4), ("multi_axis", 4)]
)
def test_same_capture_writer_supports_multiple_shard_layouts(
    tmp_path, strategy, world_size
):
    mp.spawn(
        _capture_run,
        args=(world_size, (tmp_path / "rendezvous").as_uri(), strategy),
        nprocs=world_size,
        join=True,
    )


def test_export_callback_can_supply_chunks_while_staging_replaced_storage(monkeypatch):
    model = torch.nn.Linear(2, 2, bias=False)
    capture = capture_training(
        model, state_dict=lambda module: {"weight": module.weight}, stage=True
    )
    initial = capture.snapshot()
    assert initial.materialized == {"weight"}
    monkeypatch.setattr(writer_module, "MetaServerClient", lambda *_: None)
    writer = writer_module.WeightsExchangeShardingWriter(
        capture, {"meta_server_addr": "test:1234"}
    )
    writer.target_dtypes = {"weight": model.weight.dtype}
    bound = writer.bind_parameters(
        initial, {"weight": "weight"}, torch.device("cpu"), {}
    )
    model.weight = torch.nn.Parameter(torch.full_like(model.weight, 5))
    assert (
        writer.bind_parameters(
            capture.snapshot(), {"weight": "weight"}, torch.device("cpu"), bound
        )
        is bound
    )
    torch.testing.assert_close(bound["weight"], model.weight)


def test_training_engine_publishes_fresh_weights_through_the_same_prepared_binding(
    monkeypatch,
):
    from shardfly.integrations.engine import CapturedTrainingEngine

    model = torch.nn.Linear(4, 5, bias=False)
    infer_meta, targets = _inference_layout(model.weight.detach())
    objects = {
        "infer_params_meta": infer_meta,
        "num_infer_engines": 1,
        "master_info": ("127.0.0.1", 1234),
    }
    monkeypatch.setattr(
        writer_module,
        "MetaServerClient",
        lambda *_: SimpleNamespace(
            get_object=lambda key, **_: objects[key], put_object=objects.__setitem__
        ),
    )
    monkeypatch.setattr(dist, "get_rank", lambda: 0)
    monkeypatch.setattr(dist, "get_world_size", lambda: 1)
    monkeypatch.setattr(
        dist, "all_gather_object", lambda rows, row: rows.__setitem__(0, row)
    )
    monkeypatch.setattr(dist, "broadcast_object_list", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(dist, "barrier", lambda **_: None)
    monkeypatch.setattr(writer_module.device_util, "synchronize", lambda **_: None)
    monkeypatch.setattr(writer_module.device_util, "current_device", lambda: 0)
    prepared = []

    class CPUTransport:
        def __init__(self, *_args, **_kwargs):
            pass

        def resolve_rollout_topology(self):
            pass

        def prepare_send(self, parameters, plan, **_kwargs):
            prepared.append(parameters)

        def send(self, parameters, plan, _step_id):
            assert parameters is prepared[0]
            for receiver, operations in plan.operations.items():
                for op in operations:
                    targets[receiver][op.inf_slices].copy_(
                        parameters[op.send_shard_meta.tensor_key][op.train_slices]
                    )
            return {}

    monkeypatch.setattr(writer_module, "Transport", CPUTransport)
    engine = CapturedTrainingEngine(
        capture_training(model), meta_server_addr="test:1234"
    )
    writer = engine.weights_exchange_writer
    writer._set_device = lambda: None
    writer._init_weights_exchange_process_group = lambda: setattr(
        writer, "weights_update_group", None
    )
    writer._shake_hands_with_reader = lambda: None
    writer.bind_source = lambda snapshot, required: writer.bind_parameters(
        snapshot, required, torch.device("cpu"), writer.device_parameters
    )

    with torch.no_grad():
        for step in (1, 2):
            model.weight.fill_(step)
            engine.set_global_step(step)
            engine.write_weights()
            for target in targets:
                torch.testing.assert_close(target, torch.full_like(target, step))
    assert len(prepared) == 1
    assert objects["training_params_meta"][0].dtype is torch.bfloat16
