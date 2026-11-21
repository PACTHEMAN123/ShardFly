import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch
from shardfly.integrations.loader_capture import CAPTURE_ATTRIBUTE, WeightLoadCapture
from shardfly.integrations.megatron_loader import MegatronLoadSource
from shardfly.integrations.metadata.inference import InferParamMetaResolver
from shardfly.integrations.metadata.loader_capture import build_captured_parameters_meta
from shardfly.integrations.reader.base import WorkerWeightsReader
from shardfly.integrations.sharding.vllm import get_vllm_rank_info
from shardfly.integrations.vllm.loader import register_capture_loader
from shardfly.plan import TransferPlanBuilder


class _FusedModel(torch.nn.Module):
    # An architecture without a registered ShardFly converter or sharding rule.
    config = SimpleNamespace(
        architectures=["UnregisteredForCausalLM"], num_hidden_layers=1
    )

    def __init__(self, rank):
        super().__init__()
        self.rank = rank
        self.fused_qkv = torch.nn.Parameter(torch.empty(6, 4), requires_grad=False)
        self.output = torch.nn.Parameter(torch.empty(4, 2), requires_grad=False)
        self.embedding = torch.nn.Parameter(torch.empty(4, 4), requires_grad=False)

    def load_weights(self, weights):
        for name, weight in weights:
            if name in {"query", "key", "value"}:
                fused_offset = {"query": 0, "key": 2, "value": 4}[name]
                self.fused_qkv.data.narrow(0, fused_offset, 2).copy_(
                    weight.narrow(0, self.rank * 2, 2)
                )
            elif name == "output":
                self.output.data.copy_(weight.narrow(1, self.rank * 2, 2))
            elif name == "embedding":
                start = self.rank * 4
                shard = weight.narrow(0, start, min(weight.size(0) - start, 4))
                self.embedding[: shard.size(0)].data.copy_(shard)
                self.embedding[shard.size(0) :].data.fill_(0)


def _context(rank, size):
    return {
        "global_rank": rank,
        "local_rank": rank,
        "world_size": size,
        "tp_rank": rank,
        "tp_size": size,
        "pp_rank": 0,
        "pp_size": 1,
        "dp_rank": 0,
        "dp_size": 1,
        "attn_tp_rank": rank,
        "attn_tp_size": size,
        "attn_dp_rank": 0,
        "scheduler": SimpleNamespace(),
        "infer_engine_config": SimpleNamespace(),
    }


def _weights():
    # A shared backing storage also occurs with mmap-based checkpoint readers.
    checkpoint = torch.arange(84, dtype=torch.float32)
    names = ("query", "key", "value", "output", "embedding")
    return {
        name: value.view(-1, 4)
        for name, value in zip(names, checkpoint.split([16, 16, 16, 16, 20]))
    }


def _capture(model, weights):
    with torch.inference_mode(), WeightLoadCapture() as capture:
        capture.start(model)
        model.load_weights(capture.weights(weights.items()))
        capture.end()
    return capture.finish(model)


def test_loader_capture_reuses_transfer_plan_for_changed_weights(monkeypatch):
    def fail(*args, **kwargs):
        pytest.fail("The capture path must bypass the inference model registry")

    monkeypatch.setattr(
        "shardfly.integrations.metadata.inference.get_infer_weights_converter", fail
    )
    monkeypatch.setattr(
        "shardfly.integrations.metadata.inference.get_sharding_strategy_builder", fail
    )
    monkeypatch.setattr(
        "shardfly.integrations.reader.base.get_infer_weights_converter", fail
    )
    monkeypatch.setattr(
        "shardfly.integrations.reader.base.MetaServerClient", lambda *args: None
    )
    weights = _weights()
    models = [_FusedModel(rank) for rank in range(2)]
    contexts = [_context(rank, 2) for rank in range(2)]
    for model in models:
        setattr(model, CAPTURE_ATTRIBUTE, _capture(model, weights))

    class Engine:
        config = SimpleNamespace(enable_debug_mode=False)
        hf_config = _FusedModel.config
        engine_name = "vllm"

        def execute_task_in_model_worker(self, task, **kwargs):
            return [
                task(model=model, model_context=context, **kwargs)
                for model, context in zip(models, contexts)
            ]

    resolver = InferParamMetaResolver(Engine(), convert_params=True)
    infer_meta = resolver.get_parameters_meta()
    assert resolver.uses_loader_capture
    assert {meta.name: meta.global_shape for meta in infer_meta} == {
        name: tensor.shape for name, tensor in weights.items()
    }
    assert [shard.shape for shard in infer_meta[0].shards] == [(4, 4), (1, 4)]

    # The trainer supplies a separate TP2 layout sharded on rows for all weights.
    # The output projection therefore intersects two senders per receive shard.
    train_workers = []
    for rank in range(2):
        rows = []
        for name, tensor in weights.items():
            split = (tensor.size(0) + 1) // 2
            start = rank * split
            rows.append(
                {
                    "name": name,
                    "global_shape": tuple(tensor.shape),
                    "global_offset": (start, 0),
                    "shape": (min(split, tensor.size(0) - start), tensor.size(1)),
                    "dtype": "float32",
                }
            )
        train_workers.append(
            {
                "rank_info": resolver._params_raw_meta[rank]["rank_info"],
                "params_meta": rows,
            }
        )
    train_meta = build_captured_parameters_meta(train_workers)
    builder = TransferPlanBuilder(2, 2, group_replicated_inference=False)
    receive_plans = [
        builder.build_local_transfer_plan(infer_meta, train_meta, rank)
        for rank in range(2)
    ]
    receivers = [
        WorkerWeightsReader(
            "vllm",
            model,
            context,
            {"hf_config": model.config, "loader_capture": True},
            0,
            1,
            "localhost:1234",
            infer_meta,
            train_meta,
        )
        for model, context in zip(models, contexts)
    ]
    for reader in receivers:
        reader.initialize()

    for delta in (1000, 2000):
        changed = {name: tensor + delta for name, tensor in weights.items()}
        for reader, plan in zip(receivers, receive_plans):
            for operations in plan.operations.values():
                for op in operations:
                    shard = op.send_shard_meta
                    source = changed[shard.name].narrow(
                        0, shard.global_offset[0], shard.shape[0]
                    )
                    target = reader.parameters[op.recv_shard_meta.name]
                    target[op.inf_slices].copy_(source[op.train_slices])
        for model in models:
            expected = _FusedModel(model.rank)
            expected.load_weights(changed.items())
            for key, param in model.named_parameters():
                torch.testing.assert_close(
                    param, expected.get_parameter(key), rtol=0, atol=0
                )


@pytest.mark.parametrize(
    ("write", "error"),
    [
        (lambda dst, src: dst.copy_(src * 2), "Unsupported checkpoint operation"),
        (lambda dst, src: dst.copy_(src.t()), "Unsupported checkpoint slice"),
        (lambda dst, src: dst[:1].copy_(src[:1]), "Incomplete writes"),
        (lambda dst, src: (dst.copy_(src), dst.zero_()), "overlapping writes"),
    ],
)
def test_capture_rejects_unsupported_writes(write, error):
    model = torch.nn.Linear(2, 2, bias=False)
    with pytest.raises(ValueError, match=error):
        with torch.no_grad(), WeightLoadCapture() as capture:
            capture.start(model)
            for _, weight in capture.weights(
                [("source", torch.arange(4.0).view(2, 2))]
            ):
                write(model.weight.data, weight)
            capture.end()
        capture.finish(model)


def test_capture_refuses_replaced_native_parameter():
    model = _FusedModel(0)
    plan = _capture(model, _weights())
    model.output = torch.nn.Parameter(model.output.clone(), requires_grad=False)
    with pytest.raises(ValueError, match="changed the storage or layout of output"):
        plan.bind(model)


def test_capture_keeps_checkpoint_coordinates_when_loader_squeezes_weight():
    model = torch.nn.Linear(2, 3, bias=False)
    checkpoint = torch.arange(12.0).view(3, 2, 2)
    with torch.no_grad(), WeightLoadCapture() as capture:
        capture.start(model)
        for _, weight in capture.weights([("weight", checkpoint)]):
            model.weight.copy_(weight.select(1, 1))
        capture.end()

    plan = capture.finish(model)
    metadata = plan.metadata(model)[0]
    assert metadata["global_offset"] == (0, 1, 0)
    assert metadata["shape"] == (3, 1, 2)
    torch.testing.assert_close(plan.bind(model)["weight"], checkpoint[:, 1:2])


def test_capture_refreshes_parameters_derived_after_loading():
    model = torch.nn.Linear(2, 3, bias=False)
    checkpoint = torch.arange(6.0).view(3, 2)
    with torch.no_grad(), WeightLoadCapture() as capture:
        capture.start(model)
        for _, weight in capture.weights([("weight", checkpoint)]):
            model.weight.copy_(weight)
        capture.end()
        model.derived = torch.nn.Parameter(model.weight * 2, requires_grad=False)

    def prepare_refresh(names):
        assert names == {"derived"}

        def refresh(target):
            target.derived.copy_(target.weight * 2)

        return refresh

    plan = capture.finish(model, prepare_refresh)
    assert set(plan.bind(model)) == {"weight"}
    assert [param["name"] for param in plan.metadata(model)] == ["weight"]
    plan.bind(model)["weight"].copy_(checkpoint + 1)
    plan.refresh(model)
    torch.testing.assert_close(model.derived, (checkpoint + 1) * 2)


def test_mfsdp_loader_borrows_parameters_for_each_publication():
    calls = []
    module = SimpleNamespace(
        all_gather_pipeline=object(),
        data_parallel_sharding_strategy="optim_grads_params",
        start_param_sync=lambda **kw: calls.append(("gather", kw["force_sync"])),
        synchronize_param_gather=lambda: calls.append("reshard"),
    )
    source = MegatronLoadSource([SimpleNamespace(module=module)])
    assert source.needs_rebinding
    with source.publication():
        calls.append("publish")
    with pytest.raises(RuntimeError, match="failed update"):
        with source.publication():
            raise RuntimeError("failed update")
    assert calls == [("gather", True), "publish", "reshard", ("gather", True), "reshard"]


def test_captured_metadata_rejects_incomplete_checkpoint_coverage():
    models = [_FusedModel(rank) for rank in range(2)]
    workers = [
        {
            "rank_info": get_vllm_rank_info(_context(rank, 2), engine_rank=0),
            "params_meta": _capture(model, _weights()).metadata(model),
        }
        for rank, model in enumerate(models)
    ]
    workers[1]["params_meta"] = [
        param for param in workers[1]["params_meta"] if param["name"] != "query"
    ]
    with pytest.raises(ValueError, match="Incomplete checkpoint coverage for query"):
        build_captured_parameters_meta(workers)


def test_vllm_loader_captures_inside_default_startup(monkeypatch):
    class DefaultModelLoader:
        def __init__(self, load_config):
            assert load_config.load_format == "auto"

        def load_model(self, vllm_config, model_config):
            model = _FusedModel(0)
            self.load_weights(model, model_config)
            if vllm_config.mutate_after_load:
                model.output.zero_()
            return model

        def load_weights(self, model, model_config):
            model.load_weights(self.get_all_weights(model_config, model))

        def get_all_weights(self, model_config, model):
            yield from _weights().items()

    registered = {}
    model_executor = ModuleType("vllm.model_executor")
    model_executor.model_loader = SimpleNamespace(
        register_model_loader=lambda name: (
            lambda loader: registered.setdefault(name, loader)
        ),
        DefaultModelLoader=DefaultModelLoader,
    )
    vllm = ModuleType("vllm")
    vllm.model_executor = model_executor
    monkeypatch.setitem(sys.modules, "vllm", vllm)
    monkeypatch.setitem(sys.modules, "vllm.model_executor", model_executor)
    register_capture_loader()
    config = SimpleNamespace(load_format="shardfly_capture")
    loader = registered["shardfly_capture"](config)
    model_config = SimpleNamespace(quantization=None)
    vllm_config = SimpleNamespace(speculative_config=None, mutate_after_load=False)
    with torch.inference_mode():
        model = loader.load_model(vllm_config, model_config)
        assert config.load_format == "shardfly_capture"
        assert set(getattr(model, CAPTURE_ATTRIBUTE).bind(model)) == set(_weights())
        vllm_config.mutate_after_load = True
        with pytest.raises(ValueError, match="modified output after loading"):
            loader.load_model(vllm_config, model_config)
