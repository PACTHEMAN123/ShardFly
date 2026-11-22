from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch

from shardfly.capture import capture_inference
from shardfly.integrations import rtp_llm
from shardfly.integrations.rtp_llm import load_weights_with_capture


class _Database:
    def __init__(self):
        self.tensors = {
            "q": torch.arange(12, dtype=torch.bfloat16).reshape(4, 3),
            "k": torch.arange(6, dtype=torch.bfloat16).reshape(2, 3) + 20,
            "norm": torch.ones(3, dtype=torch.bfloat16),
        }

    def get_tensor_shape(self, name):
        return self.tensors[name].shape

    def get_pretrain_tensor_names(self):
        return self.tensors.keys()

    def load_tensor(self, name, data_type=None):
        return [self.tensors[name]]


class _Fused:
    def load(self, source, _layer, _device, _config):
        q, k = [source.load_tensor(name)[0] for name in ("q", "k")]
        padding = torch.zeros((3, 1)).half()
        fused = torch.cat((q[2:].T, k.T, padding), dim=1).contiguous()
        return {"fused": torch.ops.aten.to.dtype_layout(fused, dtype=torch.bfloat16)}


class _Direct:
    def load(self, source, _layer, _device, _config):
        return {"norm": source.load_tensor("norm")[0].clone()}


class _Loader:
    def __init__(self, descriptor):
        self.config = SimpleNamespace(
            database=_Database(), is_ft_style_weight=False, merge_lora=False
        )
        self.units = [(descriptor, 0, None), (_Direct(), None, None)]

    def get_load_config(self):
        return self.config

    def _generate_weight_info(self):
        return {}, self.units

    @torch.inference_mode()
    def load_weights(self, device):
        fused = self.units[0][0].load(self.config.database, 0, device, self.config)
        direct = self.units[1][0].load(self.config.database, None, device, self.config)
        return SimpleNamespace(global_weights=direct, weights=[fused])


def test_capture_reuses_native_storage_and_applies_fused_destinations():
    loader = _Loader(_Fused())
    weights = load_weights_with_capture(loader, "cpu")
    capture = capture_inference(weights)
    snapshot = capture.snapshot()
    assert set(snapshot.tensors) == {("q", (2, 0)), ("k", (0, 0)), ("norm", (0,))}
    fused = weights.weights[0]["fused"]
    original_pointer = fused.data_ptr()
    assert snapshot.tensors["norm", (0,)].data_ptr() == weights.global_weights["norm"].data_ptr()
    with torch.inference_mode():
        for step in range(1, 3):
            for row in snapshot.shards:
                name = row["name"]
                slices = tuple(slice(offset, offset + count) for offset, count in
                               zip(row["global_offset"], row["shape"]))
                current = loader.config.database.tensors[name]
                current.add_(step)
                snapshot.tensors[row["binding_key"]].copy_(current[slices])
            capture.refresh()
            expected = loader.load_weights("cpu")
            torch.testing.assert_close(fused, expected.weights[0]["fused"], rtol=0, atol=0)
            torch.testing.assert_close(weights.global_weights["norm"],
                                       expected.global_weights["norm"], rtol=0, atol=0)
            assert fused.data_ptr() == original_pointer
            assert capture.snapshot() is snapshot


def test_capture_rejects_loader_arithmetic():
    class Scaled(_Fused):
        def load(self, *args):
            return {"fused": super().load(*args)["fused"] * 2}

    with pytest.raises(ValueError, match="cannot replay aten.mul"):
        load_weights_with_capture(_Loader(Scaled()), "cpu")


def test_capture_binds_stacked_expert_matrices():
    class Stacked(_Fused):
        def load(self, source, _layer, _device, _config):
            q, k = [source.load_tensor(name)[0] for name in ("q", "k")]
            return {"fused": torch.stack((q[2:].T.contiguous(), k.T.contiguous()))}

    loader = _Loader(Stacked())
    weights = load_weights_with_capture(loader, "cpu")
    capture = capture_inference(weights)
    snapshot = capture.snapshot()
    assert set(snapshot.tensors) == {("q", (2, 0)), ("k", (0, 0)), ("norm", (0,))}
    with torch.inference_mode():
        snapshot.tensors[("q", (2, 0))].fill_(7)
        snapshot.tensors[("k", (0, 0))].fill_(11)
        capture.refresh()
    assert torch.all(weights.weights[0]["fused"][0] == 7)
    assert torch.all(weights.weights[0]["fused"][1] == 11)


def test_single_worker_initializes_control_group_once(monkeypatch):
    groups = []
    store = object()
    monkeypatch.setattr(rtp_llm, "dist", SimpleNamespace(
        HashStore=lambda: store,
        is_initialized=lambda: bool(groups),
        init_process_group=lambda backend, **kwargs: groups.append((backend, kwargs)),
    ))
    devices = []
    monkeypatch.setattr(torch.cuda, "device", lambda device: devices.append(device) or nullcontext())
    worker = rtp_llm.RtpLlmWorker(
        SimpleNamespace(device="cuda:3"), {"world_size": 1}, lambda: True,
    )
    for _ in range(2):
        assert worker.execute(lambda **kwargs: kwargs["model"]) is worker.weights
    assert groups == [("gloo", {"store": store, "rank": 0, "world_size": 1})]
    assert devices == [torch.device("cuda", 3)] * 2
    assert worker.rank_context["scheduler"].gpu_id == 3
