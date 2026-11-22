import importlib
import sys
from importlib.machinery import ModuleSpec
from types import ModuleType

import pytest

from shardfly.integrations import vllm


@pytest.fixture(autouse=True)
def clear_worker_extension_cache():
    vllm._load_worker_extension.cache_clear()
    yield
    vllm._load_worker_extension.cache_clear()


@pytest.mark.parametrize("enabled", [None, "0", "1"])
def test_spawned_worker_resolves_extension_and_only_applies_opt_in_fp8(monkeypatch, enabled):
    from shardfly.integrations import loader_capture

    calls = []
    plugin = ModuleType("shardfly.integrations.vllm_plugin")
    plugin._patch_shardfly_worker = lambda: calls.append("patch")
    quant = ModuleType("verl.utils.vllm.vllm_quant_utils")
    quant.apply_vllm_quant_patches = lambda: calls.append("fp8")
    monkeypatch.setitem(sys.modules, plugin.__name__, plugin)
    monkeypatch.setitem(sys.modules, quant.__name__, quant)
    monkeypatch.setattr(loader_capture, "register_capture_loader", lambda: calls.append("loader"))
    if enabled is None:
        monkeypatch.delenv("VERL_VLLM_FP8_QUANT_ENABLED", raising=False)
    else:
        monkeypatch.setenv("VERL_VLLM_FP8_QUANT_ENABLED", enabled)

    module_name, cls_name = "shardfly.integrations.vllm.ShardFlyWorkerExtension".rsplit(".", 1)
    module = importlib.import_module(module_name)
    assert calls == []
    extension = getattr(module, cls_name)
    assert getattr(module, cls_name) is extension
    assert calls == (["fp8", "loader", "patch"] if enabled == "1" else ["loader", "patch"])


def test_general_plugin_registers_loader_and_routes_outside_model_inspection(monkeypatch):
    from shardfly.integrations import loader_capture

    calls = []
    plugin = ModuleType("shardfly.integrations.vllm_plugin")
    plugin.register_shardfly_plugin = lambda: calls.append("routes")
    monkeypatch.setitem(sys.modules, plugin.__name__, plugin)
    monkeypatch.setattr(loader_capture, "register_capture_loader", lambda: calls.append("loader"))
    monkeypatch.setattr(
        sys.modules["__main__"], "__spec__",
        ModuleSpec("vllm.model_executor.models.registry", loader=None),
    )
    vllm.register_shardfly_plugin()
    assert calls == []
    monkeypatch.setattr(sys.modules["__main__"], "__spec__", None)
    vllm.register_shardfly_plugin()
    assert calls == ["loader", "routes"]
