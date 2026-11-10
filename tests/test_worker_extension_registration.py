import runpy
import sys
import types
from pathlib import Path

import pytest


def test_spawned_worker_patches_shardfly(monkeypatch):
    calls = []
    module = types.ModuleType("shardfly.integrations.vllm.plugin")
    module._patch_shardfly_worker = lambda: calls.append("_patch_shardfly_worker")
    monkeypatch.setitem(sys.modules, "shardfly.integrations.vllm.plugin", module)
    monkeypatch.delenv("VERL_VLLM_FP8_QUANT_ENABLED", raising=False)
    runpy.run_path(
        str(
            Path(__file__).parents[1]
            / "python/shardfly/integrations/vllm/worker_extension.py"
        )
    )
    assert calls == ["_patch_shardfly_worker"]


@pytest.mark.parametrize("enabled", [None, "0", "1"])
def test_verl_fp8_patch_is_opt_in(monkeypatch, enabled):
    calls = []
    patch_module = types.ModuleType("shardfly.integrations.vllm.plugin")
    patch_module._patch_shardfly_worker = lambda: None
    quant_module = types.ModuleType("verl.utils.vllm.vllm_quant_utils")
    quant_module.apply_vllm_quant_patches = lambda: calls.append("apply")
    monkeypatch.setitem(sys.modules, "shardfly.integrations.vllm.plugin", patch_module)
    monkeypatch.setitem(
        sys.modules, "verl.utils.vllm.vllm_quant_utils", quant_module
    )
    if enabled is None:
        monkeypatch.delenv("VERL_VLLM_FP8_QUANT_ENABLED", raising=False)
    else:
        monkeypatch.setenv("VERL_VLLM_FP8_QUANT_ENABLED", enabled)
    runpy.run_path(
        str(
            Path(__file__).parents[1]
            / "python/shardfly/integrations/vllm/worker_extension.py"
        )
    )
    assert calls == (["apply"] if enabled == "1" else [])
