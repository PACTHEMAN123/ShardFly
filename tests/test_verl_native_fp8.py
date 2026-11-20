"""Native FP8 refits must consume reusable NCCL bucket views synchronously."""
import sys
from types import SimpleNamespace

import torch

from shardfly.integrations.publication import _load_fp8_weights


def test_fp8_native_loader_consumes_views_before_bucket_reuse(monkeypatch):
    events = []
    loaded = []
    parameter = torch.nn.Parameter(torch.ones(1), requires_grad=False)

    class Model:
        def named_parameters(self):
            return [('weight', parameter)]

        def parameters(self):
            return [parameter]

        def load_weights(self, weights):
            for name, tensor in weights:
                events.append(('load', name))
                loaded.append(tensor.clone())

    def quant_weights(weights, model, config, dtype):
        assert events == ['prepare']
        for name, weight in weights:
            yield name, weight * 2

    state = object()

    def prepare(model):
        events.append('prepare')
        return state

    def finish(model, actual_state):
        assert actual_state is state
        events.append('finish')

    monkeypatch.setitem(sys.modules, 'verl.utils.vllm.vllm_quant_utils', SimpleNamespace(
        prepare_quanted_weights_for_loading=prepare,
        process_quanted_weights_after_loading=finish,
        quant_weights=quant_weights,
    ))
    buffer = torch.zeros(4)

    def weights():
        for version in range(4):
            buffer.fill_(version)
            yield str(version), buffer

    runner = SimpleNamespace(model=Model(), vllm_config=SimpleNamespace(
        quant_config=object(), model_config=SimpleNamespace(dtype=torch.bfloat16)))
    _load_fp8_weights(runner, weights())
    assert [tensor.tolist() for tensor in loaded] == [[2 * v] * 4 for v in range(4)]
    assert events == ['prepare', *[('load', str(v)) for v in range(4)], 'finish']


def test_native_fp8_oracle_accepts_only_adjacent_midpoint_roundings(monkeypatch):
    import pytest

    from shardfly.profiling.fp8_model import FP8ModelWeightProfile
    from shardfly.profiling.model import version_marker_factor

    monkeypatch.setattr(torch.cuda, 'synchronize', lambda: None)
    # Amax 448 gives scale 1. 1.0625 is exactly halfway between FP8 1 and
    # 1.125, while 1 is unambiguous. Both sources are exactly BF16 values.
    source = torch.full((128, 128), 1.0625, dtype=torch.bfloat16)
    source[0, 0], source[0, 1] = 448, 1
    parameters = {
        'layer.weight': source.to(torch.float8_e4m3fn),
        'layer.weight_scale_inv': torch.ones(1, 1),
        'model.norm.weight': torch.ones(8, dtype=torch.bfloat16),
    }
    oracle = FP8ModelWeightProfile(parameters, {
        'layer.weight': source, 'model.norm.weight': torch.ones(8, dtype=torch.bfloat16),
    }, 'scale', 35)
    parameters['model.norm.weight'].mul_(version_marker_factor(-1, 35))
    oracle.quantization_rounding = 'fp32_interval'
    parameters['layer.weight'][1, 1] = 1.125
    result = oracle.command('verify', 0)
    assert not result['fp8_bytes_exact']
    assert result['fp8_rounding_mode'] == 'fp32_interval'
    assert result['fp8_relative_bound'] == 4 * torch.finfo(torch.float32).eps
    assert result['fp8_ambiguous_elements'] == source.numel() - 2
    assert result['fp8_alternative_elements'] == 1
    # The next FP8 value is outside the interval, even at a midpoint.
    parameters['layer.weight'][1, 1] = 1.25
    with pytest.raises(AssertionError, match='FP8 byte mismatch'):
        oracle.command('verify', 0)
    parameters['layer.weight'][1, 1] = 1.125
    # A one-step corruption at an unambiguous source must also fail.
    parameters['layer.weight'][0, 1] = 1.125
    with pytest.raises(AssertionError, match='FP8 byte mismatch'):
        oracle.command('verify', 0)
    parameters['layer.weight'][0, 1] = 1
    parameters['layer.weight_scale_inv'].mul_(1.01)
    with pytest.raises(AssertionError, match='scale_inv'):
        oracle.command('verify', 0)
