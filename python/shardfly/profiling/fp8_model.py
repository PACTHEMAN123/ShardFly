"""Independent checkpoint oracle for opt-in native FP8 model benchmarks.

Checkpoint reads, reference quantization, clearing and comparisons are outside
the timed publication. Expected bytes are computed from BF16 checkpoint data,
never copied from the receiver. Scales are checked against an independent amax
reference before using their FP32 rounding to define the byte oracle.
"""

import json
from contextlib import ExitStack
from pathlib import Path

import torch

from shardfly.profiling.model import ModelWeightProfile, version_marker_factor


class FP8ModelWeightProfile(ModelWeightProfile):
    def __init__(self, parameters, references, marker_mode="scale", marker_period=None,
                 block_shape=(128, 128)):
        if marker_mode not in ("constant", "scale"):
            raise ValueError("Unknown final norm marker mode")
        version_marker_factor(0, marker_period)
        self.parameters = dict(parameters)
        self.references = dict(references)
        self.marker_mode, self.marker_period = marker_mode, marker_period
        self.block_shape = block_shape
        self.compute = None
        self.norm_name = "model.norm.weight"
        self.encoded = {}
        self.encoding_scales = {}
        self.quantization_rounding = "exact"
        self.rounding_alternatives = {}
        self.quantized = {}
        expected_names = set(self.references)
        for name, actual in self.parameters.items():
            if actual.dtype != torch.float8_e4m3fn:
                continue
            br, bc = block_shape
            if actual.ndim != 2 or actual.shape[0] % br or actual.shape[1] % bc:
                raise ValueError(f"FP8 reference requires complete blocks: {name}")
            source = self.references[name]
            if source.dtype != torch.bfloat16 or source.shape != actual.shape:
                raise ValueError(f"No matching BF16 checkpoint reference: {name}")
            scale_name = name + "_scale_inv"
            if scale_name not in self.parameters:
                raise ValueError(f"No FP8 block scales: {name}")
            blocks = source.float().reshape(source.shape[0] // br, br,
                                           source.shape[1] // bc, bc)
            scales = blocks.abs().amax(dim=(1, 3)).clamp_min(1e-12) / 448.0
            if (self.parameters[scale_name].dtype != torch.float32
                    or self.parameters[scale_name].shape != scales.shape):
                raise ValueError(f"Invalid FP8 scale storage: {scale_name}")
            self.references[scale_name] = scales
            expected_names.add(scale_name)
            self.quantized[name] = scale_name
        if not self.quantized or expected_names != set(self.parameters):
            raise ValueError("Checkpoint references must cover every FP8, scale and mixed parameter")
        if self.norm_name not in self.references or self.parameters[self.norm_name].dtype not in (
            torch.bfloat16, torch.float32
        ):
            raise ValueError("No unquantized final norm version marker")
        for name, actual in self.parameters.items():
            if actual.dtype not in (torch.float8_e4m3fn, torch.bfloat16, torch.float32):
                raise ValueError(f"Unsupported benchmark parameter dtype: {name}")
            if actual.shape != self.references[name].shape:
                raise ValueError(f"Checkpoint reference shape mismatch: {name}")
        self.pointers = {name: p.data_ptr() for name, p in self.parameters.items()}

    @classmethod
    def from_checkpoint(cls, reader, checkpoint, marker_mode, marker_period):
        from safetensors import safe_open

        checkpoint = Path(checkpoint)
        index = json.loads((checkpoint / "model.safetensors.index.json").read_text())["weight_map"]
        parameters = dict(reader.parameters)
        metadata = {m.name: m for m in reader.current_worker_parameters_meta if m.shards}
        if set(metadata) != set(parameters):
            raise ValueError("FP8 checkpoint oracle must cover the complete local catalog")
        references = {}
        with ExitStack() as stack:
            files = {}
            for key, meta in metadata.items():
                name = key
                if name.endswith("_scale_inv"):
                    continue
                if len(meta.shards) != 1:
                    raise ValueError(f"Expected one canonical shard: {key}")
                shard = meta.shards[0]
                filename = index[name]
                if filename not in files:
                    files[filename] = stack.enter_context(safe_open(
                        checkpoint / filename, framework="pt", device="cpu"))
                data = files[filename].get_slice(name)
                shape = data.get_shape()
                slices = tuple(slice(offset, offset + size)
                               for offset, size in zip(shard.global_offset, shard.shape))
                actual = parameters[key]
                dtype = torch.bfloat16 if actual.dtype == torch.float8_e4m3fn else actual.dtype
                loaded = data[slices].to(device=actual.device, dtype=dtype)
                if tuple(loaded.shape) != tuple(shard.shape) or len(shape) != len(shard.shape):
                    raise ValueError(f"Checkpoint does not cover the recorded shard: {name}")
                references[key] = loaded
        profile = cls(parameters, references, marker_mode, marker_period)
        torch.cuda.synchronize()
        return profile

    def _verify_parameters(self, version):
        br, bc = self.block_shape
        ambiguous_elements = 0
        alternative_elements = 0
        native_rounding = self.quantization_rounding == "fp32_interval"
        relative_bound = 4 * torch.finfo(torch.float32).eps
        for name, scale_name in self.quantized.items():
            actual_scale = self.parameters[scale_name]
            torch.testing.assert_close(actual_scale, self.references[scale_name],
                                       rtol=5e-7, atol=0, msg=scale_name)
            if name not in self.encoded or not torch.equal(actual_scale, self.encoding_scales[name]):
                source = self.references[name]
                blocks = source.float().reshape(source.shape[0] // br, br,
                                               source.shape[1] // bc, bc)
                if native_rounding:
                    # Use independent FP64 arithmetic with a tight FP32 error
                    # envelope, rather than rerunning the upstream kernel.
                    # The scales have already passed the independent amax check.
                    ideal = blocks.double() / actual_scale.double()[:, None, :, None]
                    lower = (ideal * (1 - relative_bound)).clamp(-448, 448).float().reshape(
                        source.shape).to(torch.float8_e4m3fn).view(torch.uint8)
                    upper = (ideal * (1 + relative_bound)).clamp(-448, 448).float().reshape(
                        source.shape).to(torch.float8_e4m3fn).view(torch.uint8)
                    indices = (lower.flatten() != upper.flatten()).nonzero().flatten()
                    self.encoded[name] = lower
                    self.rounding_alternatives[name] = (indices, upper.flatten()[indices])
                else:
                    # Preserve the GIN exact-byte checkpoint oracle.
                    inverse = (1.0 / actual_scale.double()).float()
                    self.encoded[name] = (blocks * inverse[:, None, :, None]).reshape(
                        source.shape).to(torch.float8_e4m3fn).view(torch.uint8)
                self.encoding_scales[name] = actual_scale.detach().clone()
            actual_bytes = self.parameters[name].view(torch.uint8)
            if native_rounding:
                indices, alternatives = self.rounding_alternatives[name]
                invalid = actual_bytes.flatten() != self.encoded[name].flatten()
                alternate = invalid[indices] & (actual_bytes.flatten()[indices] == alternatives)
                invalid[indices] &= ~alternate
                ambiguous_elements += indices.numel()
                alternative_elements += int(alternate.sum())
                mismatch = bool(invalid.any())
            else:
                mismatch = not torch.equal(actual_bytes, self.encoded[name])
            if mismatch:
                raise AssertionError(f"FP8 byte mismatch at version {version}: {name}")
        for name, actual in self.parameters.items():
            if name in self.quantized or name in self.quantized.values():
                continue
            expected = self.references[name]
            if name == self.norm_name:
                factor = version_marker_factor(version - 1, self.marker_period)
                expected = (expected * factor if self.marker_mode == "scale"
                            else torch.full_like(expected, factor))
            if not torch.equal(actual, expected):
                raise AssertionError(f"Mixed parameter mismatch at version {version}: {name}")
        rounding = (dict(fp8_rounding_mode="fp32_interval", fp8_relative_bound=relative_bound,
                         fp8_ambiguous_elements=ambiguous_elements,
                         fp8_alternative_elements=alternative_elements)
                    if native_rounding else {})
        return dict(full_checkpoint_reference=True, fp8_bytes_exact=not native_rounding,
                    **rounding,
                    scale_rtol=5e-7, scale_atol=0,
                    fp8_elements=sum(self.parameters[n].numel() for n in self.quantized),
                    scale_elements=sum(self.parameters[n].numel() for n in self.quantized.values()),
                    bf16_fp32_elements=sum(p.numel() for n, p in self.parameters.items()
                                           if n not in self.quantized and n not in self.quantized.values()))
