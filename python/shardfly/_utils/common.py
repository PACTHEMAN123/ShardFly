# Copyright 2026 ShardFly contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Portions derived from the Awex project (https://github.com/inclusionAI/Awex),
# Copyright the Awex developers, licensed under the Apache License, Version 2.0.


import math
import os
import pickle
import socket
import struct
from enum import Enum

import torch

from shardfly import logging

logger = logging.getLogger(__name__)


def _is_allowed_infer_only_alias(
    extra_key: str, infer_keys: set[str], train_keys: set[str]
) -> bool:
    """Allow the scale tensor that transport generates for an FP8 destination."""
    if os.environ.get("SHARDFLY_FP8_BLOCKWISE", "0") == "1" and extra_key.endswith(
        ".weight_scale_inv"
    ):
        weight_key = extra_key.removesuffix("_scale_inv")
        return weight_key in infer_keys and weight_key in train_keys
    return False


def ensure_divisibility(numerator, denominator):
    """Ensure that numerator is divisible by the denominator."""
    assert numerator % denominator == 0, (
        f"{numerator} is not divisible by {denominator}"
    )


def divide(numerator, denominator):
    """Ensure that numerator is divisible by the denominator and return
    the division value."""
    ensure_divisibility(numerator, denominator)
    return numerator // denominator


def get_ip_address():
    override = os.environ.get("SHARDFLY_MASTER_ADDR")
    if override:
        return override
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("8.8.8.8", 80))
            return s.getsockname()[0]
    except Exception:
        return socket.gethostbyname(socket.gethostname())


def get_free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def to_binary(data):
    pickled_data = pickle.dumps(data)
    data_len = len(pickled_data)
    return struct.pack("!I", data_len) + pickled_data


def from_binary(binary):
    data_len = struct.unpack("!I", binary[:4])[0]
    pickled_data = binary[4 : 4 + data_len]
    data = pickle.loads(pickled_data)
    return data


def to_dict(param_meta, ignore_keys=None) -> dict:
    """Convert the parameter meta to a dict."""
    ignore_keys = ignore_keys or set()

    def convert_value(v):
        if isinstance(v, Enum):
            return v.value
        if isinstance(v, (tuple, list)):
            return [convert_value(x) for x in v]
        if isinstance(v, torch.dtype):
            return str(v)
        if isinstance(v, slice):
            return str(v)
        if isinstance(v, dict):
            return {k: convert_value(v) for (k, v) in v.items() if k not in ignore_keys}
        if hasattr(v, "__dict__"):
            return {
                k: convert_value(v)
                for (k, v) in v.__dict__.items()
                if not k.startswith("_") and k not in ignore_keys
            }
        if hasattr(v, "__slots__"):
            return {
                k: convert_value(getattr(v, k))
                for k in v.__slots__
                if not k.startswith("_") and k not in ignore_keys
            }
        return v

    param_dict = convert_value(param_meta)
    return param_dict


def compute_statistics(stage_history: dict, step_id: int, duration: float, stage: str):
    if stage not in stage_history:
        stage_history[stage] = []
    history = stage_history[stage]
    history.append(duration)
    if len(history) > 10000:
        history.pop(0)
    if step_id == 2 and len(history) > 1:
        history.pop(history.index(max(history)))
    num_updates = len(history)
    stage_history[stage] = history = sorted(history)
    avg_time = sum(history) / num_updates
    median_time = history[num_updates // 2]
    max_time = history[-1]
    min_time = history[0]
    logger.info(
        f"{stage} time statistics for step {step_id}: average time: {avg_time:.4f} seconds, median time: {median_time:.4f} seconds, min time: {min_time:.4f} seconds,  max time: {max_time:.4f} seconds"
    )


def pretty_bytes(size_bytes):
    if size_bytes == 0:
        return "0B"
    units = ["B", "KB", "MB", "GB", "TB"]
    index = int(math.floor(math.log(size_bytes, 1024)))
    power = math.pow(1024, index)
    converted_size = round(size_bytes / power, 2)
    return f"{converted_size} {units[index]}"


def stripped_env_vars():
    keys = (
        "RANK",
        "LOCAL_RANK",
        "WORLD_SIZE",
        "LOCAL_WORLD_SIZE",
        "CUDA_VISIBLE_DEVICES",
        "CUDA_DEVICE_MAX_CONNECTIONS",
        "SHARDFLY_NODE_LOCAL_RANK_OFFSET",
        "SHARDFLY_NODE_LOCAL_WORLD_SIZE",
        "SHARDFLY_NODE_LOCAL_GPU_IDS",
        "SHARDFLY_HCA_POLICY",
        "SHARDFLY_SELECTED_HCA_BANDWIDTH_GBPS",
        "NCCL_IB_HCA",
        "NCCL_IB_GID_INDEX",
        "NCCL_IB_DISABLE",
        "NCCL_NET",
        "NCCL_SOCKET_IFNAME",
        "NCCL_CROSS_NIC",
        "NCCL_IGNORE_CPU_AFFINITY",
    )
    bindings = {key: os.environ[key] for key in keys if key in os.environ}
    if hasattr(os, "sched_getaffinity"):
        bindings["process_cpu_affinity"] = sorted(os.sched_getaffinity(0))
    return bindings


class AttrDict(dict):
    """Dictionary that allows attribute-style access."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__dict__ = self


def simple_hf_config(hf_config):
    return AttrDict(**hf_config.to_dict())
