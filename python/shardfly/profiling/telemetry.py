"""Emit publication timings and read host RDMA counters."""

import json
import logging
import os
import socket
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any


def profile_enabled() -> bool:
    return os.environ.get("SHARDFLY_PROFILE", "0").lower() in {"1", "true", "yes"}


def profile_phase(step_id: int) -> str:
    warmup_updates = int(os.environ.get("SHARDFLY_PROFILE_WARMUP_UPDATES", "0"))
    update_index = int(step_id) + 1
    return "warmup" if update_index < warmup_updates else "measure"


def emit_profile(logger, **values: Any) -> None:
    if not profile_enabled():
        return
    record = (
        "SHARDFLY_PROFILE "
        + json.dumps(values, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode()
    try:
        # One write keeps records intact when torchrun ranks share stdout.
        os.write(1, record)
    except OSError:
        logger.info("%s", record.decode().rstrip())


def _rdma_stats(netdev):
    for attempt in range(2):
        try:
            return subprocess.check_output(
                ["ethtool", "-S", netdev], text=True, timeout=15
            ), attempt + 1
        except subprocess.TimeoutExpired:
            if attempt:
                raise
            logging.getLogger(__name__).warning(
                "RDMA counter read timed out on %s; retrying once", netdev
            )


def node_rdma_counters(root="/sys/class/infiniband"):
    from shardfly.topology import _active_rdma_endpoints

    started_ns = time.time_ns()
    def capture(endpoint):
        device = Path(root) / endpoint.name
        interfaces = list((device / "device/net").iterdir())
        if len(interfaces) != 1:
            raise ValueError(f"Ambiguous HCA/netdev mapping: {endpoint.name}")
        netdev = interfaces[0].name
        before = time.time_ns()
        text, attempts = _rdma_stats(netdev)
        raw = {}
        for line in text.splitlines():
            if ":" not in line:
                continue
            name, value = line.strip().split(":", 1)
            if name in ("tx_vport_rdma_unicast_bytes", "rx_vport_rdma_unicast_bytes"):
                raw[name] = int(value)
        if len(raw) != 2:
            raise ValueError(f"RDMA byte counters unavailable on {netdev}")
        return endpoint.name + ":" + str(endpoint.port), {
            "tx_bytes": raw["tx_vport_rdma_unicast_bytes"],
            "rx_bytes": raw["rx_vport_rdma_unicast_bytes"],
            "netdev": netdev, "source": "ethtool RDMA unicast bytes",
            "read_attempts": attempts,
            "started_ns": before, "ended_ns": time.time_ns(),
        }
    with ThreadPoolExecutor(max_workers=4) as executor:
        ports = dict(executor.map(capture, _active_rdma_endpoints()))
    return {"hostname": socket.gethostname(), "pid": os.getpid(),
            "started_ns": started_ns, "ended_ns": time.time_ns(), "rdma": ports}
