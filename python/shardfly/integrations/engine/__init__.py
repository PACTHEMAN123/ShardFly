"""Training-side publication over a captured set of checkpoint weights."""

from shardfly.capture import CapturePlan
from shardfly.integrations.writer.worker import TransportWriter


class CapturedTrainingEngine:
    def __init__(self, capture: CapturePlan, *, meta_server_addr: str, **config):
        self.global_step = -1
        self.weights_exchange_writer = TransportWriter(
            capture, dict(config, meta_server_addr=meta_server_addr)
        )

    def set_global_step(self, global_step: int):
        self.global_step = global_step

    def write_weights(self, **kwargs):
        if self.global_step < 0:
            raise ValueError("Set the training step before publishing")
        self.weights_exchange_writer.write_weights(self.global_step, **kwargs)

    def close(self):
        self.weights_exchange_writer.close()
