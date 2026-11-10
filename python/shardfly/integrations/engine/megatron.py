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


import os
import time
from typing import Any, Dict, List, Optional

import torch.distributed as dist

from shardfly import logging
from shardfly._utils import device as device_util
from shardfly.integrations.engine.base import TrainingEngine
from shardfly.integrations.writer.base import get_weights_exchange_writer

logger = logging.getLogger(__name__)


class MegatronEngine(TrainingEngine):
    def __init__(self, config: Dict[str, Any], hf_config, model):
        super().__init__(hf_config)
        self.config = config
        logger.info(f"config {self.config}")
        self.model = model
        self.enable_debug_mode = self.config.get("enable_debug_mode", False)
        self.comm_backend = self.config.get("comm_backend", "transport")
        self.meta_server_addr = self.config.get("meta_server_addr", "")
        os.environ["SHARDFLY_META_SERVER_ADDR"] = self.meta_server_addr or ""
        (ip, port) = (self.meta_server_addr or ":").split(":")
        os.environ["SHARDFLY_META_SERVER_IP"] = ip
        os.environ["SHARDFLY_META_SERVER_PORT"] = port
        self.weights_exchange_writer = None

    @property
    def engine_name(self):
        return "mcore"

    def initialize(self) -> None:
        self.weights_exchange_writer = get_weights_exchange_writer(self)
        self.weights_exchange_writer.initialize()
        device = device_util.current_device()
        logger.info(f"Finish initialize on device {device}")

    def write_weights(self, **kwargs):
        logger.info(
            f"Start to write weights for step {self.global_step} for rank {dist.get_rank()}"
        )
        start_time = time.time()
        self.weights_exchange_writer.write_weights(step_id=self.global_step, **kwargs)
        duration = time.time() - start_time
        logger.info(
            f"Finished writing weights for step {self.global_step} for rank {dist.get_rank()}, took {duration:.3f} seconds"
        )

    def close(self) -> None:
        if self.weights_exchange_writer is not None:
            self.weights_exchange_writer.close()
            self.weights_exchange_writer = None

    def release_memory_occupation(self, tags: Optional[List[str]] = None) -> None:
        """Release memory occupation.

        tags: optimizer, weights, default is both
        """

    def release_grad_memory(self, empty_cache=True):
        raise NotImplementedError

    def resume_memory_occupation(self, tags: Optional[List[str]] = None) -> None:
        """Resume memory occupation.

        tags: optimizer, weights, default is both
        """
        raise NotImplementedError
