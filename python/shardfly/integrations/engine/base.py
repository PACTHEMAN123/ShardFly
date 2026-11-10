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


from abc import ABC, abstractmethod
from typing import List, Optional

from shardfly import logging

logger = logging.getLogger(__name__)


class Engine(ABC):
    def __init__(self, hf_config):
        self.global_step = -1
        self.hf_config = hf_config

    @property
    @abstractmethod
    def engine_name(self):
        pass

    @abstractmethod
    def initialize(self) -> None:
        """Initialize the engine with configuration."""
        pass

    def set_global_step(self, global_step: int):
        logger.info(f"set global step from {self.global_step} to {global_step}")
        self.global_step = global_step

    @abstractmethod
    def release_memory_occupation(self, tags: Optional[List[str]] = None) -> None:
        """Release memory occupied by the engine."""
        pass

    @abstractmethod
    def resume_memory_occupation(self, tags: Optional[List[str]] = None) -> None:
        """Resume memory occupation for the engine."""
        pass


class TrainingEngine(Engine):
    @abstractmethod
    def write_weights(self, **kwargs):
        """Write weights for training engine."""
        pass

    @abstractmethod
    def release_grad_memory(self, empty_cache=True):
        pass
