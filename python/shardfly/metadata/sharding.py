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


from enum import Enum


class ShardingType(Enum):
    """
    Enum representing the type of sharding applied to a parameter.
    """

    NO_SHARDING = "NO_SHARDING"
    TP_SHARDING = "TP_SHARDING"
    DP_TP_SHARDING = "DP_TP_SHARDING"
    EP_SHARDING = "EP_SHARDING"
    EP_TP_SHARDING = "EP_TP_SHARDING"
