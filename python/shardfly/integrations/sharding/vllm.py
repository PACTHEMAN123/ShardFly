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

from shardfly.metadata.rank import RankInfo


def get_vllm_expert_coordinates(enable_expert_parallel, get_ep_group):
    # vLLM allocates the group even when experts use TP instead of EP.
    if not enable_expert_parallel:
        return 0, 1
    try:
        group = get_ep_group()
        return group.rank_in_group, group.world_size
    except AssertionError:
        return 0, 1


def get_vllm_rank_info(model_context, engine_rank) -> RankInfo:
    tp_size = model_context["tp_size"]
    tp_rank = model_context["tp_rank"]
    dp_size = model_context["dp_size"]
    dp_rank = model_context["dp_rank"]
    ep_size = model_context.get("ep_size", 1)
    ep_rank = model_context.get("ep_rank", 0)
    # vLLM expert parallel groups are initialized independently of TP.
    # Use explicit ep_tp_size/ep_tp_rank when provided; otherwise treat EP as a
    # standalone dimension (ep_tp_size=1) to avoid tp_size < ep_size issues.
    ep_tp_size = model_context.get("ep_tp_size", 1)
    ep_tp_rank = model_context.get("ep_tp_rank", 0)
    if ep_size <= 1:
        ep_rank = 0
        ep_tp_size = 1
        ep_tp_rank = 0
    cp_size = int(model_context.get("cp_size", 1) or 1)
    cp_rank = int(model_context.get("cp_rank", 0) or 0)
    cp_mode = model_context.get("cp_mode")
    if not cp_mode:
        cp_mode = "ring" if cp_size > 1 else "none"
    return RankInfo(
        tp_rank=tp_rank,
        tp_size=tp_size,
        pp_rank=model_context["pp_rank"],
        pp_size=model_context["pp_size"],
        dp_size=dp_size,
        dp_rank=dp_rank,
        ep_rank=ep_rank,
        ep_size=ep_size,
        ep_tp_rank=ep_tp_rank,
        ep_tp_size=ep_tp_size,
        attn_tp_rank=model_context["attn_tp_rank"],
        attn_tp_size=model_context["attn_tp_size"],
        attn_dp_rank=model_context["attn_dp_rank"],
        world_size=model_context["world_size"],
        global_rank=model_context["global_rank"],
        engine_rank=engine_rank,
        local_rank=model_context["local_rank"],
        is_infer=True,
        cp_rank=cp_rank,
        cp_size=cp_size,
        cp_mode=cp_mode,
    )
