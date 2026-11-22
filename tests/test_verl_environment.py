from shardfly.topology import configure_ray_locality


def test_single_gpu_ray_actor_uses_physical_gpu_as_local_rank_offset():
    env = {
        "SHARDFLY_HCA_POLICY": "balanced",
        "CUDA_VISIBLE_DEVICES": "7",
        "LOCAL_RANK": "0",
        "LOCAL_WORLD_SIZE": "8",
        "RAY_LOCAL_WORLD_SIZE": "8",
    }

    configure_ray_locality(env)

    assert env["SHARDFLY_NODE_LOCAL_RANK_OFFSET"] == "7"
    assert env["SHARDFLY_NODE_LOCAL_WORLD_SIZE"] == "8"


def test_multi_gpu_worker_keeps_native_local_rank_mapping():
    env = {
        "SHARDFLY_HCA_POLICY": "balanced",
        "CUDA_VISIBLE_DEVICES": "0,1,2,3",
        "LOCAL_RANK": "2",
        "LOCAL_WORLD_SIZE": "4",
    }

    configure_ray_locality(env)

    assert "SHARDFLY_NODE_LOCAL_RANK_OFFSET" not in env


def test_vllm_worker_maps_subset_rank_to_node_local_gpu():
    env = {
        "SHARDFLY_HCA_POLICY": "balanced",
        "CUDA_VISIBLE_DEVICES": "4,5,6,7",
        "RAY_LOCAL_WORLD_SIZE": "8",
    }

    configure_ray_locality(env, worker_local_rank=2)

    assert env["LOCAL_RANK"] == "2"
    assert env["SHARDFLY_NODE_LOCAL_RANK_OFFSET"] == "4"
    assert env["SHARDFLY_NODE_LOCAL_WORLD_SIZE"] == "8"


def test_explicit_local_rank_offset_is_preserved():
    env = {
        "SHARDFLY_HCA_POLICY": "balanced",
        "SHARDFLY_NODE_LOCAL_RANK_OFFSET": "3",
        "CUDA_VISIBLE_DEVICES": "7",
        "LOCAL_RANK": "0",
        "LOCAL_WORLD_SIZE": "8",
    }

    configure_ray_locality(env)

    assert env["SHARDFLY_NODE_LOCAL_RANK_OFFSET"] == "3"


def test_vllm_worker_updates_local_rank_with_explicit_engine_offset():
    env = {
        "SHARDFLY_HCA_POLICY": "balanced",
        "SHARDFLY_NODE_LOCAL_RANK_OFFSET": "4",
        "SHARDFLY_NODE_LOCAL_WORLD_SIZE": "8",
        "CUDA_VISIBLE_DEVICES": "4,5,6,7",
        "LOCAL_RANK": "0",
    }

    configure_ray_locality(env, worker_local_rank=2)

    assert env["LOCAL_RANK"] == "2"
    assert env["SHARDFLY_NODE_LOCAL_RANK_OFFSET"] == "4"
    assert env["SHARDFLY_NODE_LOCAL_WORLD_SIZE"] == "8"


def test_topology_policy_does_not_rewrite_ray_locality():
    env = {
        "SHARDFLY_HCA_POLICY": "topology",
        "CUDA_VISIBLE_DEVICES": "7",
        "LOCAL_RANK": "0",
        "LOCAL_WORLD_SIZE": "8",
    }

    configure_ray_locality(env)

    assert "SHARDFLY_NODE_LOCAL_RANK_OFFSET" not in env
