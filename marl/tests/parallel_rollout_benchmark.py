"""Real-environment fixed-action throughput benchmark; no policy update."""

from __future__ import annotations

import argparse
import json
import multiprocessing
import os
import time


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, choices=(1, 2, 4), required=True)
    parser.add_argument("--transitions", type=int, default=96)
    parser.add_argument("--seed", type=int, default=7110)
    args = parser.parse_args()
    if args.transitions % args.workers:
        raise ValueError("transitions must be divisible by workers")
    for name in (
        "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"
    ):
        os.environ[name] = "1"

    import numpy as np

    from marl.checkpointing.environment_state import single_environment_state_dict
    from marl.envs.harl_env_factory import make_harl_train_env

    created = time.perf_counter()
    env = make_harl_train_env(seed=args.seed, n_rollout_threads=args.workers)
    creation_seconds = time.perf_counter() - created
    child_pids = list(getattr(env, "process_ids", ()))
    try:
        reset_started = time.perf_counter()
        env.reset()
        reset_seconds = time.perf_counter() - reset_started
        actions = np.full((args.workers, 2, 22), 0.5, dtype=np.float32)
        vector_steps = args.transitions // args.workers
        episode_segments = []
        rollout_started = time.perf_counter()
        segment_started = rollout_started
        for index in range(vector_steps):
            env.step(actions)
            if (index + 1) % 24 == 0:
                now = time.perf_counter()
                episode_segments.append(now - segment_started)
                segment_started = now
        rollout_seconds = time.perf_counter() - rollout_started
        if hasattr(env, "get_env_states"):
            states = env.get_env_states()
        else:
            states = [single_environment_state_dict(item) for item in env.envs]
        cache = [state["grid"]["cache"] for state in states]
        result = {
            "workers": args.workers,
            "seed": args.seed,
            "worker_seeds": list(env.worker_seeds),
            "vec_env_type": env.vec_env_type,
            "multiprocessing_start_method": env.multiprocessing_start_method,
            "child_process_count": len(child_pids),
            "child_pids": child_pids,
            "transitions": args.transitions,
            "vector_steps": vector_steps,
            "complete_episodes": args.transitions // 24,
            "creation_seconds": creation_seconds,
            "reset_seconds": reset_seconds,
            "rollout_seconds": rollout_seconds,
            "total_seconds": creation_seconds + reset_seconds + rollout_seconds,
            "transitions_per_second_rollout": args.transitions / rollout_seconds,
            "transitions_per_second_total": args.transitions / (creation_seconds + reset_seconds + rollout_seconds),
            "episodes_per_second_rollout": (args.transitions / 24) / rollout_seconds,
            "episode_segment_seconds": episode_segments,
            "opf_hits": sum(int(item["opf_hit_count"]) for item in cache),
            "opf_misses": sum(int(item["opf_miss_count"]) for item in cache),
            "mef_hits": sum(int(item["mef_hit_count"]) for item in cache),
            "mef_misses": sum(int(item["mef_miss_count"]) for item in cache),
            "opf_entries": sum(len(item["opf_entries"]) for item in cache),
            "mef_entries": sum(len(item["mef_entries"]) for item in cache),
            "per_worker_cache": [
                {
                    "opf_hits": int(item["opf_hit_count"]),
                    "opf_misses": int(item["opf_miss_count"]),
                    "mef_hits": int(item["mef_hit_count"]),
                    "mef_misses": int(item["mef_miss_count"]),
                    "opf_entries": len(item["opf_entries"]),
                    "mef_entries": len(item["mef_entries"]),
                }
                for item in cache
            ],
        }
    finally:
        env.close()
    result["workers_alive_after_close"] = list(getattr(env, "alive", ()))
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()

