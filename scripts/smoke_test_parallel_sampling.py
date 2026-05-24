"""Smoke test CPU synchronous VecEnv sampling for GridCoupledEnv.

Run with:
    python -m scripts.smoke_test_parallel_sampling
    python -m scripts.smoke_test_parallel_sampling --n-envs 2 --vec-env subproc --steps 48
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _ensure_project_root_on_path() -> Path:
    for candidate in Path(__file__).resolve().parents:
        if (candidate / "envs").exists() and (candidate / "grid_model").exists():
            root = candidate
            break
    else:
        raise RuntimeError("Cannot locate project root containing envs/ and grid_model/")

    root_str = str(root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)
    return root


PROJECT_ROOT = _ensure_project_root_on_path()

from configs.config_ultimate import PPO_CONFIG  # noqa: E402
from configs.experiment_cases import get_experiment_case, print_experiment_case  # noqa: E402
from train.train_ppo_ultimate import (  # noqa: E402
    build_vec_env,
    run_random_vec_env_smoke,
    set_cpu_thread_env,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-envs", type=int, default=2)
    parser.add_argument("--vec-env", choices=["dummy", "subproc", "auto"], default="subproc")
    parser.add_argument("--steps", type=int, default=48)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--cpu-threads-per-worker", type=int, default=1)
    parser.add_argument("--start-method", type=str, default="spawn")
    args = parser.parse_args()

    set_cpu_thread_env(args.cpu_threads_per_worker, force=True)

    case_config = get_experiment_case("main")
    print_experiment_case(case_config)

    vec_env = None
    try:
        vec_env = build_vec_env(
            n_envs=args.n_envs,
            vec_env_type=args.vec_env,
            seed=args.seed,
            env_config=case_config["env_config"],
            reward_config=case_config["reward_config"],
            data_config=case_config["data_config"],
            start_method=args.start_method,
            cpu_threads_per_worker=args.cpu_threads_per_worker,
            n_steps=PPO_CONFIG["n_steps"],
            batch_size=PPO_CONFIG["batch_size"],
        )
        run_random_vec_env_smoke(
            vec_env=vec_env,
            n_envs=max(int(args.n_envs), 1),
            steps=args.steps,
            seed=args.seed,
            expected_obs_dim=264,
            expected_action_dim=23,
        )
    finally:
        if vec_env is not None:
            vec_env.close()


if __name__ == "__main__":
    import multiprocessing as mp

    mp.freeze_support()
    main()
