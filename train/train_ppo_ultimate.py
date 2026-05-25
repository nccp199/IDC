"""Train PPO for IDCPriceEnv20D experiment cases.

Examples:
    python -m train.train_ppo_ultimate --case main --timesteps 10000 --run-name smoke
    python -m train.train_ppo_ultimate --parallel-smoke --n-envs 2 --vec-env subproc --parallel-smoke-steps 48
    python -m train.train_ppo_ultimate --case main --timesteps 500000 --run-name report
"""

from __future__ import annotations

import argparse
import json
import math
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np


THREAD_ENV_VARS = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
)


def set_cpu_thread_env(cpu_threads_per_worker: int, force: bool = True) -> None:
    value = str(max(int(cpu_threads_per_worker), 1))
    for name in THREAD_ENV_VARS:
        if force or name not in os.environ:
            os.environ[name] = value


# Keep numerical libraries from fanning out before argparse has a chance to run.
set_cpu_thread_env(1, force=False)

from configs.config_ultimate import (  # noqa: E402
    DEFAULT_EVAL_SEED,
    GRID_CACHE_CONFIG,
    GRID_CONFIG,
    GRID_REWARD_CONFIG,
    GRID_SCENARIO_CONFIG,
    IDC_SCALE_CONFIG,
    PPO_CONFIG,
    resolve_output_path,
)
from configs.experiment_cases import (  # noqa: E402
    get_experiment_case,
    key_env_config,
    key_reward_config,
    print_experiment_case,
)
from data_io.data_loader import build_external_series_from_config  # noqa: E402
from env_wrappers import GridCoupledEnv  # noqa: E402
from envs.idc_price_env import IDCPriceEnv20D  # noqa: E402


def _worker_seed(seed: Optional[int], rank: int) -> Optional[int]:
    return None if seed is None else int(seed) + int(rank)


def make_single_env(
    env_config: Dict[str, Any],
    reward_config: Dict[str, Any],
    data_config: Dict[str, Any],
    seed: Optional[int] = None,
    rank: int = 0,
    verbose: bool = False,
    monitor: bool = True,
    grid_cache_config: Optional[Dict[str, Any]] = None,
) -> Any:
    """Create one IDC + GridCoupledEnv inside the current process."""
    worker_seed = _worker_seed(seed, rank)
    env_kwargs = {
        **env_config,
        **reward_config,
        **IDC_SCALE_CONFIG,
        **build_external_series_from_config(data_config, env_config["horizon"]),
        "server_seed": worker_seed,
        "task_seed": worker_seed,
    }
    base_env = IDCPriceEnv20D(**env_kwargs)
    env = GridCoupledEnv(
        base_env,
        GRID_CONFIG,
        GRID_REWARD_CONFIG,
        GRID_SCENARIO_CONFIG,
        grid_cache_config=grid_cache_config,
    )
    if worker_seed is not None:
        env.action_space.seed(worker_seed)
        env.observation_space.seed(worker_seed)
    if verbose:
        print(f"[env rank={rank}] seed={worker_seed}")
    if not monitor:
        return env

    try:
        from stable_baselines3.common.monitor import Monitor
    except ModuleNotFoundError as exc:
        if exc.name == "stable_baselines3":
            return env
        raise

    return Monitor(env)


def make_env(
    env_config: Dict[str, Any],
    reward_config: Dict[str, Any],
    data_config: Dict[str, Any],
    seed: Optional[int] = None,
    grid_cache_config: Optional[Dict[str, Any]] = None,
) -> Any:
    """Backward-compatible single monitored environment factory."""
    return make_single_env(
        env_config,
        reward_config,
        data_config,
        seed=seed,
        rank=0,
        monitor=True,
        grid_cache_config=grid_cache_config,
    )


def make_unmonitored_env(
    env_config: Dict[str, Any],
    reward_config: Dict[str, Any],
    data_config: Dict[str, Any],
    seed: Optional[int] = None,
    grid_cache_config: Optional[Dict[str, Any]] = None,
) -> GridCoupledEnv:
    return make_single_env(
        env_config,
        reward_config,
        data_config,
        seed=seed,
        rank=0,
        monitor=False,
        grid_cache_config=grid_cache_config,
    )


def make_env_fn(
    rank: int,
    base_seed: Optional[int],
    env_config: Dict[str, Any],
    reward_config: Dict[str, Any],
    data_config: Dict[str, Any],
    cpu_threads_per_worker: int = 1,
    verbose: bool = False,
    grid_cache_config: Optional[Dict[str, Any]] = None,
):
    """Return a pickleable VecEnv worker factory."""

    def _init():
        set_cpu_thread_env(cpu_threads_per_worker, force=True)
        return make_single_env(
            env_config=env_config,
            reward_config=reward_config,
            data_config=data_config,
            seed=base_seed,
            rank=rank,
            verbose=verbose,
            monitor=True,
            grid_cache_config=grid_cache_config,
        )

    return _init


class _SimpleDummyVecEnv:
    """Small VecEnv fallback used only when stable_baselines3 is unavailable."""

    def __init__(self, env_fns):
        self.envs = [env_fn() for env_fn in env_fns]
        if not self.envs:
            raise ValueError("At least one environment is required.")
        self.observation_space = self.envs[0].observation_space
        self.action_space = self.envs[0].action_space

    def reset(self):
        observations = []
        for env in self.envs:
            obs, _info = env.reset()
            observations.append(obs)
        return _stack_obs(observations)

    def step(self, actions):
        observations = []
        rewards = []
        dones = []
        infos = []
        for env, action in zip(self.envs, actions):
            obs, reward, terminated, truncated, info = env.step(action)
            done = bool(terminated or truncated)
            info = dict(info)
            if done:
                info["terminal_observation"] = obs
                obs, reset_info = env.reset()
                info["reset_info"] = reset_info
            observations.append(obs)
            rewards.append(float(reward))
            dones.append(done)
            infos.append(info)
        return _stack_obs(observations), np.asarray(rewards, dtype=np.float32), np.asarray(dones, dtype=bool), infos

    def close(self):
        for env in self.envs:
            env.close()


class _SimpleSubprocVecEnv:
    """Small subprocess VecEnv fallback for smoke tests and cache benchmarking."""

    def __init__(
        self,
        n_envs: int,
        base_seed: Optional[int],
        env_config: Dict[str, Any],
        reward_config: Dict[str, Any],
        data_config: Dict[str, Any],
        start_method: str,
        cpu_threads_per_worker: int,
        verbose_workers: bool,
        grid_cache_config: Optional[Dict[str, Any]],
    ):
        import multiprocessing as mp

        self.n_envs = max(int(n_envs), 1)
        ctx = mp.get_context(start_method)
        self.remotes = []
        self.processes = []
        for rank in range(self.n_envs):
            parent_remote, child_remote = ctx.Pipe()
            process = ctx.Process(
                target=_simple_subproc_worker,
                args=(
                    child_remote,
                    rank,
                    base_seed,
                    env_config,
                    reward_config,
                    data_config,
                    cpu_threads_per_worker,
                    verbose_workers,
                    grid_cache_config,
                ),
            )
            process.daemon = True
            process.start()
            child_remote.close()
            self.remotes.append(parent_remote)
            self.processes.append(process)

        self.remotes[0].send(("get_spaces", None))
        self.observation_space, self.action_space = self.remotes[0].recv()

    def reset(self):
        for remote in self.remotes:
            remote.send(("reset", None))
        return _stack_obs([remote.recv()[0] for remote in self.remotes])

    def step(self, actions):
        for remote, action in zip(self.remotes, actions):
            remote.send(("step", action))
        results = [remote.recv() for remote in self.remotes]
        observations, rewards, dones, infos = zip(*results)
        return (
            _stack_obs(observations),
            np.asarray(rewards, dtype=np.float32),
            np.asarray(dones, dtype=bool),
            list(infos),
        )

    def close(self):
        for remote in self.remotes:
            try:
                remote.send(("close", None))
            except Exception:
                pass
        for process in self.processes:
            process.join(timeout=5)
            if process.is_alive():
                process.terminate()


def _simple_subproc_worker(
    remote,
    rank: int,
    base_seed: Optional[int],
    env_config: Dict[str, Any],
    reward_config: Dict[str, Any],
    data_config: Dict[str, Any],
    cpu_threads_per_worker: int,
    verbose: bool,
    grid_cache_config: Optional[Dict[str, Any]],
) -> None:
    set_cpu_thread_env(cpu_threads_per_worker, force=True)
    env = make_single_env(
        env_config=env_config,
        reward_config=reward_config,
        data_config=data_config,
        seed=base_seed,
        rank=rank,
        verbose=verbose,
        monitor=False,
        grid_cache_config=grid_cache_config,
    )
    try:
        while True:
            command, data = remote.recv()
            if command == "get_spaces":
                remote.send((env.observation_space, env.action_space))
            elif command == "reset":
                remote.send(env.reset())
            elif command == "step":
                obs, reward, terminated, truncated, info = env.step(data)
                done = bool(terminated or truncated)
                info = dict(info)
                if done:
                    info["terminal_observation"] = obs
                    obs, reset_info = env.reset()
                    info["reset_info"] = reset_info
                remote.send((obs, float(reward), done, info))
            elif command == "close":
                break
            else:
                raise RuntimeError(f"Unknown VecEnv worker command: {command!r}")
    except EOFError:
        pass
    finally:
        env.close()
        remote.close()


def _stack_obs(observations):
    import numpy as np

    return np.stack([np.asarray(obs, dtype=np.float32) for obs in observations], axis=0)


def sanity_check_env(
    env_config: Dict[str, Any],
    reward_config: Dict[str, Any],
    data_config: Dict[str, Any],
) -> tuple[int, int]:
    """Check the current PPO observation/action interface."""
    env = make_unmonitored_env(env_config, reward_config, data_config, seed=DEFAULT_EVAL_SEED)
    try:
        obs, info = env.reset()
        expected_obs_dim = int(env.observation_space.shape[0])

        print(">>> PPO environment sanity check")
        print(f"    obs.shape = {obs.shape}")
        print(f"    env.observation_space.shape = {env.observation_space.shape}")
        print(f"    action_space.shape = {env.action_space.shape}")
        print(f"    base_obs_dim = {env.base_obs_dim}")
        print(f"    grid_obs_dim = {env.grid_obs_dim}")
        print(f"    enable_grid_obs = {env.enable_grid_obs}")
        print(f"    grid_reward_mode = {info.get('grid_reward_mode')}")
        print(f"    grid_reward_enabled = {info.get('grid_reward_enabled')}")
        print(f"    server_group_size = {info.get('server_group_size')}")
        print(f"    effective_total_server_count = {info.get('effective_total_server_count')}")
        print(f"    task_workload_scale = {info.get('task_workload_scale')}")
        print(f"    obs_dim(info) = {info.get('obs_dim')}")
        print(f"    total_task_count = {info.get('total_task_count')}")

        if int(obs.shape[0]) != expected_obs_dim:
            raise RuntimeError(f"Expected obs_dim={expected_obs_dim}, got obs.shape={obs.shape}")
        if env.action_space.shape != (23,):
            raise RuntimeError(f"Expected action_space.shape=(23,), got {env.action_space.shape}")

        return int(obs.shape[0]), int(env.action_space.shape[0])
    finally:
        env.close()


def infer_env_shape(
    env_config: Dict[str, Any],
    reward_config: Dict[str, Any],
    data_config: Dict[str, Any],
) -> tuple[int, int]:
    env = make_unmonitored_env(env_config, reward_config, data_config, seed=DEFAULT_EVAL_SEED)
    try:
        return int(env.observation_space.shape[0]), int(env.action_space.shape[0])
    finally:
        env.close()


def build_vec_env(
    n_envs: int,
    vec_env_type: str,
    seed: Optional[int],
    env_config: Dict[str, Any],
    reward_config: Dict[str, Any],
    data_config: Dict[str, Any],
    start_method: str = "spawn",
    cpu_threads_per_worker: int = 1,
    n_steps: Optional[int] = None,
    batch_size: Optional[int] = None,
    verbose_workers: bool = False,
    grid_cache_config: Optional[Dict[str, Any]] = None,
):
    """Build a DummyVecEnv or SubprocVecEnv without sharing env objects."""
    sb3_vec_env_available = True
    try:
        from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
    except ModuleNotFoundError as exc:
        if exc.name == "stable_baselines3":
            sb3_vec_env_available = False
            DummyVecEnv = None
            SubprocVecEnv = None
        else:
            raise

    n_envs = max(int(n_envs), 1)
    requested_type = str(vec_env_type or "auto").strip().lower()
    if requested_type not in {"auto", "dummy", "subproc"}:
        raise ValueError(f"--vec-env must be dummy, subproc, or auto; got {vec_env_type!r}.")

    if n_envs <= 1 or requested_type == "dummy":
        actual_type = "dummy"
    elif requested_type in {"auto", "subproc"}:
        actual_type = "subproc"
    else:
        actual_type = "dummy"

    base_seed = seed
    if n_envs > 1 and base_seed is None:
        base_seed = DEFAULT_EVAL_SEED

    env_fns = [
        make_env_fn(
            rank=rank,
            base_seed=base_seed,
            env_config=env_config,
            reward_config=reward_config,
            data_config=data_config,
            cpu_threads_per_worker=cpu_threads_per_worker,
            verbose=verbose_workers,
            grid_cache_config=grid_cache_config,
        )
        for rank in range(n_envs)
    ]

    try:
        if sb3_vec_env_available:
            if actual_type == "dummy":
                vec_env = DummyVecEnv(env_fns)
            else:
                vec_env = SubprocVecEnv(env_fns, start_method=start_method)
        elif actual_type == "dummy":
            vec_env = _SimpleDummyVecEnv(env_fns)
        else:
            vec_env = _SimpleSubprocVecEnv(
                n_envs=n_envs,
                base_seed=base_seed,
                env_config=env_config,
                reward_config=reward_config,
                data_config=data_config,
                start_method=start_method,
                cpu_threads_per_worker=cpu_threads_per_worker,
                verbose_workers=verbose_workers,
                grid_cache_config=grid_cache_config,
            )
    except Exception as exc:
        raise RuntimeError(
            f"Failed to create {actual_type} VecEnv with n_envs={n_envs}, "
            f"start_method={start_method!r}: {type(exc).__name__}: {exc}"
        ) from exc

    rollout_steps = int(n_steps if n_steps is not None else PPO_CONFIG["n_steps"])
    mini_batch = int(batch_size if batch_size is not None else PPO_CONFIG["batch_size"])
    total_rollout_size = n_envs * rollout_steps
    obs_shape = (n_envs, *vec_env.observation_space.shape)
    action_shape = (n_envs, *vec_env.action_space.shape)

    print(">>> VecEnv configuration")
    print(f"    vec_env_type = {actual_type} (requested={requested_type})")
    print(f"    vec_env_backend = {'stable_baselines3' if sb3_vec_env_available else 'simple_fallback'}")
    print(f"    n_envs = {n_envs}")
    print(f"    obs_shape = {obs_shape}")
    print(f"    action_shape = {action_shape}")
    print(f"    n_steps = {rollout_steps}")
    print(f"    batch_size = {mini_batch}")
    print(f"    total_rollout_size = {total_rollout_size}")

    if mini_batch > total_rollout_size:
        print(
            "WARNING: batch_size is larger than n_envs * n_steps; "
            "Stable-Baselines3 may use a truncated minibatch."
        )
    elif total_rollout_size % mini_batch != 0:
        print(
            "WARNING: batch_size does not evenly divide n_envs * n_steps; "
            "Stable-Baselines3 will leave a truncated minibatch."
        )

    return vec_env


def run_random_vec_env_smoke(
    vec_env,
    n_envs: int,
    steps: int,
    seed: int = DEFAULT_EVAL_SEED,
    expected_obs_dim: Optional[int] = 264,
    expected_action_dim: Optional[int] = 23,
) -> Dict[str, Any]:
    """Step a VecEnv with random actions and return grid diagnostic counts."""
    import numpy as np

    rng = np.random.default_rng(seed)
    obs = vec_env.reset()
    action_low = np.asarray(vec_env.action_space.low, dtype=np.float32)
    action_high = np.asarray(vec_env.action_space.high, dtype=np.float32)
    action_shape = (int(n_envs), *vec_env.action_space.shape)

    if expected_obs_dim is not None and tuple(obs.shape) != (int(n_envs), int(expected_obs_dim)):
        raise RuntimeError(f"Expected obs shape {(int(n_envs), int(expected_obs_dim))}, got {tuple(obs.shape)}")
    if expected_action_dim is not None and action_shape != (int(n_envs), int(expected_action_dim)):
        raise RuntimeError(f"Expected action shape {(int(n_envs), int(expected_action_dim))}, got {action_shape}")

    opf_success_count = 0
    opf_fail_count = 0
    mef_success_count = 0
    mef_fail_count = 0
    lmp_values: list[float] = []
    mef_values: list[float] = []
    total_safe_violation_cost = 0.0
    reward_mismatch_count = 0
    step_counts = [0 for _ in range(int(n_envs))]
    opf_cache_hit_counts = [0 for _ in range(int(n_envs))]
    opf_cache_miss_counts = [0 for _ in range(int(n_envs))]
    mef_cache_hit_counts = [0 for _ in range(int(n_envs))]
    mef_cache_miss_counts = [0 for _ in range(int(n_envs))]
    opf_cache_sizes = [0 for _ in range(int(n_envs))]
    mef_cache_sizes = [0 for _ in range(int(n_envs))]
    cache_enabled = False
    cache_load_bin_mw = math.nan

    for _ in range(int(steps)):
        actions = rng.uniform(low=action_low, high=action_high, size=action_shape).astype(np.float32)
        obs, rewards, dones, infos = vec_env.step(actions)
        if tuple(obs.shape) != (int(n_envs), int(expected_obs_dim or obs.shape[-1])):
            raise RuntimeError(f"Unexpected obs shape during step: {tuple(obs.shape)}")

        for env_idx, info in enumerate(infos):
            step_counts[env_idx] += 1
            if bool(info.get("grid_opf_success", False)):
                opf_success_count += 1
            else:
                opf_fail_count += 1
            if bool(info.get("grid_mef_success", False)):
                mef_success_count += 1
            else:
                mef_fail_count += 1

            grid_lmp = _finite_float(info.get("grid_lmp"))
            grid_mef_plus = _finite_float(info.get("grid_mef_plus"))
            if math.isfinite(grid_lmp):
                lmp_values.append(grid_lmp)
            if math.isfinite(grid_mef_plus):
                mef_values.append(grid_mef_plus)
            total_safe_violation_cost += _finite_or_zero(info.get("safe_violation_cost"))
            cache_enabled = cache_enabled or bool(info.get("grid_cache_enabled", False))
            bin_value = _finite_float(info.get("grid_cache_load_bin_mw"))
            if math.isfinite(bin_value):
                cache_load_bin_mw = bin_value
            opf_cache_hit_counts[env_idx] = int(_finite_or_zero(info.get("grid_opf_cache_hit_count")))
            opf_cache_miss_counts[env_idx] = int(_finite_or_zero(info.get("grid_opf_cache_miss_count")))
            mef_cache_hit_counts[env_idx] = int(_finite_or_zero(info.get("grid_mef_cache_hit_count")))
            mef_cache_miss_counts[env_idx] = int(_finite_or_zero(info.get("grid_mef_cache_miss_count")))
            opf_cache_sizes[env_idx] = int(_finite_or_zero(info.get("grid_cache_opf_size")))
            mef_cache_sizes[env_idx] = int(_finite_or_zero(info.get("grid_cache_mef_size")))

            if not bool(info.get("grid_reward_enabled", False)):
                base_reward = _finite_float(info.get("base_reward"))
                adjusted_reward = _finite_float(info.get("grid_adjusted_reward"))
                if math.isfinite(base_reward) and math.isfinite(adjusted_reward):
                    if abs(base_reward - adjusted_reward) > 1e-9:
                        reward_mismatch_count += 1

    if any(count <= 0 for count in step_counts):
        raise RuntimeError(f"At least one environment did not step: step_counts={step_counts}")

    opf_cache_hit_count = int(sum(opf_cache_hit_counts))
    opf_cache_miss_count = int(sum(opf_cache_miss_counts))
    mef_cache_hit_count = int(sum(mef_cache_hit_counts))
    mef_cache_miss_count = int(sum(mef_cache_miss_counts))
    opf_cache_total = opf_cache_hit_count + opf_cache_miss_count
    mef_cache_total = mef_cache_hit_count + mef_cache_miss_count

    summary = {
        "n_envs": int(n_envs),
        "vec_env_type": type(vec_env).__name__,
        "steps": int(steps),
        "obs_shape": tuple(obs.shape),
        "action_shape": action_shape,
        "opf_success_count": int(opf_success_count),
        "opf_fail_count": int(opf_fail_count),
        "mef_success_count": int(mef_success_count),
        "mef_fail_count": int(mef_fail_count),
        "avg_grid_lmp": _mean_or_nan(lmp_values),
        "avg_grid_mef_plus": _mean_or_nan(mef_values),
        "total_safe_violation_cost": float(total_safe_violation_cost),
        "reward_mismatch_count": int(reward_mismatch_count),
        "cache_enabled": bool(cache_enabled),
        "cache_load_bin_mw": float(cache_load_bin_mw),
        "opf_cache_hit_count": opf_cache_hit_count,
        "opf_cache_miss_count": opf_cache_miss_count,
        "opf_cache_hit_rate": float(opf_cache_hit_count / opf_cache_total) if opf_cache_total else 0.0,
        "mef_cache_hit_count": mef_cache_hit_count,
        "mef_cache_miss_count": mef_cache_miss_count,
        "mef_cache_hit_rate": float(mef_cache_hit_count / mef_cache_total) if mef_cache_total else 0.0,
        "opf_cache_size": int(sum(opf_cache_sizes)),
        "mef_cache_size": int(sum(mef_cache_sizes)),
    }
    print_parallel_smoke_summary(summary)
    return summary


def print_parallel_smoke_summary(summary: Dict[str, Any]) -> None:
    print("\n[PARALLEL SMOKE SUMMARY]")
    for key in [
        "n_envs",
        "vec_env_type",
        "steps",
        "obs_shape",
        "action_shape",
        "opf_success_count",
        "opf_fail_count",
        "mef_success_count",
        "mef_fail_count",
        "avg_grid_lmp",
        "avg_grid_mef_plus",
        "total_safe_violation_cost",
        "reward_mismatch_count",
        "cache_enabled",
        "cache_load_bin_mw",
        "opf_cache_hit_count",
        "opf_cache_miss_count",
        "opf_cache_hit_rate",
        "mef_cache_hit_count",
        "mef_cache_miss_count",
        "mef_cache_hit_rate",
        "opf_cache_size",
        "mef_cache_size",
    ]:
        print(f"{key}: {summary.get(key)}")


def _finite_float(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return math.nan
    return number if math.isfinite(number) else math.nan


def _finite_or_zero(value: Any) -> float:
    number = _finite_float(value)
    return number if math.isfinite(number) else 0.0


def _mean_or_nan(values: list[float]) -> float:
    return float(sum(values) / len(values)) if values else math.nan


def save_training_metadata(
    output_dir: Path,
    case_config: Dict[str, Any],
    run_name: str,
    timesteps: int,
    obs_dim: int,
    action_dim: int,
    final_model_path: Path,
    best_model_path: Path,
    n_envs: int,
    vec_env_type: str,
    n_steps: int,
    batch_size: int,
) -> Path:
    metadata = {
        "case": case_config["case"],
        "run_name": run_name,
        "timesteps": int(timesteps),
        "obs_dim": int(obs_dim),
        "action_dim": int(action_dim),
        "n_envs": int(n_envs),
        "vec_env_type": vec_env_type,
        "n_steps": int(n_steps),
        "batch_size": int(batch_size),
        "total_rollout_size": int(n_envs) * int(n_steps),
        "ENV_CONFIG_key_params": key_env_config(case_config["env_config"]),
        "REWARD_CONFIG_key_params": key_reward_config(case_config["reward_config"]),
        "DATA_CONFIG": case_config["data_config"],
        "GRID_REWARD_CONFIG": GRID_REWARD_CONFIG,
        "GRID_CACHE_CONFIG": GRID_CACHE_CONFIG,
        "training_time": datetime.now().isoformat(timespec="seconds"),
        "model_save_path": str(final_model_path.with_suffix(".zip")),
        "best_model_path": str(best_model_path),
    }

    metadata_path = output_dir / "training_metadata.json"
    with metadata_path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    return metadata_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", type=str, default="main", help="Experiment case: main, no_bess, carbon_w0, carbon_w03, carbon_w05.")
    parser.add_argument("--timesteps", type=int, default=500_000, help="PPO training timesteps.")
    parser.add_argument("--run-name", type=str, default="ultimate", help="Run name used in output directory.")
    parser.add_argument("--no-sanity-check", action="store_true", help="Skip observation/action shape sanity check.")
    parser.add_argument("--seed", type=int, default=None, help="Base worker seed. Defaults to random for one env and 2026 for multi-env.")
    parser.add_argument("--n-envs", type=int, default=1, help="Number of synchronous sampling environments.")
    parser.add_argument("--vec-env", choices=["dummy", "subproc", "auto"], default="auto", help="VecEnv backend.")
    parser.add_argument("--start-method", type=str, default="spawn", help="SubprocVecEnv multiprocessing start method.")
    parser.add_argument("--n-steps", type=int, default=None, help="Override PPO n_steps.")
    parser.add_argument("--batch-size", type=int, default=None, help="Override PPO batch_size.")
    parser.add_argument("--cpu-threads-per-worker", type=int, default=1, help="OMP/MKL/NUMEXPR/OPENBLAS threads per worker.")
    parser.add_argument("--parallel-smoke", action="store_true", help="Only run short random VecEnv sampling; do not create PPO.")
    parser.add_argument("--parallel-smoke-steps", type=int, default=48, help="Random VecEnv smoke-test steps.")
    args = parser.parse_args()

    set_cpu_thread_env(args.cpu_threads_per_worker, force=True)

    case_config = get_experiment_case(args.case)
    print_experiment_case(case_config)
    env_config = case_config["env_config"]
    reward_config = case_config["reward_config"]
    data_config = case_config["data_config"]

    ppo_config = dict(PPO_CONFIG)
    if args.n_steps is not None:
        ppo_config["n_steps"] = int(args.n_steps)
    if args.batch_size is not None:
        ppo_config["batch_size"] = int(args.batch_size)

    if args.no_sanity_check:
        obs_dim, action_dim = infer_env_shape(env_config, reward_config, data_config)
    else:
        obs_dim, action_dim = sanity_check_env(env_config, reward_config, data_config)

    vec_env = None
    if args.parallel_smoke:
        try:
            vec_env = build_vec_env(
                n_envs=args.n_envs,
                vec_env_type=args.vec_env,
                seed=args.seed,
                env_config=env_config,
                reward_config=reward_config,
                data_config=data_config,
                start_method=args.start_method,
                cpu_threads_per_worker=args.cpu_threads_per_worker,
                n_steps=ppo_config["n_steps"],
                batch_size=ppo_config["batch_size"],
            )
            run_random_vec_env_smoke(
                vec_env=vec_env,
                n_envs=max(int(args.n_envs), 1),
                steps=args.parallel_smoke_steps,
                seed=args.seed if args.seed is not None else DEFAULT_EVAL_SEED,
                expected_obs_dim=obs_dim,
                expected_action_dim=action_dim,
            )
        finally:
            if vec_env is not None:
                vec_env.close()
        return

    output_dir = resolve_output_path(f"ppo_outputs_{args.run_name}_{case_config['case']}")
    model_dir = output_dir / "models"
    log_dir = output_dir / "logs"
    best_model_dir = output_dir / "best_model"

    model_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    best_model_dir.mkdir(parents=True, exist_ok=True)

    try:
        from stable_baselines3 import PPO
        from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
    except Exception as exc:
        raise SystemExit(f"stable_baselines3 is required for PPO training: {exc}") from exc

    eval_env = None
    try:
        vec_env = build_vec_env(
            n_envs=args.n_envs,
            vec_env_type=args.vec_env,
            seed=args.seed,
            env_config=env_config,
            reward_config=reward_config,
            data_config=data_config,
            start_method=args.start_method,
            cpu_threads_per_worker=args.cpu_threads_per_worker,
            n_steps=ppo_config["n_steps"],
            batch_size=ppo_config["batch_size"],
        )
        eval_env = make_env(env_config, reward_config, data_config, seed=DEFAULT_EVAL_SEED)

        checkpoint_callback = CheckpointCallback(
            save_freq=50_000,
            save_path=str(model_dir),
            name_prefix="ppo_idc_ultimate",
        )

        eval_callback = EvalCallback(
            eval_env,
            best_model_save_path=str(best_model_dir),
            log_path=str(log_dir),
            eval_freq=20_000,
            n_eval_episodes=5,
            deterministic=True,
            render=False,
        )

        model = PPO(
            policy="MlpPolicy",
            env=vec_env,
            tensorboard_log=str(log_dir),
            **ppo_config,
        )

        print("\n>>> Start PPO training")
        print(f">>> case: {case_config['case']}")
        print(f">>> timesteps: {args.timesteps}")
        print(f">>> output dir: {output_dir}")

        model.learn(
            total_timesteps=args.timesteps,
            callback=[checkpoint_callback, eval_callback],
            progress_bar=False,
        )

        final_model_path = model_dir / "ppo_idc_ultimate_final"
        model.save(str(final_model_path))
        best_model_path = best_model_dir / "best_model.zip"
        metadata_path = save_training_metadata(
            output_dir=output_dir,
            case_config=case_config,
            run_name=args.run_name,
            timesteps=args.timesteps,
            obs_dim=obs_dim,
            action_dim=action_dim,
            final_model_path=final_model_path,
            best_model_path=best_model_path,
            n_envs=max(int(args.n_envs), 1),
            vec_env_type=args.vec_env,
            n_steps=int(ppo_config["n_steps"]),
            batch_size=int(ppo_config["batch_size"]),
        )

        print("\n>>> PPO training finished")
        print(f"Final model: {final_model_path}.zip")
        print(f"Best model dir: {best_model_dir}")
        print(f"Training logs: {log_dir}")
        print(f"Metadata: {metadata_path}")
    finally:
        if eval_env is not None:
            eval_env.close()
        if vec_env is not None:
            vec_env.close()


if __name__ == "__main__":
    import multiprocessing as mp

    mp.freeze_support()
    main()
