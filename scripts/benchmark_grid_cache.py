"""Benchmark GridCoupledEnv sampling with OPF/MEF cache on and off.

Run with:
    python -m scripts.benchmark_grid_cache
    python -m scripts.benchmark_grid_cache --n-envs 6 --vec-env subproc --steps 240 --cache both
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path
from typing import Any


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

import numpy as np  # noqa: E402

from configs.config_ultimate import GRID_CACHE_CONFIG, PPO_CONFIG  # noqa: E402
from configs.experiment_cases import get_experiment_case, print_experiment_case  # noqa: E402
from train.train_ppo_ultimate import build_vec_env, set_cpu_thread_env  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-envs", type=int, default=1)
    parser.add_argument("--vec-env", choices=["dummy", "subproc"], default="dummy")
    parser.add_argument("--steps", type=int, default=240)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--cache", choices=["on", "off", "both"], default="both")
    parser.add_argument("--load-bin-mw", type=float, default=0.1)
    parser.add_argument("--load-scale-bin", type=float, default=0.005)
    parser.add_argument("--cpu-threads-per-worker", type=int, default=1)
    parser.add_argument("--start-method", type=str, default="spawn")
    args = parser.parse_args()

    set_cpu_thread_env(args.cpu_threads_per_worker, force=True)

    case_config = get_experiment_case("main")
    print_experiment_case(case_config)

    modes = ["off", "on"] if args.cache == "both" else [args.cache]
    rows = []
    baseline_off: dict[str, Any] | None = None
    for mode in modes:
        row = run_one_mode(args, case_config, cache_mode=mode)
        if mode == "off":
            baseline_off = row
        if mode == "on" and baseline_off is not None:
            row["reward_delta_vs_off"] = row["total_reward"] - baseline_off["total_reward"]
            row["safe_cost_delta_vs_off"] = row["total_safe_violation_cost"] - baseline_off["total_safe_violation_cost"]
        rows.append(row)

    print_comparison_table(rows)


def run_one_mode(args: argparse.Namespace, case_config: dict[str, Any], cache_mode: str) -> dict[str, Any]:
    n_envs = max(int(args.n_envs), 1)
    steps = max(int(args.steps), 1)
    cache_config = make_cache_config(cache_mode, args.load_bin_mw, args.load_scale_bin)

    vec_env = None
    try:
        vec_env = build_vec_env(
            n_envs=n_envs,
            vec_env_type=args.vec_env,
            seed=args.seed,
            env_config=case_config["env_config"],
            reward_config=case_config["reward_config"],
            data_config=case_config["data_config"],
            start_method=args.start_method,
            cpu_threads_per_worker=args.cpu_threads_per_worker,
            n_steps=PPO_CONFIG["n_steps"],
            batch_size=PPO_CONFIG["batch_size"],
            grid_cache_config=cache_config,
        )

        rng = np.random.default_rng(int(args.seed))
        action_low = np.asarray(vec_env.action_space.low, dtype=np.float32)
        action_high = np.asarray(vec_env.action_space.high, dtype=np.float32)
        action_shape = (n_envs, *vec_env.action_space.shape)

        start = time.perf_counter()
        obs = vec_env.reset()

        opf_success_count = 0
        opf_fail_count = 0
        mef_success_count = 0
        mef_fail_count = 0
        reward_mismatch_count = 0
        total_reward = 0.0
        total_safe_violation_cost = 0.0
        total_safe_cost = 0.0
        opf_cache_hit_counts = [0 for _ in range(n_envs)]
        opf_cache_miss_counts = [0 for _ in range(n_envs)]
        mef_cache_hit_counts = [0 for _ in range(n_envs)]
        mef_cache_miss_counts = [0 for _ in range(n_envs)]
        opf_cache_sizes = [0 for _ in range(n_envs)]
        mef_cache_sizes = [0 for _ in range(n_envs)]

        for _ in range(steps):
            actions = rng.uniform(low=action_low, high=action_high, size=action_shape).astype(np.float32)
            obs, rewards, dones, infos = vec_env.step(actions)
            total_reward += float(np.sum(rewards))
            for env_idx, info in enumerate(infos):
                if bool(info.get("grid_opf_success", False)):
                    opf_success_count += 1
                else:
                    opf_fail_count += 1
                if bool(info.get("grid_mef_success", False)):
                    mef_success_count += 1
                else:
                    mef_fail_count += 1

                total_safe_violation_cost += _finite_or_zero(info.get("safe_violation_cost"))
                total_safe_cost += _finite_or_zero(info.get("safe_cost_total"))
                if not bool(info.get("grid_reward_enabled", False)):
                    base_reward = _finite_float(info.get("base_reward"))
                    adjusted_reward = _finite_float(info.get("grid_adjusted_reward"))
                    if math.isfinite(base_reward) and math.isfinite(adjusted_reward):
                        if abs(base_reward - adjusted_reward) > 1e-9:
                            reward_mismatch_count += 1

                opf_cache_hit_counts[env_idx] = int(_finite_or_zero(info.get("grid_opf_cache_hit_count")))
                opf_cache_miss_counts[env_idx] = int(_finite_or_zero(info.get("grid_opf_cache_miss_count")))
                mef_cache_hit_counts[env_idx] = int(_finite_or_zero(info.get("grid_mef_cache_hit_count")))
                mef_cache_miss_counts[env_idx] = int(_finite_or_zero(info.get("grid_mef_cache_miss_count")))
                opf_cache_sizes[env_idx] = int(_finite_or_zero(info.get("grid_cache_opf_size")))
                mef_cache_sizes[env_idx] = int(_finite_or_zero(info.get("grid_cache_mef_size")))

        total_seconds = time.perf_counter() - start
        env_steps = steps * n_envs
        opf_cache_hit_count = int(sum(opf_cache_hit_counts))
        opf_cache_miss_count = int(sum(opf_cache_miss_counts))
        mef_cache_hit_count = int(sum(mef_cache_hit_counts))
        mef_cache_miss_count = int(sum(mef_cache_miss_counts))
        opf_cache_total = opf_cache_hit_count + opf_cache_miss_count
        mef_cache_total = mef_cache_hit_count + mef_cache_miss_count

        return {
            "cache_mode": cache_mode,
            "cache_enabled": cache_mode == "on",
            "n_envs": n_envs,
            "vec_env": args.vec_env,
            "steps": steps,
            "env_steps": env_steps,
            "obs_shape": tuple(obs.shape),
            "total_seconds": float(total_seconds),
            "steps_per_second": float(env_steps / total_seconds) if total_seconds > 0.0 else math.nan,
            "total_reward": float(total_reward),
            "reward_mismatch_count": int(reward_mismatch_count),
            "total_safe_violation_cost": float(total_safe_violation_cost),
            "total_safe_cost": float(total_safe_cost),
            "opf_success_count": int(opf_success_count),
            "opf_fail_count": int(opf_fail_count),
            "mef_success_count": int(mef_success_count),
            "mef_fail_count": int(mef_fail_count),
            "opf_cache_hit_count": opf_cache_hit_count,
            "opf_cache_miss_count": opf_cache_miss_count,
            "opf_cache_hit_rate": float(opf_cache_hit_count / opf_cache_total) if opf_cache_total else 0.0,
            "mef_cache_hit_count": mef_cache_hit_count,
            "mef_cache_miss_count": mef_cache_miss_count,
            "mef_cache_hit_rate": float(mef_cache_hit_count / mef_cache_total) if mef_cache_total else 0.0,
            "opf_cache_size": int(sum(opf_cache_sizes)),
            "mef_cache_size": int(sum(mef_cache_sizes)),
            "cache_load_bin_mw": float(cache_config["cache_load_bin_mw"]),
            "cache_load_scale_bin": float(cache_config["cache_load_scale_bin"]),
            "reward_delta_vs_off": math.nan,
            "safe_cost_delta_vs_off": math.nan,
        }
    finally:
        if vec_env is not None:
            vec_env.close()


def make_cache_config(cache_mode: str, load_bin_mw: float, load_scale_bin: float) -> dict[str, Any]:
    enabled = cache_mode == "on"
    config = dict(GRID_CACHE_CONFIG)
    config.update(
        {
            "enable_grid_cache": enabled,
            "cache_opf": enabled,
            "cache_mef": enabled,
            "cache_load_bin_mw": float(load_bin_mw),
            "cache_load_scale_bin": float(load_scale_bin),
            "cache_scope": "per_worker",
            "cache_failed_results": False,
        }
    )
    return config


def print_comparison_table(rows: list[dict[str, Any]]) -> None:
    print("\n[GRID CACHE BENCHMARK]")
    columns = [
        ("cache", "cache_mode", 7, "s"),
        ("envs", "n_envs", 5, "d"),
        ("vec", "vec_env", 8, "s"),
        ("env_steps", "env_steps", 9, "d"),
        ("seconds", "total_seconds", 10, ".3f"),
        ("steps/s", "steps_per_second", 10, ".3f"),
        ("opf_ok", "opf_success_count", 8, "d"),
        ("mef_ok", "mef_success_count", 8, "d"),
        ("opf_hit", "opf_cache_hit_rate", 9, ".4f"),
        ("mef_hit", "mef_cache_hit_rate", 9, ".4f"),
        ("opf_size", "opf_cache_size", 9, "d"),
        ("mef_size", "mef_cache_size", 9, "d"),
        ("reward_mis", "reward_mismatch_count", 10, "d"),
        ("safe_cost", "total_safe_violation_cost", 10, ".3f"),
        ("reward_delta", "reward_delta_vs_off", 12, ".6f"),
    ]
    header = " ".join(f"{label:>{width}}" for label, _, width, _ in columns)
    print(header)
    print("-" * len(header))
    for row in rows:
        cells = []
        for _, key, width, fmt in columns:
            value = row.get(key)
            cells.append(_format_cell(value, width, fmt))
        print(" ".join(cells))

    print("\n[DETAILS]")
    for row in rows:
        print(
            f"cache={row['cache_mode']} "
            f"opf_hit/miss={row['opf_cache_hit_count']}/{row['opf_cache_miss_count']} "
            f"mef_hit/miss={row['mef_cache_hit_count']}/{row['mef_cache_miss_count']} "
            f"load_bin_mw={row['cache_load_bin_mw']} "
            f"load_scale_bin={row['cache_load_scale_bin']} "
            f"total_safe_cost={row['total_safe_cost']:.6f} "
            f"safe_cost_delta_vs_off={_fmt_float(row['safe_cost_delta_vs_off'])}"
        )


def _format_cell(value: Any, width: int, fmt: str) -> str:
    if fmt == "s":
        return f"{str(value):>{width}}"
    if fmt == "d":
        try:
            return f"{int(value):>{width}d}"
        except Exception:
            return f"{'nan':>{width}}"
    try:
        number = float(value)
    except Exception:
        return f"{'nan':>{width}}"
    if not math.isfinite(number):
        return f"{'nan':>{width}}"
    return f"{number:>{width}{fmt}}"


def _fmt_float(value: Any) -> str:
    number = _finite_float(value)
    return "nan" if not math.isfinite(number) else f"{number:.6f}"


def _finite_float(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return math.nan
    return number if math.isfinite(number) else math.nan


def _finite_or_zero(value: Any) -> float:
    number = _finite_float(value)
    return number if math.isfinite(number) else 0.0


if __name__ == "__main__":
    import multiprocessing as mp

    mp.freeze_support()
    main()
