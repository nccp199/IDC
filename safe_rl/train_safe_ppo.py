"""Train Safe PPO / Lagrangian PPO for single-IDC bus9 scenarios.

Examples:
    python -m safe_rl.train_safe_ppo --scenario single_idc_bus9_normal --connectivity-only
    python -m safe_rl.train_safe_ppo --scenario single_idc_bus9_normal --timesteps 100000 --enable-safe-reward --run-name safe_ppo_bus9_normal_100k
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from configs.config_ultimate import (
    DATA_CONFIG,
    DEFAULT_EVAL_SEED,
    ENV_CONFIG,
    GRID_CACHE_CONFIG,
    GRID_CONFIG,
    GRID_REWARD_CONFIG,
    GRID_SCENARIO_CONFIG,
    IDC_SCALE_CONFIG,
    PPO_CONFIG,
    REWARD_CONFIG,
    resolve_output_path,
)
from configs.single_idc_scenarios import apply_single_idc_scenario, get_single_idc_scenario
from data_io.data_loader import build_external_series_from_config
from env_wrappers import GridCoupledEnv
from envs.idc_price_env import IDCPriceEnv20D
from safe_rl.callbacks import SafePPOCallback
from safe_rl.config_safe import LAGRANGIAN_CONFIG, SAFE_COST_CONFIG, SAFE_TRAIN_CONFIG
from safe_rl.lagrangian import LagrangianMultiplierManager
from safe_rl.safe_wrapper import SafeRewardWrapper


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


set_cpu_thread_env(1, force=False)


def build_safe_scenario_configs(scenario_name: str) -> dict[str, Any]:
    env_config, grid_config, idc_scale_config, scenario = apply_single_idc_scenario(
        env_config=ENV_CONFIG,
        grid_config=GRID_CONFIG,
        idc_scale_config=IDC_SCALE_CONFIG,
        scenario_name=scenario_name,
    )
    reward_config = deepcopy(REWARD_CONFIG)
    data_config = deepcopy(DATA_CONFIG)
    grid_reward_config = deepcopy(GRID_REWARD_CONFIG)
    grid_reward_config["enable_grid_reward"] = False
    grid_reward_config["grid_reward_mode"] = "none"

    grid_scenario_config = deepcopy(GRID_SCENARIO_CONFIG)
    grid_scenario_config["enable_dynamic_grid_load"] = False
    grid_scenario_config["fallback_load_scale"] = 1.0

    return {
        "scenario": scenario,
        "env_config": env_config,
        "reward_config": reward_config,
        "data_config": data_config,
        "grid_config": grid_config,
        "grid_reward_config": grid_reward_config,
        "grid_scenario_config": grid_scenario_config,
        "idc_scale_config": idc_scale_config,
    }


def build_safe_train_config(args: argparse.Namespace) -> dict[str, Any]:
    config = deepcopy(SAFE_TRAIN_CONFIG)
    dry_run = bool(getattr(args, "dry_run", False))
    config["report_only"] = bool(args.report_only)
    config["shadow_lambda_update"] = bool(getattr(args, "shadow_lambda_update", False))
    config["safe_reward_active"] = (
        bool(args.enable_safe_reward)
        and not bool(args.report_only)
        and not dry_run
    )
    config["enable_safe_reward"] = bool(config["safe_reward_active"])
    config["lambda_update_active"] = (
        bool(config["safe_reward_active"]) or bool(config["shadow_lambda_update"])
    ) and not dry_run
    config["dry_run"] = dry_run
    config["terminate_on_opf_failure"] = bool(args.terminate_on_opf_failure)
    return config


def make_safe_single_env(
    configs: dict[str, Any],
    lagrangian_manager: LagrangianMultiplierManager,
    safe_train_config: dict[str, Any],
    seed: int | None = None,
    monitor: bool = True,
    grid_cache_config: dict[str, Any] | None = None,
):
    env_config = configs["env_config"]
    reward_config = configs["reward_config"]
    data_config = configs["data_config"]
    worker_seed = None if seed is None else int(seed)
    env_kwargs = {
        **env_config,
        **reward_config,
        **configs["idc_scale_config"],
        **build_external_series_from_config(data_config, env_config["horizon"]),
        "server_seed": worker_seed,
        "task_seed": worker_seed,
    }
    base_env = IDCPriceEnv20D(**env_kwargs)
    grid_env = GridCoupledEnv(
        base_env,
        configs["grid_config"],
        configs["grid_reward_config"],
        configs["grid_scenario_config"],
        grid_cache_config=grid_cache_config if grid_cache_config is not None else GRID_CACHE_CONFIG,
    )
    safe_env = SafeRewardWrapper(
        grid_env,
        cost_config=SAFE_COST_CONFIG,
        lagrangian_manager=lagrangian_manager,
        train_config=safe_train_config,
    )
    if worker_seed is not None:
        safe_env.action_space.seed(worker_seed)
        safe_env.observation_space.seed(worker_seed)

    if not monitor:
        return safe_env

    try:
        from stable_baselines3.common.monitor import Monitor
    except Exception:
        return safe_env
    return Monitor(safe_env)


def make_safe_env_fn(
    rank: int,
    base_seed: int | None,
    configs: dict[str, Any],
    lagrangian_manager: LagrangianMultiplierManager,
    safe_train_config: dict[str, Any],
    grid_cache_config: dict[str, Any] | None = None,
):
    def _init():
        seed = None if base_seed is None else int(base_seed) + int(rank)
        return make_safe_single_env(
            configs=configs,
            lagrangian_manager=lagrangian_manager,
            safe_train_config=safe_train_config,
            seed=seed,
            monitor=True,
            grid_cache_config=grid_cache_config,
        )

    return _init


def infer_env_shape(configs: dict[str, Any], manager: LagrangianMultiplierManager, safe_train_config: dict[str, Any]) -> tuple[int, int]:
    env = make_safe_single_env(configs, manager, safe_train_config, seed=DEFAULT_EVAL_SEED, monitor=False)
    try:
        return int(env.observation_space.shape[0]), int(env.action_space.shape[0])
    finally:
        env.close()


def run_connectivity_check(
    configs: dict[str, Any],
    manager: LagrangianMultiplierManager,
    safe_train_config: dict[str, Any],
    seed: int = DEFAULT_EVAL_SEED,
) -> dict[str, Any]:
    env = make_safe_single_env(configs, manager, safe_train_config, seed=seed, monitor=False)
    try:
        obs, reset_info = env.reset()
        action = np.ones(env.action_space.shape, dtype=np.float32)
        if action.shape[0] >= 1:
            action[-1] = 0.5

        rows = []
        total_reward = 0.0
        for _ in range(int(getattr(env, "horizon", 24))):
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += float(reward)
            rows.append(dict(info))
            if terminated or truncated:
                break

        idc_loads = [_finite_or_nan(row.get("grid_bus_net_load_mw", row.get("grid_idc_load_mw"))) for row in rows]
        min_v = [_finite_or_nan(row.get("grid_min_voltage_pu")) for row in rows]
        lmp = [_finite_or_nan(row.get("grid_lmp")) for row in rows]
        mef = [_finite_or_nan(row.get("grid_mef_plus")) for row in rows]
        opf_success = [bool(row.get("grid_opf_success", False)) for row in rows]
        safe_costs = [dict(row.get("safe_costs", {})) for row in rows]

        summary = {
            "obs_dim": int(env.observation_space.shape[0]),
            "action_dim": int(env.action_space.shape[0]),
            "steps": len(rows),
            "idc_load_peak_mw": _max_finite(idc_loads),
            "idc_load_mean_mw": _mean_finite(idc_loads),
            "total_pv_available_kWh": _finite_or_nan(rows[-1].get("total_pv_available_kWh")) if rows else np.nan,
            "total_pv_used_kWh": _finite_or_nan(rows[-1].get("total_pv_used_kWh")) if rows else np.nan,
            "total_pv_curtail_kWh": _finite_or_nan(rows[-1].get("total_pv_curtail_kWh")) if rows else np.nan,
            "pv_utilization_rate": _finite_or_nan(rows[-1].get("pv_utilization_rate")) if rows else np.nan,
            "renewable_share": _finite_or_nan(rows[-1].get("renewable_share")) if rows else np.nan,
            "opf_success_count": int(sum(opf_success)),
            "opf_step_count": int(len(opf_success)),
            "opf_success_rate": float(sum(opf_success) / max(len(opf_success), 1)),
            "minV_min": _min_finite(min_v),
            "minV_mean": _mean_finite(min_v),
            "LMP_bus9_mean": _mean_finite(lmp),
            "MEF_bus9_mean": _mean_finite(mef),
            "reward_sum": float(total_reward),
            "reward_has_nan": bool(any(not np.isfinite(_finite_or_nan(row.get("reward_safe", row.get("reward_base")))) for row in rows)),
            "has_safe_costs": bool(all("C_opf" in item and "C_voltage" in item for item in safe_costs)),
            "lambda_opf": float(manager.get_lambdas().get("lambda_opf", 0.0)),
            "lambda_voltage": float(manager.get_lambdas().get("lambda_voltage", 0.0)),
            "ema_C_opf": float(manager.get_ema_costs().get("ema_C_opf", 0.0)),
            "ema_C_voltage": float(manager.get_ema_costs().get("ema_C_voltage", 0.0)),
        }
        print_connectivity_summary(summary)
        return summary
    finally:
        env.close()


def print_safe_startup(
    configs: dict[str, Any],
    safe_train_config: dict[str, Any],
    manager: LagrangianMultiplierManager,
    obs_dim: int,
    action_dim: int,
    output_dir: Path,
    timesteps: int,
    ppo_n_steps: int,
    n_envs: int,
) -> None:
    scenario = configs["scenario"]
    if safe_train_config.get("dry_run", False):
        mode = "dry-run diagnostic"
    elif safe_train_config.get("report_only", False):
        mode = "report-only training"
    elif safe_train_config.get("safe_reward_active", False):
        mode = "true Safe PPO training"
    else:
        mode = "PPO training with safe monitoring disabled"
    print("\n>>> Safe PPO startup")
    print(f"mode: {mode}")
    print(f"scenario: {scenario['scenario_name']}")
    print(f"idc_bus: {scenario['idc_bus']}")
    print(f"target_peak_mw: {scenario['target_peak_mw']}")
    print(f"target_mean_mw: {scenario['target_mean_mw']:.3f}")
    print(f"server_group_size: {scenario['server_group_size']}")
    print(f"IEEE14 base load scale: {configs['grid_scenario_config'].get('fallback_load_scale')} (dynamic disabled)")
    print(f"obs_dim: {obs_dim}")
    print(f"action_dim: {action_dim}")
    print(f"Requested timesteps: {timesteps}")
    print(f"PPO n_steps: {ppo_n_steps}")
    print("Actual timesteps may be rounded up to a multiple of n_steps.")
    rollout_chunk = max(int(ppo_n_steps) * max(int(n_envs), 1), 1)
    aligned_timesteps = int(np.ceil(max(int(timesteps), 1) / rollout_chunk) * rollout_chunk)
    print(f"Rollout-aligned timesteps for current settings: {aligned_timesteps}")
    if int(n_envs) > 1:
        print(f"With n_envs={n_envs}, rollout chunks are n_envs * n_steps = {rollout_chunk}.")
    print(f"enable_safe_reward: {safe_train_config['enable_safe_reward']}")
    print(f"report_only: {safe_train_config['report_only']}")
    print(f"shadow_lambda_update: {safe_train_config.get('shadow_lambda_update', False)}")
    print(f"safe_reward_active: {safe_train_config.get('safe_reward_active', False)}")
    print(f"lambda_update_active: {safe_train_config.get('lambda_update_active', False)}")
    print(f"terminate_on_opf_failure: {safe_train_config['terminate_on_opf_failure']}")
    print(f"enabled_constraints: {LAGRANGIAN_CONFIG['enabled_constraints']}")
    print(f"initial_lambdas: {manager.get_lambdas()}")
    print(
        "cost_limits: "
        f"opf={LAGRANGIAN_CONFIG['cost_limit_opf']}, "
        f"voltage={LAGRANGIAN_CONFIG['cost_limit_voltage']}"
    )
    print(f"output_dir: {output_dir}")


def print_connectivity_summary(summary: dict[str, Any]) -> None:
    print("\n>>> Single IDC bus9 connectivity check")
    for key in [
        "obs_dim",
        "action_dim",
        "steps",
        "idc_load_peak_mw",
        "idc_load_mean_mw",
        "total_pv_available_kWh",
        "total_pv_used_kWh",
        "total_pv_curtail_kWh",
        "pv_utilization_rate",
        "renewable_share",
        "opf_success_count",
        "opf_step_count",
        "opf_success_rate",
        "minV_min",
        "minV_mean",
        "LMP_bus9_mean",
        "MEF_bus9_mean",
        "reward_sum",
        "reward_has_nan",
        "has_safe_costs",
        "lambda_opf",
        "lambda_voltage",
        "ema_C_opf",
        "ema_C_voltage",
    ]:
        print(f"{key}: {summary.get(key)}")


def make_dry_run_action(env, policy: str, rng: np.random.Generator) -> np.ndarray:
    policy_name = str(policy).strip().lower()
    action_dim = int(env.action_space.shape[0])
    action = np.zeros(action_dim, dtype=np.float32)
    base_env = env.unwrapped

    if policy_name == "zero":
        return action
    if policy_name == "one":
        return np.ones(action_dim, dtype=np.float32)
    if policy_name in {"full_neutral_bess", "one_neutral_bess"}:
        action[:] = 1.0
        if action_dim >= 1:
            action[-1] = 0.5
        return action
    if policy_name == "random":
        return rng.uniform(0.0, 1.0, size=action_dim).astype(np.float32)
    if policy_name == "rule":
        model_n = int(getattr(getattr(base_env, "model", None), "N", max(action_dim - 3, 1)))
        t = int(getattr(base_env, "current_step", 0))
        price = np.asarray(getattr(base_env, "price_t", np.ones(24)), dtype=np.float64)
        price_now = float(price[min(max(t, 0), len(price) - 1)])
        if np.isclose(price_now, float(np.min(price))):
            server_level = 0.95
            bess = 0.20
        elif np.isclose(price_now, float(np.max(price))):
            server_level = 0.25
            bess = 0.80
        else:
            server_level = 0.55
            bess = 0.50
        action[:model_n] = server_level
        if action_dim > model_n:
            action[model_n] = 0.85
        if action_dim > model_n + 1:
            action[model_n + 1] = 0.80
        if action_dim > model_n + 2:
            action[model_n + 2] = bess
        return np.clip(action, 0.0, 1.0).astype(np.float32)

    valid = "full_neutral_bess, rule, random, zero, one"
    raise ValueError(f"Unknown --dry-run-policy {policy!r}. Valid policies: {valid}")


def run_dry_run(
    configs: dict[str, Any],
    manager: LagrangianMultiplierManager,
    safe_train_config: dict[str, Any],
    output_dir: Path,
    run_name: str,
    policy: str,
    episodes: int,
    seed: int,
    obs_dim: int,
    action_dim: int,
    write_tensorboard: bool = False,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    hourly_rows: list[dict[str, Any]] = []
    rng = np.random.default_rng(seed)

    for episode in range(max(int(episodes), 1)):
        env = make_safe_single_env(
            configs,
            manager,
            safe_train_config,
            seed=int(seed) + episode,
            monitor=False,
            grid_cache_config=GRID_CACHE_CONFIG,
        )
        try:
            obs, info = env.reset()
            for step in range(int(configs["env_config"].get("horizon", 24))):
                action = make_dry_run_action(env, policy, rng)
                obs, reward, terminated, truncated, info = env.step(action)
                safe_costs = dict(info.get("safe_costs", {}))
                hourly_rows.append(
                    {
                        "episode": int(episode),
                        "seed": int(seed) + episode,
                        "hour": int(info.get("hour", step)),
                        "reward_base": _finite_or_nan(info.get("reward_base")),
                        "reward_safe": _finite_or_nan(info.get("reward_safe")),
                        "P_local_demand_kW": _finite_or_nan(info.get("P_local_demand_kW")),
                        "P_local_net_before_pv_kW": _finite_or_nan(info.get("P_local_net_before_pv_kW")),
                        "P_bus_net_kW": _finite_or_nan(info.get("P_bus_net_kW")),
                        "P_grid_kW": _finite_or_nan(info.get("P_grid_kW")),
                        "pv_available_kW": _finite_or_nan(info.get("pv_available_kW")),
                        "pv_used_kW": _finite_or_nan(info.get("pv_used_kW")),
                        "pv_curtail_kW": _finite_or_nan(info.get("pv_curtail_kW")),
                        "total_pv_available_kWh": _finite_or_nan(info.get("total_pv_available_kWh")),
                        "total_pv_used_kWh": _finite_or_nan(info.get("total_pv_used_kWh")),
                        "total_pv_curtail_kWh": _finite_or_nan(info.get("total_pv_curtail_kWh")),
                        "pv_utilization_rate": _finite_or_nan(info.get("pv_utilization_rate")),
                        "renewable_share": _finite_or_nan(info.get("renewable_share")),
                        "grid_bus_net_load_mw": _finite_or_nan(info.get("grid_bus_net_load_mw")),
                        "idc_load_mw": _finite_or_nan(info.get("grid_idc_load_mw")),
                        "opf_success": bool(safe_costs.get("opf_success", info.get("grid_opf_success", False))),
                        "minV": _finite_or_nan(safe_costs.get("raw_minV", info.get("grid_min_voltage_pu"))),
                        "LMP_bus9": _finite_or_nan(safe_costs.get("raw_lmp", info.get("grid_lmp"))),
                        "MEF_bus9": _finite_or_nan(safe_costs.get("raw_mef", info.get("grid_mef_plus"))),
                        "C_opf": _finite_or_nan(safe_costs.get("C_opf")),
                        "C_voltage": _finite_or_nan(safe_costs.get("C_voltage")),
                        "C_total": _finite_or_nan(safe_costs.get("C_total")),
                        "safe_penalty_raw": _finite_or_nan(info.get("safe_penalty_raw")),
                        "safe_penalty_applied": _finite_or_nan(info.get("safe_penalty_applied")),
                        "safe_reward_active": bool(info.get("safe_reward_active", False)),
                        "lambda_update_active": bool(info.get("lambda_update_active", False)),
                    }
                )
                if terminated or truncated:
                    break
        finally:
            env.close()

    summary = build_dry_run_summary(
        configs=configs,
        run_name=run_name,
        policy=policy,
        seed=seed,
        episodes=episodes,
        obs_dim=obs_dim,
        action_dim=action_dim,
        rows=hourly_rows,
        manager=manager,
        safe_train_config=safe_train_config,
    )
    write_dry_run_outputs(output_dir, summary, hourly_rows)
    if write_tensorboard:
        write_dry_run_tensorboard(output_dir / "logs", hourly_rows, summary)
    print_dry_run_summary(summary)
    return summary


def build_dry_run_summary(
    configs: dict[str, Any],
    run_name: str,
    policy: str,
    seed: int,
    episodes: int,
    obs_dim: int,
    action_dim: int,
    rows: list[dict[str, Any]],
    manager: LagrangianMultiplierManager,
    safe_train_config: dict[str, Any],
) -> dict[str, Any]:
    opf_values = [bool(row.get("opf_success", False)) for row in rows]
    summary = {
        "mode": "dry_run",
        "run_name": run_name,
        "scenario": configs["scenario"]["scenario_name"],
        "policy": policy,
        "seed": int(seed),
        "episodes": int(max(episodes, 1)),
        "steps": int(len(rows)),
        "obs_dim": int(obs_dim),
        "action_dim": int(action_dim),
        "idc_load_mw_min": _min_finite([_finite_or_nan(row.get("idc_load_mw")) for row in rows]),
        "idc_load_mw_mean": _mean_finite([_finite_or_nan(row.get("idc_load_mw")) for row in rows]),
        "idc_load_mw_max": _max_finite([_finite_or_nan(row.get("idc_load_mw")) for row in rows]),
        "grid_bus_net_load_mw_min": _min_finite([_finite_or_nan(row.get("grid_bus_net_load_mw")) for row in rows]),
        "grid_bus_net_load_mw_mean": _mean_finite([_finite_or_nan(row.get("grid_bus_net_load_mw")) for row in rows]),
        "grid_bus_net_load_mw_max": _max_finite([_finite_or_nan(row.get("grid_bus_net_load_mw")) for row in rows]),
        "P_bus_net_kW_mean": _mean_finite([_finite_or_nan(row.get("P_bus_net_kW")) for row in rows]),
        "P_grid_kW_mean": _mean_finite([_finite_or_nan(row.get("P_grid_kW")) for row in rows]),
        "pv_available_kW_mean": _mean_finite([_finite_or_nan(row.get("pv_available_kW")) for row in rows]),
        "pv_used_kW_mean": _mean_finite([_finite_or_nan(row.get("pv_used_kW")) for row in rows]),
        "pv_curtail_kW_mean": _mean_finite([_finite_or_nan(row.get("pv_curtail_kW")) for row in rows]),
        "total_pv_available_kWh": _finite_or_nan(rows[-1].get("total_pv_available_kWh")) if rows else np.nan,
        "total_pv_used_kWh": _finite_or_nan(rows[-1].get("total_pv_used_kWh")) if rows else np.nan,
        "total_pv_curtail_kWh": _finite_or_nan(rows[-1].get("total_pv_curtail_kWh")) if rows else np.nan,
        "pv_utilization_rate": _finite_or_nan(rows[-1].get("pv_utilization_rate")) if rows else np.nan,
        "renewable_share": _finite_or_nan(rows[-1].get("renewable_share")) if rows else np.nan,
        "opf_success_count": int(sum(opf_values)),
        "opf_step_count": int(len(opf_values)),
        "opf_success_rate": float(sum(opf_values) / max(len(opf_values), 1)),
        "minV_min": _min_finite([_finite_or_nan(row.get("minV")) for row in rows]),
        "minV_mean": _mean_finite([_finite_or_nan(row.get("minV")) for row in rows]),
        "LMP_bus9_mean": _mean_finite([_finite_or_nan(row.get("LMP_bus9")) for row in rows]),
        "LMP_bus9_max": _max_finite([_finite_or_nan(row.get("LMP_bus9")) for row in rows]),
        "MEF_bus9_mean": _mean_finite([_finite_or_nan(row.get("MEF_bus9")) for row in rows]),
        "MEF_bus9_max": _max_finite([_finite_or_nan(row.get("MEF_bus9")) for row in rows]),
        "C_opf_mean": _mean_finite([_finite_or_nan(row.get("C_opf")) for row in rows]),
        "C_opf_sum": _sum_finite([_finite_or_nan(row.get("C_opf")) for row in rows]),
        "C_voltage_mean": _mean_finite([_finite_or_nan(row.get("C_voltage")) for row in rows]),
        "C_voltage_sum": _sum_finite([_finite_or_nan(row.get("C_voltage")) for row in rows]),
        "C_total_mean": _mean_finite([_finite_or_nan(row.get("C_total")) for row in rows]),
        "C_total_sum": _sum_finite([_finite_or_nan(row.get("C_total")) for row in rows]),
        "reward_base_sum": _sum_finite([_finite_or_nan(row.get("reward_base")) for row in rows]),
        "reward_safe_sum": _sum_finite([_finite_or_nan(row.get("reward_safe")) for row in rows]),
        "safe_penalty_raw_sum": _sum_finite([_finite_or_nan(row.get("safe_penalty_raw")) for row in rows]),
        "safe_penalty_applied_sum": _sum_finite([_finite_or_nan(row.get("safe_penalty_applied")) for row in rows]),
        "safe_reward_active": bool(safe_train_config.get("safe_reward_active", False)),
        "lambda_update_active": bool(safe_train_config.get("lambda_update_active", False)),
        "lambdas": manager.get_lambdas(),
        "ema_costs": manager.get_ema_costs(),
    }
    return summary


def write_dry_run_outputs(output_dir: Path, summary: dict[str, Any], hourly_rows: list[dict[str, Any]]) -> None:
    summary_path = output_dir / "safe_dry_run_summary.json"
    hourly_path = output_dir / "safe_dry_run_hourly.csv"
    report_path = output_dir / "safe_dry_run_report.md"

    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    if hourly_rows:
        fieldnames = list(hourly_rows[0].keys())
        with hourly_path.open("w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(hourly_rows)

    lines = [
        "# Safe PPO Dry-Run Report",
        "",
        f"- scenario: `{summary['scenario']}`",
        f"- policy: `{summary['policy']}`",
        f"- safe_reward_active: `{summary['safe_reward_active']}`",
        f"- lambda_update_active: `{summary['lambda_update_active']}`",
        "",
        "| metric | value |",
        "|---|---:|",
    ]
    for key in [
        "obs_dim",
        "action_dim",
        "idc_load_mw_min",
        "idc_load_mw_mean",
        "idc_load_mw_max",
        "grid_bus_net_load_mw_min",
        "grid_bus_net_load_mw_mean",
        "grid_bus_net_load_mw_max",
        "P_bus_net_kW_mean",
        "P_grid_kW_mean",
        "pv_available_kW_mean",
        "pv_used_kW_mean",
        "pv_curtail_kW_mean",
        "total_pv_available_kWh",
        "total_pv_used_kWh",
        "total_pv_curtail_kWh",
        "pv_utilization_rate",
        "renewable_share",
        "opf_success_count",
        "opf_success_rate",
        "minV_min",
        "minV_mean",
        "LMP_bus9_mean",
        "LMP_bus9_max",
        "MEF_bus9_mean",
        "MEF_bus9_max",
        "C_opf_mean",
        "C_opf_sum",
        "C_voltage_mean",
        "C_voltage_sum",
        "C_total_mean",
        "C_total_sum",
        "reward_base_sum",
        "reward_safe_sum",
        "safe_penalty_raw_sum",
        "safe_penalty_applied_sum",
    ]:
        lines.append(f"| {key} | {summary.get(key)} |")
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_dry_run_tensorboard(log_dir: Path, rows: list[dict[str, Any]], summary: dict[str, Any]) -> None:
    try:
        from torch.utils.tensorboard import SummaryWriter
    except Exception as exc:
        print(f">>> Dry-run TensorBoard skipped: {exc}")
        return

    log_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(log_dir))
    try:
        for idx, row in enumerate(rows):
            writer.add_scalar("safe/C_opf", _finite_or_nan(row.get("C_opf")), idx)
            writer.add_scalar("safe/C_voltage", _finite_or_nan(row.get("C_voltage")), idx)
            writer.add_scalar("safe/C_total", _finite_or_nan(row.get("C_total")), idx)
            writer.add_scalar("safe/safe_penalty_raw", _finite_or_nan(row.get("safe_penalty_raw")), idx)
            writer.add_scalar("safe/safe_penalty_applied", _finite_or_nan(row.get("safe_penalty_applied")), idx)
            writer.add_scalar("safe/reward_base", _finite_or_nan(row.get("reward_base")), idx)
            writer.add_scalar("safe/reward_safe", _finite_or_nan(row.get("reward_safe")), idx)
            writer.add_scalar("safe/min_voltage", _finite_or_nan(row.get("minV")), idx)
            writer.add_scalar("safe/LMP_bus9", _finite_or_nan(row.get("LMP_bus9")), idx)
            writer.add_scalar("safe/MEF_bus9", _finite_or_nan(row.get("MEF_bus9")), idx)
            writer.add_scalar("pv/available_kW", _finite_or_nan(row.get("pv_available_kW")), idx)
            writer.add_scalar("pv/used_kW", _finite_or_nan(row.get("pv_used_kW")), idx)
            writer.add_scalar("pv/curtail_kW", _finite_or_nan(row.get("pv_curtail_kW")), idx)
            writer.add_scalar("grid/P_bus_net_kW", _finite_or_nan(row.get("P_bus_net_kW")), idx)
            writer.add_scalar("grid/P_grid_kW", _finite_or_nan(row.get("P_grid_kW")), idx)
            writer.add_scalar("safe/safe_reward_active", 1.0 if row.get("safe_reward_active") else 0.0, idx)
            writer.add_scalar("safe/lambda_update_active", 1.0 if row.get("lambda_update_active") else 0.0, idx)
        writer.add_scalar("safe/opf_failure_rate", 1.0 - float(summary.get("opf_success_rate", 0.0)), len(rows))
        writer.add_scalar("pv/total_available_kWh", _finite_or_nan(summary.get("total_pv_available_kWh")), len(rows))
        writer.add_scalar("pv/total_used_kWh", _finite_or_nan(summary.get("total_pv_used_kWh")), len(rows))
        writer.add_scalar("pv/total_curtail_kWh", _finite_or_nan(summary.get("total_pv_curtail_kWh")), len(rows))
        writer.add_scalar("pv/utilization_rate", _finite_or_nan(summary.get("pv_utilization_rate")), len(rows))
        writer.add_scalar("pv/renewable_share", _finite_or_nan(summary.get("renewable_share")), len(rows))
    finally:
        writer.close()


def print_dry_run_summary(summary: dict[str, Any]) -> None:
    print("\n>>> Safe PPO dry-run summary")
    for key in [
        "mode",
        "scenario",
        "policy",
        "obs_dim",
        "action_dim",
        "idc_load_mw_min",
        "idc_load_mw_mean",
        "idc_load_mw_max",
        "grid_bus_net_load_mw_min",
        "grid_bus_net_load_mw_mean",
        "grid_bus_net_load_mw_max",
        "P_bus_net_kW_mean",
        "P_grid_kW_mean",
        "pv_available_kW_mean",
        "pv_used_kW_mean",
        "pv_curtail_kW_mean",
        "total_pv_available_kWh",
        "total_pv_used_kWh",
        "total_pv_curtail_kWh",
        "pv_utilization_rate",
        "renewable_share",
        "opf_success_count",
        "opf_step_count",
        "opf_success_rate",
        "minV_min",
        "minV_mean",
        "LMP_bus9_mean",
        "LMP_bus9_max",
        "MEF_bus9_mean",
        "MEF_bus9_max",
        "C_opf_mean",
        "C_opf_sum",
        "C_voltage_mean",
        "C_voltage_sum",
        "C_total_mean",
        "C_total_sum",
        "reward_base_sum",
        "reward_safe_sum",
        "safe_penalty_raw_sum",
        "safe_penalty_applied_sum",
        "safe_reward_active",
        "lambda_update_active",
        "lambdas",
        "ema_costs",
    ]:
        print(f"{key}: {summary.get(key)}")


def save_safe_training_metadata(
    output_dir: Path,
    configs: dict[str, Any],
    safe_train_config: dict[str, Any],
    manager: LagrangianMultiplierManager,
    obs_dim: int,
    action_dim: int,
    timesteps: int,
    run_name: str,
    final_model_path: Path,
    connectivity_summary: dict[str, Any] | None,
) -> Path:
    metadata = {
        "run_name": run_name,
        "timesteps": int(timesteps),
        "obs_dim": int(obs_dim),
        "action_dim": int(action_dim),
        "scenario": configs["scenario"],
        "safe_cost_config": SAFE_COST_CONFIG,
        "lagrangian_config": LAGRANGIAN_CONFIG,
        "safe_train_config": safe_train_config,
        "final_lambdas": manager.get_lambdas(),
        "final_ema_costs": manager.get_ema_costs(),
        "connectivity_summary": connectivity_summary,
        "model_save_path": str(final_model_path.with_suffix(".zip")),
        "training_time": datetime.now().isoformat(timespec="seconds"),
    }
    path = output_dir / "safe_training_metadata.json"
    with path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    return path


def save_lagrangian_state(output_dir: Path, manager: LagrangianMultiplierManager) -> Path:
    path = output_dir / "lagrangian_state.json"
    with path.open("w", encoding="utf-8") as f:
        json.dump(manager.state_dict(), f, ensure_ascii=False, indent=2)
    return path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", type=str, default="single_idc_bus9_normal")
    parser.add_argument("--timesteps", type=int, default=100_000)
    parser.add_argument("--enable-safe-reward", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--report-only", action="store_true")
    parser.add_argument("--shadow-lambda-update", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--dry-run-episodes", type=int, default=1)
    parser.add_argument(
        "--dry-run-policy",
        type=str,
        default="full_neutral_bess",
        help="Dry-run policy: full_neutral_bess, rule, random, zero, or one.",
    )
    parser.add_argument("--dry-run-tensorboard", action="store_true")
    parser.add_argument("--run-name", type=str, default="safe_ppo_bus9_normal")
    parser.add_argument("--seed", type=int, default=DEFAULT_EVAL_SEED)
    parser.add_argument("--n-envs", type=int, default=1)
    parser.add_argument("--n-steps", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--cpu-threads-per-worker", type=int, default=1)
    parser.add_argument("--skip-connectivity-check", action="store_true")
    parser.add_argument("--connectivity-only", action="store_true")
    parser.add_argument("--terminate-on-opf-failure", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    if args.connectivity_only:
        args.dry_run = True

    set_cpu_thread_env(args.cpu_threads_per_worker, force=True)

    configs = build_safe_scenario_configs(args.scenario)
    safe_train_config = build_safe_train_config(args)
    manager = LagrangianMultiplierManager(LAGRANGIAN_CONFIG)

    ppo_config = deepcopy(PPO_CONFIG)
    if args.n_steps is not None:
        ppo_config["n_steps"] = int(args.n_steps)
    if args.batch_size is not None:
        ppo_config["batch_size"] = int(args.batch_size)

    obs_dim, action_dim = infer_env_shape(configs, manager, safe_train_config)
    if safe_train_config["dry_run"]:
        output_dir = resolve_output_path(f"safe_dry_run_{args.run_name}")
        log_dir = output_dir / "logs"
    else:
        output_dir = resolve_output_path(f"ppo_outputs_{args.run_name}")
        model_dir = output_dir / "models"
        log_dir = output_dir / "logs"
        checkpoint_dir = output_dir / "checkpoints"
        model_dir.mkdir(parents=True, exist_ok=True)
        log_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

    print_safe_startup(
        configs=configs,
        safe_train_config=safe_train_config,
        manager=manager,
        obs_dim=obs_dim,
        action_dim=action_dim,
        output_dir=output_dir,
        timesteps=args.timesteps,
        ppo_n_steps=int(ppo_config["n_steps"]),
        n_envs=max(int(args.n_envs), 1),
    )

    if safe_train_config["dry_run"]:
        run_dry_run(
            configs=configs,
            manager=manager,
            safe_train_config=safe_train_config,
            output_dir=output_dir,
            run_name=args.run_name,
            policy=args.dry_run_policy,
            episodes=args.dry_run_episodes,
            seed=args.seed,
            obs_dim=obs_dim,
            action_dim=action_dim,
            write_tensorboard=bool(args.dry_run_tensorboard),
        )
        print(f"\nDry-run outputs: {output_dir}")
        if args.dry_run_tensorboard:
            print(f"TensorBoard command: tensorboard --logdir {log_dir}")
        return

    connectivity_summary = None
    if not args.skip_connectivity_check:
        connectivity_summary = run_connectivity_check(configs, manager, safe_train_config, seed=args.seed)

    try:
        from stable_baselines3 import PPO
        from stable_baselines3.common.callbacks import CheckpointCallback
        from stable_baselines3.common.vec_env import DummyVecEnv
    except Exception as exc:
        raise SystemExit(f"stable_baselines3 is required for Safe PPO training: {exc}") from exc

    n_envs = max(int(args.n_envs), 1)
    env_fns = [
        make_safe_env_fn(
            rank=rank,
            base_seed=args.seed,
            configs=configs,
            lagrangian_manager=manager,
            safe_train_config=safe_train_config,
            grid_cache_config=GRID_CACHE_CONFIG,
        )
        for rank in range(n_envs)
    ]
    vec_env = DummyVecEnv(env_fns)

    checkpoint_callback = CheckpointCallback(
        save_freq=50_000,
        save_path=str(checkpoint_dir),
        name_prefix="report_only_ppo_idc_bus9" if safe_train_config["report_only"] else "safe_ppo_idc_bus9",
    )
    safe_callback = SafePPOCallback(
        manager,
        lambda_update_active=bool(safe_train_config.get("lambda_update_active", False)),
        safe_reward_active=bool(safe_train_config.get("safe_reward_active", False)),
    )

    model = PPO(
        policy="MlpPolicy",
        env=vec_env,
        tensorboard_log=str(log_dir),
        **ppo_config,
    )

    print("\n>>> Start Safe PPO training")
    print(f"TensorBoard log_dir: {log_dir}")
    model.learn(
        total_timesteps=int(args.timesteps),
        callback=[safe_callback, checkpoint_callback],
        progress_bar=False,
    )

    final_model_path = (
        model_dir / "report_only_ppo_idc_bus9_final"
        if safe_train_config["report_only"]
        else model_dir / "safe_ppo_idc_bus9_final"
    )
    model.save(str(final_model_path))
    lagrangian_path = save_lagrangian_state(output_dir, manager)
    metadata_path = save_safe_training_metadata(
        output_dir=output_dir,
        configs=configs,
        safe_train_config=safe_train_config,
        manager=manager,
        obs_dim=obs_dim,
        action_dim=action_dim,
        timesteps=args.timesteps,
        run_name=args.run_name,
        final_model_path=final_model_path,
        connectivity_summary=connectivity_summary,
    )

    vec_env.close()

    print("\n>>> Safe PPO training finished")
    print(f"Final model: {final_model_path}.zip")
    print(f"Lagrangian state: {lagrangian_path}")
    print(f"Metadata: {metadata_path}")
    print(f"TensorBoard command: tensorboard --logdir {log_dir}")


def _finite_or_nan(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return number if np.isfinite(number) else float("nan")


def _finite_values(values: list[float]) -> list[float]:
    return [float(value) for value in values if np.isfinite(value)]


def _mean_finite(values: list[float]) -> float:
    finite = _finite_values(values)
    return float(np.mean(finite)) if finite else float("nan")


def _sum_finite(values: list[float]) -> float:
    finite = _finite_values(values)
    return float(np.sum(finite)) if finite else 0.0


def _min_finite(values: list[float]) -> float:
    finite = _finite_values(values)
    return float(np.min(finite)) if finite else float("nan")


def _max_finite(values: list[float]) -> float:
    finite = _finite_values(values)
    return float(np.max(finite)) if finite else float("nan")


if __name__ == "__main__":
    main()
