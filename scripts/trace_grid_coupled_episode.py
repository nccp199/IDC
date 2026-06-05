"""Trace one 24h GridCoupledEnv episode without training.

Examples:
    python -m scripts.trace_grid_coupled_episode --policy rule --cache off
    python -m scripts.trace_grid_coupled_episode --policy ppo --ppo-model-path path/to/model.zip
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
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

from configs.config_ultimate import (  # noqa: E402
    DATA_CONFIG,
    ENV_CONFIG,
    GRID_CACHE_CONFIG,
    GRID_CONFIG,
    GRID_REWARD_CONFIG,
    GRID_SCENARIO_CONFIG,
    IDC_SCALE_CONFIG,
    REWARD_CONFIG,
)
from data_io.data_loader import build_external_series_from_config  # noqa: E402
from env_wrappers import GridCoupledEnv  # noqa: E402
from envs.idc_price_env import IDCPriceEnv20D  # noqa: E402
from eval.eval_base import POLICY_FUNCS  # noqa: E402
from grid_model.ieee14_loader import get_bus_index_by_ieee_number  # noqa: E402


POLICY_ALIASES = {
    "zero": "ZERO",
    "one": "ONE",
    "random": "RANDOM",
    "rule": "RULE",
    "fast_neutral_bess": "FAST_NEUTRAL_BESS",
    "uniform_neutral_bess": "UNIFORM_NEUTRAL_BESS",
    "rule_price_only": "RULE_PRICE_ONLY",
    "rule_price_bess": "RULE_PRICE_BESS",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", type=str, default="rule", choices=sorted(list(POLICY_ALIASES) + ["ppo"]))
    parser.add_argument("--ppo-model-path", type=str, default="")
    parser.add_argument("--seed", type=int, default=3000)
    parser.add_argument("--cache", choices=["on", "off"], default="off")
    parser.add_argument("--out", type=str, default="outputs/diagnostics/grid_episode_trace.csv")
    args = parser.parse_args()

    env = make_env(seed=args.seed, cache_mode=args.cache)
    rng = np.random.default_rng(int(args.seed) + 12345)
    ppo_model = load_ppo_model(args.ppo_model_path) if args.policy == "ppo" else None

    rows: list[dict[str, Any]] = []
    obs, reset_info = env.reset(seed=args.seed)
    done = False
    total_reward = 0.0
    case_stats = raw_case_stats(env)

    while not done:
        action = choose_action(args.policy, env, obs, rng, ppo_model)
        obs, reward, terminated, truncated, info = env.step(action)
        done = bool(terminated or truncated)
        total_reward += float(reward)
        rows.append(build_row(info, action, reward, total_reward, case_stats))

    env.close()

    out_path = PROJECT_ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    write_csv(out_path, rows)
    print(f"Saved episode trace: {out_path}")


def make_env(seed: int, cache_mode: str) -> GridCoupledEnv:
    base_env = IDCPriceEnv20D(
        **ENV_CONFIG,
        **REWARD_CONFIG,
        **IDC_SCALE_CONFIG,
        **build_external_series_from_config(DATA_CONFIG, ENV_CONFIG["horizon"]),
        server_seed=seed,
        task_seed=seed,
    )
    cache_enabled = str(cache_mode).lower() == "on"
    cache_config = dict(GRID_CACHE_CONFIG)
    cache_config.update(
        {
            "enable_grid_cache": cache_enabled,
            "cache_opf": cache_enabled,
            "cache_mef": cache_enabled,
            "cache_failed_results": False,
        }
    )
    env = GridCoupledEnv(
        base_env,
        grid_config=dict(GRID_CONFIG),
        grid_reward_config=dict(GRID_REWARD_CONFIG),
        grid_scenario_config=dict(GRID_SCENARIO_CONFIG),
        grid_cache_config=cache_config,
    )
    env.action_space.seed(seed)
    return env


def choose_action(policy: str, env: GridCoupledEnv, obs: np.ndarray, rng: np.random.Generator, ppo_model) -> np.ndarray:
    policy_name = str(policy).lower()
    if policy_name == "ppo":
        if ppo_model is None:
            raise ValueError("--ppo-model-path is required when --policy ppo.")
        action, _state = ppo_model.predict(obs, deterministic=True)
        return np.asarray(action, dtype=np.float32)

    eval_policy_name = POLICY_ALIASES[policy_name]
    return POLICY_FUNCS[eval_policy_name](env.env, rng)


def load_ppo_model(path: str):
    model_path = Path(path)
    if not model_path.exists():
        raise FileNotFoundError(f"PPO model not found: {model_path}")
    try:
        from stable_baselines3 import PPO
    except Exception as exc:
        raise RuntimeError(f"stable_baselines3 is required to load PPO models: {exc}") from exc
    return PPO.load(str(model_path))


def raw_case_stats(env: GridCoupledEnv) -> dict[str, float]:
    net = env.grid_case.raw_network
    bus9_idx = get_bus_index_by_ieee_number(env.grid_case, 9)
    load_bus9 = net.load[net.load.bus == bus9_idx]
    if len(load_bus9) and "scaling" in net.load.columns:
        bus9_base_load_mw = float((load_bus9.p_mw.astype(float) * load_bus9.scaling.astype(float)).sum())
    elif len(load_bus9):
        bus9_base_load_mw = float(load_bus9.p_mw.astype(float).sum())
    else:
        bus9_base_load_mw = 0.0

    base_total_load_mw = (
        float((net.load.p_mw.astype(float) * net.load.scaling.astype(float)).sum())
        if "scaling" in net.load.columns
        else float(net.load.p_mw.astype(float).sum())
    )
    return {
        "bus9_base_load_mw": bus9_base_load_mw,
        "base_total_load_mw": base_total_load_mw,
    }


def build_row(
    info: dict[str, Any],
    action: np.ndarray,
    reward: float,
    total_reward: float,
    case_stats: dict[str, float],
) -> dict[str, Any]:
    idc_load_mw = finite(info.get("grid_idc_load_mw"))
    grid_load_scale = finite(info.get("grid_load_scale"))
    total_grid_load_mw = finite(info.get("grid_total_load_mw"))
    bus9_scaled_load = case_stats["bus9_base_load_mw"] * grid_load_scale if math.isfinite(grid_load_scale) else math.nan
    grid_reward_penalty = finite(info.get("grid_reward_penalty"), 0.0)

    return {
        "hour": int(finite(info.get("hour"), 0.0)),
        "action_mean": finite(info.get("action_mean")),
        "P_IDC_kW": finite(info.get("P_IDC_kW")),
        "P_local_demand_kW": finite(info.get("P_local_demand_kW")),
        "P_local_net_before_pv_kW": finite(info.get("P_local_net_before_pv_kW")),
        "P_bus_net_kW": finite(info.get("P_bus_net_kW")),
        "P_grid_kW": finite(info.get("P_grid_kW")),
        "pv_available_kW": finite(info.get("pv_available_kW")),
        "pv_used_kW": finite(info.get("pv_used_kW")),
        "pv_curtail_kW": finite(info.get("pv_curtail_kW")),
        "total_pv_available_kWh": finite(info.get("total_pv_available_kWh")),
        "total_pv_used_kWh": finite(info.get("total_pv_used_kWh")),
        "total_pv_curtail_kWh": finite(info.get("total_pv_curtail_kWh")),
        "pv_utilization_rate": finite(info.get("pv_utilization_rate")),
        "renewable_share": finite(info.get("renewable_share")),
        "grid_bus_net_load_mw": finite(info.get("grid_bus_net_load_mw")),
        "idc_load_mw": idc_load_mw,
        "grid_load_scale": grid_load_scale,
        "total_grid_load_mw": total_grid_load_mw,
        "bus9_base_load_mw": case_stats["bus9_base_load_mw"],
        "bus9_scaled_load_mw": bus9_scaled_load,
        "idc_load_ratio_system": safe_div(idc_load_mw, total_grid_load_mw),
        "idc_load_ratio_bus9": safe_div(idc_load_mw, bus9_scaled_load),
        "opf_success": bool(info.get("grid_opf_success", False)),
        "minV": finite(info.get("grid_min_voltage_pu")),
        "maxV": finite(info.get("grid_max_voltage_pu")),
        "maxLine": finite(info.get("grid_max_line_loading_percent")),
        "network_loss": finite(info.get("grid_network_loss_mw")),
        "LMP": finite(info.get("grid_lmp")),
        "MEF_plus": finite(info.get("grid_mef_plus")),
        "MEF_minus": finite(info.get("grid_mef_minus")),
        "safe_cost": finite(info.get("safe_violation_cost", info.get("safe_cost_total"))),
        "voltage_violation": finite(info.get("safe_violation_voltage"), 0.0),
        "line_violation": finite(info.get("safe_violation_line"), 0.0),
        "opf_violation": finite(info.get("safe_violation_opf"), 0.0),
        "reward": float(reward),
        "total_reward": float(total_reward),
        "grid_reward": -grid_reward_penalty,
        "grid_reward_penalty": grid_reward_penalty,
        "base_reward": finite(info.get("base_reward")),
        "grid_adjusted_reward": finite(info.get("grid_adjusted_reward")),
        "grid_opf_message": info.get("grid_opf_message", ""),
        "grid_mef_success": bool(info.get("grid_mef_success", False)),
        "grid_mef_message": info.get("grid_mef_message", ""),
        "bess_soc": finite(info.get("bess_soc")),
        "bess_charge_power_kW": finite(info.get("bess_charge_power_kW")),
        "bess_discharge_power_kW": finite(info.get("bess_discharge_power_kW")),
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def finite(value: Any, default: float = math.nan) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if math.isfinite(number) else default


def safe_div(numerator: float, denominator: float) -> float:
    if not math.isfinite(numerator) or not math.isfinite(denominator) or abs(denominator) <= 1e-12:
        return math.nan
    return numerator / denominator


if __name__ == "__main__":
    main()
