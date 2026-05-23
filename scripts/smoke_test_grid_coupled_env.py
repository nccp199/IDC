"""Smoke test for GridCoupledEnv.

Run with:
    python -m scripts.smoke_test_grid_coupled_env
"""

from __future__ import annotations

import math
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

from configs.config_ultimate import (  # noqa: E402
    DATA_CONFIG,
    ENV_CONFIG,
    GRID_CONFIG,
    GRID_REWARD_CONFIG,
    REWARD_CONFIG,
)
from data_io.data_loader import build_external_series_from_config  # noqa: E402
from env_wrappers import GridCoupledEnv  # noqa: E402
from envs.idc_price_env import IDCPriceEnv20D  # noqa: E402


def main() -> None:
    base_env = IDCPriceEnv20D(
        **ENV_CONFIG,
        **REWARD_CONFIG,
        **build_external_series_from_config(DATA_CONFIG, ENV_CONFIG["horizon"]),
        server_seed=2026,
        task_seed=2026,
    )
    env = GridCoupledEnv(base_env, grid_config=GRID_CONFIG, grid_reward_config=GRID_REWARD_CONFIG)
    env.action_space.seed(2026)

    obs, info = env.reset(seed=2026)
    print(f"base_obs_dim={env.base_obs_dim}")
    print(f"grid_obs_dim={env.grid_obs_dim}")
    print(f"final_obs_dim={obs.shape[0]}")
    print(f"action_dim={env.action_space.shape[0]}")
    print(f"enable_grid_obs={env.enable_grid_obs}")
    print(f"grid_initial_obs_mode={info.get('grid_initial_obs_mode')}")
    print(
        "grid_case="
        f"{info['grid_case_name']} | mode={info['grid_opf_mode']} | "
        f"IDC IEEE bus={info['grid_idc_ieee_bus_number']} | "
        f"pandapower idx={info['grid_idc_bus_index']}"
    )

    done = False
    total_reward = 0.0
    opf_success_count = 0
    opf_fail_count = 0
    mef_success_count = 0
    mef_fail_count = 0
    lmp_values: list[float] = []
    mef_values: list[float] = []
    mef_minus_values: list[float] = []
    line_loading_values: list[float] = []
    voltage_values: list[float] = []
    total_safe_cost = 0.0
    reward_mismatch_count = 0
    final_info = info

    print("\nhour | obs_dim | P_grid_kW | load_MW | LMP | MEF+ | minV | maxLine% | OPF | base_reward | adjusted_reward")
    print("-" * 122)

    while not done:
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        final_info = info
        total_reward += float(reward)

        if info.get("grid_opf_success"):
            opf_success_count += 1
        else:
            opf_fail_count += 1
            print(f"WARNING: OPF failed at hour {info.get('hour')}: {info.get('grid_opf_message')}")

        if info.get("grid_mef_success"):
            mef_success_count += 1
        else:
            mef_fail_count += 1
            print(f"WARNING: MEF failed at hour {info.get('hour')}: {info.get('grid_mef_message')}")

        _append_finite(lmp_values, info.get("grid_lmp"))
        _append_finite(mef_values, info.get("grid_mef_plus"))
        _append_finite(mef_minus_values, info.get("grid_mef_minus"))
        _append_finite(line_loading_values, info.get("grid_max_line_loading_percent"))
        _append_finite(voltage_values, info.get("grid_min_voltage_pu"))
        total_safe_cost += _finite_or_zero(info.get("safe_cost_total"))
        if abs(_finite_or_zero(info.get("base_reward")) - _finite_or_zero(info.get("grid_adjusted_reward"))) > 1e-9:
            reward_mismatch_count += 1

        print(
            f"{int(info.get('hour', -1)):>4} | "
            f"{int(obs.shape[0]):>7} | "
            f"{_fmt(info.get('P_grid_kW')):>9} | "
            f"{_fmt(info.get('grid_idc_load_mw')):>7} | "
            f"{_fmt(info.get('grid_lmp')):>7} | "
            f"{_fmt(info.get('grid_mef_plus')):>7} | "
            f"{_fmt(info.get('grid_min_voltage_pu')):>7} | "
            f"{_fmt(info.get('grid_max_line_loading_percent')):>9} | "
            f"{str(info.get('grid_opf_success')):>5} | "
            f"{_fmt(info.get('base_reward')):>11} | "
            f"{_fmt(info.get('grid_adjusted_reward')):>15}"
        )

    print("-" * 122)
    print("[SUMMARY]")
    step_count = opf_success_count + opf_fail_count
    print(f"total_reward: {_fmt(total_reward)}")
    print(f"total_cost: {_fmt(final_info.get('total_cost'))}")
    print(f"total_carbon_emission: {_fmt(final_info.get('total_carbon_emission'))}")
    print(f"total_grid_energy_kWh: {_fmt(final_info.get('total_grid_energy_kWh'))}")
    print(f"completion_rate: {_fmt(final_info.get('completion_rate'))}")
    print(f"task_completion_rate: {_fmt(final_info.get('task_completion_rate'))}")
    print(f"grid_opf_success_count: {opf_success_count}")
    print(f"grid_opf_fail_count: {opf_fail_count}")
    print(f"grid_mef_success_count: {mef_success_count}")
    print(f"grid_mef_fail_count: {mef_fail_count}")
    print(f"grid_opf_success_rate: {_fmt(opf_success_count / max(step_count, 1))}")
    print(f"grid_mef_success_rate: {_fmt(mef_success_count / max(step_count, 1))}")
    print(f"avg_grid_lmp: {_fmt(_mean(lmp_values))}")
    print(f"avg_grid_mef_plus: {_fmt(_mean(mef_values))}")
    print(f"avg_grid_mef_minus: {_fmt(_mean(mef_minus_values))}")
    print(f"max_grid_line_loading: {_fmt(max(line_loading_values) if line_loading_values else math.nan)}")
    print(f"min_grid_voltage: {_fmt(min(voltage_values) if voltage_values else math.nan)}")
    print(f"total_safe_cost: {_fmt(total_safe_cost)}")
    print(f"reward_mismatch_count: {reward_mismatch_count}")


def _append_finite(values: list[float], value) -> None:
    number = _to_float(value)
    if math.isfinite(number):
        values.append(number)


def _mean(values: list[float]) -> float:
    if not values:
        return math.nan
    return sum(values) / len(values)


def _finite_or_zero(value) -> float:
    number = _to_float(value)
    return number if math.isfinite(number) else 0.0


def _to_float(value) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def _fmt(value) -> str:
    number = _to_float(value)
    if not math.isfinite(number):
        return "nan"
    return f"{number:.6f}"


if __name__ == "__main__":
    main()
