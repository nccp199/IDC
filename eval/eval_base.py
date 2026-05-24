"""
Unified baseline evaluation for IDCPriceEnv20D_ultimate.

This script evaluates simple policies and optionally PPO, then merges existing
GA/PSO CSV results into one final comparison table.

Run examples:
    python -m eval.eval_base --quick
    python -m eval.eval_base --start-seed 3000 --n-seeds 30
    python -m eval.eval_base --start-seed 3000 --n-seeds 30 --no-ppo
    python -m eval.eval_base --ppo-model ppo_outputs_ultimate_500k/best_model/best_model.zip

Legacy root command is still supported:
    python eval_base.py --quick
"""

from __future__ import annotations

import argparse
import csv
import math
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from envs.idc_price_env import IDCPriceEnv20D
from configs.config_ultimate import (
    GRID_CONFIG,
    GRID_REWARD_CONFIG,
    GRID_SCENARIO_CONFIG,
    IDC_SCALE_CONFIG,
    resolve_output_path,
)
from data_io.data_loader import build_external_series_from_config
from configs.experiment_cases import get_experiment_case, print_experiment_case
from env_wrappers import GridCoupledEnv


ACTIVE_CASE_CONFIG = get_experiment_case("main")


# =========================
# Shared environment config
# =========================

def make_env(seed: int) -> GridCoupledEnv:
    """
    Create evaluation environment from config_ultimate.py.

    The seed fixes both server parameters and task generation,
    so PPO / GA / PSO / rule baselines are compared on the same scenarios.
    """
    env_config = ACTIVE_CASE_CONFIG["env_config"]
    reward_config = ACTIVE_CASE_CONFIG["reward_config"]
    data_config = ACTIVE_CASE_CONFIG["data_config"]
    env_kwargs = {
        **env_config,
        **reward_config,
        **IDC_SCALE_CONFIG,
        **build_external_series_from_config(data_config, env_config["horizon"]),
        "server_seed": seed,
        "task_seed": seed,
    }
    base_env = IDCPriceEnv20D(**env_kwargs)
    return GridCoupledEnv(base_env, GRID_CONFIG, GRID_REWARD_CONFIG, GRID_SCENARIO_CONFIG)


# =========================
# Metrics utilities
# =========================

METRIC_KEYS = [
    "total_reward",
    "completion_rate",
    "task_completion_rate",
    "total_completed_work",
    "final_backlog_work",
    "overflow_work",
    "deadline_miss_rate",
    "sla_violation_rate",
    "sla_penalty",
    "avg_task_delay",
    "max_task_delay",
    "avg_waiting_time",
    "avg_turnaround_time",
    "total_cost",
    "unit_task_cost",
    "P_grid_kW",
    "grid_energy_kWh",
    "idc_energy_kWh",
    "carbon_cost",
    "total_energy_kWh",
    "total_grid_energy_kWh",
    "total_idc_energy_kWh",
    "energy_per_task",
    "idc_energy_per_task",
    "total_carbon_emission",
    "total_carbon_cost",
    "carbon_per_task",
    "episode_grid_peak_power_kW",
    "total_grid_peak_excess_kW_hour",
    "episode_peak_power_kW",
    "total_peak_excess_kW_hour",
    "load_change",
    "action_change",
    "bess_soc",
    "total_bess_charge_kWh",
    "total_bess_discharge_kWh",
    "total_bess_degradation_cost",
    "finished_task_count",
    "total_task_count",
    "deadline_miss_count",
    "sla_violation_count",
    "total_pause_count",
    "total_resume_count",
    "total_non_interruptible_interruption_count",
    "grid_opf_success_rate",
    "grid_mef_success_rate",
    "avg_grid_lmp",
    "avg_grid_mef_plus",
    "avg_grid_mef_minus",
    "avg_grid_total_generation_cost",
    "avg_grid_total_emission_kg",
    "avg_grid_network_loss_mw",
    "avg_grid_load_scale",
    "avg_grid_reference_usep",
    "min_grid_voltage_pu",
    "max_grid_line_loading_percent",
    "total_grid_security_penalty",
    "total_safe_violation_cost",
    "total_safe_violation_voltage",
    "total_safe_violation_line",
    "total_safe_violation_opf",
    "avg_safe_violation_cost",
    "max_safe_violation_cost",
    "total_safe_cost",
    "total_grid_reward_penalty",
    "avg_grid_reward_penalty",
    "total_grid_lmp_cost_penalty",
    "total_grid_mef_carbon_penalty",
    "total_grid_safe_violation_penalty",
    "reward_mismatch_count",
    "grid_opf_fail_count",
    "grid_mef_fail_count",
]

SUMMARY_KEYS = [
    "total_reward",
    "completion_rate",
    "task_completion_rate",
    "total_completed_work",
    "final_backlog_work",
    "overflow_work",
    "deadline_miss_rate",
    "sla_violation_rate",
    "sla_penalty",
    "avg_task_delay",
    "avg_waiting_time",
    "avg_turnaround_time",
    "total_cost",
    "unit_task_cost",
    "total_energy_kWh",
    "total_grid_energy_kWh",
    "total_idc_energy_kWh",
    "energy_per_task",
    "idc_energy_per_task",
    "total_carbon_emission",
    "total_carbon_cost",
    "carbon_per_task",
    "episode_grid_peak_power_kW",
    "total_grid_peak_excess_kW_hour",
    "episode_peak_power_kW",
    "total_peak_excess_kW_hour",
    "bess_soc",
    "total_bess_charge_kWh",
    "total_bess_discharge_kWh",
    "total_bess_degradation_cost",
    "grid_opf_success_rate",
    "grid_mef_success_rate",
    "avg_grid_lmp",
    "avg_grid_mef_plus",
    "avg_grid_mef_minus",
    "avg_grid_total_generation_cost",
    "avg_grid_total_emission_kg",
    "avg_grid_network_loss_mw",
    "avg_grid_load_scale",
    "avg_grid_reference_usep",
    "min_grid_voltage_pu",
    "max_grid_line_loading_percent",
    "total_grid_security_penalty",
    "total_safe_violation_cost",
    "total_safe_violation_voltage",
    "total_safe_violation_line",
    "total_safe_violation_opf",
    "avg_safe_violation_cost",
    "max_safe_violation_cost",
    "total_safe_cost",
    "total_grid_reward_penalty",
    "avg_grid_reward_penalty",
    "total_grid_lmp_cost_penalty",
    "total_grid_mef_carbon_penalty",
    "total_grid_safe_violation_penalty",
    "reward_mismatch_count",
    "grid_opf_fail_count",
    "grid_mef_fail_count",
]

PRIMARY_PRINT_KEYS = [
    "completion_rate",
    "task_completion_rate",
    "final_backlog_work",
    "deadline_miss_rate",
    "sla_violation_rate",
    "unit_task_cost",
    "carbon_per_task",
    "total_grid_energy_kWh",
    "total_idc_energy_kWh",
    "total_cost",
    "total_carbon_emission",
    "total_carbon_cost",
    "episode_grid_peak_power_kW",
    "episode_peak_power_kW",
    "total_reward",
    "avg_grid_lmp",
    "avg_grid_mef_plus",
    "avg_grid_mef_minus",
    "min_grid_voltage_pu",
    "max_grid_line_loading_percent",
    "total_safe_violation_cost",
    "total_safe_violation_voltage",
    "total_safe_violation_line",
    "total_safe_violation_opf",
    "total_safe_cost",
    "grid_opf_success_rate",
    "grid_mef_success_rate",
]


def safe_float(value: Any, default: float = np.nan) -> float:
    if value is None:
        return default
    try:
        if isinstance(value, str):
            text = value.strip()
            if not text:
                return default
            return float(text)
        return float(value)
    except Exception:
        return default


def _finite_values(infos: Sequence[Dict[str, Any]], key: str) -> List[float]:
    values = [safe_float(info.get(key)) for info in infos]
    return [float(value) for value in values if math.isfinite(float(value))]


def _mean_or_nan(values: Sequence[float]) -> float:
    return float(np.mean(values)) if values else np.nan


def _truthy(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y"}
    return bool(value)


def aggregate_grid_episode_metrics(episode_infos: Sequence[Dict[str, Any]]) -> Dict[str, float]:
    if not episode_infos:
        return {
            "grid_opf_success_rate": np.nan,
            "grid_mef_success_rate": np.nan,
            "avg_grid_lmp": np.nan,
            "avg_grid_mef_plus": np.nan,
            "avg_grid_mef_minus": np.nan,
            "avg_grid_total_generation_cost": np.nan,
            "avg_grid_total_emission_kg": np.nan,
            "avg_grid_network_loss_mw": np.nan,
            "avg_grid_load_scale": np.nan,
            "avg_grid_reference_usep": np.nan,
            "min_grid_voltage_pu": np.nan,
            "max_grid_line_loading_percent": np.nan,
            "total_grid_security_penalty": np.nan,
            "total_safe_violation_cost": np.nan,
            "total_safe_violation_voltage": np.nan,
            "total_safe_violation_line": np.nan,
            "total_safe_violation_opf": np.nan,
            "avg_safe_violation_cost": np.nan,
            "max_safe_violation_cost": np.nan,
            "total_safe_cost": np.nan,
            "total_grid_reward_penalty": np.nan,
            "avg_grid_reward_penalty": np.nan,
            "total_grid_lmp_cost_penalty": np.nan,
            "total_grid_mef_carbon_penalty": np.nan,
            "total_grid_safe_violation_penalty": np.nan,
            "reward_mismatch_count": np.nan,
            "grid_opf_fail_count": np.nan,
            "grid_mef_fail_count": np.nan,
        }

    n_steps = len(episode_infos)
    opf_success_count = sum(1 for info in episode_infos if bool(info.get("grid_opf_success", False)))
    mef_success_count = sum(1 for info in episode_infos if bool(info.get("grid_mef_success", False)))
    min_voltage_values = _finite_values(episode_infos, "grid_min_voltage_pu")
    max_line_values = _finite_values(episode_infos, "grid_max_line_loading_percent")
    safe_violation_cost_values = _finite_values(episode_infos, "safe_violation_cost")
    if not safe_violation_cost_values:
        safe_violation_cost_values = _finite_values(episode_infos, "safe_cost_total")
    safe_violation_voltage_values = _finite_values(episode_infos, "safe_violation_voltage")
    if not safe_violation_voltage_values:
        safe_violation_voltage_values = _finite_values(episode_infos, "safe_cost_voltage")
    safe_violation_line_values = _finite_values(episode_infos, "safe_violation_line")
    if not safe_violation_line_values:
        safe_violation_line_values = _finite_values(episode_infos, "safe_cost_line")
    safe_violation_opf_values = _finite_values(episode_infos, "safe_violation_opf")
    if not safe_violation_opf_values:
        safe_violation_opf_values = _finite_values(episode_infos, "safe_cost_opf")
    grid_reward_penalty_values = _finite_values(episode_infos, "grid_reward_penalty")
    reward_mismatch_count = 0
    for info in episode_infos:
        if _truthy(info.get("grid_reward_enabled", False)):
            continue
        base_reward = safe_float(info.get("base_reward"))
        adjusted_reward = safe_float(info.get("grid_adjusted_reward"))
        if math.isfinite(base_reward) and math.isfinite(adjusted_reward):
            if abs(base_reward - adjusted_reward) > 1e-9:
                reward_mismatch_count += 1

    return {
        "grid_opf_success_rate": opf_success_count / max(n_steps, 1),
        "grid_mef_success_rate": mef_success_count / max(n_steps, 1),
        "avg_grid_lmp": _mean_or_nan(_finite_values(episode_infos, "grid_lmp")),
        "avg_grid_mef_plus": _mean_or_nan(_finite_values(episode_infos, "grid_mef_plus")),
        "avg_grid_mef_minus": _mean_or_nan(_finite_values(episode_infos, "grid_mef_minus")),
        "avg_grid_total_generation_cost": _mean_or_nan(_finite_values(episode_infos, "grid_total_generation_cost")),
        "avg_grid_total_emission_kg": _mean_or_nan(_finite_values(episode_infos, "grid_total_emission_kg")),
        "avg_grid_network_loss_mw": _mean_or_nan(_finite_values(episode_infos, "grid_network_loss_mw")),
        "avg_grid_load_scale": _mean_or_nan(_finite_values(episode_infos, "grid_load_scale")),
        "avg_grid_reference_usep": _mean_or_nan(_finite_values(episode_infos, "grid_reference_usep")),
        "min_grid_voltage_pu": min(min_voltage_values) if min_voltage_values else np.nan,
        "max_grid_line_loading_percent": max(max_line_values) if max_line_values else np.nan,
        "total_grid_security_penalty": float(np.sum(_finite_values(episode_infos, "grid_security_penalty"))),
        "total_safe_violation_cost": float(np.sum(safe_violation_cost_values)),
        "total_safe_violation_voltage": float(np.sum(safe_violation_voltage_values)),
        "total_safe_violation_line": float(np.sum(safe_violation_line_values)),
        "total_safe_violation_opf": float(np.sum(safe_violation_opf_values)),
        "avg_safe_violation_cost": _mean_or_nan(safe_violation_cost_values),
        "max_safe_violation_cost": max(safe_violation_cost_values) if safe_violation_cost_values else np.nan,
        "total_safe_cost": float(np.sum(safe_violation_cost_values)),
        "total_grid_reward_penalty": float(np.sum(grid_reward_penalty_values)),
        "avg_grid_reward_penalty": _mean_or_nan(grid_reward_penalty_values),
        "total_grid_lmp_cost_penalty": float(np.sum(_finite_values(episode_infos, "grid_lmp_cost_penalty"))),
        "total_grid_mef_carbon_penalty": float(np.sum(_finite_values(episode_infos, "grid_mef_carbon_penalty"))),
        "total_grid_safe_violation_penalty": float(np.sum(_finite_values(episode_infos, "grid_safe_violation_penalty"))),
        "reward_mismatch_count": float(reward_mismatch_count),
        "grid_opf_fail_count": n_steps - opf_success_count,
        "grid_mef_fail_count": n_steps - mef_success_count,
    }


def final_metrics_from_info(
    total_reward: float,
    info: Dict[str, Any],
    episode_infos: Optional[Sequence[Dict[str, Any]]] = None,
) -> Dict[str, float]:
    metrics = {
        "total_reward": float(total_reward),
        "fitness": float(total_reward),
        "total_energy_kWh": safe_float(info.get("total_energy_kWh")),
        "P_grid_kW": safe_float(info.get("P_grid_kW")),
        "grid_energy_kWh": safe_float(info.get("grid_energy_kWh")),
        "idc_energy_kWh": safe_float(info.get("idc_energy_kWh")),
        "carbon_cost": safe_float(info.get("carbon_cost")),
        "total_grid_energy_kWh": safe_float(info.get("total_grid_energy_kWh")),
        "total_idc_energy_kWh": safe_float(info.get("total_idc_energy_kWh")),
        "total_cost": safe_float(info.get("total_cost")),
        "total_completed_work": safe_float(info.get("total_completed_work")),
        "completion_rate": safe_float(info.get("completion_rate")),
        "unit_task_cost": safe_float(info.get("unit_task_cost")),
        "energy_per_task": safe_float(info.get("energy_per_task")),
        "idc_energy_per_task": safe_float(info.get("idc_energy_per_task")),
        "final_backlog_work": safe_float(info.get("final_backlog_work", info.get("Q"))),
        "overflow_work": safe_float(info.get("overflow_work")),
        "task_completion_rate": safe_float(info.get("task_completion_rate")),
        "finished_task_count": safe_float(info.get("finished_task_count")),
        "total_task_count": safe_float(info.get("total_task_count")),
        "deadline_miss_rate": safe_float(info.get("deadline_miss_rate")),
        "deadline_miss_count": safe_float(info.get("deadline_miss_count")),
        "sla_penalty": safe_float(info.get("sla_penalty")),
        "sla_violation_rate": safe_float(info.get("sla_violation_rate")),
        "sla_violation_count": safe_float(info.get("sla_violation_count")),
        "avg_task_delay": safe_float(info.get("avg_task_delay")),
        "max_task_delay": safe_float(info.get("max_task_delay")),
        "avg_waiting_time": safe_float(info.get("avg_waiting_time")),
        "avg_turnaround_time": safe_float(info.get("avg_turnaround_time")),
        "total_carbon_emission": safe_float(info.get("total_carbon_emission")),
        "total_carbon_cost": safe_float(info.get("total_carbon_cost")),
        "carbon_per_task": safe_float(info.get("carbon_per_task")),
        "episode_grid_peak_power_kW": safe_float(info.get("episode_grid_peak_power_kW")),
        "total_grid_peak_excess_kW_hour": safe_float(info.get("total_grid_peak_excess_kW_hour")),
        "episode_peak_power_kW": safe_float(info.get("episode_peak_power_kW")),
        "total_peak_excess_kW_hour": safe_float(info.get("total_peak_excess_kW_hour")),
        "load_change": safe_float(info.get("load_change")),
        "action_change": safe_float(info.get("action_change")),
        "bess_soc": safe_float(info.get("bess_soc")),
        "total_bess_charge_kWh": safe_float(info.get("total_bess_charge_kWh")),
        "total_bess_discharge_kWh": safe_float(info.get("total_bess_discharge_kWh")),
        "total_bess_degradation_cost": safe_float(info.get("total_bess_degradation_cost")),
        "total_pause_count": safe_float(info.get("total_pause_count")),
        "total_resume_count": safe_float(info.get("total_resume_count")),
        "total_non_interruptible_interruption_count": safe_float(
            info.get("total_non_interruptible_interruption_count")
        ),
    }
    metrics.update(aggregate_grid_episode_metrics(episode_infos or []))
    return metrics


def build_hourly_row(
    algorithm: str,
    env_seed: int,
    run_idx: int,
    step: int,
    action: np.ndarray,
    reward: float,
    info: Dict[str, Any],
) -> Dict[str, Any]:
    """Build one unified hourly row for all online policies."""
    return {
        "algorithm": algorithm,
        "seed": int(env_seed),
        "run_idx": int(run_idx),
        "step": int(step),
        "hour": int(info.get("hour", step)),
        "price": safe_float(info.get("price")),
        "carbon_factor": safe_float(info.get("carbon_factor")),
        "PV": safe_float(info.get("PV")),
        "WT": safe_float(info.get("WT")),
        "lambda_t": safe_float(info.get("lambda_t")),
        "action_mean": safe_float(info.get("action_mean", np.mean(action))),
        "action_min": safe_float(info.get("action_min")),
        "action_max": safe_float(info.get("action_max")),
        "urgent_preference": safe_float(info.get("urgent_preference")),
        "continuity_preference": safe_float(info.get("continuity_preference")),
        "bess_raw_action": safe_float(info.get("bess_raw_action")),
        "planned_task_load_mean": safe_float(info.get("planned_task_load_mean")),
        "planned_total_load_mean": safe_float(info.get("planned_total_load_mean")),
        "actual_task_load_mean": safe_float(info.get("actual_task_load_mean")),
        "actual_total_load_mean": safe_float(info.get("actual_total_load_mean")),
        "planned_capacity": safe_float(info.get("planned_capacity")),
        "completed_work": safe_float(info.get("completed_work")),
        "unused_capacity": safe_float(info.get("unused_capacity")),
        "backlog_work": safe_float(info.get("backlog_work", info.get("Q"))),
        "queue_capacity_ref": safe_float(info.get("queue_capacity_ref")),
        "overflow_work": safe_float(info.get("overflow_work")),
        "server_group_size": safe_float(info.get("server_group_size")),
        "effective_total_server_count": safe_float(info.get("effective_total_server_count")),
        "task_workload_scale": safe_float(info.get("task_workload_scale")),
        "P_IDC": safe_float(info.get("P_IDC")),
        "P_IDC_kW": safe_float(info.get("P_IDC_kW")),
        "P_grid_kW": safe_float(info.get("P_grid_kW")),
        "grid_power_kW": safe_float(info.get("grid_power_kW")),
        "grid_power_limit_kW": safe_float(info.get("grid_power_limit_kW")),
        "bess_soc": safe_float(info.get("bess_soc")),
        "bess_energy_kWh": safe_float(info.get("bess_energy_kWh")),
        "bess_mode": info.get("bess_mode", ""),
        "bess_capacity_kWh": safe_float(info.get("bess_capacity_kWh")),
        "bess_charge_power_max_kW": safe_float(info.get("bess_charge_power_max_kW")),
        "bess_discharge_power_max_kW": safe_float(info.get("bess_discharge_power_max_kW")),
        "bess_scale_factor": safe_float(info.get("bess_scale_factor")),
        "desired_bess_charge_power_kW": safe_float(info.get("desired_bess_charge_power_kW")),
        "desired_bess_discharge_power_kW": safe_float(info.get("desired_bess_discharge_power_kW")),
        "bess_charge_power_kW": safe_float(info.get("bess_charge_power_kW")),
        "bess_discharge_power_kW": safe_float(info.get("bess_discharge_power_kW")),
        "bess_charge_kWh": safe_float(info.get("bess_charge_kWh")),
        "bess_discharge_kWh": safe_float(info.get("bess_discharge_kWh")),
        "bess_degradation_cost": safe_float(info.get("bess_degradation_cost")),
        "invalid_bess_action": safe_float(info.get("invalid_bess_action")),
        "soc_deviation": safe_float(info.get("soc_deviation")),
        "soc_excess": safe_float(info.get("soc_excess")),
        "total_bess_charge_kWh": safe_float(info.get("total_bess_charge_kWh")),
        "total_bess_discharge_kWh": safe_float(info.get("total_bess_discharge_kWh")),
        "total_bess_degradation_cost": safe_float(info.get("total_bess_degradation_cost")),
        "P_IT": safe_float(info.get("P_IT")),
        "P_cooling": safe_float(info.get("P_cooling")),
        "PUE": safe_float(info.get("PUE")),
        "COP": safe_float(info.get("COP")),
        "grid_peak_power_kW": safe_float(info.get("grid_peak_power_kW")),
        "grid_peak_excess_kW": safe_float(info.get("grid_peak_excess_kW")),
        "episode_grid_peak_power_kW": safe_float(info.get("episode_grid_peak_power_kW")),
        "total_grid_peak_excess_kW_hour": safe_float(info.get("total_grid_peak_excess_kW_hour")),
        "idc_peak_power_kW": safe_float(info.get("idc_peak_power_kW")),
        "peak_power_kW": safe_float(info.get("peak_power_kW")),
        "peak_excess_kW": safe_float(info.get("peak_excess_kW")),
        "episode_peak_power_kW": safe_float(info.get("episode_peak_power_kW")),
        "total_peak_excess_kW_hour": safe_float(info.get("total_peak_excess_kW_hour")),
        "energy_kWh": safe_float(info.get("energy_kWh")),
        "grid_energy_kWh": safe_float(info.get("grid_energy_kWh")),
        "idc_energy_kWh": safe_float(info.get("idc_energy_kWh")),
        "hourly_cost": safe_float(info.get("hourly_cost", info.get("cost"))),
        "carbon_emission": safe_float(info.get("carbon_emission")),
        "carbon_cost": safe_float(info.get("carbon_cost")),
        "new_deadline_miss_count": safe_float(info.get("new_deadline_miss_count")),
        "deadline_miss_count": safe_float(info.get("deadline_miss_count")),
        "sla_penalty": safe_float(info.get("sla_penalty")),
        "sla_violation_count": safe_float(info.get("sla_violation_count")),
        "sla_violation_rate": safe_float(info.get("sla_violation_rate")),
        "avg_task_delay": safe_float(info.get("avg_task_delay")),
        "max_task_delay": safe_float(info.get("max_task_delay")),
        "load_change": safe_float(info.get("load_change")),
        "action_change": safe_float(info.get("action_change")),
        "pause_count_this_step": safe_float(info.get("pause_count_this_step")),
        "resume_count_this_step": safe_float(info.get("resume_count_this_step")),
        "non_interruptible_interruption_this_step": safe_float(
            info.get("non_interruptible_interruption_this_step")
        ),
        "r_done": safe_float(info.get("r_done")),
        "r_cost": safe_float(info.get("r_cost")),
        "r_carbon": safe_float(info.get("r_carbon")),
        "r_queue": safe_float(info.get("r_queue")),
        "r_queue_overflow": safe_float(info.get("r_queue_overflow")),
        "r_urgent_backlog": safe_float(info.get("r_urgent_backlog")),
        "r_waiting": safe_float(info.get("r_waiting")),
        "r_deadline": safe_float(info.get("r_deadline")),
        "r_sla": safe_float(info.get("r_sla")),
        "r_unused": safe_float(info.get("r_unused")),
        "r_grid_peak": safe_float(info.get("r_grid_peak")),
        "r_peak_load": safe_float(info.get("r_peak_load")),
        "r_pause": safe_float(info.get("r_pause")),
        "r_resume": safe_float(info.get("r_resume")),
        "r_non_interruptible": safe_float(info.get("r_non_interruptible")),
        "r_load_smooth": safe_float(info.get("r_load_smooth")),
        "r_action_smooth": safe_float(info.get("r_action_smooth")),
        "r_bess_degradation": safe_float(info.get("r_bess_degradation")),
        "r_bess_invalid_action": safe_float(info.get("r_bess_invalid_action")),
        "r_final_queue": safe_float(info.get("r_final_queue")),
        "r_soc_final": safe_float(info.get("r_soc_final")),
        "reward": float(reward),
        "reward_total": safe_float(info.get("reward_total", reward)),
        "grid_enabled": bool(info.get("grid_enabled", False)),
        "grid_opf_mode": info.get("grid_opf_mode", ""),
        "grid_opf_success": bool(info.get("grid_opf_success", False)),
        "grid_mef_success": bool(info.get("grid_mef_success", False)),
        "grid_idc_ieee_bus_number": safe_float(info.get("grid_idc_ieee_bus_number")),
        "grid_idc_bus_index": safe_float(info.get("grid_idc_bus_index")),
        "grid_idc_load_mw": safe_float(info.get("grid_idc_load_mw")),
        "grid_load_scale": safe_float(info.get("grid_load_scale")),
        "grid_scenario_enabled": bool(info.get("grid_scenario_enabled", False)),
        "grid_reference_usep": safe_float(info.get("grid_reference_usep")),
        "grid_lmp": safe_float(info.get("grid_lmp")),
        "grid_mef_plus": safe_float(info.get("grid_mef_plus")),
        "grid_mef_minus": safe_float(info.get("grid_mef_minus")),
        "grid_total_generation_cost": safe_float(info.get("grid_total_generation_cost")),
        "grid_total_emission_kg": safe_float(info.get("grid_total_emission_kg")),
        "grid_total_load_mw": safe_float(info.get("grid_total_load_mw")),
        "grid_total_generation_mw": safe_float(info.get("grid_total_generation_mw")),
        "grid_network_loss_mw": safe_float(info.get("grid_network_loss_mw")),
        "grid_min_voltage_pu": safe_float(info.get("grid_min_voltage_pu")),
        "grid_max_voltage_pu": safe_float(info.get("grid_max_voltage_pu")),
        "grid_max_line_loading_percent": safe_float(info.get("grid_max_line_loading_percent")),
        "grid_voltage_violation_count": safe_float(info.get("grid_voltage_violation_count")),
        "grid_line_overload_count": safe_float(info.get("grid_line_overload_count")),
        "grid_voltage_violation_magnitude": safe_float(info.get("grid_voltage_violation_magnitude")),
        "grid_line_overload_magnitude": safe_float(info.get("grid_line_overload_magnitude")),
        "grid_security_penalty": safe_float(info.get("grid_security_penalty")),
        "safe_violation_voltage": safe_float(info.get("safe_violation_voltage")),
        "safe_violation_line": safe_float(info.get("safe_violation_line")),
        "safe_violation_opf": safe_float(info.get("safe_violation_opf")),
        "safe_violation_cost": safe_float(info.get("safe_violation_cost")),
        "safe_cost_voltage": safe_float(info.get("safe_cost_voltage")),
        "safe_cost_line": safe_float(info.get("safe_cost_line")),
        "safe_cost_opf": safe_float(info.get("safe_cost_opf")),
        "safe_cost_total": safe_float(info.get("safe_cost_total")),
        "grid_reward_enabled": bool(info.get("grid_reward_enabled", False)),
        "grid_reward_mode": info.get("grid_reward_mode", ""),
        "grid_lmp_cost": safe_float(info.get("grid_lmp_cost")),
        "grid_lmp_cost_norm": safe_float(info.get("grid_lmp_cost_norm")),
        "grid_lmp_cost_penalty": safe_float(info.get("grid_lmp_cost_penalty")),
        "grid_mef_carbon": safe_float(info.get("grid_mef_carbon")),
        "grid_mef_carbon_norm": safe_float(info.get("grid_mef_carbon_norm")),
        "grid_mef_carbon_penalty": safe_float(info.get("grid_mef_carbon_penalty")),
        "grid_safe_violation": safe_float(info.get("grid_safe_violation")),
        "grid_safe_violation_norm": safe_float(info.get("grid_safe_violation_norm")),
        "grid_safe_violation_penalty": safe_float(info.get("grid_safe_violation_penalty")),
        "base_reward": safe_float(info.get("base_reward")),
        "grid_reward_penalty": safe_float(info.get("grid_reward_penalty")),
        "grid_adjusted_reward": safe_float(info.get("grid_adjusted_reward")),
    }


# =========================
# Basic policy evaluation
# =========================

def action_zero(env: IDCPriceEnv20D, rng: Optional[np.random.Generator] = None) -> np.ndarray:
    # Extreme action baseline: action[22]=0 means maximum BESS charge tendency.
    action = np.zeros(env.action_dim, dtype=np.float32)
    return action


def action_one(env: IDCPriceEnv20D, rng: Optional[np.random.Generator] = None) -> np.ndarray:
    # Extreme action baseline: action[22]=1 means maximum BESS discharge tendency.
    return np.ones(env.action_dim, dtype=np.float32)


def action_random(env: IDCPriceEnv20D, rng: Optional[np.random.Generator] = None) -> np.ndarray:
    if rng is None:
        rng = np.random.default_rng()
    return rng.uniform(0.0, 1.0, size=env.action_dim).astype(np.float32)


def ensure_env_action_dim(action: np.ndarray, env: IDCPriceEnv20D) -> np.ndarray:
    action = np.asarray(action, dtype=np.float32).reshape(-1)
    if action.shape[0] == env.action_dim:
        return action
    if action.shape[0] == env.action_dim - 1:
        return np.concatenate([action, np.array([0.5], dtype=np.float32)]).astype(np.float32)
    raise ValueError(f"action shape should be ({env.action_dim},), got {action.shape}")


def action_rule(env: IDCPriceEnv20D, rng: Optional[np.random.Generator] = None) -> np.ndarray:
    """
    Price-aware rule policy:
    - valley price: high server action
    - flat price: medium server action
    - peak price: low server action
    """
    t = int(env.current_step)
    price = np.asarray(env.price_t, dtype=np.float64)
    price_now = float(price[t])
    low = float(np.min(price))
    high = float(np.max(price))

    if np.isclose(price_now, low):
        server_level = 0.95
    elif np.isclose(price_now, high):
        server_level = 0.25
    else:
        server_level = 0.55

    action = np.zeros(env.action_dim, dtype=np.float32)
    action[: env.model.N] = server_level
    action[env.model.N] = 0.85      # urgent preference
    action[env.model.N + 1] = 0.80  # continuity preference
    if env.action_dim > env.model.N + 2:
        if np.isclose(price_now, low):
            action[env.model.N + 2] = 0.20
        elif np.isclose(price_now, high):
            action[env.model.N + 2] = 0.80
        else:
            action[env.model.N + 2] = 0.50
    return np.clip(action, 0.0, 1.0).astype(np.float32)


def action_fast_neutral_bess(env: IDCPriceEnv20D, rng: Optional[np.random.Generator] = None) -> np.ndarray:
    """Traditional fast execution baseline without active BESS use."""
    action = np.zeros(env.action_dim, dtype=np.float32)
    action[: env.model.N] = 0.95
    action[env.model.N] = 0.85
    action[env.model.N + 1] = 0.80
    if env.action_dim > env.model.N + 2:
        action[env.model.N + 2] = 0.50
    return np.clip(action, 0.0, 1.0).astype(np.float32)


def action_uniform_neutral_bess(env: IDCPriceEnv20D, rng: Optional[np.random.Generator] = None) -> np.ndarray:
    """Fixed uniform-load baseline without active BESS use."""
    action = np.zeros(env.action_dim, dtype=np.float32)
    action[: env.model.N] = 0.60
    action[env.model.N] = 0.75
    action[env.model.N + 1] = 0.75
    if env.action_dim > env.model.N + 2:
        action[env.model.N + 2] = 0.50
    return np.clip(action, 0.0, 1.0).astype(np.float32)


def action_rule_price_only(env: IDCPriceEnv20D, rng: Optional[np.random.Generator] = None) -> np.ndarray:
    """Time-of-use price scheduling baseline with neutral BESS."""
    t = int(env.current_step)
    price = np.asarray(env.price_t, dtype=np.float64)
    price_now = float(price[t])
    low = float(np.min(price))
    high = float(np.max(price))

    if np.isclose(price_now, low):
        server_level = 0.95
    elif np.isclose(price_now, high):
        server_level = 0.25
    else:
        server_level = 0.55

    action = np.zeros(env.action_dim, dtype=np.float32)
    action[: env.model.N] = server_level
    action[env.model.N] = 0.85
    action[env.model.N + 1] = 0.80
    if env.action_dim > env.model.N + 2:
        action[env.model.N + 2] = 0.50
    return np.clip(action, 0.0, 1.0).astype(np.float32)


def action_rule_price_bess(env: IDCPriceEnv20D, rng: Optional[np.random.Generator] = None) -> np.ndarray:
    """Time-of-use price scheduling baseline with charge/discharge BESS rule."""
    return action_rule(env, rng)


POLICY_FUNCS = {
    "ZERO": action_zero,
    "ONE": action_one,
    "RANDOM": action_random,
    "RULE": action_rule,
    "FAST_NEUTRAL_BESS": action_fast_neutral_bess,
    "UNIFORM_NEUTRAL_BESS": action_uniform_neutral_bess,
    "RULE_PRICE_ONLY": action_rule_price_only,
    "RULE_PRICE_BESS": action_rule_price_bess,
}


def evaluate_basic_policy(
    algorithm: str,
    env_seed: int,
    rng_seed: Optional[int] = None,
    random_run: int = 0,
    hourly_rows: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    if algorithm not in POLICY_FUNCS:
        raise ValueError(f"Unknown basic policy: {algorithm}")

    env = make_env(env_seed)
    obs, reset_info = env.reset()
    rng = np.random.default_rng(rng_seed) if rng_seed is not None else None

    total_reward = 0.0
    info: Dict[str, Any] = {}
    episode_infos: List[Dict[str, Any]] = []

    for step in range(env.horizon):
        action = POLICY_FUNCS[algorithm](env, rng)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += float(reward)
        episode_infos.append(dict(info))

        if hourly_rows is not None:
            hourly_rows.append(
                build_hourly_row(
                    algorithm=algorithm,
                    env_seed=env_seed,
                    run_idx=random_run,
                    step=step,
                    action=action,
                    reward=float(reward),
                    info=info,
                )
            )

        if terminated or truncated:
            break

    row: Dict[str, Any] = {
        "algorithm": algorithm,
        "seed": int(env_seed),
        "source": "eval_policy",
        "run_idx": int(random_run),
    }
    row.update(final_metrics_from_info(total_reward, info, episode_infos))
    return row


# =========================
# PPO evaluation
# =========================

def resolve_ppo_model_path(model_arg: str) -> Optional[Path]:
    model_text = (model_arg or "").strip()
    model_lower = model_text.lower()
    if model_lower in {"", "none", "skip"}:
        return None

    if model_lower != "auto":
        path = Path(model_text).expanduser()
        if path.exists():
            if path.is_dir():
                for candidate in [path / "best_model.zip", path / "ppo_idc_ultimate_final.zip"]:
                    if candidate.exists():
                        return candidate
                raise FileNotFoundError(
                    f"PPO model path is a directory, but no best_model.zip or "
                    f"ppo_idc_ultimate_final.zip was found inside: {path}"
                )
            return path
        zip_path = path if path.suffix.lower() == ".zip" else path.with_suffix(".zip")
        if zip_path.exists():
            return zip_path
        raise FileNotFoundError(
            f"PPO model not found: {path}. Also tried: {zip_path}. "
            "Please pass an absolute path or a path relative to the project directory."
        )

    candidates = [
        resolve_output_path("ppo_outputs_ultimate/best_model/best_model.zip"),
        resolve_output_path("ppo_outputs_ultimate/models/ppo_idc_ultimate_final.zip"),
        resolve_output_path("ppo_outputs_ultimate/best_model.zip"),
        resolve_output_path("ppo_outputs_ultimate_main/best_model/best_model.zip"),
        resolve_output_path("ppo_outputs_ultimate_main/models/ppo_idc_ultimate_final.zip"),
        resolve_output_path("ppo_outputs_report_main/best_model/best_model.zip"),
        resolve_output_path("ppo_outputs_report_main/models/ppo_idc_ultimate_final.zip"),
        resolve_output_path("ppo_outputs_report_carbon_w03/best_model/best_model.zip"),
        resolve_output_path("ppo_outputs_report_carbon_w03/models/ppo_idc_ultimate_final.zip"),
        Path("ppo_outputs_ultimate/best_model/best_model.zip"),
        Path("ppo_outputs_ultimate/models/ppo_idc_ultimate_final.zip"),
        Path("ppo_outputs_ultimate/best_model.zip"),
        Path("best_model.zip"),
        Path("ppo_idc_ultimate_final.zip"),
    ]
    for path in candidates:
        if path.exists():
            return path
    return None


def evaluate_ppo(model: Any, env_seed: int, ppo_name: str = "PPO", hourly_rows=None) -> Dict[str, Any]:
    env = make_env(env_seed)
    obs, reset_info = env.reset()

    total_reward = 0.0
    info: Dict[str, Any] = {}
    episode_infos: List[Dict[str, Any]] = []

    for step in range(env.horizon):
        action, _state = model.predict(obs, deterministic=True)
        action = ensure_env_action_dim(action, env)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += float(reward)
        episode_infos.append(dict(info))

        if hourly_rows is not None:
            hourly_rows.append(
                build_hourly_row(
                    algorithm=ppo_name,
                    env_seed=env_seed,
                    run_idx=0,
                    step=step,
                    action=np.asarray(action, dtype=np.float32),
                    reward=float(reward),
                    info=info,
                )
            )

        if terminated or truncated:
            break

    row: Dict[str, Any] = {
        "algorithm": ppo_name,
        "seed": int(env_seed),
        "source": "eval_ppo",
        "run_idx": 0,
    }
    row.update(final_metrics_from_info(total_reward, info, episode_infos))
    return row


# =========================
# GA/PSO CSV importing
# =========================

def first_existing_path(paths: Sequence[str]) -> Optional[Path]:
    for p in paths:
        path = Path(p)
        if path.exists():
            return path
    return None


def import_search_csv(
    csv_path: Path,
    algorithm_name: str,
    wanted_seeds: Optional[set[int]] = None,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with csv_path.open("r", newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for raw in reader:
            seed_value = raw.get("seed", raw.get("env_seed", ""))
            seed = int(round(safe_float(seed_value, default=np.nan))) if str(seed_value).strip() else -1
            if wanted_seeds is not None and seed not in wanted_seeds:
                continue

            row: Dict[str, Any] = {
                "algorithm": algorithm_name,
                "seed": seed,
                "source": str(csv_path),
                "run_idx": 0,
            }
            for key in ["fitness"] + METRIC_KEYS:
                if key in raw:
                    row[key] = safe_float(raw.get(key))
                else:
                    row[key] = np.nan

            # GA/PSO CSVs use fitness=total_reward by default, but keep both fields robust.
            if math.isnan(safe_float(row.get("total_reward"))) and not math.isnan(safe_float(row.get("fitness"))):
                row["total_reward"] = row["fitness"]
            if math.isnan(safe_float(row.get("fitness"))) and not math.isnan(safe_float(row.get("total_reward"))):
                row["fitness"] = row["total_reward"]

            for extra in ["eval_count", "env_seed", "algo_seed"]:
                if extra in raw:
                    row[extra] = safe_float(raw.get(extra))

            rows.append(row)
    return rows


# =========================
# CSV writing and summary
# =========================

def ordered_fieldnames(rows: List[Dict[str, Any]]) -> List[str]:
    preferred = [
        "algorithm",
        "seed",
        "source",
        "run_idx",
        "fitness",
        "total_reward",
        "completion_rate",
        "task_completion_rate",
        "total_completed_work",
        "final_backlog_work",
        "overflow_work",
        "deadline_miss_rate",
        "sla_violation_rate",
        "sla_penalty",
        "avg_task_delay",
        "avg_waiting_time",
        "avg_turnaround_time",
        "total_cost",
        "unit_task_cost",
        "P_grid_kW",
        "grid_energy_kWh",
        "idc_energy_kWh",
        "carbon_cost",
        "total_energy_kWh",
        "total_grid_energy_kWh",
        "total_idc_energy_kWh",
        "energy_per_task",
        "idc_energy_per_task",
        "total_carbon_emission",
        "total_carbon_cost",
        "carbon_per_task",
        "episode_grid_peak_power_kW",
        "total_grid_peak_excess_kW_hour",
        "episode_peak_power_kW",
        "total_peak_excess_kW_hour",
        "bess_soc",
        "total_bess_charge_kWh",
        "total_bess_discharge_kWh",
        "total_bess_degradation_cost",
        "load_change",
        "action_change",
        "finished_task_count",
        "total_task_count",
        "deadline_miss_count",
        "sla_violation_count",
        "total_pause_count",
        "total_resume_count",
        "total_non_interruptible_interruption_count",
        "grid_opf_success_rate",
        "grid_mef_success_rate",
        "avg_grid_lmp",
        "avg_grid_mef_plus",
        "avg_grid_mef_minus",
        "avg_grid_total_generation_cost",
        "avg_grid_total_emission_kg",
        "avg_grid_network_loss_mw",
        "avg_grid_load_scale",
        "avg_grid_reference_usep",
        "min_grid_voltage_pu",
        "max_grid_line_loading_percent",
        "total_grid_security_penalty",
        "total_safe_violation_cost",
        "total_safe_violation_voltage",
        "total_safe_violation_line",
        "total_safe_violation_opf",
        "avg_safe_violation_cost",
        "max_safe_violation_cost",
        "total_safe_cost",
        "total_grid_reward_penalty",
        "avg_grid_reward_penalty",
        "total_grid_lmp_cost_penalty",
        "total_grid_mef_carbon_penalty",
        "total_grid_safe_violation_penalty",
        "reward_mismatch_count",
        "grid_opf_fail_count",
        "grid_mef_fail_count",
        "eval_count",
        "env_seed",
        "algo_seed",
    ]
    keys = set()
    for row in rows:
        keys.update(row.keys())
    return [k for k in preferred if k in keys] + sorted(k for k in keys if k not in preferred)


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fieldnames = ordered_fieldnames(rows)
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def save_hourly_csv(rows, out_path):
    if not rows:
        return

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = list(rows[0].keys())
    with out_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def save_hourly_mean_csv(rows, out_path):
    if not rows:
        return

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    keys = [
        "price",
        "carbon_factor",
        "action_mean",
        "actual_task_load_mean",
        "actual_total_load_mean",
        "planned_task_load_mean",
        "planned_capacity",
        "completed_work",
        "unused_capacity",
        "energy_kWh",
        "grid_energy_kWh",
        "idc_energy_kWh",
        "hourly_cost",
        "carbon_emission",
        "carbon_cost",
        "P_IDC",
        "P_IDC_kW",
        "P_grid_kW",
        "grid_power_kW",
        "grid_power_limit_kW",
        "bess_soc",
        "bess_energy_kWh",
        "desired_bess_charge_power_kW",
        "desired_bess_discharge_power_kW",
        "bess_charge_power_kW",
        "bess_discharge_power_kW",
        "bess_charge_kWh",
        "bess_discharge_kWh",
        "bess_degradation_cost",
        "invalid_bess_action",
        "soc_deviation",
        "soc_excess",
        "total_bess_charge_kWh",
        "total_bess_discharge_kWh",
        "total_bess_degradation_cost",
        "P_cooling",
        "PUE",
        "COP",
        "grid_peak_power_kW",
        "grid_peak_excess_kW",
        "episode_grid_peak_power_kW",
        "total_grid_peak_excess_kW_hour",
        "idc_peak_power_kW",
        "peak_power_kW",
        "peak_excess_kW",
        "episode_peak_power_kW",
        "total_peak_excess_kW_hour",
        "backlog_work",
        "overflow_work",
        "server_group_size",
        "effective_total_server_count",
        "task_workload_scale",
        "bess_capacity_kWh",
        "bess_charge_power_max_kW",
        "bess_discharge_power_max_kW",
        "bess_scale_factor",
        "deadline_miss_count",
        "sla_penalty",
        "sla_violation_rate",
        "avg_task_delay",
        "load_change",
        "action_change",
        "r_cost",
        "r_carbon",
        "r_queue",
        "r_queue_overflow",
        "r_urgent_backlog",
        "r_waiting",
        "r_deadline",
        "r_sla",
        "r_unused",
        "r_grid_peak",
        "r_peak_load",
        "r_pause",
        "r_resume",
        "r_non_interruptible",
        "r_load_smooth",
        "r_action_smooth",
        "r_bess_degradation",
        "r_bess_invalid_action",
        "r_final_queue",
        "r_soc_final",
        "reward",
        "grid_idc_load_mw",
        "grid_load_scale",
        "grid_reference_usep",
        "grid_lmp",
        "grid_mef_plus",
        "grid_mef_minus",
        "grid_total_generation_cost",
        "grid_total_emission_kg",
        "grid_total_load_mw",
        "grid_total_generation_mw",
        "grid_network_loss_mw",
        "grid_min_voltage_pu",
        "grid_max_voltage_pu",
        "grid_max_line_loading_percent",
        "grid_voltage_violation_count",
        "grid_line_overload_count",
        "grid_voltage_violation_magnitude",
        "grid_line_overload_magnitude",
        "grid_security_penalty",
        "safe_violation_voltage",
        "safe_violation_line",
        "safe_violation_opf",
        "safe_violation_cost",
        "safe_cost_voltage",
        "safe_cost_line",
        "safe_cost_opf",
        "safe_cost_total",
        "grid_lmp_cost",
        "grid_lmp_cost_norm",
        "grid_lmp_cost_penalty",
        "grid_mef_carbon",
        "grid_mef_carbon_norm",
        "grid_mef_carbon_penalty",
        "grid_safe_violation",
        "grid_safe_violation_norm",
        "grid_safe_violation_penalty",
        "base_reward",
        "grid_reward_penalty",
        "grid_adjusted_reward",
    ]

    groups = sorted(set((str(r["algorithm"]), int(r["hour"])) for r in rows))
    mean_rows = []

    for alg, h in groups:
        h_rows = [
            r for r in rows
            if str(r["algorithm"]) == alg and int(r["hour"]) == h
        ]
        out = {
            "algorithm": alg,
            "hour": h,
            "n": len(h_rows),
        }

        for key in keys:
            vals = []
            for r in h_rows:
                v = r.get(key)
                try:
                    vals.append(float(v))
                except Exception:
                    pass

            mean_value = sum(vals) / len(vals) if vals else ""
            out[key] = mean_value
            out[f"{key}_mean"] = mean_value

        mean_rows.append(out)

    fieldnames = list(mean_rows[0].keys())
    with out_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(mean_rows)


def build_summary(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    algorithms = sorted(set(str(row.get("algorithm", "")) for row in rows))
    summary: List[Dict[str, Any]] = []

    for alg in algorithms:
        alg_rows = [row for row in rows if str(row.get("algorithm", "")) == alg]
        out: Dict[str, Any] = {
            "algorithm": alg,
            "n_rows": len(alg_rows),
            "n_seeds": len(set(int(row.get("seed", -1)) for row in alg_rows)),
        }
        for key in SUMMARY_KEYS:
            values = np.array([safe_float(row.get(key)) for row in alg_rows], dtype=np.float64)
            values = values[~np.isnan(values)]
            if len(values) == 0:
                out[f"{key}_mean"] = np.nan
                out[f"{key}_std"] = np.nan
            else:
                out[f"{key}_mean"] = float(np.mean(values))
                out[f"{key}_std"] = float(np.std(values))
        summary.append(out)

    # Sort by core logic: higher completion, lower backlog, lower unit cost.
    def sort_key(row: Dict[str, Any]) -> Tuple[float, float, float]:
        completion = safe_float(row.get("completion_rate_mean"), default=-np.inf)
        backlog = safe_float(row.get("final_backlog_work_mean"), default=np.inf)
        unit_cost = safe_float(row.get("unit_task_cost_mean"), default=np.inf)
        return (-completion, backlog, unit_cost)

    summary.sort(key=sort_key)
    return summary


def print_summary(summary: List[Dict[str, Any]]) -> None:
    if not summary:
        print("No summary rows.")
        return

    print("\n=== Unified summary, sorted by completion/backlog/unit cost ===")
    header = (
        f"{'algorithm':<22} {'n':>4} "
        f"{'comp':>9} {'task_comp':>10} {'backlog':>12} {'miss':>9} "
        f"{'unit_cost':>10} {'cost':>10} {'reward':>10} "
        f"{'grid_lmp':>10} {'mef+':>10} {'minV':>8} {'maxLine':>9} {'opf_ok':>8}"
    )
    print(header)
    print("-" * len(header))

    for row in summary:
        print(
            f"{str(row['algorithm']):<22} {int(row['n_seeds']):>4} "
            f"{safe_float(row.get('completion_rate_mean')):>9.4f} "
            f"{safe_float(row.get('task_completion_rate_mean')):>10.4f} "
            f"{safe_float(row.get('final_backlog_work_mean')):>12.2f} "
            f"{safe_float(row.get('deadline_miss_rate_mean')):>9.4f} "
            f"{safe_float(row.get('unit_task_cost_mean')):>10.5f} "
            f"{safe_float(row.get('total_cost_mean')):>10.2f} "
            f"{safe_float(row.get('total_reward_mean')):>10.4f} "
            f"{safe_float(row.get('avg_grid_lmp_mean')):>10.4f} "
            f"{safe_float(row.get('avg_grid_mef_plus_mean')):>10.4f} "
            f"{safe_float(row.get('min_grid_voltage_pu_mean')):>8.4f} "
            f"{safe_float(row.get('max_grid_line_loading_percent_mean')):>9.4f} "
            f"{safe_float(row.get('grid_opf_success_rate_mean')):>8.4f}"
        )


# =========================
# Main
# =========================

def parse_seeds(args: argparse.Namespace) -> List[int]:
    if args.seeds.strip():
        return [int(x.strip()) for x in args.seeds.split(",") if x.strip()]
    return list(range(args.start_seed, args.start_seed + args.n_seeds))


def main() -> None:
    global ACTIVE_CASE_CONFIG

    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="Fast smoke test: seed=3000 only, no PPO unless model exists.")
    parser.add_argument("--case", type=str, default="main", help="Experiment case: main, no_bess, carbon_w0, carbon_w03, carbon_w05.")
    parser.add_argument("--start-seed", type=int, default=3000)
    parser.add_argument("--n-seeds", type=int, default=30)
    parser.add_argument("--seeds", type=str, default="", help="Comma-separated seeds, e.g. 3000,3001,3002.")
    parser.add_argument("--out", type=str, default=None, help="Output directory. Defaults to report_outputs/eval_<case>.")

    parser.add_argument("--random-runs", type=int, default=1, help="Random policy repetitions per seed.")
    parser.add_argument("--no-basic", action="store_true", help="Skip ZERO/ONE/RANDOM/RULE evaluation.")
    parser.add_argument("--no-ga", action="store_true", help="Skip importing GA CSV.")
    parser.add_argument("--no-pso", action="store_true", help="Skip importing PSO CSV.")
    parser.add_argument("--no-ppo", action="store_true", help="Skip PPO evaluation.")

    parser.add_argument("--ga-csv", type=str, default="auto", help="Path to GA CSV, or auto.")
    parser.add_argument("--pso-csv", type=str, default="auto", help="Path to PSO CSV, or auto.")
    parser.add_argument("--ppo-model", type=str, default="auto", help="PPO model .zip path, auto, or skip.")
    parser.add_argument("--ppo-name", type=str, default="PPO", help="Algorithm name used for PPO rows in output CSVs.")
    args = parser.parse_args()

    ACTIVE_CASE_CONFIG = get_experiment_case(args.case)
    print_experiment_case(ACTIVE_CASE_CONFIG)

    if args.quick:
        args.n_seeds = 1
        if not args.seeds.strip():
            args.start_seed = 3000

    out_dir = resolve_output_path(f"eval_{ACTIVE_CASE_CONFIG['case']}") if args.out is None else Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    seeds = parse_seeds(args)
    wanted_seeds = set(seeds)

    print("=== Unified baseline evaluation ===")
    print(f"case: {ACTIVE_CASE_CONFIG['case']}")
    print(f"seeds: {seeds}")
    print(f"output dir: {out_dir.resolve()}")

    all_rows: List[Dict[str, Any]] = []
    hourly_rows: List[Dict[str, Any]] = []
    start_time = time.time()

    # 1. Basic policies
    if not args.no_basic:
        deterministic_policies = [
            "ZERO",
            "ONE",
            "RULE",
            "FAST_NEUTRAL_BESS",
            "UNIFORM_NEUTRAL_BESS",
            "RULE_PRICE_ONLY",
            "RULE_PRICE_BESS",
        ]
        print("\n>>> Evaluating basic policies: " + ", ".join(deterministic_policies + ["RANDOM"]))
        for seed in seeds:
            for alg in deterministic_policies:
                row = evaluate_basic_policy(alg, seed, hourly_rows=hourly_rows)
                all_rows.append(row)
                print(
                    f"{alg:<20} seed={seed} comp={row['completion_rate']:.3f} "
                    f"cost={row['total_cost']:.2f} unit={row['unit_task_cost']:.4f} "
                    f"backlog={row['final_backlog_work']:.2f} reward={row['total_reward']:.4f}"
                )

            for run_idx in range(max(args.random_runs, 1)):
                rng_seed = seed + 30000 + 1000 * run_idx
                row = evaluate_basic_policy(
                    "RANDOM",
                    seed,
                    rng_seed=rng_seed,
                    random_run=run_idx,
                    hourly_rows=hourly_rows,
                )
                all_rows.append(row)
                print(
                    f"RANDOM seed={seed} run={run_idx} comp={row['completion_rate']:.3f} "
                    f"cost={row['total_cost']:.2f} unit={row['unit_task_cost']:.4f} "
                    f"backlog={row['final_backlog_work']:.2f} reward={row['total_reward']:.4f}"
                )

    # 2. Import GA CSV
    if not args.no_ga:
        if args.ga_csv.lower() == "auto":
            ga_path = first_existing_path([
                resolve_output_path("ga_out/ga_stress_30_30_results.csv"),
                resolve_output_path("ga_out/ga_results.csv"),
                "ga_out/ga_stress_30_30_results.csv",
                "ga_out/ga_results.csv",
            ])
        else:
            ga_path = Path(args.ga_csv)
        if ga_path is not None and ga_path.exists():
            ga_rows = import_search_csv(ga_path, "GA", wanted_seeds=wanted_seeds)
            all_rows.extend(ga_rows)
            print(f"\n>>> Imported GA rows: {len(ga_rows)} from {ga_path}")
        else:
            print("\n>>> GA CSV not found; skipped GA import.")

    # 3. Import PSO CSV
    if not args.no_pso:
        if args.pso_csv.lower() == "auto":
            pso_path = first_existing_path([
                resolve_output_path("pso_out/pso_stress_30_30_results.csv"),
                resolve_output_path("pso_out/pso_results.csv"),
                "pso_out/pso_stress_30_30_results.csv",
                "pso_out/pso_results.csv",
            ])
        else:
            pso_path = Path(args.pso_csv)
        if pso_path is not None and pso_path.exists():
            pso_rows = import_search_csv(pso_path, "PSO", wanted_seeds=wanted_seeds)
            all_rows.extend(pso_rows)
            print(f"\n>>> Imported PSO rows: {len(pso_rows)} from {pso_path}")
        else:
            print("\n>>> PSO CSV not found; skipped PSO import.")

    # 4. PPO evaluation
    if not args.no_ppo:
        ppo_model_arg = (args.ppo_model or "").strip()
        try:
            model_path = resolve_ppo_model_path(ppo_model_arg)
        except FileNotFoundError as exc:
            raise SystemExit(f"\n>>> {exc}") from exc
        if model_path is None:
            if ppo_model_arg.lower() == "auto":
                print("\n>>> 未找到 PPO 模型；请使用 --ppo-model 手动指定从台式机复制过来的模型路径。")
            else:
                print("\n>>> Skipped PPO evaluation by --ppo-model none/skip.")
        else:
            print(f"\n>>> Evaluating PPO model: {model_path}")
            print(f">>> PPO algorithm name: {args.ppo_name}")
            try:
                from stable_baselines3 import PPO
            except Exception as exc:
                print(f">>> Could not import stable_baselines3.PPO; skipped PPO. Error: {exc}")
            else:
                model = PPO.load(str(model_path))
                for seed in seeds:
                    row = evaluate_ppo(model, seed, ppo_name=args.ppo_name, hourly_rows=hourly_rows)
                    row["ppo_model_path"] = str(model_path)
                    all_rows.append(row)
                    print(
                        f"{args.ppo_name:<20} seed={seed} comp={row['completion_rate']:.3f} "
                        f"cost={row['total_cost']:.2f} unit={row['unit_task_cost']:.4f} "
                        f"backlog={row['final_backlog_work']:.2f} reward={row['total_reward']:.4f}"
                    )

    # 5. Write outputs
    all_csv = out_dir / "all_results.csv"
    summary_csv = out_dir / "summary.csv"
    hourly_csv = out_dir / "hourly_result.csv"
    hourly_mean_csv = out_dir / "hourly_mean.csv"

    write_csv(all_csv, all_rows)
    summary_rows = build_summary(all_rows)
    write_csv(summary_csv, summary_rows)
    save_hourly_csv(hourly_rows, hourly_csv)
    save_hourly_mean_csv(hourly_rows, hourly_mean_csv)
    print_summary(summary_rows)

    print(f"\nTotal wall time: {time.time() - start_time:.2f} s")
    print(f"All results CSV: {all_csv}")
    print(f"Summary CSV:     {summary_csv}")
    print(f"Hourly CSV:      {hourly_csv}")
    print(f"Hourly mean CSV: {hourly_mean_csv}")


if __name__ == "__main__":
    main()
