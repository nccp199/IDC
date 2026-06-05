"""Evaluate ordinary PPO or Safe PPO on single-IDC Safe RL scenarios."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from configs.config_ultimate import DEFAULT_EVAL_SEED
from configs.single_idc_scenarios import get_single_idc_scenario
from safe_rl.config_safe import LAGRANGIAN_CONFIG, SAFE_TRAIN_CONFIG
from safe_rl.lagrangian import LagrangianMultiplierManager
from safe_rl.train_safe_ppo import build_safe_scenario_configs, make_safe_single_env


DEFAULT_EVAL_SCENARIOS = "single_idc_bus9_normal,single_idc_bus9_strong,single_idc_bus9_stress"


def parse_csv_list(text: str) -> list[str]:
    return [item.strip() for item in str(text).split(",") if item.strip()]


def parse_seeds(text: str, default_seed: int) -> list[int]:
    items = parse_csv_list(text)
    return [int(item) for item in items] if items else [int(default_seed)]


def load_lagrangian_state(manager: LagrangianMultiplierManager, model_path: Path, state_arg: str) -> Path | None:
    state_text = str(state_arg or "auto").strip()
    if state_text.lower() in {"none", "skip"}:
        return None
    candidates: list[Path]
    if state_text.lower() == "auto":
        candidates = [
            model_path.parent / "lagrangian_state.json",
            model_path.parent.parent / "lagrangian_state.json",
            model_path.parent.parent.parent / "lagrangian_state.json",
        ]
    else:
        candidates = [Path(state_text)]

    for candidate in candidates:
        if candidate.exists():
            with candidate.open("r", encoding="utf-8") as f:
                manager.load_state_dict(json.load(f))
            return candidate
    return None


def ensure_action_dim(action: Any, action_dim: int) -> np.ndarray:
    arr = np.asarray(action, dtype=np.float32).reshape(-1)
    if arr.shape[0] == int(action_dim):
        return arr
    if arr.shape[0] == int(action_dim) - 1:
        return np.concatenate([arr, np.array([0.5], dtype=np.float32)]).astype(np.float32)
    raise ValueError(f"Model action_dim={arr.shape[0]} does not match env action_dim={action_dim}.")


def evaluate_one_episode(
    model: Any,
    scenario_name: str,
    seed: int,
    manager_state: dict[str, Any],
    model_kind: str,
    hourly_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    configs = build_safe_scenario_configs(scenario_name)
    manager = LagrangianMultiplierManager(LAGRANGIAN_CONFIG)
    manager.load_state_dict(manager_state)
    safe_train_config = dict(SAFE_TRAIN_CONFIG)
    safe_train_config["enable_safe_reward"] = True
    safe_train_config["safe_reward_active"] = True
    safe_train_config["report_only"] = False
    safe_train_config["lambda_update_active"] = False
    safe_train_config["shadow_lambda_update"] = False

    env = make_safe_single_env(configs, manager, safe_train_config, seed=seed, monitor=False)
    try:
        obs, reset_info = env.reset()
        expected_obs_shape = tuple(env.observation_space.shape)
        model_obs_space = getattr(model, "observation_space", None)
        if model_obs_space is not None and tuple(model_obs_space.shape) != expected_obs_shape:
            raise RuntimeError(
                f"Model observation shape {tuple(model_obs_space.shape)} does not match "
                f"{scenario_name} environment shape {expected_obs_shape}. Refusing to evaluate mismatched PPO weights."
            )

        episode_infos: list[dict[str, Any]] = []
        for step in range(int(getattr(env, "horizon", 24))):
            action, _state = model.predict(obs, deterministic=True)
            action = ensure_action_dim(action, int(env.action_space.shape[0]))
            obs, reward, terminated, truncated, info = env.step(action)
            row = build_hourly_row(
                model_kind=model_kind,
                scenario_name=scenario_name,
                seed=seed,
                step=step,
                action=action,
                reward=float(reward),
                info=info,
                manager=manager,
            )
            hourly_rows.append(row)
            episode_infos.append(dict(info))
            if terminated or truncated:
                break

        return aggregate_episode(
            model_kind=model_kind,
            scenario_name=scenario_name,
            seed=seed,
            scenario=configs["scenario"],
            infos=episode_infos,
            manager=manager,
        )
    finally:
        env.close()


def build_hourly_row(
    model_kind: str,
    scenario_name: str,
    seed: int,
    step: int,
    action: np.ndarray,
    reward: float,
    info: dict[str, Any],
    manager: LagrangianMultiplierManager,
) -> dict[str, Any]:
    safe_costs = dict(info.get("safe_costs", {}))
    lambdas = manager.get_lambdas()
    return {
        "model_kind": model_kind,
        "scenario": scenario_name,
        "seed": int(seed),
        "step": int(step),
        "hour": int(info.get("hour", step)),
        "reward": float(reward),
        "reward_base": safe_float(info.get("reward_base")),
        "reward_safe": safe_float(info.get("reward_safe")),
        "safe_penalty": safe_float(info.get("safe_penalty")),
        "safe_penalty_raw": safe_float(info.get("safe_penalty_raw")),
        "safe_penalty_applied": safe_float(info.get("safe_penalty_applied")),
        "safe_reward_active": bool(info.get("safe_reward_active", False)),
        "lambda_update_active": bool(info.get("lambda_update_active", False)),
        "action_mean": float(np.mean(action)),
        "P_local_demand_kW": safe_float(info.get("P_local_demand_kW")),
        "P_local_net_before_pv_kW": safe_float(info.get("P_local_net_before_pv_kW")),
        "P_bus_net_kW": safe_float(info.get("P_bus_net_kW")),
        "P_grid_kW": safe_float(info.get("P_grid_kW")),
        "pv_available_kW": safe_float(info.get("pv_available_kW")),
        "pv_used_kW": safe_float(info.get("pv_used_kW")),
        "pv_curtail_kW": safe_float(info.get("pv_curtail_kW")),
        "total_pv_available_kWh": safe_float(info.get("total_pv_available_kWh")),
        "total_pv_used_kWh": safe_float(info.get("total_pv_used_kWh")),
        "total_pv_curtail_kWh": safe_float(info.get("total_pv_curtail_kWh")),
        "pv_utilization_rate": safe_float(info.get("pv_utilization_rate")),
        "renewable_share": safe_float(info.get("renewable_share")),
        "grid_bus_net_load_mw": safe_float(info.get("grid_bus_net_load_mw")),
        "grid_idc_load_mw": safe_float(info.get("grid_idc_load_mw")),
        "opf_success": bool(safe_costs.get("opf_success", info.get("grid_opf_success", False))),
        "minV": safe_float(safe_costs.get("raw_minV", info.get("grid_min_voltage_pu"))),
        "maxV": safe_float(safe_costs.get("raw_maxV", info.get("grid_max_voltage_pu"))),
        "maxLine": safe_float(safe_costs.get("raw_maxLine", info.get("grid_max_line_loading_percent"))),
        "maxTrafo": safe_float(safe_costs.get("raw_maxTrafo", 0.0), default=0.0),
        "LMP_bus9": safe_float(safe_costs.get("raw_lmp", info.get("grid_lmp"))),
        "MEF_bus9": safe_float(safe_costs.get("raw_mef", info.get("grid_mef_plus"))),
        "C_opf": safe_float(safe_costs.get("C_opf")),
        "C_voltage": safe_float(safe_costs.get("C_voltage")),
        "C_line": safe_float(safe_costs.get("C_line")),
        "C_trafo": safe_float(safe_costs.get("C_trafo")),
        "C_lmp": safe_float(safe_costs.get("C_lmp")),
        "C_mef": safe_float(safe_costs.get("C_mef")),
        "C_total": safe_float(safe_costs.get("C_total")),
        "lambda_opf": safe_float(lambdas.get("lambda_opf"), default=0.0),
        "lambda_voltage": safe_float(lambdas.get("lambda_voltage"), default=0.0),
        "completion_rate": safe_float(info.get("completion_rate")),
        "task_completion_rate": safe_float(info.get("task_completion_rate")),
        "final_backlog_work": safe_float(info.get("final_backlog_work", info.get("Q"))),
        "deadline_miss_rate": safe_float(info.get("deadline_miss_rate")),
        "total_cost": safe_float(info.get("total_cost")),
        "unit_task_cost": safe_float(info.get("unit_task_cost")),
        "total_carbon_emission": safe_float(info.get("total_carbon_emission")),
        "carbon_per_task": safe_float(info.get("carbon_per_task")),
    }


def aggregate_episode(
    model_kind: str,
    scenario_name: str,
    seed: int,
    scenario: dict[str, Any],
    infos: list[dict[str, Any]],
    manager: LagrangianMultiplierManager,
) -> dict[str, Any]:
    if not infos:
        raise RuntimeError(f"No evaluation steps collected for {scenario_name} seed={seed}.")
    final = infos[-1]
    safe_cost_rows = [dict(info.get("safe_costs", {})) for info in infos]
    lambdas = manager.get_lambdas()
    row = {
        "model_kind": model_kind,
        "scenario": scenario_name,
        "seed": int(seed),
        "target_peak_mw": safe_float(scenario.get("target_peak_mw")),
        "target_mean_mw": safe_float(scenario.get("target_mean_mw")),
        "steps": len(infos),
        "reward_base": sum_finite(info.get("reward_base") for info in infos),
        "reward_safe": sum_finite(info.get("reward_safe") for info in infos),
        "completion_rate": safe_float(final.get("completion_rate")),
        "task_completion_rate": safe_float(final.get("task_completion_rate")),
        "final_backlog_work": safe_float(final.get("final_backlog_work", final.get("Q"))),
        "deadline_miss_rate": safe_float(final.get("deadline_miss_rate")),
        "total_cost": safe_float(final.get("total_cost")),
        "unit_task_cost": safe_float(final.get("unit_task_cost")),
        "total_carbon_emission": safe_float(final.get("total_carbon_emission")),
        "carbon_per_task": safe_float(final.get("carbon_per_task")),
        "total_pv_available_kWh": safe_float(final.get("total_pv_available_kWh")),
        "total_pv_used_kWh": safe_float(final.get("total_pv_used_kWh")),
        "total_pv_curtail_kWh": safe_float(final.get("total_pv_curtail_kWh")),
        "pv_utilization_rate": safe_float(final.get("pv_utilization_rate")),
        "renewable_share": safe_float(final.get("renewable_share")),
        "P_bus_net_kW_final": safe_float(final.get("P_bus_net_kW")),
        "P_grid_kW_final": safe_float(final.get("P_grid_kW")),
        "opf_success_rate": mean_bool(info.get("grid_opf_success", False) for info in infos),
        "minV_min": min_finite(cost.get("raw_minV", info.get("grid_min_voltage_pu")) for cost, info in zip(safe_cost_rows, infos)),
        "minV_mean": mean_finite(cost.get("raw_minV", info.get("grid_min_voltage_pu")) for cost, info in zip(safe_cost_rows, infos)),
        "maxLine_max": max_finite(cost.get("raw_maxLine", info.get("grid_max_line_loading_percent")) for cost, info in zip(safe_cost_rows, infos)),
        "maxTrafo_max": max_finite(cost.get("raw_maxTrafo", 0.0) for cost in safe_cost_rows),
        "LMP_bus9_mean": mean_finite(cost.get("raw_lmp", info.get("grid_lmp")) for cost, info in zip(safe_cost_rows, infos)),
        "LMP_bus9_max": max_finite(cost.get("raw_lmp", info.get("grid_lmp")) for cost, info in zip(safe_cost_rows, infos)),
        "MEF_bus9_mean": mean_finite(cost.get("raw_mef", info.get("grid_mef_plus")) for cost, info in zip(safe_cost_rows, infos)),
        "MEF_bus9_max": max_finite(cost.get("raw_mef", info.get("grid_mef_plus")) for cost, info in zip(safe_cost_rows, infos)),
        "lambda_opf": safe_float(lambdas.get("lambda_opf"), default=0.0),
        "lambda_voltage": safe_float(lambdas.get("lambda_voltage"), default=0.0),
        "safe_penalty_raw_mean": mean_finite(info.get("safe_penalty_raw") for info in infos),
        "safe_penalty_raw_sum": sum_finite(info.get("safe_penalty_raw") for info in infos),
        "safe_penalty_applied_mean": mean_finite(info.get("safe_penalty_applied") for info in infos),
        "safe_penalty_applied_sum": sum_finite(info.get("safe_penalty_applied") for info in infos),
        "safe_penalty_mean": mean_finite(info.get("safe_penalty") for info in infos),
        "safe_penalty_sum": sum_finite(info.get("safe_penalty") for info in infos),
        "safe_reward_active": bool(final.get("safe_reward_active", False)),
        "lambda_update_active": bool(final.get("lambda_update_active", False)),
    }

    for name in ["opf", "voltage", "line", "trafo", "lmp", "mef", "total"]:
        key = f"C_{name}"
        row[f"{key}_mean"] = mean_finite(cost.get(key) for cost in safe_cost_rows)
        row[f"{key}_sum"] = sum_finite(cost.get(key) for cost in safe_cost_rows)

    return row


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    keys: list[str] = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                keys.append(key)
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_report(path: Path, summary_rows: list[dict[str, Any]], model_path: Path, state_path: Path | None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Safe PPO Evaluation Report",
        "",
        f"- model_path: `{model_path}`",
        f"- lagrangian_state: `{state_path}`" if state_path is not None else "- lagrangian_state: not loaded",
        "",
        "| scenario | seed | opf_success_rate | minV_min | C_voltage_mean | safe_penalty_mean | completion_rate | unit_task_cost |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary_rows:
        lines.append(
            "| {scenario} | {seed} | {opf:.4f} | {minv:.4f} | {cv:.4f} | {pen:.4f} | {comp:.4f} | {unit:.6f} |".format(
                scenario=row.get("scenario"),
                seed=int(row.get("seed", 0)),
                opf=safe_float(row.get("opf_success_rate")),
                minv=safe_float(row.get("minV_min")),
                cv=safe_float(row.get("C_voltage_mean")),
                pen=safe_float(row.get("safe_penalty_mean")),
                comp=safe_float(row.get("completion_rate")),
                unit=safe_float(row.get("unit_task_cost")),
            )
        )
    lines.append("")
    lines.append("Focus checks: OPF failure reduction, voltage margin, stable lambdas, completion rate, and unit task cost.")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--model-kind", choices=["ppo", "safe_ppo"], default="safe_ppo")
    parser.add_argument("--scenarios", type=str, default=DEFAULT_EVAL_SCENARIOS)
    parser.add_argument("--seeds", type=str, default=str(DEFAULT_EVAL_SEED))
    parser.add_argument("--lagrangian-state", type=str, default="auto")
    parser.add_argument("--out-dir", type=str, default="outputs/safe_eval")
    args = parser.parse_args()

    model_path = Path(args.model_path)
    if not model_path.exists():
        zip_path = model_path if model_path.suffix.lower() == ".zip" else model_path.with_suffix(".zip")
        if zip_path.exists():
            model_path = zip_path
        else:
            raise SystemExit(f"Model path not found: {args.model_path}")

    try:
        from stable_baselines3 import PPO
    except Exception as exc:
        raise SystemExit(f"stable_baselines3 is required for PPO evaluation: {exc}") from exc

    base_manager = LagrangianMultiplierManager(LAGRANGIAN_CONFIG)
    state_path = load_lagrangian_state(base_manager, model_path, args.lagrangian_state)
    manager_state = base_manager.state_dict()
    model = PPO.load(str(model_path))

    scenarios = parse_csv_list(args.scenarios)
    for scenario_name in scenarios:
        get_single_idc_scenario(scenario_name)
    seeds = parse_seeds(args.seeds, DEFAULT_EVAL_SEED)

    out_dir = Path(args.out_dir)
    summary_rows: list[dict[str, Any]] = []
    hourly_rows: list[dict[str, Any]] = []
    for scenario_name in scenarios:
        for seed in seeds:
            row = evaluate_one_episode(
                model=model,
                scenario_name=scenario_name,
                seed=seed,
                manager_state=manager_state,
                model_kind=args.model_kind,
                hourly_rows=hourly_rows,
            )
            summary_rows.append(row)
            print(
                f"{args.model_kind} {scenario_name} seed={seed} "
                f"opf={row['opf_success_rate']:.3f} minV={row['minV_min']:.4f} "
                f"C_voltage={row['C_voltage_mean']:.4f} comp={row['completion_rate']:.3f}"
            )

    summary_path = out_dir / "safe_ppo_eval_summary.csv"
    hourly_path = out_dir / "safe_ppo_eval_hourly.csv"
    report_path = out_dir / "safe_ppo_eval_report.md"
    write_csv(summary_path, summary_rows)
    write_csv(hourly_path, hourly_rows)
    write_report(report_path, summary_rows, model_path, state_path)

    print("\n>>> Safe PPO evaluation finished")
    print(f"Summary CSV: {summary_path}")
    print(f"Hourly CSV: {hourly_path}")
    print(f"Report: {report_path}")


def safe_float(value: Any, default: float = math.nan) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if math.isfinite(number) else default


def finite_values(values) -> list[float]:
    out = []
    for value in values:
        number = safe_float(value)
        if math.isfinite(number):
            out.append(float(number))
    return out


def mean_finite(values) -> float:
    vals = finite_values(values)
    return float(np.mean(vals)) if vals else math.nan


def sum_finite(values) -> float:
    vals = finite_values(values)
    return float(np.sum(vals)) if vals else 0.0


def min_finite(values) -> float:
    vals = finite_values(values)
    return float(np.min(vals)) if vals else math.nan


def max_finite(values) -> float:
    vals = finite_values(values)
    return float(np.max(vals)) if vals else math.nan


def mean_bool(values) -> float:
    items = [bool(value) for value in values]
    return float(sum(items) / max(len(items), 1))


if __name__ == "__main__":
    main()
