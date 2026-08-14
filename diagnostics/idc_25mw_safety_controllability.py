"""Rated-25-MW IDC action controllability diagnostic on IEEE-14 bus 9.

This script does not train an agent or mutate formal configuration.  It keeps
the current 20 server groups and current PV/BESS parameters, calibrates only a
private diagnostic fleet, and evaluates five prescribed legal action cases.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from copy import deepcopy
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from configs.config_ultimate import (  # noqa: E402
    DATA_CONFIG,
    ENV_CONFIG,
    GRID_CONFIG,
    GRID_REWARD_CONFIG,
    GRID_SCENARIO_CONFIG,
    IDC_SCALE_CONFIG,
    REWARD_CONFIG,
)
from diagnostics.idc_25mw_acopf_probe import (  # noqa: E402
    SiteHour,
    _make_site_env,
    run_opf_case,
    validate_no_nonfinite,
)


SEED = 2026
RATED_FACILITY_MW = 25.0
NUM_GROUPS = 20
RATED_SERVER_ACTION = 1.0
LOW_SERVER_ACTION = 0.0
NORMAL_SERVER_ACTION = 0.5
URGENT_ACTION = 0.5
CONTINUITY_ACTION = 0.5
BESS_ACTIONS = {"charge": 0.0, "idle": 0.5, "discharge": 1.0}


@dataclass(frozen=True)
class ControlHour:
    case: str
    hour: int
    server_action: float
    bess_action: float
    actual_task_load_mean: float
    actual_total_load_mean: float
    completed_work: float
    backlog_work: float
    p_it_mw: float
    p_cooling_mw: float
    p_others_mw: float
    p_idc_mw: float
    pv_available_mw: float
    pv_used_mw: float
    desired_bess_charge_mw: float
    desired_bess_discharge_mw: float
    bess_charge_mw: float
    bess_discharge_mw: float
    bess_soc: float
    p_bus_net_mw: float
    grid_load_scale: float


CASES = (
    ("low_idc_discharge", LOW_SERVER_ACTION, BESS_ACTIONS["discharge"]),
    ("low_idc_charge", LOW_SERVER_ACTION, BESS_ACTIONS["charge"]),
    ("high_idc_idle", RATED_SERVER_ACTION, BESS_ACTIONS["idle"]),
    ("high_idc_charge", RATED_SERVER_ACTION, BESS_ACTIONS["charge"]),
    ("high_idc_discharge", RATED_SERVER_ACTION, BESS_ACTIONS["discharge"]),
)


def _formal_config_snapshot() -> dict[str, Any]:
    return deepcopy(
        {
            "env": ENV_CONFIG,
            "scale": IDC_SCALE_CONFIG,
            "data": DATA_CONFIG,
            "reward": REWARD_CONFIG,
            "grid": GRID_CONFIG,
            "grid_reward": GRID_REWARD_CONFIG,
            "grid_scenario": GRID_SCENARIO_CONFIG,
        }
    )


def _facility_power_at_load(env, group_size: int, load: float, ambient_c: float) -> dict[str, float]:
    model = env.env.model
    loads = np.full((1, model.N), float(load), dtype=np.float64)
    p_idc, p_it, pue, cop, p_cooling = model.calc_pue_and_total_power(
        loads, np.asarray([ambient_c], dtype=np.float64)
    )
    return {
        "group_size": int(group_size),
        "total_load": float(load),
        "ambient_c": float(ambient_c),
        "p_it_mw": float(p_it[0]) / 1e6,
        "p_cooling_mw": float(p_cooling[0]) / 1e6,
        "p_others_mw": float(model.P_others) / 1e6,
        "p_idc_mw": float(p_idc[0]) / 1e6,
        "pue": float(pue[0]),
        "cop": float(cop[0]),
    }


def calibrate_rated_group_size(seed: int = SEED) -> dict[str, Any]:
    """Solve the integer fleet size at max legal planned load and worst 24h temperature."""

    reference_size = int(IDC_SCALE_CONFIG["server_group_size"])
    env = _make_site_env(reference_size, seed)
    try:
        base = env.env
        maximum_total_load = float(base.base_load + base.max_task_load_per_server)
        maximum_ambient = float(np.max(base.T_amb[:24]))
        reference = _facility_power_at_load(env, reference_size, maximum_total_load, maximum_ambient)
        fixed_it_mw = float(base.model.delta_P_loss) / 1e6
        cop = reference["cop"]
        scalable_it_per_group_mw = (reference["p_it_mw"] - fixed_it_mw) / reference_size
        target_it_mw = (
            RATED_FACILITY_MW - reference["p_others_mw"]
        ) / (1.0 + 1.0 / cop)
        exact_size = (target_it_mw - fixed_it_mw) / scalable_it_per_group_mw
        chosen_size = max(int(round(exact_size)), 1)
    finally:
        env.close()

    rated_env = _make_site_env(chosen_size, seed)
    old_env = _make_site_env(2395, seed)
    try:
        rated = _facility_power_at_load(rated_env, chosen_size, maximum_total_load, maximum_ambient)
        old_calibration_rated = _facility_power_at_load(
            old_env, 2395, maximum_total_load, maximum_ambient
        )
    finally:
        rated_env.close()
        old_env.close()

    return {
        "definition": (
            "25 MW is facility-level rated demand P_IT + P_cooling + P_others at "
            "maximum legal compute action (server action=1, planned total load=0.65) "
            "and the maximum ambient temperature in the fixed 24h scenario."
        ),
        "reference_group_size": reference_size,
        "exact_group_size": float(exact_size),
        "chosen_group_size": chosen_size,
        "fleet_scale_vs_current_formal": chosen_size / float(reference_size),
        "effective_server_count": chosen_size * NUM_GROUPS,
        "maximum_legal_server_action": RATED_SERVER_ACTION,
        "maximum_planned_total_load": maximum_total_load,
        "maximum_scenario_ambient_c": maximum_ambient,
        "rated_power": rated,
        "old_action_0_5_calibration_group_size": 2395,
        "old_group_size_rated_power": old_calibration_rated,
    }


def collect_control_trace(
    case_name: str,
    group_size: int,
    server_action: float,
    bess_action: float,
    seed: int = SEED,
) -> tuple[list[SiteHour], list[ControlHour], dict[str, Any]]:
    env = _make_site_env(group_size, seed)
    base = env.env
    action = np.full(env.action_space.shape, 0.5, dtype=np.float32)
    action[: base.model.N] = float(server_action)
    action[base.model.N] = URGENT_ACTION
    action[base.model.N + 1] = CONTINUITY_ACTION
    action[base.model.N + 2] = float(bess_action)
    env.reset(seed=seed)
    site_rows: list[SiteHour] = []
    control_rows: list[ControlHour] = []
    try:
        for hour in range(24):
            load_scale = float(env.grid_load_scale_t[hour % len(env.grid_load_scale_t)])
            _, _, terminated, truncated, info = env.step(action)
            site = SiteHour(
                hour=hour,
                grid_load_scale=load_scale,
                p_it_mw=float(info["P_IT"]) / 1e6,
                p_cooling_mw=float(info["P_cooling"]) / 1e6,
                p_others_mw=float(base.model.P_others) / 1e6,
                p_idc_mw=float(info["P_IDC_kW"]) / 1000.0,
                pv_available_mw=float(info["pv_available_kW"]) / 1000.0,
                pv_used_mw=float(info["pv_used_kW"]) / 1000.0,
                bess_charge_mw=float(info["bess_charge_power_kW"]) / 1000.0,
                bess_discharge_mw=float(info["bess_discharge_power_kW"]) / 1000.0,
                p_bus_net_mw=max(float(info["P_bus_net_kW"]), 0.0) / 1000.0,
            )
            site_rows.append(site)
            control_rows.append(
                ControlHour(
                    case=case_name,
                    hour=hour,
                    server_action=float(server_action),
                    bess_action=float(bess_action),
                    actual_task_load_mean=float(info["actual_task_load_mean"]),
                    actual_total_load_mean=float(info["actual_total_load_mean"]),
                    completed_work=float(info["completed_work"]),
                    backlog_work=float(info["backlog_work"]),
                    p_it_mw=site.p_it_mw,
                    p_cooling_mw=site.p_cooling_mw,
                    p_others_mw=site.p_others_mw,
                    p_idc_mw=site.p_idc_mw,
                    pv_available_mw=site.pv_available_mw,
                    pv_used_mw=site.pv_used_mw,
                    desired_bess_charge_mw=float(info["desired_bess_charge_power_kW"]) / 1000.0,
                    desired_bess_discharge_mw=float(info["desired_bess_discharge_power_kW"]) / 1000.0,
                    bess_charge_mw=site.bess_charge_mw,
                    bess_discharge_mw=site.bess_discharge_mw,
                    bess_soc=float(info["bess_soc"]),
                    p_bus_net_mw=site.p_bus_net_mw,
                    grid_load_scale=load_scale,
                )
            )
            if terminated or truncated:
                break
    finally:
        env.close()
    if len(site_rows) != 24:
        raise RuntimeError(f"{case_name}: expected 24 hours, got {len(site_rows)}")
    static = {
        "action_shape": list(action.shape),
        "server_action": float(server_action),
        "urgent_action": URGENT_ACTION,
        "continuity_action": CONTINUITY_ACTION,
        "bess_action": float(bess_action),
        "server_group_size": int(group_size),
        "num_server_groups": int(base.model.N),
        "effective_server_count": int(base.model.effective_total_server_count),
        "bess_capacity_mwh": float(base.bess_capacity_kWh) / 1000.0,
        "bess_charge_power_max_mw": float(base.bess_charge_power_max_kW) / 1000.0,
        "bess_discharge_power_max_mw": float(base.bess_discharge_power_max_kW) / 1000.0,
        "pv_peak_mw": float(max(base.pv_t)) / 1000.0,
    }
    return site_rows, control_rows, static


def summarize_control(rows: list[ControlHour]) -> dict[str, Any]:
    def stats(field: str) -> dict[str, float]:
        values = np.asarray([getattr(row, field) for row in rows], dtype=np.float64)
        return {"min": float(values.min()), "mean": float(values.mean()), "max": float(values.max())}

    return {
        "actual_task_load_mean": stats("actual_task_load_mean"),
        "actual_total_load_mean": stats("actual_total_load_mean"),
        "p_it_mw": stats("p_it_mw"),
        "p_cooling_mw": stats("p_cooling_mw"),
        "p_idc_mw": stats("p_idc_mw"),
        "p_bus_net_mw": stats("p_bus_net_mw"),
        "bess_soc": stats("bess_soc"),
        "bess_charge_peak_mw": max(row.bess_charge_mw for row in rows),
        "bess_discharge_peak_mw": max(row.bess_discharge_mw for row in rows),
        "bess_charge_energy_mwh": sum(row.bess_charge_mw for row in rows),
        "bess_discharge_energy_mwh": sum(row.bess_discharge_mw for row in rows),
        "bess_nonzero_hours": sum(
            row.bess_charge_mw > 1e-9 or row.bess_discharge_mw > 1e-9 for row in rows
        ),
        "completed_work_total": sum(row.completed_work for row in rows),
        "backlog_work_final": rows[-1].backlog_work,
    }


def _numeric_delta(a: dict[str, Any], b: dict[str, Any], metrics: tuple[str, ...]) -> dict[str, float]:
    return {metric: float(a[metric]) - float(b[metric]) for metric in metrics}


def _delta_stats(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "min": float(array.min()),
        "mean": float(array.mean()),
        "max": float(array.max()),
        "max_absolute": float(np.abs(array).max()),
    }


def paired_hourly_effect(
    opf_rows: list[dict[str, Any]],
    generator_rows: list[dict[str, Any]],
    case_a: str,
    case_b: str,
) -> dict[str, Any]:
    """Return same-hour action effects as case_a minus case_b."""

    opf_a = {int(row["hour"]): row for row in opf_rows if row["case"] == case_a}
    opf_b = {int(row["hour"]): row for row in opf_rows if row["case"] == case_b}
    opf_metrics = (
        "p_bus_net_mw",
        "min_voltage_pu",
        "idc_bus_voltage_pu",
        "network_loss_mw",
        "ext_grid_import_mw",
        "idc_bus_lmp",
        "lmp_spread",
    )
    result: dict[str, Any] = {
        "definition": f"{case_a} minus {case_b}, paired at the same hour",
        "opf": {
            metric: _delta_stats(
                [float(opf_a[hour][metric]) - float(opf_b[hour][metric]) for hour in range(24)]
            )
            for metric in opf_metrics
        },
        "sources": {},
    }
    gen_a = {
        (str(row["element_id"]), int(row["hour"])): row
        for row in generator_rows
        if row["case"] == case_a
    }
    gen_b = {
        (str(row["element_id"]), int(row["hour"])): row
        for row in generator_rows
        if row["case"] == case_b
    }
    source_metrics = (
        "p_mw",
        "q_mvar",
        "p_upper_headroom_mw",
        "q_upper_headroom_mvar",
    )
    for element_id in sorted({key[0] for key in gen_a} & {key[0] for key in gen_b}):
        result["sources"][element_id] = {
            metric: _delta_stats(
                [
                    float(gen_a[(element_id, hour)][metric])
                    - float(gen_b[(element_id, hour)][metric])
                    for hour in range(24)
                ]
            )
            for metric in source_metrics
        }
    return result


def comparisons(
    summaries: dict[str, dict[str, Any]],
    opf_rows: list[dict[str, Any]],
    generator_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    metrics = (
        "p_idc_mw_max",
        "p_bus_net_mw_max",
        "opf_success_count",
        "min_voltage_pu",
        "minimum_lower_voltage_margin_pu",
        "idc_bus_voltage_min_pu",
        "idc_bus_voltage_mean_pu",
        "minimum_source_q_upper_headroom_mvar",
        "minimum_source_p_upper_headroom_mw",
        "network_loss_mean_mw",
        "network_loss_max_mw",
        "peak_ext_grid_import_mw",
        "idc_bus_lmp_mean",
        "lmp_spread_max",
        "max_line_loading_percent",
        "max_transformer_loading_percent",
    )
    return {
        "high_charge_minus_high_discharge": _numeric_delta(
            summaries["high_idc_charge"], summaries["high_idc_discharge"], metrics
        ),
        "high_charge_minus_low_charge": _numeric_delta(
            summaries["high_idc_charge"], summaries["low_idc_charge"], metrics
        ),
        "high_discharge_minus_low_discharge": _numeric_delta(
            summaries["high_idc_discharge"], summaries["low_idc_discharge"], metrics
        ),
        "paired_hourly_high_charge_minus_high_discharge": paired_hourly_effect(
            opf_rows,
            generator_rows,
            "high_idc_charge",
            "high_idc_discharge",
        ),
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    flat = [
        {key: value for key, value in row.items() if not isinstance(value, (dict, list, tuple))}
        for row in rows
    ]
    fields = sorted({key for row in flat for key in row})
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(flat)


def run(output_dir: Path, seed: int = SEED) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    config_before = _formal_config_snapshot()
    calibration = calibrate_rated_group_size(seed)
    group_size = int(calibration["chosen_group_size"])

    # A separate site-only action=0.5 trace supplies the requested normal-range audit;
    # it is not an additional grid controllability case.
    normal_site, normal_control, normal_static = collect_control_trace(
        "normal_reference_site_only", group_size, NORMAL_SERVER_ACTION, BESS_ACTIONS["idle"], seed
    )

    case_summaries: dict[str, dict[str, Any]] = {}
    case_control_summaries: dict[str, dict[str, Any]] = {}
    case_static: dict[str, dict[str, Any]] = {}
    all_control: list[ControlHour] = []
    all_opf: list[dict[str, Any]] = []
    all_generators: list[dict[str, Any]] = []
    for case_name, server_action, bess_action in CASES:
        site, control, static = collect_control_trace(
            case_name, group_size, server_action, bess_action, seed
        )
        summary, opf_rows, generator_rows = run_opf_case(case_name, site)
        case_summaries[case_name] = summary
        case_control_summaries[case_name] = summarize_control(control)
        case_static[case_name] = static
        all_control.extend(control)
        all_opf.extend(opf_rows)
        all_generators.extend(generator_rows)

    config_after = _formal_config_snapshot()
    if config_before != config_after:
        raise AssertionError("Diagnostic mutated formal configuration dictionaries.")
    if any(static["action_shape"] != [23] for static in case_static.values()):
        raise AssertionError("Actor/action dimension changed from 23.")

    payload = {
        "metadata": {
            "seed": seed,
            "background_multiplier": 1.0,
            "idc_ieee_bus_number": 9,
            "num_grid_cases": len(CASES),
            "grid_cases": [case[0] for case in CASES],
            "branch_rating_note": (
                "The 9900 MVA branch ratings are not validated thermal limits; line and transformer "
                "loading are response metrics only."
            ),
        },
        "calibration": calibration,
        "normal_reference": {
            "static": normal_static,
            "summary": summarize_control(normal_control),
            "site_hours": [asdict(row) for row in normal_site],
        },
        "case_static": case_static,
        "case_control_summaries": case_control_summaries,
        "case_opf_summaries": case_summaries,
        "comparisons": comparisons(case_summaries, all_opf, all_generators),
        "formal_config_unchanged": config_before == config_after,
        "control_hours": [asdict(row) for row in all_control],
        "opf_hours": all_opf,
        "generator_hours": all_generators,
    }
    validate_no_nonfinite(payload)
    with (output_dir / "idc_25mw_safety_controllability.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, allow_nan=False)
    write_csv(output_dir / "control_hourly.csv", payload["control_hours"])
    write_csv(output_dir / "opf_hourly.csv", all_opf)
    write_csv(output_dir / "generator_capability_hourly.csv", all_generators)
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/idc_25mw_safety_controllability"),
    )
    parser.add_argument("--seed", type=int, default=SEED)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    result = run(args.output_dir, args.seed)
    print(
        json.dumps(
            {
                "output_dir": str(args.output_dir),
                "rated_group_size": result["calibration"]["chosen_group_size"],
                "rated_power_mw": result["calibration"]["rated_power"]["p_idc_mw"],
                "opf_success": {
                    name: summary["opf_success_count"]
                    for name, summary in result["case_opf_summaries"].items()
                },
            },
            indent=2,
        )
    )
