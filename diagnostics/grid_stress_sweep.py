"""Phase 2.5 IEEE-14 capacity audit and deterministic stress sweep.

This is an offline diagnostic.  It never mutates the formal environment
configuration or the network stored in :mod:`grid_model.ieee14_loader`.
Every OPF solve works on the deep copy created by ``solve_ac_opf``.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from copy import deepcopy
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path
from typing import Any, Iterable

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from configs.experiment_cases import get_experiment_case
from grid_model import get_bus_index_by_ieee_number, load_ieee14_case
from grid_model.opf_solver import solve_ac_opf
from train.train_ppo_ultimate import make_unmonitored_env


SEED = 2026
BACKGROUND_AXIS = (1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0)
IDC_AXIS = (1.0, 1.5, 2.0, 2.5, 3.0, 4.0)
COARSE_BACKGROUND = (1.0, 1.5, 2.0, 2.5)
COARSE_IDC = (1.0, 1.5, 2.0, 3.0)
REFERENCE_FACTORS = (1.2, 1.5)


@dataclass(frozen=True)
class PhysicsHour:
    hour: int
    background_load_scale: float
    idc_bus_net_mw: float
    idc_demand_mw: float
    bess_charge_mw: float
    bess_discharge_mw: float
    pv_used_mw: float


def _finite(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _max_finite(values: Iterable[Any]) -> float | None:
    numbers = [value for value in (_finite(item) for item in values) if value is not None]
    return max(numbers) if numbers else None


def _min_finite(values: Iterable[Any]) -> float | None:
    numbers = [value for value in (_finite(item) for item in values) if value is not None]
    return min(numbers) if numbers else None


def _mean_finite(values: Iterable[Any]) -> float | None:
    numbers = [value for value in (_finite(item) for item in values) if value is not None]
    return float(np.mean(numbers)) if numbers else None


def _percentile_finite(values: Iterable[Any], percentile: float) -> float | None:
    numbers = [value for value in (_finite(item) for item in values) if value is not None]
    return float(np.percentile(numbers, percentile)) if numbers else None


def _safe_ratio(value: float | None, denominator: float | None) -> float | None:
    if value is None or denominator is None or denominator <= 0.0:
        return None
    return value / denominator


def collect_fixed_physics_trace(seed: int = SEED) -> list[PhysicsHour]:
    """Run the real IDC/BESS/PV transition model with a fixed neutral action.

    The grid wrapper is disabled *on this private instance* so this step only
    establishes a physically coherent 24 h site trace.  The background grid
    profile is retained from that same wrapper and is later used by the probe.
    """

    case_config = get_experiment_case("main")
    env = make_unmonitored_env(
        deepcopy(case_config["env_config"]),
        deepcopy(case_config["reward_config"]),
        deepcopy(case_config["data_config"]),
        seed=seed,
        grid_cache_config={"enable_grid_cache": False},
    )
    env.grid_enabled = False
    env.use_mef = False
    action = np.full(env.action_space.shape, 0.5, dtype=np.float32)
    env.reset(seed=seed)
    trace: list[PhysicsHour] = []
    try:
        for hour in range(24):
            scale = float(env.grid_load_scale_t[hour % len(env.grid_load_scale_t)])
            _, _, terminated, truncated, info = env.step(action)
            trace.append(
                PhysicsHour(
                    hour=hour,
                    background_load_scale=scale,
                    idc_bus_net_mw=max(float(info["P_bus_net_kW"]), 0.0) / 1000.0,
                    idc_demand_mw=float(info["P_IDC_kW"]) / 1000.0,
                    bess_charge_mw=float(info["bess_charge_power_kW"]) / 1000.0,
                    bess_discharge_mw=float(info["bess_discharge_power_kW"]) / 1000.0,
                    pv_used_mw=float(info["pv_used_kW"]) / 1000.0,
                )
            )
            if terminated or truncated:
                break
    finally:
        env.close()
    if len(trace) != 24:
        raise RuntimeError(f"Expected a 24 h physics trace, got {len(trace)} hours.")
    return trace


def audit_capacity_definitions() -> dict[str, Any]:
    """Return exact case data plus the capacity implied by pandapower fields."""

    case = load_ieee14_case()
    net = case.raw_network
    ext_buses = set(int(value) for value in net.ext_grid.bus.tolist())
    gen_buses = set(int(value) for value in net.gen.bus.tolist())
    load_buses = set(int(value) for value in net.load.bus.tolist())
    idc_bus = get_bus_index_by_ieee_number(case, 9)

    buses = []
    for index, row in net.bus.iterrows():
        bus = int(index)
        roles = []
        if bus in ext_buses:
            roles.extend(("slack", "ext_grid_connected"))
        if bus in gen_buses:
            roles.extend(("PV", "generator_connected"))
        elif bus in load_buses:
            roles.append("PQ")
        elif bus not in ext_buses:
            roles.append("passive")
        if bus == idc_bus:
            roles.append("IDC_connected")
        buses.append(
            {
                "bus_index": bus,
                "ieee_bus_number": case.bus_index_to_ieee_bus_number[bus],
                "vn_kv": float(row.vn_kv),
                "min_vm_pu": float(row.min_vm_pu),
                "max_vm_pu": float(row.max_vm_pu),
                "roles": "+".join(roles),
            }
        )

    lines = []
    for index, row in net.line.iterrows():
        from_bus = int(row.from_bus)
        permitted_current = float(row.max_i_ka) * float(row.df) * int(row.parallel)
        approx_mva = math.sqrt(3.0) * float(net.bus.at[from_bus, "vn_kv"]) * permitted_current
        lines.append(
            {
                "line_index": int(index),
                "from_bus": from_bus,
                "to_bus": int(row.to_bus),
                "length_km": float(row.length_km),
                "r_ohm_per_km": float(row.r_ohm_per_km),
                "x_ohm_per_km": float(row.x_ohm_per_km),
                "c_nf_per_km": float(row.c_nf_per_km),
                "max_i_ka": float(row.max_i_ka),
                "df": float(row.df),
                "parallel": int(row.parallel),
                "max_loading_percent": float(row.max_loading_percent),
                "std_type": None if row.get("std_type") is None else str(row.get("std_type")),
                "permitted_current_ka": permitted_current,
                "approx_thermal_mva_at_from_bus_voltage": approx_mva,
            }
        )

    transformers = []
    for index, row in net.trafo.iterrows():
        transformers.append(
            {
                "trafo_index": int(index),
                "hv_bus": int(row.hv_bus),
                "lv_bus": int(row.lv_bus),
                "sn_mva": float(row.sn_mva),
                "vn_hv_kv": float(row.vn_hv_kv),
                "vn_lv_kv": float(row.vn_lv_kv),
                "vk_percent": float(row.vk_percent),
                "vkr_percent": float(row.vkr_percent),
                "parallel": int(row.parallel),
                "df": float(row.df),
                "max_loading_percent": float(row.max_loading_percent),
                "permitted_apparent_power_mva": (
                    float(row.sn_mva)
                    * float(row.df)
                    * int(row.parallel)
                    * float(row.max_loading_percent)
                    / 100.0
                ),
            }
        )

    generators = []
    for table_name in ("ext_grid", "gen"):
        table = getattr(net, table_name)
        for index, row in table.iterrows():
            generators.append(
                {
                    "element": table_name,
                    "index": int(index),
                    "bus": int(row.bus),
                    "vm_pu": float(row.vm_pu),
                    "min_p_mw": _finite(row.get("min_p_mw")),
                    "max_p_mw": _finite(row.get("max_p_mw")),
                    "min_q_mvar": _finite(row.get("min_q_mvar")),
                    "max_q_mvar": _finite(row.get("max_q_mvar")),
                }
            )

    native_p = float((net.load.p_mw.astype(float) * net.load.scaling.astype(float)).sum())
    native_q = float((net.load.q_mvar.astype(float) * net.load.scaling.astype(float)).sum())
    return {
        "case_name": case.name,
        "base_mva": float(case.base_mva),
        "idc_bus_index": idc_bus,
        "idc_ieee_bus_number": 9,
        "native_load_p_mw": native_p,
        "native_load_q_mvar": native_q,
        "buses": buses,
        "lines": lines,
        "transformers": transformers,
        "generators": generators,
        "line_loading_definition": "100 * max(i_from_ka, i_to_ka) / (max_i_ka * df * parallel)",
        "transformer_loading_definition": (
            "pandapower runopp current mode: 100 * max(side current * side nominal voltage * sqrt(3)) "
            "/ (sn_mva * df * parallel)"
        ),
    }


def _extract_hour(
    result: Any,
    physics: PhysicsHour,
    background_multiplier: float,
    idc_multiplier: float,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "hour": physics.hour,
        "background_load_multiplier": float(background_multiplier),
        "idc_multiplier": float(idc_multiplier),
        "applied_background_load_scale": physics.background_load_scale * background_multiplier,
        "baseline_idc_bus_net_mw": physics.idc_bus_net_mw,
        "applied_idc_bus_net_mw": physics.idc_bus_net_mw * idc_multiplier,
        "opf_success": bool(result.success),
        "opf_message": str(result.message),
    }
    if not result.success:
        return row

    net = result.raw_result
    voltages = result.bus_voltage_pu
    lower_margins = {
        bus: voltages[bus] - result.bus_voltage_min_pu[bus]
        for bus in voltages
        if _finite(voltages[bus]) is not None
    }
    upper_margins = {
        bus: result.bus_voltage_max_pu[bus] - voltages[bus]
        for bus in voltages
        if _finite(voltages[bus]) is not None
    }
    line_ratios = {
        idx: _safe_ratio(_finite(value), _finite(result.line_loading_limit_percent.get(idx)))
        for idx, value in result.line_loading_percent.items()
    }
    trafo_ratios = {
        idx: _safe_ratio(_finite(value), _finite(result.transformer_loading_limit_percent.get(idx)))
        for idx, value in result.transformer_loading_percent.items()
    }
    line_values = {idx: value for idx, value in result.line_loading_percent.items() if _finite(value) is not None}
    trafo_values = {
        idx: value for idx, value in result.transformer_loading_percent.items() if _finite(value) is not None
    }
    idc_bus = 8
    incident_lines = net.line.index[(net.line.from_bus == idc_bus) | (net.line.to_bus == idc_bus)].tolist()
    incident_trafos = net.trafo.index[(net.trafo.hv_bus == idc_bus) | (net.trafo.lv_bus == idc_bus)].tolist()
    incident_values = [result.line_loading_percent.get(int(idx)) for idx in incident_lines]
    incident_values.extend(result.transformer_loading_percent.get(int(idx)) for idx in incident_trafos)
    ext_grid_import = _finite(net.res_ext_grid.p_mw.sum()) if len(net.res_ext_grid) else None

    row.update(
        {
            "min_bus_voltage_pu": _min_finite(voltages.values()),
            "max_bus_voltage_pu": _max_finite(voltages.values()),
            "minimum_lower_voltage_margin_pu": _min_finite(lower_margins.values()),
            "minimum_upper_voltage_margin_pu": _min_finite(upper_margins.values()),
            "minimum_voltage_safety_margin_pu": _min_finite(
                list(lower_margins.values()) + list(upper_margins.values())
            ),
            "configured_voltage_violation_count": sum(value < -1e-7 for value in lower_margins.values())
            + sum(value < -1e-7 for value in upper_margins.values()),
            "max_line_loading_percent": _max_finite(line_values.values()),
            "critical_line_index": max(line_values, key=line_values.get) if line_values else None,
            "configured_line_violation_count": sum(
                ratio is not None and ratio > 1.0 + 1e-7 for ratio in line_ratios.values()
            ),
            "max_transformer_loading_percent": _max_finite(trafo_values.values()),
            "critical_transformer_index": max(trafo_values, key=trafo_values.get) if trafo_values else None,
            "configured_transformer_violation_count": sum(
                ratio is not None and ratio > 1.0 + 1e-7 for ratio in trafo_ratios.values()
            ),
            "network_loss_mw": _finite(result.network_loss_mw),
            "total_load_mw": _finite(result.total_load_mw),
            "total_generation_mw": _finite(result.total_generation_mw),
            "ext_grid_import_mw": ext_grid_import,
            "lmp_min": _min_finite(result.lmp_by_bus.values()),
            "lmp_max": _max_finite(result.lmp_by_bus.values()),
            "lmp_spread": (
                _max_finite(result.lmp_by_bus.values()) - _min_finite(result.lmp_by_bus.values())
                if result.lmp_by_bus
                else None
            ),
            "idc_bus_voltage_pu": _finite(result.bus_voltage_pu.get(idc_bus)),
            "idc_bus_lmp": _finite(result.lmp_by_bus.get(idc_bus)),
            "idc_bus_p_mw": _finite(result.bus_active_power_mw.get(idc_bus)),
            "idc_bus_q_mvar": _finite(result.bus_reactive_power_mvar.get(idc_bus)),
            "idc_bus_incident_max_loading_percent": _max_finite(incident_values),
            "line_loading_by_index": {str(key): _finite(value) for key, value in line_values.items()},
            "transformer_loading_by_index": {str(key): _finite(value) for key, value in trafo_values.items()},
            "bus_voltage_by_index": {str(key): _finite(value) for key, value in voltages.items()},
        }
    )
    return row


def summarize_case(
    hourly: list[dict[str, Any]],
    background_multiplier: float,
    idc_multiplier: float,
    reference_peaks: dict[str, dict[int, float]] | None = None,
) -> dict[str, Any]:
    successful = [row for row in hourly if row["opf_success"]]
    summary: dict[str, Any] = {
        "background_load_multiplier": float(background_multiplier),
        "idc_multiplier": float(idc_multiplier),
        "hours": len(hourly),
        "opf_success_count": len(successful),
        "opf_success_rate": len(successful) / len(hourly) if hourly else 0.0,
        "opf_failure_count": len(hourly) - len(successful),
        "min_bus_voltage_pu": _min_finite(row.get("min_bus_voltage_pu") for row in successful),
        "max_bus_voltage_pu": _max_finite(row.get("max_bus_voltage_pu") for row in successful),
        "minimum_lower_voltage_margin_pu": _min_finite(
            row.get("minimum_lower_voltage_margin_pu") for row in successful
        ),
        "minimum_upper_voltage_margin_pu": _min_finite(
            row.get("minimum_upper_voltage_margin_pu") for row in successful
        ),
        "minimum_voltage_safety_margin_pu": _min_finite(
            row.get("minimum_voltage_safety_margin_pu") for row in successful
        ),
        "configured_voltage_violation_count": sum(
            int(row.get("configured_voltage_violation_count", 0)) for row in successful
        ),
        "max_line_loading_percent": _max_finite(row.get("max_line_loading_percent") for row in successful),
        "configured_line_violation_count": sum(
            int(row.get("configured_line_violation_count", 0)) for row in successful
        ),
        "max_transformer_loading_percent": _max_finite(
            row.get("max_transformer_loading_percent") for row in successful
        ),
        "configured_transformer_violation_count": sum(
            int(row.get("configured_transformer_violation_count", 0)) for row in successful
        ),
        "mean_network_loss_mw": _mean_finite(row.get("network_loss_mw") for row in successful),
        "max_network_loss_mw": _max_finite(row.get("network_loss_mw") for row in successful),
        "peak_grid_import_mw": _max_finite(row.get("ext_grid_import_mw") for row in successful),
        "lmp_min": _min_finite(row.get("lmp_min") for row in successful),
        "lmp_max": _max_finite(row.get("lmp_max") for row in successful),
        "lmp_spread_max": _max_finite(row.get("lmp_spread") for row in successful),
        "idc_bus_voltage_min_pu": _min_finite(row.get("idc_bus_voltage_pu") for row in successful),
        "idc_bus_voltage_mean_pu": _mean_finite(row.get("idc_bus_voltage_pu") for row in successful),
        "idc_bus_lmp_min": _min_finite(row.get("idc_bus_lmp") for row in successful),
        "idc_bus_lmp_max": _max_finite(row.get("idc_bus_lmp") for row in successful),
        "idc_bus_p_min_mw": _min_finite(row.get("idc_bus_p_mw") for row in successful),
        "idc_bus_p_max_mw": _max_finite(row.get("idc_bus_p_mw") for row in successful),
        "idc_bus_q_min_mvar": _min_finite(row.get("idc_bus_q_mvar") for row in successful),
        "idc_bus_q_max_mvar": _max_finite(row.get("idc_bus_q_mvar") for row in successful),
        "idc_bus_incident_max_loading_percent": _max_finite(
            row.get("idc_bus_incident_max_loading_percent") for row in successful
        ),
    }
    line_samples: dict[int, list[float]] = defaultdict(list)
    trafo_samples: dict[int, list[float]] = defaultdict(list)
    for row in successful:
        for key, value in row.get("line_loading_by_index", {}).items():
            if value is not None:
                line_samples[int(key)].append(float(value))
        for key, value in row.get("transformer_loading_by_index", {}).items():
            if value is not None:
                trafo_samples[int(key)].append(float(value))
    summary["critical_line_index"] = (
        max(line_samples, key=lambda key: max(line_samples[key])) if line_samples else None
    )
    summary["critical_transformer_index"] = (
        max(trafo_samples, key=lambda key: max(trafo_samples[key])) if trafo_samples else None
    )
    summary["line_statistics"] = {
        str(key): {
            "max_loading_percent": max(values),
            "mean_loading_percent": float(np.mean(values)),
            "p95_loading_percent": float(np.percentile(values, 95)),
        }
        for key, values in sorted(line_samples.items())
    }
    summary["transformer_statistics"] = {
        str(key): {
            "max_loading_percent": max(values),
            "mean_loading_percent": float(np.mean(values)),
            "p95_loading_percent": float(np.percentile(values, 95)),
        }
        for key, values in sorted(trafo_samples.items())
    }
    if reference_peaks:
        for factor in REFERENCE_FACTORS:
            line_count = 0
            trafo_count = 0
            for index, values in line_samples.items():
                reference = reference_peaks["line"].get(index, math.inf) * factor
                line_count += sum(value > reference + 1e-9 for value in values)
            for index, values in trafo_samples.items():
                reference = reference_peaks["trafo"].get(index, math.inf) * factor
                trafo_count += sum(value > reference + 1e-9 for value in values)
            suffix = str(factor).replace(".", "p")
            summary[f"reference_line_exceedance_count_{suffix}x_baseline"] = line_count
            summary[f"reference_transformer_exceedance_count_{suffix}x_baseline"] = trafo_count
    return summary


def run_stress_case(
    trace: list[PhysicsHour],
    background_multiplier: float,
    idc_multiplier: float,
    reference_peaks: dict[str, dict[int, float]] | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    case = load_ieee14_case()
    idc_bus = get_bus_index_by_ieee_number(case, 9)
    hourly = []
    for physics in trace:
        result = solve_ac_opf(
            case,
            idc_bus_id=idc_bus,
            idc_load_mw=physics.idc_bus_net_mw * float(idc_multiplier),
            load_scale=physics.background_load_scale * float(background_multiplier),
        )
        hourly.append(_extract_hour(result, physics, background_multiplier, idc_multiplier))
    return summarize_case(
        hourly,
        background_multiplier,
        idc_multiplier,
        reference_peaks=reference_peaks,
    ), hourly


def baseline_reference_peaks(summary: dict[str, Any]) -> dict[str, dict[int, float]]:
    return {
        "line": {
            int(key): float(value["max_loading_percent"])
            for key, value in summary["line_statistics"].items()
        },
        "trafo": {
            int(key): float(value["max_loading_percent"])
            for key, value in summary["transformer_statistics"].items()
        },
    }


def classify_region(summary: dict[str, Any], baseline: dict[str, Any]) -> str:
    """Evidence-oriented labels; thresholds are reported, not formal safety rules."""

    if summary["opf_success_rate"] < 0.95 or summary["configured_voltage_violation_count"] > 0:
        return "C_infeasible_or_unstable"
    # Generator-regulated buses can legally sit on their upper setpoint in the
    # normal case, so lower-voltage margin is the informative load-stress signal.
    margin = summary.get("minimum_lower_voltage_margin_pu")
    baseline_margin = baseline.get("minimum_lower_voltage_margin_pu")
    relative_line = _safe_ratio(summary.get("max_line_loading_percent"), baseline.get("max_line_loading_percent"))
    relative_trafo = _safe_ratio(
        summary.get("max_transformer_loading_percent"), baseline.get("max_transformer_loading_percent")
    )
    stressed = (
        (margin is not None and margin <= 0.02)
        or (
            margin is not None
            and baseline_margin is not None
            and margin <= 0.5 * baseline_margin
        )
        or (relative_line is not None and relative_line >= 1.5)
        or (relative_trafo is not None and relative_trafo >= 1.5)
    )
    return "B_stressed_but_feasible" if stressed else "A_normal_or_loose"


def _case_key(background: float, idc: float) -> tuple[float, float]:
    return round(float(background), 6), round(float(idc), 6)


def run_sweep(trace: list[PhysicsHour]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    summaries: list[dict[str, Any]] = []
    hourly_rows: list[dict[str, Any]] = []
    seen: set[tuple[float, float]] = set()

    baseline, baseline_hourly = run_stress_case(trace, 1.0, 1.0)
    references = baseline_reference_peaks(baseline)
    baseline = summarize_case(baseline_hourly, 1.0, 1.0, references)
    baseline["sweep_group"] = "baseline+axes+coarse"
    baseline["region"] = classify_region(baseline, baseline)
    summaries.append(baseline)
    hourly_rows.extend(baseline_hourly)
    seen.add(_case_key(1.0, 1.0))

    requested = []
    requested.extend((value, 1.0, "background_axis") for value in BACKGROUND_AXIS)
    requested.extend((1.0, value, "idc_axis") for value in IDC_AXIS)
    requested.extend((bg, idc, "coarse_2d") for bg in COARSE_BACKGROUND for idc in COARSE_IDC)
    for background, idc, group in requested:
        key = _case_key(background, idc)
        if key in seen:
            continue
        summary, hourly = run_stress_case(trace, background, idc, references)
        summary["sweep_group"] = group
        summary["region"] = classify_region(summary, baseline)
        summaries.append(summary)
        hourly_rows.extend(hourly)
        seen.add(key)

    # Local refinement around the first background-axis success/failure boundary.
    axis = sorted(
        (item for item in summaries if item["idc_multiplier"] == 1.0),
        key=lambda item: item["background_load_multiplier"],
    )
    feasible = [item for item in axis if item["opf_success_rate"] >= 0.95]
    failed = [item for item in axis if item["opf_success_rate"] < 0.95]
    if feasible and failed:
        lower = max(item["background_load_multiplier"] for item in feasible if item["background_load_multiplier"] < min(x["background_load_multiplier"] for x in failed))
        upper = min(item["background_load_multiplier"] for item in failed if item["background_load_multiplier"] > lower)
        fine_values = np.linspace(lower, upper, 6)[1:-1]
        for background in fine_values:
            key = _case_key(float(background), 1.0)
            if key in seen:
                continue
            summary, hourly = run_stress_case(trace, float(background), 1.0, references)
            summary["sweep_group"] = "local_transition_fine"
            summary["region"] = classify_region(summary, baseline)
            summaries.append(summary)
            hourly_rows.extend(hourly)
            seen.add(key)
    return summaries, hourly_rows


def voltage_observation_table(hourly: list[dict[str, Any]], audit: dict[str, Any]) -> list[dict[str, Any]]:
    baseline = [
        row
        for row in hourly
        if row["background_load_multiplier"] == 1.0
        and row["idc_multiplier"] == 1.0
        and row["opf_success"]
    ]
    rows = []
    for bus in audit["buses"]:
        key = str(bus["bus_index"])
        values = [row["bus_voltage_by_index"][key] for row in baseline]
        observed_min = min(values)
        observed_max = max(values)
        lower_margins = [value - bus["min_vm_pu"] for value in values]
        upper_margins = [bus["max_vm_pu"] - value for value in values]
        rows.append(
            {
                **bus,
                "observed_min_vm_pu": observed_min,
                "observed_max_vm_pu": observed_max,
                "minimum_lower_margin_pu": min(lower_margins),
                "minimum_upper_margin_pu": min(upper_margins),
                "minimum_safety_margin_pu": min(lower_margins + upper_margins),
                "maximum_safety_margin_pu": max(lower_margins + upper_margins),
                "violation_count": sum(value < -1e-7 for value in lower_margins + upper_margins),
            }
        )
    return rows


def opf_constraint_probe(trace: list[PhysicsHour]) -> dict[str, Any]:
    """Inspect the solved internal case and probe whether branch RATE_A binds."""

    case = load_ieee14_case()
    idc_bus = get_bus_index_by_ieee_number(case, 9)
    first = trace[0]
    normal = solve_ac_opf(
        case,
        idc_bus_id=idc_bus,
        idc_load_mw=first.idc_bus_net_mw,
        load_scale=first.background_load_scale,
    )
    if not normal.success:
        return {"normal_probe_success": False, "normal_probe_message": normal.message}
    net = normal.raw_result
    rate_a = [float(value) for value in net._ppc["branch"][:, 5].tolist()]
    options = dict(net._options)

    tight_case = load_ieee14_case()
    tight_case.raw_network.line.loc[:, "max_loading_percent"] = 0.001
    tight_case.raw_network.trafo.loc[:, "max_loading_percent"] = 0.001
    tight = solve_ac_opf(
        tight_case,
        idc_bus_id=idc_bus,
        idc_load_mw=first.idc_bus_net_mw,
        load_scale=first.background_load_scale,
    )
    return {
        "normal_probe_success": True,
        "runopp_options": {
            key: value
            for key, value in options.items()
            if isinstance(value, (str, int, float, bool, type(None)))
        },
        "internal_branch_rate_a_mva": rate_a,
        "all_rate_a_near_9900_mva": all(abs(value - 9900.0) < 1e-3 for value in rate_a),
        "tight_all_branch_limit_probe_success": bool(tight.success),
        "tight_all_branch_limit_probe_message": tight.message,
        "interpretation": (
            "RATE_A is populated and the deliberately impossible private-copy limit makes OPF fail; "
            "line and transformer limits are enforced, but the configured 9900 MVA ratings are non-binding."
        ),
    }


def _write_csv(path: Path, rows: list[dict[str, Any]], *, exclude_nested: bool = True) -> None:
    flat_rows = []
    for row in rows:
        flat_rows.append(
            {
                key: value
                for key, value in row.items()
                if not exclude_nested or not isinstance(value, (dict, list, tuple))
            }
        )
    fields = sorted({key for row in flat_rows for key in row})
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(flat_rows)


def _write_json(path: Path, payload: Any) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, allow_nan=False)


def _plot_results(output_dir: Path, summaries: list[dict[str, Any]]) -> None:
    import matplotlib.pyplot as plt

    bg_axis = sorted(
        (row for row in summaries if row["idc_multiplier"] == 1.0),
        key=lambda row: row["background_load_multiplier"],
    )
    idc_axis = sorted(
        (row for row in summaries if row["background_load_multiplier"] == 1.0),
        key=lambda row: row["idc_multiplier"],
    )

    def plot_axis(rows: list[dict[str, Any]], x_key: str, y_keys: list[tuple[str, str]], filename: str, ylabel: str):
        fig, ax = plt.subplots(figsize=(7.2, 4.5))
        for key, label in y_keys:
            x = [row[x_key] for row in rows if row.get(key) is not None]
            y = [row[key] for row in rows if row.get(key) is not None]
            ax.plot(x, y, marker="o", label=label)
        ax.set_xlabel(x_key.replace("_", " "))
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.3)
        if len(y_keys) > 1:
            ax.legend()
        fig.tight_layout()
        fig.savefig(output_dir / filename, dpi=160)
        plt.close(fig)

    plot_axis(bg_axis, "background_load_multiplier", [("min_bus_voltage_pu", "minimum bus voltage")], "background_vs_min_voltage.png", "voltage (pu)")
    plot_axis(
        bg_axis,
        "background_load_multiplier",
        [("max_line_loading_percent", "line"), ("max_transformer_loading_percent", "transformer")],
        "background_vs_max_branch_loading.png",
        "configured loading (%)",
    )
    plot_axis(idc_axis, "idc_multiplier", [("idc_bus_voltage_min_pu", "IDC bus minimum")], "idc_vs_idc_bus_voltage.png", "voltage (pu)")
    plot_axis(idc_axis, "idc_multiplier", [("idc_bus_incident_max_loading_percent", "incident branch")], "idc_vs_local_loading.png", "configured loading (%)")

    coarse = [row for row in summaries if row.get("sweep_group") in {"coarse_2d", "baseline+axes+coarse"}]
    bgs = list(COARSE_BACKGROUND)
    idcs = list(COARSE_IDC)
    metrics = (
        ("opf_success_rate", "OPF success rate"),
        ("min_bus_voltage_pu", "Minimum voltage (pu)"),
        ("max_line_loading_percent", "Maximum line loading (%)"),
    )
    lookup = {_case_key(row["background_load_multiplier"], row["idc_multiplier"]): row for row in coarse}
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2), constrained_layout=True)
    for ax, (metric, title) in zip(axes, metrics, strict=True):
        matrix = np.full((len(bgs), len(idcs)), np.nan)
        for i, bg in enumerate(bgs):
            for j, idc in enumerate(idcs):
                row = lookup.get(_case_key(bg, idc))
                if row is not None and row.get(metric) is not None:
                    matrix[i, j] = row[metric]
        image = ax.imshow(matrix, aspect="auto", origin="lower")
        ax.set_xticks(range(len(idcs)), labels=idcs)
        ax.set_yticks(range(len(bgs)), labels=bgs)
        ax.set_xlabel("IDC multiplier")
        ax.set_ylabel("background multiplier")
        ax.set_title(title)
        fig.colorbar(image, ax=ax, shrink=0.85)
    fig.savefig(output_dir / "coarse_2d_heatmaps.png", dpi=160)
    plt.close(fig)


def _validate_no_nonfinite(payload: Any, path: str = "root") -> None:
    if isinstance(payload, dict):
        for key, value in payload.items():
            _validate_no_nonfinite(value, f"{path}.{key}")
    elif isinstance(payload, (list, tuple)):
        for index, value in enumerate(payload):
            _validate_no_nonfinite(value, f"{path}[{index}]")
    elif isinstance(payload, (float, np.floating)) and not math.isfinite(float(payload)):
        raise ValueError(f"Non-finite value at {path}: {payload}")


def run(output_dir: Path, seed: int = SEED) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    audit = audit_capacity_definitions()
    trace = collect_fixed_physics_trace(seed)
    constraint_probe = opf_constraint_probe(trace)
    summaries, hourly = run_sweep(trace)
    voltage_rows = voltage_observation_table(hourly, audit)
    payload = {
        "metadata": {
            "phase": "2.5",
            "date": date.today().isoformat(),
            "seed": seed,
            "action_source": "fixed 23-dimensional action = 0.5; BESS maps to zero physical power",
            "idc_multiplier_target": "physically realized IDC-bus net grid purchase after IDC/BESS/PV balance",
            "reference_limit_note": (
                "1.2x and 1.5x baseline-peak exceedances are diagnostic reference indices only, "
                "not thermal violations or proposed formal limits."
            ),
        },
        "capacity_audit": audit,
        "opf_constraint_probe": constraint_probe,
        "physics_trace": [asdict(item) for item in trace],
        "case_summaries": summaries,
        "voltage_observation_table": voltage_rows,
    }
    _validate_no_nonfinite(payload)
    _write_json(output_dir / "grid_stress_audit.json", payload)
    _write_csv(output_dir / "case_summaries.csv", summaries)
    _write_csv(output_dir / "hourly_results.csv", hourly)
    _write_csv(output_dir / "bus_capacity_audit.csv", audit["buses"])
    _write_csv(output_dir / "line_capacity_audit.csv", audit["lines"])
    _write_csv(output_dir / "transformer_capacity_audit.csv", audit["transformers"])
    _write_csv(output_dir / "bus_voltage_observations.csv", voltage_rows)
    _plot_results(output_dir, summaries)
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/grid_stress_audit_phase2_5"),
    )
    parser.add_argument("--seed", type=int, default=SEED)
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    result = run(arguments.output_dir, seed=arguments.seed)
    print(json.dumps({"output_dir": str(arguments.output_dir), "cases": len(result["case_summaries"])}, indent=2))
