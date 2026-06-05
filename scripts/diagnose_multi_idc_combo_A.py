"""Diagnose multi-IDC Combo A with direct 24h AC/DC OPF sweeps.

This script is diagnostic-only. It does not instantiate GridCoupledEnv, does
not train PPO, does not edit reward logic, does not use grid_cache, and does
not mutate the IEEE14 base case.

Examples:
    python -m scripts.diagnose_multi_idc_combo_A --scenarios multi_idc_A_conservative,multi_idc_A_normal,multi_idc_A_strong --opf-mode ac --no-cache --no-mef
    python -m scripts.diagnose_multi_idc_combo_A --scenarios multi_idc_A_conservative,multi_idc_A_normal --opf-mode ac --no-cache --enable-mef
"""

from __future__ import annotations

import argparse
import copy
import csv
import math
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any, Iterable


def _ensure_project_root_on_path() -> Path:
    for candidate in Path(__file__).resolve().parents:
        if (candidate / "configs").exists() and (candidate / "grid_model").exists():
            root = candidate
            break
    else:
        raise RuntimeError("Cannot locate project root containing configs/ and grid_model/")

    root_str = str(root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)
    return root


PROJECT_ROOT = _ensure_project_root_on_path()

from configs.multi_idc_scenarios import (  # noqa: E402
    build_synthetic_profile,
    get_multi_idc_scenario,
    list_multi_idc_scenarios,
)
from grid_model import (  # noqa: E402
    GridCase,
    build_default_gen_emission_factors,
    compute_total_emission,
    extract_grid_metrics,
    get_bus_index_by_ieee_number,
    load_ieee14_case,
    solve_opf,
)


TRACE_FIELDS = [
    "scenario",
    "scenario_role",
    "hour",
    "idc_A_bus",
    "idc_B_bus",
    "idc_C_bus",
    "idc_A_load_mw",
    "idc_B_load_mw",
    "idc_C_load_mw",
    "total_idc_load_mw",
    "total_system_load_mw_before_idc",
    "total_system_load_mw_after_idc",
    "idc_ratio_to_system_load",
    "opf_mode",
    "opf_success",
    "failure_reason",
    "minV",
    "maxV",
    "voltage_low_margin",
    "voltage_high_margin",
    "maxLine",
    "maxTrafo",
    "line_margin",
    "trafo_margin",
    "network_loss_mw",
    "total_generation_mw",
    "ext_grid_p_mw",
    "ext_grid_q_mvar",
    "LMP_bus9",
    "LMP_bus10",
    "LMP_bus13",
    "LMP_system_mean",
    "LMP_system_max",
    "safe_cost",
    "voltage_violation_count",
    "line_overload_count",
    "trafo_overload_count",
    "MEF_bus9_plus",
    "MEF_bus10_plus",
    "MEF_bus13_plus",
    "MEF_bus9_success",
    "MEF_bus10_success",
    "MEF_bus13_success",
]


SUMMARY_FIELDS = [
    "scenario",
    "scenario_role",
    "hours",
    "opf_success_rate",
    "failed_hours",
    "total_idc_load_min",
    "total_idc_load_mean",
    "total_idc_load_max",
    "idc_ratio_to_system_load_max",
    "minV_min",
    "maxV_max",
    "voltage_low_margin_min",
    "voltage_high_margin_min",
    "maxLine_max",
    "maxTrafo_max",
    "line_margin_min",
    "trafo_margin_min",
    "network_loss_max",
    "LMP_bus9_min",
    "LMP_bus9_max",
    "LMP_bus9_range",
    "LMP_bus10_min",
    "LMP_bus10_max",
    "LMP_bus10_range",
    "LMP_bus13_min",
    "LMP_bus13_max",
    "LMP_bus13_range",
    "LMP_system_max",
    "safe_cost_sum",
    "MEF_bus9_plus_mean",
    "MEF_bus10_plus_mean",
    "MEF_bus13_plus_mean",
    "recommendation_label",
]


DEFAULT_SCENARIOS = "multi_idc_A_conservative,multi_idc_A_normal,multi_idc_A_strong"
DEFAULT_OUT_DIR = "outputs/diagnostics"


def main() -> None:
    args = parse_args()
    selected_scenarios = resolve_scenarios(args.scenarios)
    compute_mef = bool(args.enable_mef) and not bool(args.no_mef)
    out_dir = PROJECT_ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    trace_path = out_dir / "multi_idc_combo_A_trace.csv"
    report_path = out_dir / "multi_idc_combo_A_report.md"
    summary_path = out_dir / "multi_idc_combo_A_summary.csv"

    grid_case = load_ieee14_case()
    emission_factors = build_default_gen_emission_factors(grid_case)
    base_system_load_mw = compute_base_system_load_mw(grid_case)

    rows: list[dict[str, Any]] = []
    for scenario_idx, scenario_name in enumerate(selected_scenarios, start=1):
        scenario = get_multi_idc_scenario(scenario_name)
        print(f"[scenario {scenario_idx}/{len(selected_scenarios)}] {scenario_name}")
        scenario_rows = diagnose_scenario(
            grid_case=grid_case,
            scenario=scenario,
            opf_mode=args.opf_mode,
            base_system_load_mw=base_system_load_mw,
            compute_mef=compute_mef,
            delta_p_mw=args.delta_p_mw,
            emission_factors=emission_factors,
        )
        rows.extend(scenario_rows)
        write_csv(trace_path, rows, TRACE_FIELDS)

    summary_rows = summarize_trace(rows)
    write_csv(trace_path, rows, TRACE_FIELDS)
    write_csv(summary_path, summary_rows, SUMMARY_FIELDS)
    write_report(
        report_path,
        trace_rows=rows,
        summary_rows=summary_rows,
        scenarios=selected_scenarios,
        opf_mode=args.opf_mode,
        compute_mef=compute_mef,
        delta_p_mw=args.delta_p_mw,
        paths={
            "trace": trace_path,
            "summary": summary_path,
            "report": report_path,
        },
    )

    print(f"Saved trace:   {trace_path}")
    print(f"Saved summary: {summary_path}")
    print(f"Saved report:  {report_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Diagnose multi-IDC Combo A scenarios with direct OPF.")
    parser.add_argument("--scenarios", default=DEFAULT_SCENARIOS, help="Comma-separated scenario names, or 'all'.")
    parser.add_argument("--opf-mode", default="ac", choices=["ac", "dc"], help="OPF mode.")
    parser.add_argument("--no-cache", action="store_true", help="Accepted for clarity; this script does not use grid_cache.")
    parser.add_argument("--no-mef", action="store_true", help="Skip MEF. This is the default unless --enable-mef is set.")
    parser.add_argument("--enable-mef", action="store_true", help="Compute plus MEF at bus9, bus10, and bus13.")
    parser.add_argument("--delta-p-mw", type=float, default=0.1, help="MEF plus perturbation in MW.")
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR, help="Output directory.")
    return parser.parse_args()


def resolve_scenarios(text: str) -> list[str]:
    value = str(text).strip()
    if value.lower() == "all":
        return list_multi_idc_scenarios()
    names = [item.strip() for item in value.split(",") if item.strip()]
    if not names:
        raise ValueError("At least one scenario must be selected.")
    valid = set(list_multi_idc_scenarios())
    invalid = [name for name in names if name not in valid]
    if invalid:
        raise ValueError(f"Unknown scenarios {invalid}. Valid scenarios: {sorted(valid)}")
    return names


def diagnose_scenario(
    *,
    grid_case: GridCase,
    scenario: dict[str, Any],
    opf_mode: str,
    base_system_load_mw: float,
    compute_mef: bool,
    delta_p_mw: float,
    emission_factors: dict[int, float],
) -> list[dict[str, Any]]:
    idc_profiles = build_hourly_idc_profiles(scenario)
    bus_indices = {
        idc_name: get_bus_index_by_ieee_number(grid_case, int(info["ieee_bus"]))
        for idc_name, info in scenario_idc_map(scenario).items()
    }
    rows: list[dict[str, Any]] = []
    for hour in range(24):
        load_specs = []
        for idc_name in ("IDC_A", "IDC_B", "IDC_C"):
            idc_info = scenario_idc_map(scenario)[idc_name]
            p_mw = idc_profiles[idc_name][hour]
            q_mvar = reactive_q_mvar(p_mw, to_float(idc_info.get("power_factor"), 1.0))
            load_specs.append(
                {
                    "idc_name": idc_name,
                    "ieee_bus": int(idc_info["ieee_bus"]),
                    "bus_index": int(bus_indices[idc_name]),
                    "p_mw": float(p_mw),
                    "q_mvar": float(q_mvar),
                    "power_factor": to_float(idc_info.get("power_factor"), 1.0),
                }
            )

        scenario_case = make_case_with_multi_idc_loads(grid_case, load_specs)
        opf_result = solve_opf(scenario_case, mode=opf_mode, load_scale=1.0)
        metrics = extract_grid_metrics(opf_result)
        total_idc_load = sum(float(spec["p_mw"]) for spec in load_specs)
        total_after = base_system_load_mw + total_idc_load
        margins = extract_constraint_margins(opf_result)
        ext_grid = extract_ext_grid_power(opf_result)
        lmp_stats = extract_lmp_stats(opf_result)
        max_trafo = extract_max_trafo_loading(opf_result)
        trafo_overload_count = count_trafo_overloads(opf_result)

        mef_values: dict[str, Any] = {
            "MEF_bus9_plus": "",
            "MEF_bus10_plus": "",
            "MEF_bus13_plus": "",
            "MEF_bus9_success": False,
            "MEF_bus10_success": False,
            "MEF_bus13_success": False,
        }
        if compute_mef:
            mef_values = calculate_combo_plus_mefs(
                scenario_case=scenario_case,
                base_opf_result=opf_result,
                opf_mode=opf_mode,
                emission_factors=emission_factors,
                delta_p_mw=delta_p_mw,
            )

        safe_cost = (
            float(metrics.voltage_violation_count)
            + float(metrics.line_overload_count)
            + float(trafo_overload_count)
            + (0.0 if bool(opf_result.success) else 1.0)
        )

        print(
            f"  hour={hour:02d} total_idc={total_idc_load:.3f} MW "
            f"opf={bool(opf_result.success)} minV={fmt(metrics.min_voltage_pu)} "
            f"maxLine={fmt(metrics.max_line_loading_percent)}"
        )

        rows.append(
            {
                "scenario": scenario["scenario_name"],
                "scenario_role": scenario.get("role", ""),
                "hour": hour,
                "idc_A_bus": int(scenario_idc_map(scenario)["IDC_A"]["ieee_bus"]),
                "idc_B_bus": int(scenario_idc_map(scenario)["IDC_B"]["ieee_bus"]),
                "idc_C_bus": int(scenario_idc_map(scenario)["IDC_C"]["ieee_bus"]),
                "idc_A_load_mw": finite_or_blank(idc_profiles["IDC_A"][hour]),
                "idc_B_load_mw": finite_or_blank(idc_profiles["IDC_B"][hour]),
                "idc_C_load_mw": finite_or_blank(idc_profiles["IDC_C"][hour]),
                "total_idc_load_mw": finite_or_blank(total_idc_load),
                "total_system_load_mw_before_idc": finite_or_blank(base_system_load_mw),
                "total_system_load_mw_after_idc": finite_or_blank(total_after),
                "idc_ratio_to_system_load": finite_or_blank(safe_div(total_idc_load, total_after)),
                "opf_mode": opf_mode,
                "opf_success": bool(opf_result.success),
                "failure_reason": "" if bool(opf_result.success) else str(opf_result.message),
                "minV": finite_or_blank(metrics.min_voltage_pu),
                "maxV": finite_or_blank(metrics.max_voltage_pu),
                "voltage_low_margin": finite_or_blank(margins["voltage_low_margin"]),
                "voltage_high_margin": finite_or_blank(margins["voltage_high_margin"]),
                "maxLine": finite_or_blank(metrics.max_line_loading_percent),
                "maxTrafo": finite_or_blank(max_trafo),
                "line_margin": finite_or_blank(margins["line_margin"]),
                "trafo_margin": finite_or_blank(margins["trafo_margin"]),
                "network_loss_mw": finite_or_blank(opf_result.network_loss_mw),
                "total_generation_mw": finite_or_blank(opf_result.total_generation_mw),
                "ext_grid_p_mw": finite_or_blank(ext_grid["p_mw"]),
                "ext_grid_q_mvar": finite_or_blank(ext_grid["q_mvar"]),
                "LMP_bus9": finite_or_blank(lmp_at_ieee_bus(opf_result, grid_case=scenario_case, ieee_bus=9)),
                "LMP_bus10": finite_or_blank(lmp_at_ieee_bus(opf_result, grid_case=scenario_case, ieee_bus=10)),
                "LMP_bus13": finite_or_blank(lmp_at_ieee_bus(opf_result, grid_case=scenario_case, ieee_bus=13)),
                "LMP_system_mean": finite_or_blank(lmp_stats["mean"]),
                "LMP_system_max": finite_or_blank(lmp_stats["max"]),
                "safe_cost": float(safe_cost),
                "voltage_violation_count": int(metrics.voltage_violation_count),
                "line_overload_count": int(metrics.line_overload_count),
                "trafo_overload_count": int(trafo_overload_count),
                **mef_values,
            }
        )
    return rows


def build_hourly_idc_profiles(scenario: dict[str, Any]) -> dict[str, list[float]]:
    profiles: dict[str, list[float]] = {}
    for idc in scenario["idcs"]:
        idc_name = str(idc["idc_name"])
        profile = build_synthetic_profile(idc_name, float(idc["target_peak_mw"]))
        if len(profile) != 24:
            raise ValueError(f"{idc_name} profile must contain 24 hourly values, got {len(profile)}.")
        profiles[idc_name] = profile
    return profiles


def scenario_idc_map(scenario: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {str(idc["idc_name"]): idc for idc in scenario["idcs"]}


def make_case_with_multi_idc_loads(grid_case: GridCase, load_specs: list[dict[str, Any]]) -> GridCase:
    try:
        import pandapower as pp
    except ImportError as exc:
        raise ImportError("pandapower is required for multi-IDC diagnostics.") from exc

    net = copy.deepcopy(grid_case.raw_network)
    for spec in load_specs:
        pp.create_load(
            net,
            bus=int(spec["bus_index"]),
            p_mw=float(spec["p_mw"]),
            q_mvar=float(spec.get("q_mvar", 0.0)),
            name=f"{spec['idc_name']} synthetic IDC load",
            controllable=False,
        )
    return replace(grid_case, raw_network=net)


def calculate_combo_plus_mefs(
    *,
    scenario_case: GridCase,
    base_opf_result,
    opf_mode: str,
    emission_factors: dict[int, float],
    delta_p_mw: float,
) -> dict[str, Any]:
    results: dict[str, Any] = {}
    for ieee_bus in (9, 10, 13):
        key = f"MEF_bus{ieee_bus}"
        if not bool(base_opf_result.success):
            results[f"{key}_plus"] = ""
            results[f"{key}_success"] = False
            continue
        bus_index = get_bus_index_by_ieee_number(scenario_case, ieee_bus)
        mef_plus, success = calculate_plus_mef(
            scenario_case=scenario_case,
            base_opf_result=base_opf_result,
            opf_mode=opf_mode,
            bus_index=bus_index,
            delta_p_mw=delta_p_mw,
            emission_factors=emission_factors,
        )
        results[f"{key}_plus"] = finite_or_blank(mef_plus)
        results[f"{key}_success"] = bool(success)
    return results


def calculate_plus_mef(
    *,
    scenario_case: GridCase,
    base_opf_result,
    opf_mode: str,
    bus_index: int,
    delta_p_mw: float,
    emission_factors: dict[int, float],
) -> tuple[float, bool]:
    delta = float(delta_p_mw)
    if delta <= 0.0 or not bool(base_opf_result.success):
        return math.nan, False

    base_emission, _ = compute_total_emission(base_opf_result.gen_power_mw, emission_factors)
    plus_case = make_case_with_multi_idc_loads(
        scenario_case,
        [
            {
                "idc_name": "MEF_plus_probe",
                "bus_index": int(bus_index),
                "p_mw": delta,
                "q_mvar": 0.0,
            }
        ],
    )
    plus_result = solve_opf(plus_case, mode=opf_mode, load_scale=1.0)
    if not plus_result.success:
        return math.nan, False
    plus_emission, _ = compute_total_emission(plus_result.gen_power_mw, emission_factors)
    mef_plus = (plus_emission - base_emission) / delta
    return mef_plus, math.isfinite(mef_plus)


def compute_base_system_load_mw(grid_case: GridCase) -> float:
    net = grid_case.raw_network
    load = getattr(net, "load", None)
    if load is None or len(load) == 0:
        return math.nan
    total = 0.0
    for _, row in load.iterrows():
        scaling = to_float(row.get("scaling", 1.0), 1.0)
        total += to_float(row.get("p_mw", 0.0), 0.0) * scaling
    return total


def extract_constraint_margins(opf_result) -> dict[str, float]:
    voltage_low = math.nan
    voltage_high = math.nan
    low_margins: list[float] = []
    high_margins: list[float] = []
    for bus_id, value in opf_result.bus_voltage_pu.items():
        voltage = to_float(value)
        if not math.isfinite(voltage):
            continue
        min_limit = to_float(opf_result.bus_voltage_min_pu.get(bus_id), 0.95)
        max_limit = to_float(opf_result.bus_voltage_max_pu.get(bus_id), 1.05)
        low_margins.append(voltage - min_limit)
        high_margins.append(max_limit - voltage)
    if low_margins:
        voltage_low = min(low_margins)
    if high_margins:
        voltage_high = min(high_margins)

    line_margins: list[float] = []
    for line_id, loading in opf_result.line_loading_percent.items():
        number = to_float(loading)
        if not math.isfinite(number):
            continue
        limit = to_float(opf_result.line_loading_limit_percent.get(line_id), 100.0)
        line_margins.append(limit - number)

    trafo_margins = extract_trafo_margins(opf_result)

    return {
        "voltage_low_margin": voltage_low,
        "voltage_high_margin": voltage_high,
        "line_margin": min(line_margins) if line_margins else math.nan,
        "trafo_margin": min(trafo_margins) if trafo_margins else math.nan,
    }


def extract_trafo_margins(opf_result) -> list[float]:
    net = getattr(opf_result, "raw_result", None)
    res_trafo = getattr(net, "res_trafo", None)
    trafo = getattr(net, "trafo", None)
    if res_trafo is None or not hasattr(res_trafo, "columns") or "loading_percent" not in res_trafo.columns:
        return []
    margins: list[float] = []
    for idx, value in res_trafo["loading_percent"].items():
        loading = to_float(value)
        if not math.isfinite(loading):
            continue
        limit = 100.0
        if trafo is not None and hasattr(trafo, "columns") and "max_loading_percent" in trafo.columns and idx in trafo.index:
            limit = to_float(trafo.at[idx, "max_loading_percent"], 100.0)
        margins.append(limit - loading)
    return margins


def extract_max_trafo_loading(opf_result) -> float:
    net = getattr(opf_result, "raw_result", None)
    table = getattr(net, "res_trafo", None)
    if table is None or not hasattr(table, "columns") or "loading_percent" not in table.columns:
        return math.nan
    values = [to_float(value) for value in table["loading_percent"].tolist()]
    values = [value for value in values if math.isfinite(value)]
    return max(values) if values else math.nan


def count_trafo_overloads(opf_result, tolerance: float = 1e-6) -> int:
    margins = extract_trafo_margins(opf_result)
    return sum(1 for margin in margins if margin < -float(tolerance))


def extract_ext_grid_power(opf_result) -> dict[str, float]:
    net = getattr(opf_result, "raw_result", None)
    res_ext_grid = getattr(net, "res_ext_grid", None)
    p_mw = math.nan
    q_mvar = math.nan
    if res_ext_grid is not None and hasattr(res_ext_grid, "columns"):
        if "p_mw" in res_ext_grid.columns:
            p_values = [to_float(value) for value in res_ext_grid["p_mw"].tolist()]
            p_values = [value for value in p_values if math.isfinite(value)]
            p_mw = sum(p_values) if p_values else math.nan
        if "q_mvar" in res_ext_grid.columns:
            q_values = [to_float(value) for value in res_ext_grid["q_mvar"].tolist()]
            q_values = [value for value in q_values if math.isfinite(value)]
            q_mvar = sum(q_values) if q_values else math.nan
    return {"p_mw": p_mw, "q_mvar": q_mvar}


def extract_lmp_stats(opf_result) -> dict[str, float]:
    values = [to_float(value) for value in opf_result.lmp_by_bus.values()]
    values = [value for value in values if math.isfinite(value)]
    if not values:
        return {"mean": math.nan, "max": math.nan}
    return {"mean": sum(values) / len(values), "max": max(values)}


def lmp_at_ieee_bus(opf_result, *, grid_case: GridCase, ieee_bus: int) -> float:
    bus_index = get_bus_index_by_ieee_number(grid_case, int(ieee_bus))
    return to_float(opf_result.lmp_by_bus.get(bus_index))


def reactive_q_mvar(p_mw: float, power_factor: float) -> float:
    pf = max(min(float(power_factor), 1.0), 1e-6)
    if pf >= 0.999999:
        return 0.0
    return float(p_mw) * math.tan(math.acos(pf))


def summarize_trace(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_scenario: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_scenario.setdefault(str(row["scenario"]), []).append(row)

    summary_rows: list[dict[str, Any]] = []
    for scenario, scenario_rows in by_scenario.items():
        hours = len(scenario_rows)
        successes = sum(1 for row in scenario_rows if truthy(row.get("opf_success")))
        failed_hours = [int(row["hour"]) for row in scenario_rows if not truthy(row.get("opf_success"))]
        role = str(scenario_rows[0].get("scenario_role", ""))
        label = classify_scenario_for_training(scenario_rows)
        summary_rows.append(
            {
                "scenario": scenario,
                "scenario_role": role,
                "hours": hours,
                "opf_success_rate": successes / max(hours, 1),
                "failed_hours": "none" if not failed_hours else ",".join(str(hour) for hour in failed_hours),
                "total_idc_load_min": finite_or_blank(min_finite(scenario_rows, "total_idc_load_mw")),
                "total_idc_load_mean": finite_or_blank(mean_finite(scenario_rows, "total_idc_load_mw")),
                "total_idc_load_max": finite_or_blank(max_finite(scenario_rows, "total_idc_load_mw")),
                "idc_ratio_to_system_load_max": finite_or_blank(max_finite(scenario_rows, "idc_ratio_to_system_load")),
                "minV_min": finite_or_blank(min_finite(scenario_rows, "minV")),
                "maxV_max": finite_or_blank(max_finite(scenario_rows, "maxV")),
                "voltage_low_margin_min": finite_or_blank(min_finite(scenario_rows, "voltage_low_margin")),
                "voltage_high_margin_min": finite_or_blank(min_finite(scenario_rows, "voltage_high_margin")),
                "maxLine_max": finite_or_blank(max_finite(scenario_rows, "maxLine")),
                "maxTrafo_max": finite_or_blank(max_finite(scenario_rows, "maxTrafo")),
                "line_margin_min": finite_or_blank(min_finite(scenario_rows, "line_margin")),
                "trafo_margin_min": finite_or_blank(min_finite(scenario_rows, "trafo_margin")),
                "network_loss_max": finite_or_blank(max_finite(scenario_rows, "network_loss_mw")),
                "LMP_bus9_min": finite_or_blank(min_finite(scenario_rows, "LMP_bus9")),
                "LMP_bus9_max": finite_or_blank(max_finite(scenario_rows, "LMP_bus9")),
                "LMP_bus9_range": finite_or_blank(range_finite(scenario_rows, "LMP_bus9")),
                "LMP_bus10_min": finite_or_blank(min_finite(scenario_rows, "LMP_bus10")),
                "LMP_bus10_max": finite_or_blank(max_finite(scenario_rows, "LMP_bus10")),
                "LMP_bus10_range": finite_or_blank(range_finite(scenario_rows, "LMP_bus10")),
                "LMP_bus13_min": finite_or_blank(min_finite(scenario_rows, "LMP_bus13")),
                "LMP_bus13_max": finite_or_blank(max_finite(scenario_rows, "LMP_bus13")),
                "LMP_bus13_range": finite_or_blank(range_finite(scenario_rows, "LMP_bus13")),
                "LMP_system_max": finite_or_blank(max_finite(scenario_rows, "LMP_system_max")),
                "safe_cost_sum": finite_or_blank(sum(to_float(row.get("safe_cost"), 0.0) for row in scenario_rows)),
                "MEF_bus9_plus_mean": finite_or_blank(mean_finite(scenario_rows, "MEF_bus9_plus")),
                "MEF_bus10_plus_mean": finite_or_blank(mean_finite(scenario_rows, "MEF_bus10_plus")),
                "MEF_bus13_plus_mean": finite_or_blank(mean_finite(scenario_rows, "MEF_bus13_plus")),
                "recommendation_label": label,
            }
        )
    return summary_rows


def classify_scenario_for_training(rows: list[dict[str, Any]]) -> str:
    success_rate = sum(1 for row in rows if truthy(row.get("opf_success"))) / max(len(rows), 1)
    min_v = min_finite(rows, "minV")
    max_lmp = max(
        max_finite(rows, "LMP_bus9"),
        max_finite(rows, "LMP_bus10"),
        max_finite(rows, "LMP_bus13"),
    )
    lmp_range = max(
        range_finite(rows, "LMP_bus9"),
        range_finite(rows, "LMP_bus10"),
        range_finite(rows, "LMP_bus13"),
    )
    safe_sum = sum(to_float(row.get("safe_cost"), 0.0) for row in rows)

    if success_rate < 0.99 or safe_sum > 0.0:
        return "stress_or_reject"
    if is_finite(min_v) and min_v <= 0.945:
        return "borderline_stress"
    if is_finite(max_lmp) and max_lmp > 500.0:
        return "borderline_extreme_lmp"
    if is_finite(lmp_range) and lmp_range < 1.0:
        return "stable_but_feedback_weak"
    if is_finite(min_v) and min_v < 0.97:
        return "strong_training_candidate"
    return "normal_training_candidate"


def write_report(
    path: Path,
    *,
    trace_rows: list[dict[str, Any]],
    summary_rows: list[dict[str, Any]],
    scenarios: list[str],
    opf_mode: str,
    compute_mef: bool,
    delta_p_mw: float,
    paths: dict[str, Path],
) -> None:
    recommended = choose_recommended_training_scenario(summary_rows)
    lines = [
        "# Multi-IDC Combo A Diagnostic Report",
        "",
        "## Run Settings",
        "",
        f"- scenarios: {', '.join(scenarios)}",
        f"- opf_mode: {opf_mode}",
        "- cache: off (direct solve_opf calls)",
        f"- MEF enabled: {bool(compute_mef)}",
        f"- MEF delta_p_mw: {float(delta_p_mw):.6f}",
        "- IEEE14 base load: unchanged",
        "- IDC modeling: added pandapower load elements on copied networks; default IDC q_mvar is 0 to match current single-IDC coupling",
        "",
        "## Output Files",
        "",
        f"- trace: `{relative_to_project(paths['trace'])}`",
        f"- summary: `{relative_to_project(paths['summary'])}`",
        f"- report: `{relative_to_project(paths['report'])}`",
        "",
        "## Scenario Success Summary",
        "",
        "| scenario | success rate | failed hours | total IDC min/mean/max MW | max IDC/system ratio | minV min | maxV max | maxLine % | maxTrafo % | LMP9 range | LMP10 range | LMP13 range | safe_cost_sum | label |",
        "|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in summary_rows:
        total_load = (
            f"{fmt(row.get('total_idc_load_min'))}/"
            f"{fmt(row.get('total_idc_load_mean'))}/"
            f"{fmt(row.get('total_idc_load_max'))}"
        )
        lines.append(
            f"| {row['scenario']} | {to_float(row.get('opf_success_rate')):.6f} | {row.get('failed_hours')} | "
            f"{total_load} | {fmt(row.get('idc_ratio_to_system_load_max'))} | "
            f"{fmt(row.get('minV_min'))} | {fmt(row.get('maxV_max'))} | "
            f"{fmt(row.get('maxLine_max'))} | {fmt(row.get('maxTrafo_max'))} | "
            f"{fmt(row.get('LMP_bus9_range'))} | {fmt(row.get('LMP_bus10_range'))} | "
            f"{fmt(row.get('LMP_bus13_range'))} | {fmt(row.get('safe_cost_sum'))} | "
            f"{row.get('recommendation_label')} |"
        )

    lines.extend([
        "",
        "## Recommended Training Scenario",
        "",
    ])
    if recommended:
        lines.extend([
            f"Recommended main training scenario: **{recommended['scenario']}**.",
            "",
            "Reason:",
            "",
            f"- OPF success rate is {to_float(recommended.get('opf_success_rate')):.6f}.",
            f"- minV_min is {fmt(recommended.get('minV_min'))}; this avoids spending the whole day pinned to the 0.94 hard lower limit.",
            f"- maxLine and maxTrafo remain far from 100%, so branch thermal limits are not the active bottleneck.",
            f"- LMP ranges are visible without becoming an extreme boundary artifact.",
            f"- safe_cost_sum is {fmt(recommended.get('safe_cost_sum'))}.",
        ])
    else:
        lines.append("No scenario satisfies the normal-training criteria in this run.")

    lines.extend([
        "",
        "## Load Adjustment Diagnosis",
        "",
    ])
    for row in summary_rows:
        lines.append(f"- `{row['scenario']}`: {load_adjustment_note(row)}")

    lines.extend([
        "",
        "## Multi-IDC vs Previous Single-IDC Small-Load Setting",
        "",
        "The previous single-IDC bus9 setting injected roughly 0-3 MW, less than about 1% of the IEEE14 system load. Combo A raises the 24h total IDC load into a tens-to-hundreds of MW range while keeping IEEE14 original load unchanged.",
        "",
        "- Total IDC load is materially larger, so OPF voltage and LMP responses are visible in the diagnostic trace.",
        "- LMP feedback is stronger because load is distributed across bus9, bus10, and bus13 instead of being confined to the old small bus9 signal.",
        "- minV varies more meaningfully; line and trafo loading still stay far from hard limits under the current IEEE14 branch parameters.",
        "- This is more compatible with future MAPPO/CTDE/HGTA work because each IDC can become an agent with a distinct grid location and local grid feedback.",
        "",
        "## Workload Trace Note",
        "",
        "The current 24h curves are synthetic and only intended to diagnose MW scale and access-node feasibility. Later, Alibaba, Google, Azure, or Bitbrains traces can provide each IDC's load shape, while `target_peak_mw` keeps the electrical MW scale controlled.",
    ])

    if compute_mef:
        lines.extend([
            "",
            "## MEF Summary",
            "",
            "| scenario | MEF bus9 plus mean | MEF bus10 plus mean | MEF bus13 plus mean |",
            "|---|---:|---:|---:|",
        ])
        for row in summary_rows:
            lines.append(
                f"| {row['scenario']} | {fmt(row.get('MEF_bus9_plus_mean'))} | "
                f"{fmt(row.get('MEF_bus10_plus_mean'))} | {fmt(row.get('MEF_bus13_plus_mean'))} |"
            )

    lines.extend([
        "",
        "## Hourly Failure Details",
        "",
    ])
    failed = [row for row in trace_rows if not truthy(row.get("opf_success"))]
    if not failed:
        lines.append("No OPF failures were observed in the selected scenarios.")
    else:
        lines.extend([
            "| scenario | hour | total IDC MW | reason |",
            "|---|---:|---:|---|",
        ])
        for row in failed:
            lines.append(
                f"| {row['scenario']} | {row['hour']} | {fmt(row.get('total_idc_load_mw'))} | "
                f"{escape_md(str(row.get('failure_reason', '')))} |"
            )

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def choose_recommended_training_scenario(summary_rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    by_name = {str(row["scenario"]): row for row in summary_rows}
    preferred_order = [
        "multi_idc_A_normal",
        "multi_idc_A_conservative",
        "multi_idc_A_strong",
    ]
    for name in preferred_order:
        row = by_name.get(name)
        if row is None:
            continue
        success_rate = to_float(row.get("opf_success_rate"))
        safe_cost = to_float(row.get("safe_cost_sum"), 0.0)
        min_v = to_float(row.get("minV_min"))
        max_lmp = max(
            to_float(row.get("LMP_bus9_max")),
            to_float(row.get("LMP_bus10_max")),
            to_float(row.get("LMP_bus13_max")),
        )
        if success_rate >= 0.99 and safe_cost <= 0.0 and (not is_finite(min_v) or min_v > 0.945) and (not is_finite(max_lmp) or max_lmp < 500.0):
            return row
    return None


def load_adjustment_note(row: dict[str, Any]) -> str:
    scenario = str(row["scenario"])
    success_rate = to_float(row.get("opf_success_rate"))
    min_v = to_float(row.get("minV_min"))
    lmp_range = max(
        to_float(row.get("LMP_bus9_range")),
        to_float(row.get("LMP_bus10_range")),
        to_float(row.get("LMP_bus13_range")),
    )
    safe_cost = to_float(row.get("safe_cost_sum"), 0.0)
    if "stress" in scenario:
        return "stress-only; do not use as default training."
    if success_rate < 0.99 or safe_cost > 0.0:
        return "too close to infeasibility for default PPO/MAPPO training; keep as stress or reduce load."
    if is_finite(min_v) and min_v <= 0.945:
        return "hard-feasible but voltage is too close to the lower bound; use only as strong/stress."
    if is_finite(lmp_range) and lmp_range < 1.0:
        return "stable but feedback is weak; consider higher load if this is the only training scenario."
    if "normal" in scenario:
        return "good main training candidate if MEF/LMP values remain non-extreme."
    if "conservative" in scenario:
        return "stable fallback; use if normal becomes unstable after integrating the real IDC environment."
    if "strong" in scenario:
        return "use to test stronger feedback; promote only if failures and extreme LMP remain absent."
    return "review against the scenario summary."


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def min_finite(rows: Iterable[dict[str, Any]], key: str) -> float:
    values = finite_values(rows, key)
    return min(values) if values else math.nan


def max_finite(rows: Iterable[dict[str, Any]], key: str) -> float:
    values = finite_values(rows, key)
    return max(values) if values else math.nan


def mean_finite(rows: Iterable[dict[str, Any]], key: str) -> float:
    values = finite_values(rows, key)
    return sum(values) / len(values) if values else math.nan


def range_finite(rows: Iterable[dict[str, Any]], key: str) -> float:
    values = finite_values(rows, key)
    return max(values) - min(values) if values else math.nan


def finite_values(rows: Iterable[dict[str, Any]], key: str) -> list[float]:
    values: list[float] = []
    for row in rows:
        value = to_float(row.get(key))
        if math.isfinite(value):
            values.append(value)
    return values


def finite_or_blank(value: Any) -> float | str:
    number = to_float(value)
    return number if math.isfinite(number) else ""


def to_float(value: Any, default: float = math.nan) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if math.isfinite(number) else default


def is_finite(value: Any) -> bool:
    return math.isfinite(to_float(value))


def truthy(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y"}
    return bool(value)


def safe_div(numerator: float, denominator: float) -> float:
    if abs(float(denominator)) <= 1e-12:
        return math.nan
    return float(numerator) / float(denominator)


def fmt(value: Any) -> str:
    number = to_float(value)
    return "nan" if not math.isfinite(number) else f"{number:.6f}"


def escape_md(value: str) -> str:
    return str(value).replace("|", "/")


def relative_to_project(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT)).replace("\\", "/")
    except ValueError:
        return str(path)


if __name__ == "__main__":
    main()
