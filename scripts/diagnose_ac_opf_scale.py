"""Diagnose IDC load scale and IEEE14 AC/DC OPF constraint margins.

Examples:
    python -m scripts.diagnose_ac_opf_scale --quick --no-mef
    python -m scripts.diagnose_ac_opf_scale --idc-buses 3,4,5,9,14 --opf-modes ac,dc
"""

from __future__ import annotations

import argparse
import copy
import csv
import math
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Any, Iterable


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

from grid_model import (  # noqa: E402
    build_default_gen_emission_factors,
    calculate_nodal_mef,
    extract_grid_metrics,
    get_bus_index_by_ieee_number,
    load_ieee14_case,
    solve_opf,
)


DEFAULT_IDC_BUSES = "3,4,5,9,14"
DEFAULT_IDC_LOAD_MULTIPLIERS = "0,0.5,1,1.5,2,3,5,10"
DEFAULT_GRID_LOAD_SCALE_MULTIPLIERS = "0.8,1.0,1.2,1.5"
DEFAULT_LINE_CAPACITY_MULTIPLIERS = "1.0,0.8,0.6"
DEFAULT_VOLTAGE_LIMIT_MODES = "normal,tight"
DEFAULT_OPF_MODES = "ac,dc"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--idc-buses", type=str, default=DEFAULT_IDC_BUSES, help="Comma-separated IEEE bus numbers, or 'all_load_buses'.")
    parser.add_argument("--idc-load-multipliers", type=str, default=DEFAULT_IDC_LOAD_MULTIPLIERS)
    parser.add_argument("--base-idc-load-mw", type=float, default=1.0)
    parser.add_argument("--grid-load-scale-multipliers", type=str, default=DEFAULT_GRID_LOAD_SCALE_MULTIPLIERS)
    parser.add_argument("--line-capacity-multipliers", type=str, default=DEFAULT_LINE_CAPACITY_MULTIPLIERS)
    parser.add_argument("--voltage-limit-modes", type=str, default=DEFAULT_VOLTAGE_LIMIT_MODES)
    parser.add_argument("--opf-modes", type=str, default=DEFAULT_OPF_MODES)
    parser.add_argument("--delta-p-mw", type=float, default=0.1)
    parser.add_argument("--no-mef", action="store_true", help="Skip MEF perturbation OPFs for faster scans.")
    parser.add_argument("--tight-min-v", type=float, default=0.97)
    parser.add_argument("--tight-max-v", type=float, default=1.03)
    parser.add_argument("--out-dir", type=str, default="outputs/diagnostics")
    parser.add_argument("--quick", action="store_true", help="Small smoke scan: bus 9, load multipliers 0/1/5/10, AC only.")
    args = parser.parse_args()

    if args.quick:
        args.idc_buses = "9"
        args.idc_load_multipliers = "0,1,5,10"
        args.grid_load_scale_multipliers = "1.0"
        args.line_capacity_multipliers = "1.0"
        args.voltage_limit_modes = "normal"
        args.opf_modes = "ac"

    out_dir = PROJECT_ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "ac_opf_scale_sweep.csv"
    summary_path = out_dir / "ac_opf_scale_summary.md"

    base_case = load_ieee14_case()
    idc_buses = resolve_idc_buses(base_case, args.idc_buses)
    idc_load_multipliers = parse_float_list(args.idc_load_multipliers)
    grid_load_scale_multipliers = parse_float_list(args.grid_load_scale_multipliers)
    line_capacity_multipliers = parse_float_list(args.line_capacity_multipliers)
    voltage_limit_modes = parse_str_list(args.voltage_limit_modes)
    opf_modes = parse_str_list(args.opf_modes)

    rows: list[dict[str, Any]] = []
    total_cases = (
        len(idc_buses)
        * len(idc_load_multipliers)
        * len(grid_load_scale_multipliers)
        * len(line_capacity_multipliers)
        * len(voltage_limit_modes)
        * len(opf_modes)
    )
    case_idx = 0

    for ieee_bus_number in idc_buses:
        bus_index = get_bus_index_by_ieee_number(base_case, ieee_bus_number)
        for idc_load_multiplier in idc_load_multipliers:
            idc_load_mw = float(args.base_idc_load_mw) * float(idc_load_multiplier)
            for grid_load_scale_multiplier in grid_load_scale_multipliers:
                for line_capacity_multiplier in line_capacity_multipliers:
                    for voltage_limit_mode in voltage_limit_modes:
                        scan_case = make_diagnostic_case(
                            base_case,
                            line_capacity_multiplier=line_capacity_multiplier,
                            voltage_limit_mode=voltage_limit_mode,
                            tight_min_v=args.tight_min_v,
                            tight_max_v=args.tight_max_v,
                        )
                        emission_factors = build_default_gen_emission_factors(scan_case)
                        for opf_mode in opf_modes:
                            case_idx += 1
                            print(
                                f"[{case_idx}/{total_cases}] bus={ieee_bus_number} "
                                f"idc={idc_load_mw:.3f}MW grid_scale={grid_load_scale_multiplier:.3f} "
                                f"line_cap={line_capacity_multiplier:.3f} vmode={voltage_limit_mode} mode={opf_mode}"
                            )
                            row = run_one_case(
                                grid_case=scan_case,
                                ieee_bus_number=ieee_bus_number,
                                bus_index=bus_index,
                                idc_load_multiplier=idc_load_multiplier,
                                idc_load_mw=idc_load_mw,
                                grid_load_scale_multiplier=grid_load_scale_multiplier,
                                line_capacity_multiplier=line_capacity_multiplier,
                                voltage_limit_mode=voltage_limit_mode,
                                opf_mode=opf_mode,
                                delta_p_mw=args.delta_p_mw,
                                compute_mef=not args.no_mef,
                                emission_factors=emission_factors,
                            )
                            rows.append(row)
                            write_csv(csv_path, rows)

    write_csv(csv_path, rows)
    write_summary(summary_path, rows, args)
    print(f"Saved sweep CSV: {csv_path}")
    print(f"Saved summary:   {summary_path}")


def run_one_case(
    *,
    grid_case,
    ieee_bus_number: int,
    bus_index: int,
    idc_load_multiplier: float,
    idc_load_mw: float,
    grid_load_scale_multiplier: float,
    line_capacity_multiplier: float,
    voltage_limit_mode: str,
    opf_mode: str,
    delta_p_mw: float,
    compute_mef: bool,
    emission_factors: dict[int, float],
) -> dict[str, Any]:
    start = time.perf_counter()
    opf_result = solve_opf(
        grid_case,
        mode=opf_mode,
        idc_bus_id=bus_index,
        idc_load_mw=idc_load_mw,
        load_scale=grid_load_scale_multiplier,
    )
    solve_time_sec = time.perf_counter() - start
    metrics = extract_grid_metrics(opf_result)

    mef_plus = math.nan
    mef_minus = math.nan
    mef_success = False
    mef_message = "MEF skipped."
    if compute_mef:
        mef_start = time.perf_counter()
        mef_result = calculate_nodal_mef(
            grid_case,
            bus_id=bus_index,
            mode=opf_mode,
            delta_p_mw=delta_p_mw,
            load_scale=grid_load_scale_multiplier,
            base_idc_load_mw=idc_load_mw,
            clamp_minus_load=True,
            gen_emission_factors_kg_per_mwh=emission_factors,
        )
        mef_time_sec = time.perf_counter() - mef_start
        mef_plus = mef_result.mef_plus_kg_per_mwh
        mef_minus = mef_result.mef_minus_kg_per_mwh
        mef_success = bool(mef_result.success)
        mef_message = mef_result.message
    else:
        mef_time_sec = 0.0

    safe_cost = (
        float(metrics.voltage_violation_count)
        + float(metrics.line_overload_count)
        + (0.0 if bool(opf_result.success) else 1.0)
    )

    return {
        "ieee_bus_number": int(ieee_bus_number),
        "pandapower_bus_index": int(bus_index),
        "idc_load_multiplier": float(idc_load_multiplier),
        "idc_load_mw": float(idc_load_mw),
        "grid_load_scale_multiplier": float(grid_load_scale_multiplier),
        "line_capacity_multiplier": float(line_capacity_multiplier),
        "voltage_limit_mode": str(voltage_limit_mode),
        "opf_mode": str(opf_mode),
        "opf_success": bool(opf_result.success),
        "failure_reason": "" if bool(opf_result.success) else opf_result.message,
        "minV": finite_or_blank(metrics.min_voltage_pu),
        "maxV": finite_or_blank(metrics.max_voltage_pu),
        "maxLine": finite_or_blank(metrics.max_line_loading_percent),
        "network_loss": finite_or_blank(opf_result.network_loss_mw),
        "total_generation": finite_or_blank(opf_result.total_generation_mw),
        "total_load": finite_or_blank(opf_result.total_load_mw),
        "lmp_bus": finite_or_blank(opf_result.lmp_by_bus.get(bus_index, math.nan)),
        "mef_success": bool(mef_success),
        "mef_plus": finite_or_blank(mef_plus),
        "mef_minus": finite_or_blank(mef_minus),
        "mef_message": mef_message,
        "safe_cost": float(safe_cost),
        "voltage_violation_count": int(metrics.voltage_violation_count),
        "line_overload_count": int(metrics.line_overload_count),
        "opf_violation": 0 if bool(opf_result.success) else 1,
        "solve_time": float(solve_time_sec),
        "mef_time": float(mef_time_sec),
        "row_time": float(solve_time_sec + mef_time_sec),
    }


def make_diagnostic_case(
    base_case,
    *,
    line_capacity_multiplier: float,
    voltage_limit_mode: str,
    tight_min_v: float,
    tight_max_v: float,
):
    net = copy.deepcopy(base_case.raw_network)
    line_multiplier = float(line_capacity_multiplier)
    if line_multiplier <= 0.0:
        raise ValueError(f"line_capacity_multiplier must be positive, got {line_capacity_multiplier!r}.")
    if hasattr(net, "line") and len(net.line) > 0:
        if "max_i_ka" in net.line.columns:
            net.line.loc[:, "max_i_ka"] = net.line["max_i_ka"].astype(float) * line_multiplier
        if "max_loading_percent" in net.line.columns:
            net.line.loc[:, "max_loading_percent"] = net.line["max_loading_percent"].astype(float)
    if hasattr(net, "trafo") and len(net.trafo) > 0:
        if "sn_mva" in net.trafo.columns:
            net.trafo.loc[:, "sn_mva"] = net.trafo["sn_mva"].astype(float) * line_multiplier
        if "max_loading_percent" in net.trafo.columns:
            net.trafo.loc[:, "max_loading_percent"] = net.trafo["max_loading_percent"].astype(float)

    mode = str(voltage_limit_mode).strip().lower()
    if mode == "normal":
        pass
    elif mode == "tight":
        if hasattr(net, "bus") and len(net.bus) > 0:
            net.bus.loc[:, "min_vm_pu"] = float(tight_min_v)
            net.bus.loc[:, "max_vm_pu"] = float(tight_max_v)
        if hasattr(net, "gen") and len(net.gen) > 0 and "vm_pu" in net.gen.columns:
            net.gen.loc[:, "vm_pu"] = net.gen["vm_pu"].astype(float).clip(lower=float(tight_min_v), upper=float(tight_max_v))
        if hasattr(net, "ext_grid") and len(net.ext_grid) > 0 and "vm_pu" in net.ext_grid.columns:
            net.ext_grid.loc[:, "vm_pu"] = net.ext_grid["vm_pu"].astype(float).clip(lower=float(tight_min_v), upper=float(tight_max_v))
    else:
        raise ValueError(f"Unknown voltage_limit_mode {voltage_limit_mode!r}; expected normal or tight.")

    return replace(base_case, raw_network=net)


def resolve_idc_buses(grid_case, text: str) -> list[int]:
    value = str(text).strip().lower()
    if value == "all_load_buses":
        net = grid_case.raw_network
        load_bus_indices = sorted(set(int(bus) for bus in net.load.bus.tolist()))
        reverse = grid_case.bus_index_to_ieee_bus_number
        return [int(reverse[idx]) for idx in load_bus_indices if idx in reverse]
    return [int(x) for x in parse_str_list(text)]


def parse_float_list(text: str) -> list[float]:
    values = [float(item.strip()) for item in str(text).split(",") if item.strip()]
    if not values:
        raise ValueError(f"Expected at least one numeric value in {text!r}.")
    return values


def parse_str_list(text: str) -> list[str]:
    values = [item.strip() for item in str(text).split(",") if item.strip()]
    if not values:
        raise ValueError(f"Expected at least one value in {text!r}.")
    return values


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_summary(path: Path, rows: list[dict[str, Any]], args: argparse.Namespace) -> None:
    total = len(rows)
    success = sum(1 for row in rows if truthy(row.get("opf_success")))
    lines = [
        "# AC OPF Scale Sweep Summary",
        "",
        f"- total_cases: {total}",
        f"- opf_success_rate: {success / max(total, 1):.6f}",
        f"- base_idc_load_mw: {float(args.base_idc_load_mw):.6f}",
        f"- delta_p_mw: {float(args.delta_p_mw):.6f}",
        f"- mef_enabled: {not bool(args.no_mef)}",
        "",
        "## Success Rate By Mode",
        "",
        "| opf_mode | voltage_mode | line_cap | cases | success_rate | minV_min | maxLine_max | safe_cost_sum |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    groups = sorted(set((row["opf_mode"], row["voltage_limit_mode"], row["line_capacity_multiplier"]) for row in rows))
    for opf_mode, voltage_mode, line_cap in groups:
        group_rows = [
            row for row in rows
            if row["opf_mode"] == opf_mode
            and row["voltage_limit_mode"] == voltage_mode
            and row["line_capacity_multiplier"] == line_cap
        ]
        min_v = finite_values(group_rows, "minV")
        max_line = finite_values(group_rows, "maxLine")
        safe_cost = sum(float(row.get("safe_cost", 0.0)) for row in group_rows)
        ok = sum(1 for row in group_rows if truthy(row.get("opf_success")))
        lines.append(
            f"| {opf_mode} | {voltage_mode} | {float(line_cap):.3f} | {len(group_rows)} | "
            f"{ok / max(len(group_rows), 1):.6f} | {fmt(min(min_v) if min_v else math.nan)} | "
            f"{fmt(max(max_line) if max_line else math.nan)} | {safe_cost:.3f} |"
        )

    lines.extend([
        "",
        "## First Failure By Bus And Mode",
        "",
        "| ieee_bus | opf_mode | voltage_mode | line_cap | grid_scale | first_failed_idc_multiplier | first_failed_idc_mw | reason |",
        "|---:|---|---|---:|---:|---:|---:|---|",
    ])
    failure_groups = sorted(set(
        (
            row["ieee_bus_number"],
            row["opf_mode"],
            row["voltage_limit_mode"],
            row["line_capacity_multiplier"],
            row["grid_load_scale_multiplier"],
        )
        for row in rows
    ))
    for key in failure_groups:
        group_rows = [
            row for row in rows
            if (
                row["ieee_bus_number"],
                row["opf_mode"],
                row["voltage_limit_mode"],
                row["line_capacity_multiplier"],
                row["grid_load_scale_multiplier"],
            ) == key
        ]
        failed = [row for row in sorted(group_rows, key=lambda r: float(r["idc_load_mw"])) if not truthy(row.get("opf_success"))]
        if not failed:
            continue
        row = failed[0]
        reason = str(row.get("failure_reason", "")).replace("|", "/")
        lines.append(
            f"| {row['ieee_bus_number']} | {row['opf_mode']} | {row['voltage_limit_mode']} | "
            f"{float(row['line_capacity_multiplier']):.3f} | {float(row['grid_load_scale_multiplier']):.3f} | "
            f"{float(row['idc_load_multiplier']):.3f} | {float(row['idc_load_mw']):.3f} | {reason} |"
        )

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def finite_values(rows: Iterable[dict[str, Any]], key: str) -> list[float]:
    values: list[float] = []
    for row in rows:
        number = to_float(row.get(key))
        if math.isfinite(number):
            values.append(number)
    return values


def finite_or_blank(value: Any) -> float | str:
    number = to_float(value)
    return number if math.isfinite(number) else ""


def to_float(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return math.nan
    return number if math.isfinite(number) else math.nan


def truthy(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y"}
    return bool(value)


def fmt(value: Any) -> str:
    number = to_float(value)
    return "nan" if not math.isfinite(number) else f"{number:.6f}"


if __name__ == "__main__":
    main()
