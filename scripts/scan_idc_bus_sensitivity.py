"""Scan IEEE14 bus sensitivity for added IDC loads.

This script is intentionally standalone: it calls the grid model directly,
does not use GridCoupledEnv, does not train PPO, and does not touch reward
configuration. Cache is off by construction because solve_opf() is called
directly for each scan point.

Examples:
    python -m scripts.scan_idc_bus_sensitivity --opf-mode ac --no-cache --no-mef
    python -m scripts.scan_idc_bus_sensitivity --opf-mode ac --top-k-mef 5
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
import time
from pathlib import Path
from typing import Any, Iterable


def _ensure_project_root_on_path() -> Path:
    for candidate in Path(__file__).resolve().parents:
        if (candidate / "grid_model").exists() and (candidate / "scripts").exists():
            root = candidate
            break
    else:
        raise RuntimeError("Cannot locate project root containing grid_model/ and scripts/")

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


DEFAULT_CANDIDATE_BUSES = "all_load_buses"
DEFAULT_IDC_LOADS_MW = "0,5,10,20,30,50,80,100,150,200"
DEFAULT_EXTRA_LOADS_MW = "250,300"
DEFAULT_MEF_LOADS_MW = "0,20,50,100,150,200"
DEFAULT_OUT_DIR = "outputs/diagnostics"


BUS_INVENTORY_FIELDS = [
    "ieee_bus_number",
    "pandapower_bus_index",
    "bus_name",
    "vn_kv",
    "original_p_mw",
    "original_q_mvar",
    "is_original_load_bus",
    "is_generator_bus",
    "is_slack_bus",
    "is_zero_load_bus",
    "is_current_default_idc_bus",
]


SCAN_FIELDS = [
    "scan_round",
    "ieee_bus_number",
    "pandapower_bus_index",
    "bus_name",
    "original_p_mw",
    "original_q_mvar",
    "is_original_load_bus",
    "is_generator_bus",
    "is_slack_bus",
    "is_current_default_idc_bus",
    "idc_load_mw",
    "opf_mode",
    "opf_success",
    "failure_reason",
    "minV",
    "maxV",
    "voltage_margin_to_limit",
    "maxLine",
    "maxTrafo",
    "network_loss",
    "total_load",
    "total_generation",
    "lmp_bus",
    "lmp_delta_from_zero",
    "lmp_slope_from_zero_per_mw",
    "safe_cost",
    "voltage_violation_count",
    "line_overload_count",
    "trafo_overload_count",
    "opf_violation",
    "solve_time_sec",
    "mef_success",
    "mef_plus",
    "mef_minus",
    "mef_mean",
    "mef_sensitivity",
    "mef_message",
    "mef_time_sec",
]


SUMMARY_FIELDS = [
    "ieee_bus_number",
    "pandapower_bus_index",
    "bus_name",
    "original_p_mw",
    "original_q_mvar",
    "is_original_load_bus",
    "is_generator_bus",
    "is_slack_bus",
    "is_current_default_idc_bus",
    "scanned_points",
    "success_points",
    "success_rate",
    "max_successful_idc_mw",
    "first_failed_idc_mw",
    "max_idc_mw_with_minV_ge_0_97",
    "max_idc_mw_with_minV_ge_0_98",
    "minV_min_success",
    "maxV_max_success",
    "voltage_margin_min_success",
    "maxLine_max_success",
    "maxTrafo_max_success",
    "lmp_at_0_mw",
    "lmp_at_max_success_mw",
    "lmp_range_success",
    "lmp_slope_per_mw",
    "loss_at_0_mw",
    "loss_at_max_success_mw",
    "loss_slope_per_mw",
    "mef_mean_min",
    "mef_mean_max",
    "mef_sensitivity_range",
    "sensitivity_score",
    "suggested_role",
    "suggested_normal_range_mw",
    "stress_note",
]


def main() -> None:
    args = parse_args()
    out_dir = PROJECT_ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    grid_case = load_ieee14_case()
    current_idc_bus = resolve_current_default_idc_bus()
    bus_inventory = build_bus_inventory(grid_case, current_idc_bus)
    candidate_buses = resolve_candidate_buses(grid_case, bus_inventory, args.candidate_buses)
    primary_loads = sorted_unique(parse_float_list(args.idc_loads_mw))
    extra_loads = sorted_unique(parse_float_list(args.extra_loads_mw))
    mef_loads = sorted_unique(parse_float_list(args.mef_loads_mw))
    emission_factors = build_default_gen_emission_factors(grid_case)

    inventory_path = out_dir / "idc_bus_inventory.csv"
    scan_path = out_dir / "idc_bus_sensitivity_scan.csv"
    summary_path = out_dir / "idc_bus_sensitivity_summary.csv"
    mef_path = out_dir / "idc_bus_sensitivity_mef_topk.csv"
    markdown_path = out_dir / "idc_bus_sensitivity_summary.md"

    write_csv(inventory_path, bus_inventory, BUS_INVENTORY_FIELDS)

    print(f"Loaded {grid_case.name}: {len(bus_inventory)} buses")
    print(f"Candidate IEEE buses: {candidate_buses}")
    print("Cache: off (direct solve_opf calls; --no-cache is accepted for CLI clarity)")

    primary_rows = run_primary_scan(
        grid_case=grid_case,
        bus_inventory=bus_inventory,
        candidate_buses=candidate_buses,
        primary_loads=primary_loads,
        extra_loads=extra_loads,
        include_extra_loads=not args.no_extra_loads,
        stop_on_failure=not args.no_stop_on_failure,
        opf_mode=args.opf_mode,
        compute_mef=args.mef_all and not args.no_mef,
        delta_p_mw=args.delta_p_mw,
        emission_factors=emission_factors,
    )
    write_csv(scan_path, primary_rows, SCAN_FIELDS)

    summary_rows = build_bus_summary(primary_rows, bus_inventory)
    write_csv(summary_path, summary_rows, SUMMARY_FIELDS)

    mef_rows: list[dict[str, Any]] = []
    if args.no_mef:
        print("MEF scan skipped by --no-mef")
    elif args.top_k_mef > 0:
        selected_buses = select_top_k_mef_buses(summary_rows, args.top_k_mef)
        print(f"Running MEF refinement for IEEE buses: {selected_buses}")
        mef_rows = run_mef_refinement(
            grid_case=grid_case,
            bus_inventory=bus_inventory,
            selected_buses=selected_buses,
            loads_mw=mef_loads,
            opf_mode=args.opf_mode,
            delta_p_mw=args.delta_p_mw,
            emission_factors=emission_factors,
        )
        write_csv(mef_path, mef_rows, SCAN_FIELDS)
        summary_rows = build_bus_summary(primary_rows + mef_rows, bus_inventory)
        write_csv(summary_path, summary_rows, SUMMARY_FIELDS)
    else:
        write_csv(mef_path, mef_rows, SCAN_FIELDS)

    write_markdown_report(
        markdown_path,
        args=args,
        inventory_rows=bus_inventory,
        primary_rows=primary_rows,
        summary_rows=summary_rows,
        mef_rows=mef_rows,
        paths={
            "inventory": inventory_path,
            "scan": scan_path,
            "summary": summary_path,
            "mef": mef_path,
            "markdown": markdown_path,
        },
    )

    print(f"Saved bus inventory: {inventory_path}")
    print(f"Saved primary scan:  {scan_path}")
    print(f"Saved bus summary:   {summary_path}")
    print(f"Saved MEF top-k:     {mef_path}")
    print(f"Saved markdown:      {markdown_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Scan IEEE14 IDC access-bus sensitivity.")
    parser.add_argument("--candidate-buses", default=DEFAULT_CANDIDATE_BUSES, help="Comma-separated IEEE bus numbers, 'all_load_buses', or 'all_buses'.")
    parser.add_argument("--idc-loads-mw", default=DEFAULT_IDC_LOADS_MW, help="Comma-separated added IDC load levels in MW.")
    parser.add_argument("--extra-loads-mw", default=DEFAULT_EXTRA_LOADS_MW, help="Extra loads tested when the largest primary load still succeeds.")
    parser.add_argument("--no-extra-loads", action="store_true", help="Do not auto-test extra loads after the primary list succeeds.")
    parser.add_argument("--no-stop-on-failure", action="store_true", help="Continue scanning higher loads after first OPF failure.")
    parser.add_argument("--opf-mode", default="ac", choices=["ac", "dc"], help="OPF mode for the scan.")
    parser.add_argument("--no-cache", action="store_true", help="Accepted for clarity; this script does not use grid_cache.")
    parser.add_argument("--no-mef", action="store_true", help="Skip MEF completely.")
    parser.add_argument("--mef-all", action="store_true", help="Compute MEF for every primary scan point.")
    parser.add_argument("--top-k-mef", type=int, default=0, help="After the primary scan, compute MEF for the top K sensitive buses.")
    parser.add_argument("--mef-loads-mw", default=DEFAULT_MEF_LOADS_MW, help="Load levels for top-K MEF refinement.")
    parser.add_argument("--delta-p-mw", type=float, default=0.1, help="MEF perturbation in MW.")
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR, help="Output directory.")
    return parser.parse_args()


def build_bus_inventory(grid_case, current_idc_bus: int) -> list[dict[str, Any]]:
    net = grid_case.raw_network
    p_by_bus: dict[int, float] = {}
    q_by_bus: dict[int, float] = {}
    if hasattr(net, "load") and len(net.load) > 0:
        for _, load in net.load.iterrows():
            bus_idx = int(load["bus"])
            scaling = to_float(load.get("scaling", 1.0), 1.0)
            p_by_bus[bus_idx] = p_by_bus.get(bus_idx, 0.0) + to_float(load.get("p_mw", 0.0), 0.0) * scaling
            q_by_bus[bus_idx] = q_by_bus.get(bus_idx, 0.0) + to_float(load.get("q_mvar", 0.0), 0.0) * scaling

    gen_buses = set()
    if hasattr(net, "gen") and len(net.gen) > 0:
        gen_buses.update(int(bus) for bus in net.gen["bus"].tolist())

    slack_buses = set()
    if hasattr(net, "ext_grid") and len(net.ext_grid) > 0:
        slack_buses.update(int(bus) for bus in net.ext_grid["bus"].tolist())

    rows: list[dict[str, Any]] = []
    for bus_idx in sorted(int(idx) for idx in net.bus.index.tolist()):
        ieee = int(grid_case.bus_index_to_ieee_bus_number[bus_idx])
        p_mw = float(p_by_bus.get(bus_idx, 0.0))
        q_mvar = float(q_by_bus.get(bus_idx, 0.0))
        bus = net.bus.loc[bus_idx]
        rows.append(
            {
                "ieee_bus_number": ieee,
                "pandapower_bus_index": bus_idx,
                "bus_name": str(bus.get("name", "")),
                "vn_kv": finite_or_blank(bus.get("vn_kv", math.nan)),
                "original_p_mw": p_mw,
                "original_q_mvar": q_mvar,
                "is_original_load_bus": p_mw > 1e-9 or q_mvar > 1e-9,
                "is_generator_bus": bus_idx in gen_buses,
                "is_slack_bus": bus_idx in slack_buses,
                "is_zero_load_bus": abs(p_mw) <= 1e-9 and abs(q_mvar) <= 1e-9,
                "is_current_default_idc_bus": ieee == int(current_idc_bus),
            }
        )
    return rows


def resolve_current_default_idc_bus() -> int:
    try:
        from configs.config_ultimate import GRID_CONFIG  # noqa: PLC0415

        return int(GRID_CONFIG.get("idc_ieee_bus_number", 9))
    except Exception:
        return 9


def resolve_candidate_buses(grid_case, inventory_rows: list[dict[str, Any]], text: str) -> list[int]:
    value = str(text).strip().lower()
    if value == "all_load_buses":
        return [
            int(row["ieee_bus_number"])
            for row in inventory_rows
            if truthy(row.get("is_original_load_bus")) and not truthy(row.get("is_slack_bus"))
        ]
    if value == "all_buses":
        return sorted(int(key) for key in grid_case.ieee_bus_number_to_bus_index)
    buses = [int(item) for item in parse_str_list(text)]
    valid = set(int(key) for key in grid_case.ieee_bus_number_to_bus_index)
    invalid = [bus for bus in buses if bus not in valid]
    if invalid:
        raise ValueError(f"Unknown IEEE bus numbers {invalid}; valid buses are {sorted(valid)}")
    return buses


def run_primary_scan(
    *,
    grid_case,
    bus_inventory: list[dict[str, Any]],
    candidate_buses: list[int],
    primary_loads: list[float],
    extra_loads: list[float],
    include_extra_loads: bool,
    stop_on_failure: bool,
    opf_mode: str,
    compute_mef: bool,
    delta_p_mw: float,
    emission_factors: dict[int, float],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    inventory_by_bus = {int(row["ieee_bus_number"]): row for row in bus_inventory}
    base_lmp_by_bus: dict[int, float] = {}

    for bus_pos, ieee_bus in enumerate(candidate_buses, start=1):
        bus_index = get_bus_index_by_ieee_number(grid_case, ieee_bus)
        loads_to_scan = list(primary_loads)
        if include_extra_loads:
            loads_to_scan.extend([value for value in extra_loads if value not in loads_to_scan])
        loads_to_scan = sorted_unique(loads_to_scan)

        first_failed = False
        print(f"[bus {bus_pos}/{len(candidate_buses)}] IEEE bus {ieee_bus} -> pp index {bus_index}")
        for load_pos, idc_load_mw in enumerate(loads_to_scan, start=1):
            if stop_on_failure and first_failed:
                break
            print(f"  [{load_pos}/{len(loads_to_scan)}] {opf_mode.upper()} OPF IDC={idc_load_mw:.3f} MW")
            row = run_one_scan_point(
                scan_round="primary",
                grid_case=grid_case,
                inventory_row=inventory_by_bus[ieee_bus],
                bus_index=bus_index,
                idc_load_mw=idc_load_mw,
                opf_mode=opf_mode,
                compute_mef=compute_mef,
                delta_p_mw=delta_p_mw,
                emission_factors=emission_factors,
                base_lmp=base_lmp_by_bus.get(ieee_bus, math.nan),
            )
            if abs(float(idc_load_mw)) < 1e-12 and is_finite(row.get("lmp_bus")):
                base_lmp_by_bus[ieee_bus] = float(row["lmp_bus"])
                row["lmp_delta_from_zero"] = 0.0
                row["lmp_slope_from_zero_per_mw"] = ""
            elif ieee_bus in base_lmp_by_bus and is_finite(row.get("lmp_bus")) and idc_load_mw > 0:
                delta = float(row["lmp_bus"]) - base_lmp_by_bus[ieee_bus]
                row["lmp_delta_from_zero"] = delta
                row["lmp_slope_from_zero_per_mw"] = delta / float(idc_load_mw)

            rows.append(row)
            if not truthy(row.get("opf_success")):
                first_failed = True

    return rows


def run_mef_refinement(
    *,
    grid_case,
    bus_inventory: list[dict[str, Any]],
    selected_buses: list[int],
    loads_mw: list[float],
    opf_mode: str,
    delta_p_mw: float,
    emission_factors: dict[int, float],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    inventory_by_bus = {int(row["ieee_bus_number"]): row for row in bus_inventory}
    for bus_pos, ieee_bus in enumerate(selected_buses, start=1):
        bus_index = get_bus_index_by_ieee_number(grid_case, ieee_bus)
        base_lmp = math.nan
        print(f"[MEF {bus_pos}/{len(selected_buses)}] IEEE bus {ieee_bus} -> pp index {bus_index}")
        for load_pos, idc_load_mw in enumerate(loads_mw, start=1):
            print(f"  [{load_pos}/{len(loads_mw)}] {opf_mode.upper()} OPF+MEF IDC={idc_load_mw:.3f} MW")
            row = run_one_scan_point(
                scan_round="mef_topk",
                grid_case=grid_case,
                inventory_row=inventory_by_bus[ieee_bus],
                bus_index=bus_index,
                idc_load_mw=idc_load_mw,
                opf_mode=opf_mode,
                compute_mef=True,
                delta_p_mw=delta_p_mw,
                emission_factors=emission_factors,
                base_lmp=base_lmp,
            )
            if abs(float(idc_load_mw)) < 1e-12 and is_finite(row.get("lmp_bus")):
                base_lmp = float(row["lmp_bus"])
                row["lmp_delta_from_zero"] = 0.0
                row["lmp_slope_from_zero_per_mw"] = ""
            elif is_finite(base_lmp) and is_finite(row.get("lmp_bus")) and idc_load_mw > 0:
                delta = float(row["lmp_bus"]) - base_lmp
                row["lmp_delta_from_zero"] = delta
                row["lmp_slope_from_zero_per_mw"] = delta / float(idc_load_mw)
            rows.append(row)
    return rows


def run_one_scan_point(
    *,
    scan_round: str,
    grid_case,
    inventory_row: dict[str, Any],
    bus_index: int,
    idc_load_mw: float,
    opf_mode: str,
    compute_mef: bool,
    delta_p_mw: float,
    emission_factors: dict[int, float],
    base_lmp: float,
) -> dict[str, Any]:
    start = time.perf_counter()
    opf_result = solve_opf(
        grid_case,
        mode=opf_mode,
        idc_bus_id=int(bus_index),
        idc_load_mw=float(idc_load_mw),
        load_scale=1.0,
    )
    solve_time = time.perf_counter() - start
    metrics = extract_grid_metrics(opf_result)
    max_trafo = extract_max_trafo_loading(opf_result)
    trafo_overload_count = count_trafo_overloads(opf_result)
    voltage_margin = compute_voltage_margin_to_limit(opf_result)
    lmp = opf_result.lmp_by_bus.get(int(bus_index), math.nan)

    mef_success = False
    mef_plus = math.nan
    mef_minus = math.nan
    mef_mean = math.nan
    mef_sensitivity = math.nan
    mef_message = "MEF skipped."
    mef_time = 0.0
    if compute_mef:
        mef_start = time.perf_counter()
        mef_result = calculate_nodal_mef(
            grid_case,
            bus_id=int(bus_index),
            mode=opf_mode,
            delta_p_mw=float(delta_p_mw),
            base_idc_load_mw=float(idc_load_mw),
            load_scale=1.0,
            clamp_minus_load=True,
            gen_emission_factors_kg_per_mwh=emission_factors,
        )
        mef_time = time.perf_counter() - mef_start
        mef_success = bool(mef_result.success)
        mef_plus = mef_result.mef_plus_kg_per_mwh
        mef_minus = mef_result.mef_minus_kg_per_mwh
        if is_finite(mef_plus) and is_finite(mef_minus):
            mef_mean = 0.5 * (float(mef_plus) + float(mef_minus))
            mef_sensitivity = abs(float(mef_plus) - float(mef_minus))
        mef_message = mef_result.message

    safe_cost = (
        float(metrics.voltage_violation_count)
        + float(metrics.line_overload_count)
        + float(trafo_overload_count)
        + (0.0 if bool(opf_result.success) else 1.0)
    )

    row = {
        "scan_round": scan_round,
        "ieee_bus_number": int(inventory_row["ieee_bus_number"]),
        "pandapower_bus_index": int(bus_index),
        "bus_name": str(inventory_row.get("bus_name", "")),
        "original_p_mw": finite_or_blank(inventory_row.get("original_p_mw")),
        "original_q_mvar": finite_or_blank(inventory_row.get("original_q_mvar")),
        "is_original_load_bus": bool(inventory_row.get("is_original_load_bus")),
        "is_generator_bus": bool(inventory_row.get("is_generator_bus")),
        "is_slack_bus": bool(inventory_row.get("is_slack_bus")),
        "is_current_default_idc_bus": bool(inventory_row.get("is_current_default_idc_bus")),
        "idc_load_mw": float(idc_load_mw),
        "opf_mode": str(opf_mode),
        "opf_success": bool(opf_result.success),
        "failure_reason": "" if bool(opf_result.success) else str(opf_result.message),
        "minV": finite_or_blank(metrics.min_voltage_pu),
        "maxV": finite_or_blank(metrics.max_voltage_pu),
        "voltage_margin_to_limit": finite_or_blank(voltage_margin),
        "maxLine": finite_or_blank(metrics.max_line_loading_percent),
        "maxTrafo": finite_or_blank(max_trafo),
        "network_loss": finite_or_blank(opf_result.network_loss_mw),
        "total_load": finite_or_blank(opf_result.total_load_mw),
        "total_generation": finite_or_blank(opf_result.total_generation_mw),
        "lmp_bus": finite_or_blank(lmp),
        "lmp_delta_from_zero": "",
        "lmp_slope_from_zero_per_mw": "",
        "safe_cost": float(safe_cost),
        "voltage_violation_count": int(metrics.voltage_violation_count),
        "line_overload_count": int(metrics.line_overload_count),
        "trafo_overload_count": int(trafo_overload_count),
        "opf_violation": 0 if bool(opf_result.success) else 1,
        "solve_time_sec": float(solve_time),
        "mef_success": bool(mef_success),
        "mef_plus": finite_or_blank(mef_plus),
        "mef_minus": finite_or_blank(mef_minus),
        "mef_mean": finite_or_blank(mef_mean),
        "mef_sensitivity": finite_or_blank(mef_sensitivity),
        "mef_message": mef_message,
        "mef_time_sec": float(mef_time),
    }
    if is_finite(base_lmp) and is_finite(row.get("lmp_bus")) and float(idc_load_mw) > 0:
        delta = float(row["lmp_bus"]) - float(base_lmp)
        row["lmp_delta_from_zero"] = delta
        row["lmp_slope_from_zero_per_mw"] = delta / float(idc_load_mw)
    return row


def build_bus_summary(scan_rows: list[dict[str, Any]], bus_inventory: list[dict[str, Any]]) -> list[dict[str, Any]]:
    inventory_by_bus = {int(row["ieee_bus_number"]): row for row in bus_inventory}
    primary_rows = [row for row in scan_rows if row.get("scan_round") == "primary"]
    by_bus: dict[int, list[dict[str, Any]]] = {}
    for row in primary_rows:
        by_bus.setdefault(int(row["ieee_bus_number"]), []).append(row)

    summary: list[dict[str, Any]] = []
    for ieee_bus in sorted(by_bus):
        rows = sorted(by_bus[ieee_bus], key=lambda item: float(item["idc_load_mw"]))
        ok_rows = [row for row in rows if truthy(row.get("opf_success"))]
        failed_rows = [row for row in rows if not truthy(row.get("opf_success"))]
        inv = inventory_by_bus[ieee_bus]

        max_success = max_float(ok_rows, "idc_load_mw")
        first_failed = min_float(failed_rows, "idc_load_mw")
        max_v97 = max_load_with_condition(ok_rows, lambda row: to_float(row.get("minV")) >= 0.97)
        max_v98 = max_load_with_condition(ok_rows, lambda row: to_float(row.get("minV")) >= 0.98)
        lmp_slope = linear_slope(ok_rows, "idc_load_mw", "lmp_bus")
        loss_slope = linear_slope(ok_rows, "idc_load_mw", "network_loss")
        lmp_at_zero = value_at_load(ok_rows, 0.0, "lmp_bus")
        loss_at_zero = value_at_load(ok_rows, 0.0, "network_loss")
        lmp_at_max_success = value_at_load(ok_rows, max_success, "lmp_bus")
        loss_at_max_success = value_at_load(ok_rows, max_success, "network_loss")

        all_mef_rows = [
            row
            for row in scan_rows
            if int(row.get("ieee_bus_number", -1)) == ieee_bus and is_finite(row.get("mef_mean"))
        ]
        positive_mef_rows = [row for row in all_mef_rows if to_float(row.get("idc_load_mw")) > 0.0]
        mef_rows = positive_mef_rows if positive_mef_rows else all_mef_rows
        mef_means = finite_values(mef_rows, "mef_mean")
        mef_sens = finite_values(mef_rows, "mef_sensitivity")

        min_v_success = min_finite(ok_rows, "minV")
        voltage_margin_min = min_finite(ok_rows, "voltage_margin_to_limit")
        max_line_success = max_finite(ok_rows, "maxLine")
        max_trafo_success = max_finite(ok_rows, "maxTrafo")
        lmp_range = range_finite(ok_rows, "lmp_bus")
        sensitivity_score = compute_sensitivity_score(
            max_successful_mw=max_success,
            first_failed_mw=first_failed,
            min_voltage=min_v_success,
            lmp_range=lmp_range,
        )
        role, normal_range, stress_note = suggest_role(
            inv,
            max_successful_mw=max_success,
            first_failed_mw=first_failed,
            max_v97=max_v97,
            min_voltage=min_v_success,
            max_line=max_line_success,
            max_trafo=max_trafo_success,
        )

        summary.append(
            {
                "ieee_bus_number": ieee_bus,
                "pandapower_bus_index": int(inv["pandapower_bus_index"]),
                "bus_name": str(inv.get("bus_name", "")),
                "original_p_mw": finite_or_blank(inv.get("original_p_mw")),
                "original_q_mvar": finite_or_blank(inv.get("original_q_mvar")),
                "is_original_load_bus": bool(inv.get("is_original_load_bus")),
                "is_generator_bus": bool(inv.get("is_generator_bus")),
                "is_slack_bus": bool(inv.get("is_slack_bus")),
                "is_current_default_idc_bus": bool(inv.get("is_current_default_idc_bus")),
                "scanned_points": len(rows),
                "success_points": len(ok_rows),
                "success_rate": len(ok_rows) / max(len(rows), 1),
                "max_successful_idc_mw": finite_or_blank(max_success),
                "first_failed_idc_mw": finite_or_blank(first_failed),
                "max_idc_mw_with_minV_ge_0_97": finite_or_blank(max_v97),
                "max_idc_mw_with_minV_ge_0_98": finite_or_blank(max_v98),
                "minV_min_success": finite_or_blank(min_v_success),
                "maxV_max_success": finite_or_blank(max_finite(ok_rows, "maxV")),
                "voltage_margin_min_success": finite_or_blank(voltage_margin_min),
                "maxLine_max_success": finite_or_blank(max_line_success),
                "maxTrafo_max_success": finite_or_blank(max_trafo_success),
                "lmp_at_0_mw": finite_or_blank(lmp_at_zero),
                "lmp_at_max_success_mw": finite_or_blank(lmp_at_max_success),
                "lmp_range_success": finite_or_blank(lmp_range),
                "lmp_slope_per_mw": finite_or_blank(lmp_slope),
                "loss_at_0_mw": finite_or_blank(loss_at_zero),
                "loss_at_max_success_mw": finite_or_blank(loss_at_max_success),
                "loss_slope_per_mw": finite_or_blank(loss_slope),
                "mef_mean_min": finite_or_blank(min(mef_means) if mef_means else math.nan),
                "mef_mean_max": finite_or_blank(max(mef_means) if mef_means else math.nan),
                "mef_sensitivity_range": finite_or_blank(max(mef_sens) if mef_sens else math.nan),
                "sensitivity_score": finite_or_blank(sensitivity_score),
                "suggested_role": role,
                "suggested_normal_range_mw": normal_range,
                "stress_note": stress_note,
            }
        )

    summary.sort(key=lambda row: to_float(row.get("sensitivity_score")), reverse=True)
    return summary


def select_top_k_mef_buses(summary_rows: list[dict[str, Any]], top_k: int) -> list[int]:
    ranked = [
        row for row in summary_rows
        if truthy(row.get("is_original_load_bus")) and not truthy(row.get("is_slack_bus"))
    ]
    ranked.sort(key=lambda row: to_float(row.get("sensitivity_score")), reverse=True)
    return [int(row["ieee_bus_number"]) for row in ranked[: max(int(top_k), 0)]]


def compute_sensitivity_score(
    *,
    max_successful_mw: float,
    first_failed_mw: float,
    min_voltage: float,
    lmp_range: float,
) -> float:
    max_success = max_successful_mw if is_finite(max_successful_mw) else 0.0
    failure_component = max(0.0, 300.0 - max_success)
    if is_finite(first_failed_mw):
        failure_component += max(0.0, 300.0 - float(first_failed_mw)) * 0.5
    voltage_component = max(0.0, 1.0 - (min_voltage if is_finite(min_voltage) else 1.0)) * 1000.0
    lmp_component = abs(lmp_range if is_finite(lmp_range) else 0.0) * 5.0
    return float(failure_component + voltage_component + lmp_component)


def suggest_role(
    inventory_row: dict[str, Any],
    *,
    max_successful_mw: float,
    first_failed_mw: float,
    max_v97: float,
    min_voltage: float,
    max_line: float,
    max_trafo: float,
) -> tuple[str, str, str]:
    if truthy(inventory_row.get("is_slack_bus")):
        return "avoid_for_idc_baseline", "not recommended", "Slack bus would confound load-placement effects."

    if not is_finite(max_successful_mw):
        return "avoid_or_debug", "not recommended", "No successful OPF point found."

    soft_capacity = max_v97 if is_finite(max_v97) else max_successful_mw
    hard_capacity = max_successful_mw
    line_tight = is_finite(max_line) and max_line >= 80.0
    trafo_tight = is_finite(max_trafo) and max_trafo >= 80.0

    if hard_capacity >= 200.0 and soft_capacity >= 100.0 and not line_tight and not trafo_tight:
        return "normal_training_candidate", f"30-{min(100.0, soft_capacity):.0f}", "Good headroom under the current IEEE14 constraints."
    if hard_capacity >= 100.0 and soft_capacity >= 50.0:
        return "normal_or_moderate_stress", f"20-{min(80.0, soft_capacity):.0f}", "Useful for training with visible voltage/LMP response."
    if hard_capacity >= 50.0:
        return "stress_candidate", f"10-{min(50.0, soft_capacity):.0f}", "Use cautiously; OPF boundary is nearby."
    if is_finite(first_failed_mw):
        return "stress_only_or_avoid", "0-20", f"First OPF failure at {first_failed_mw:.0f} MW."
    if is_finite(min_voltage) and min_voltage < 0.97:
        return "stress_candidate", "0-30", "Voltage enters the soft-margin region."
    return "needs_review", "review required", "Sensitivity classification is inconclusive."


def write_markdown_report(
    path: Path,
    *,
    args: argparse.Namespace,
    inventory_rows: list[dict[str, Any]],
    primary_rows: list[dict[str, Any]],
    summary_rows: list[dict[str, Any]],
    mef_rows: list[dict[str, Any]],
    paths: dict[str, Path],
) -> None:
    total_points = len(primary_rows)
    successes = sum(1 for row in primary_rows if truthy(row.get("opf_success")))
    current_default = [row for row in summary_rows if truthy(row.get("is_current_default_idc_bus"))]
    lines = [
        "# IDC Bus Sensitivity Scan Summary",
        "",
        "## Run Settings",
        "",
        f"- opf_mode: {args.opf_mode}",
        "- cache: off (direct solve_opf calls)",
        f"- candidate_buses: {args.candidate_buses}",
        f"- idc_loads_mw: {args.idc_loads_mw}",
        f"- extra_loads_mw: {'disabled' if args.no_extra_loads else args.extra_loads_mw}",
        f"- stop_on_failure: {not bool(args.no_stop_on_failure)}",
        f"- mef_all: {bool(args.mef_all and not args.no_mef)}",
        f"- top_k_mef: {0 if args.no_mef else int(args.top_k_mef)}",
        f"- delta_p_mw: {float(args.delta_p_mw):.6f}",
        "",
        "## Output Files",
        "",
        f"- bus_inventory: `{relative_to_project(paths['inventory'])}`",
        f"- primary_scan: `{relative_to_project(paths['scan'])}`",
        f"- bus_summary: `{relative_to_project(paths['summary'])}`",
        f"- mef_topk: `{relative_to_project(paths['mef'])}`",
        "",
        "## Overall Result",
        "",
        f"- primary_scan_points: {total_points}",
        f"- primary_opf_success_rate: {successes / max(total_points, 1):.6f}",
    ]
    if current_default:
        row = current_default[0]
        lines.append(
            f"- current_default_bus9: max_success={fmt(row.get('max_successful_idc_mw'))} MW, "
            f"first_failure={fmt(row.get('first_failed_idc_mw'))} MW, "
            f"role={row.get('suggested_role')}"
        )

    lines.extend([
        "",
        "## IEEE14 Bus Inventory",
        "",
        "| IEEE bus | pp index | name | P MW | Q Mvar | load | gen | slack | default IDC |",
        "|---:|---:|---|---:|---:|---|---|---|---|",
    ])
    for row in sorted(inventory_rows, key=lambda item: int(item["ieee_bus_number"])):
        lines.append(
            f"| {row['ieee_bus_number']} | {row['pandapower_bus_index']} | {row['bus_name']} | "
            f"{fmt(row.get('original_p_mw'))} | {fmt(row.get('original_q_mvar'))} | "
            f"{yes_no(row.get('is_original_load_bus'))} | {yes_no(row.get('is_generator_bus'))} | "
            f"{yes_no(row.get('is_slack_bus'))} | {yes_no(row.get('is_current_default_idc_bus'))} |"
        )

    lines.extend([
        "",
        "## Sensitivity Ranking",
        "",
        "| rank | IEEE bus | max OK MW | first fail MW | minV min | maxLine max % | maxTrafo max % | LMP range | LMP slope/MW | role | normal range MW |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|",
    ])
    for rank, row in enumerate(summary_rows, start=1):
        lines.append(
            f"| {rank} | {row['ieee_bus_number']} | {fmt(row.get('max_successful_idc_mw'))} | "
            f"{fmt(row.get('first_failed_idc_mw'))} | {fmt(row.get('minV_min_success'))} | "
            f"{fmt(row.get('maxLine_max_success'))} | {fmt(row.get('maxTrafo_max_success'))} | "
            f"{fmt(row.get('lmp_range_success'))} | {fmt(row.get('lmp_slope_per_mw'))} | "
            f"{row.get('suggested_role')} | {row.get('suggested_normal_range_mw')} |"
        )

    lines.extend([
        "",
        "## First Failure By Bus",
        "",
        "| IEEE bus | first_failed_idc_mw | max_successful_idc_mw | failure reason |",
        "|---:|---:|---:|---|",
    ])
    for row in sorted(summary_rows, key=lambda item: (not is_finite(item.get("first_failed_idc_mw")), to_float(item.get("first_failed_idc_mw")))):
        if not is_finite(row.get("first_failed_idc_mw")):
            continue
        reason = first_failure_reason(primary_rows, int(row["ieee_bus_number"]))
        lines.append(
            f"| {row['ieee_bus_number']} | {fmt(row.get('first_failed_idc_mw'))} | "
            f"{fmt(row.get('max_successful_idc_mw'))} | {escape_md(reason)} |"
        )

    lines.extend([
        "",
        "## Voltage Sensitivity",
        "",
        "| rank | IEEE bus | minV min success | voltage margin min | max IDCs with minV >= 0.97 MW |",
        "|---:|---:|---:|---:|---:|",
    ])
    voltage_rank = sorted(
        [row for row in summary_rows if is_finite(row.get("minV_min_success"))],
        key=lambda item: to_float(item.get("minV_min_success")),
    )
    for rank, row in enumerate(voltage_rank[:10], start=1):
        lines.append(
            f"| {rank} | {row['ieee_bus_number']} | {fmt(row.get('minV_min_success'))} | "
            f"{fmt(row.get('voltage_margin_min_success'))} | "
            f"{fmt(row.get('max_idc_mw_with_minV_ge_0_97'))} |"
        )

    lines.extend([
        "",
        "## Branch Loading Sensitivity",
        "",
        "| rank | IEEE bus | maxLine max % | maxTrafo max % |",
        "|---:|---:|---:|---:|",
    ])
    branch_rank = sorted(
        summary_rows,
        key=lambda item: max(to_float(item.get("maxLine_max_success")), to_float(item.get("maxTrafo_max_success"))),
        reverse=True,
    )
    for rank, row in enumerate(branch_rank[:10], start=1):
        lines.append(
            f"| {rank} | {row['ieee_bus_number']} | {fmt(row.get('maxLine_max_success'))} | "
            f"{fmt(row.get('maxTrafo_max_success'))} |"
        )

    lines.extend([
        "",
        "## LMP Sensitivity",
        "",
        "| rank | IEEE bus | LMP at 0 MW | LMP at max OK | LMP range | LMP slope per MW |",
        "|---:|---:|---:|---:|---:|---:|",
    ])
    lmp_rank = sorted(summary_rows, key=lambda item: abs(to_float(item.get("lmp_slope_per_mw"))), reverse=True)
    for rank, row in enumerate(lmp_rank[:10], start=1):
        lines.append(
            f"| {rank} | {row['ieee_bus_number']} | {fmt(row.get('lmp_at_0_mw'))} | "
            f"{fmt(row.get('lmp_at_max_success_mw'))} | {fmt(row.get('lmp_range_success'))} | "
            f"{fmt(row.get('lmp_slope_per_mw'))} |"
        )

    if mef_rows:
        lines.extend([
            "",
            "## MEF Refinement",
            "",
            "The MEF summary below prefers positive-load points so the 0 MW minus-perturbation clamp does not dominate the sensitivity ranking.",
            "",
            "| IEEE bus | MEF mean min | MEF mean max | MEF sensitivity range |",
            "|---:|---:|---:|---:|",
        ])
        for row in sorted(summary_rows, key=lambda item: to_float(item.get("mef_sensitivity_range")), reverse=True):
            if not is_finite(row.get("mef_mean_min")):
                continue
            lines.append(
                f"| {row['ieee_bus_number']} | {fmt(row.get('mef_mean_min'))} | "
                f"{fmt(row.get('mef_mean_max'))} | {fmt(row.get('mef_sensitivity_range'))} |"
            )
    else:
        lines.extend([
            "",
            "## MEF Refinement",
            "",
            "MEF was not calculated in this run. Run with `--top-k-mef 5` for the second-stage refinement.",
        ])

    lines.extend([
        "",
        "## Suggested Next Use",
        "",
        "- Use `normal_training_candidate` buses for main single-IDC or multi-IDC experiments.",
        "- Use `normal_or_moderate_stress` buses when visible grid feedback is desired but OPF should remain mostly feasible.",
        "- Keep `stress_candidate` and `stress_only_or_avoid` buses out of default PPO training unless the experiment is explicitly a stress test.",
    ])

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def extract_max_trafo_loading(opf_result) -> float:
    net = getattr(opf_result, "raw_result", None)
    table = getattr(net, "res_trafo", None)
    if table is None or not hasattr(table, "columns") or "loading_percent" not in table.columns:
        return math.nan
    values = [to_float(value) for value in table["loading_percent"].tolist()]
    values = [value for value in values if math.isfinite(value)]
    return max(values) if values else math.nan


def count_trafo_overloads(opf_result, fallback_limit: float = 100.0, tolerance: float = 1e-6) -> int:
    net = getattr(opf_result, "raw_result", None)
    res_trafo = getattr(net, "res_trafo", None)
    trafo = getattr(net, "trafo", None)
    if res_trafo is None or not hasattr(res_trafo, "columns") or "loading_percent" not in res_trafo.columns:
        return 0
    count = 0
    for idx, value in res_trafo["loading_percent"].items():
        loading = to_float(value)
        if not math.isfinite(loading):
            continue
        limit = fallback_limit
        if trafo is not None and hasattr(trafo, "columns") and "max_loading_percent" in trafo.columns and idx in trafo.index:
            limit = to_float(trafo.at[idx, "max_loading_percent"], fallback_limit)
        if loading - limit > tolerance:
            count += 1
    return count


def compute_voltage_margin_to_limit(opf_result) -> float:
    margins: list[float] = []
    for bus_id, value in opf_result.bus_voltage_pu.items():
        voltage = to_float(value)
        if not math.isfinite(voltage):
            continue
        min_limit = to_float(opf_result.bus_voltage_min_pu.get(bus_id), 0.95)
        max_limit = to_float(opf_result.bus_voltage_max_pu.get(bus_id), 1.05)
        margins.append(voltage - min_limit)
        margins.append(max_limit - voltage)
    return min(margins) if margins else math.nan


def first_failure_reason(rows: list[dict[str, Any]], ieee_bus: int) -> str:
    failed = [
        row for row in rows
        if int(row.get("ieee_bus_number", -1)) == int(ieee_bus) and not truthy(row.get("opf_success"))
    ]
    if not failed:
        return ""
    failed.sort(key=lambda row: float(row.get("idc_load_mw", math.inf)))
    return str(failed[0].get("failure_reason", ""))


def max_load_with_condition(rows: list[dict[str, Any]], predicate) -> float:
    values = [to_float(row.get("idc_load_mw")) for row in rows if predicate(row)]
    values = [value for value in values if math.isfinite(value)]
    return max(values) if values else math.nan


def value_at_load(rows: list[dict[str, Any]], load_mw: float, key: str) -> float:
    if not is_finite(load_mw):
        return math.nan
    for row in rows:
        if abs(to_float(row.get("idc_load_mw")) - float(load_mw)) <= 1e-9:
            return to_float(row.get(key))
    return math.nan


def linear_slope(rows: list[dict[str, Any]], x_key: str, y_key: str) -> float:
    pairs = [
        (to_float(row.get(x_key)), to_float(row.get(y_key)))
        for row in rows
        if is_finite(row.get(x_key)) and is_finite(row.get(y_key))
    ]
    if len(pairs) < 2:
        return math.nan
    xs = [pair[0] for pair in pairs]
    ys = [pair[1] for pair in pairs]
    x_mean = sum(xs) / len(xs)
    y_mean = sum(ys) / len(ys)
    denom = sum((x - x_mean) ** 2 for x in xs)
    if abs(denom) <= 1e-12:
        return math.nan
    return sum((x - x_mean) * (y - y_mean) for x, y in pairs) / denom


def min_float(rows: list[dict[str, Any]], key: str) -> float:
    values = finite_values(rows, key)
    return min(values) if values else math.nan


def max_float(rows: list[dict[str, Any]], key: str) -> float:
    values = finite_values(rows, key)
    return max(values) if values else math.nan


def min_finite(rows: Iterable[dict[str, Any]], key: str) -> float:
    values = finite_values(rows, key)
    return min(values) if values else math.nan


def max_finite(rows: Iterable[dict[str, Any]], key: str) -> float:
    values = finite_values(rows, key)
    return max(values) if values else math.nan


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


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def sorted_unique(values: Iterable[float]) -> list[float]:
    unique = sorted({float(value) for value in values})
    if not unique:
        raise ValueError("Expected at least one numeric value.")
    return unique


def parse_float_list(text: str) -> list[float]:
    return [float(item.strip()) for item in str(text).split(",") if item.strip()]


def parse_str_list(text: str) -> list[str]:
    values = [item.strip() for item in str(text).split(",") if item.strip()]
    if not values:
        raise ValueError(f"Expected at least one value in {text!r}.")
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


def fmt(value: Any) -> str:
    number = to_float(value)
    return "nan" if not math.isfinite(number) else f"{number:.6f}"


def yes_no(value: Any) -> str:
    return "yes" if truthy(value) else "no"


def escape_md(value: str) -> str:
    return str(value).replace("|", "/")


def relative_to_project(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT)).replace("\\", "/")
    except ValueError:
        return str(path)


if __name__ == "__main__":
    main()
