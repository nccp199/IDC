"""Scan IEEE14 bus sensitivity for single-IDC grid access.

Run with:
    python -m scripts.scan_ieee14_bus_sensitivity
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any


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
    build_generator_emission_table,
    calculate_nodal_mef,
    compute_total_emission,
    extract_grid_metrics,
    get_bus_index_by_ieee_number,
    load_ieee14_case,
    solve_opf,
)


CSV_FIELDS = [
    "case_name",
    "ieee_bus_number",
    "pandapower_bus_index",
    "idc_load_mw",
    "delta_p_mw",
    "dc_opf_success",
    "dc_opf_message",
    "dc_lmp",
    "dc_total_generation_cost",
    "dc_total_load_mw",
    "dc_total_generation_mw",
    "dc_network_loss_mw",
    "dc_total_emission_kg",
    "dc_mef_success",
    "dc_mef_plus_kg_per_mwh",
    "dc_mef_minus_kg_per_mwh",
    "dc_main_plus_marginal_gen",
    "dc_main_plus_delta_mw",
    "dc_main_plus_ef",
    "dc_main_minus_marginal_gen",
    "dc_main_minus_delta_mw",
    "dc_main_minus_ef",
    "dc_min_voltage_pu",
    "dc_max_voltage_pu",
    "dc_max_line_loading_percent",
    "dc_voltage_violation_count",
    "dc_line_overload_count",
    "dc_grid_security_penalty",
    "ac_opf_success",
    "ac_opf_message",
    "ac_lmp",
    "ac_total_generation_cost",
    "ac_total_load_mw",
    "ac_total_generation_mw",
    "ac_network_loss_mw",
    "ac_total_emission_kg",
    "ac_mef_success",
    "ac_mef_plus_kg_per_mwh",
    "ac_mef_minus_kg_per_mwh",
    "ac_main_plus_marginal_gen",
    "ac_main_plus_delta_mw",
    "ac_main_plus_ef",
    "ac_main_minus_marginal_gen",
    "ac_main_minus_delta_mw",
    "ac_main_minus_ef",
    "ac_min_voltage_pu",
    "ac_max_voltage_pu",
    "ac_max_line_loading_percent",
    "ac_voltage_violation_count",
    "ac_line_overload_count",
    "ac_grid_security_penalty",
    "diagnosis",
]


def main() -> None:
    args = parse_args()
    if args.skip_ac and args.skip_dc:
        raise SystemExit("At least one OPF mode must be enabled; do not use --skip-ac and --skip-dc together.")

    try:
        grid_case = load_ieee14_case()
    except ImportError as exc:
        raise SystemExit(f"Cannot run IEEE14 bus sensitivity scan: {exc}") from exc
    gen_emission_factors = build_default_gen_emission_factors(grid_case)
    ieee_bus_numbers = sorted(grid_case.ieee_bus_number_to_bus_index)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loaded IEEE14: {len(ieee_bus_numbers)} buses")

    summary_rows: list[dict[str, Any]] = []
    detail_buses: list[dict[str, Any]] = []

    for position, ieee_bus_number in enumerate(ieee_bus_numbers, start=1):
        print(f"Scanning bus {position}/{len(ieee_bus_numbers)} ...")
        bus_index = get_bus_index_by_ieee_number(grid_case, ieee_bus_number)
        row, detail = scan_one_bus(
            grid_case=grid_case,
            case_name=args.case_name,
            ieee_bus_number=ieee_bus_number,
            bus_index=bus_index,
            idc_load_mw=args.idc_load_mw,
            delta_p_mw=args.delta_p_mw,
            gen_emission_factors=gen_emission_factors,
            run_dc=not args.skip_dc,
            run_ac=not args.skip_ac,
        )
        summary_rows.append(row)
        detail_buses.append(detail)
        if args.verbose:
            print(_format_verbose_row(row))

    csv_path = out_dir / f"{args.case_name}_node_sensitivity_static.csv"
    json_path = out_dir / f"{args.case_name}_node_sensitivity_detail.json"
    save_summary_csv(csv_path, summary_rows)
    save_detail_json(
        json_path,
        {
            "case_name": args.case_name,
            "idc_load_mw": args.idc_load_mw,
            "delta_p_mw": args.delta_p_mw,
            "buses": detail_buses,
        },
    )

    print(f"Saved summary CSV: {csv_path}")
    print(f"Saved detail JSON: {json_path}")
    print_rankings(summary_rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Scan IEEE14 node sensitivity for single-IDC access.")
    parser.add_argument("--idc-load-mw", type=float, default=1.0, help="IDC load added at each bus in MW.")
    parser.add_argument("--delta-p-mw", type=float, default=0.1, help="MEF perturbation load in MW.")
    parser.add_argument("--out-dir", default="data/grid_node_sensitivity", help="Output directory.")
    parser.add_argument("--case-name", default="ieee14", help="Case/output file prefix.")
    parser.add_argument("--skip-ac", action="store_true", help="Skip AC OPF and AC-MEF.")
    parser.add_argument("--skip-dc", action="store_true", help="Skip DC OPF and DC-MEF.")
    parser.add_argument("--verbose", action="store_true", help="Print detailed bus results while scanning.")
    return parser.parse_args()


def scan_one_bus(
    grid_case,
    case_name: str,
    ieee_bus_number: int,
    bus_index: int,
    idc_load_mw: float,
    delta_p_mw: float,
    gen_emission_factors: dict[int, float],
    run_dc: bool,
    run_ac: bool,
) -> tuple[dict[str, Any], dict[str, Any]]:
    row = _base_csv_row(case_name, ieee_bus_number, bus_index, idc_load_mw, delta_p_mw)
    detail: dict[str, Any] = {
        "ieee_bus_number": ieee_bus_number,
        "pandapower_bus_index": bus_index,
    }
    mode_results: dict[str, dict[str, Any]] = {}

    if run_dc:
        dc_data = run_mode_scan(grid_case, "dc", bus_index, idc_load_mw, delta_p_mw, gen_emission_factors)
        fill_mode_columns(row, "dc", dc_data, bus_index)
        detail["dc"] = dc_data["detail"]
        mode_results["dc"] = dc_data
    else:
        detail["dc"] = {"skipped": True}

    if run_ac:
        ac_data = run_mode_scan(grid_case, "ac", bus_index, idc_load_mw, delta_p_mw, gen_emission_factors)
        fill_mode_columns(row, "ac", ac_data, bus_index)
        detail["ac"] = ac_data["detail"]
        mode_results["ac"] = ac_data
    else:
        detail["ac"] = {"skipped": True}

    row["diagnosis"] = build_diagnosis(mode_results)
    return row, detail


def run_mode_scan(
    grid_case,
    mode: str,
    bus_index: int,
    idc_load_mw: float,
    delta_p_mw: float,
    gen_emission_factors: dict[int, float],
) -> dict[str, Any]:
    opf_result = solve_opf(grid_case, mode=mode, idc_bus_id=bus_index, idc_load_mw=idc_load_mw)
    metrics = extract_grid_metrics(opf_result)
    total_emission, _ = compute_total_emission(opf_result.gen_power_mw, gen_emission_factors)
    if not opf_result.gen_power_mw:
        total_emission = math.nan

    mef_result = calculate_nodal_mef(
        grid_case,
        bus_id=bus_index,
        mode=mode,
        delta_p_mw=delta_p_mw,
        base_idc_load_mw=idc_load_mw,
        gen_emission_factors_kg_per_mwh=gen_emission_factors,
    )
    main_plus = find_main_marginal_generator(mef_result.delta_gen_power_plus_mw, gen_emission_factors)
    main_minus = find_main_marginal_generator(mef_result.delta_gen_power_minus_mw, gen_emission_factors)
    generator_emission_table = build_generator_emission_table(opf_result, gen_emission_factors)

    return {
        "opf_result": opf_result,
        "mef_result": mef_result,
        "metrics": metrics,
        "total_emission_kg": total_emission,
        "main_plus": main_plus,
        "main_minus": main_minus,
        "detail": {
            "opf": serialize_opf_result(opf_result, total_emission),
            "mef": serialize_mef_result(mef_result),
            "metrics": serialize_grid_metrics(metrics),
            "generator_emission_table": normalize_json(generator_emission_table),
        },
    }


def find_main_marginal_generator(
    delta_gen_power_mw: dict[int, float],
    gen_emission_factors: dict[int, float],
) -> dict[str, Any]:
    finite_items = [
        (int(gen_id), _safe_float(delta_mw))
        for gen_id, delta_mw in delta_gen_power_mw.items()
        if math.isfinite(_safe_float(delta_mw))
    ]
    if not finite_items:
        return {"gen_id": "", "delta_mw": math.nan, "emission_factor": math.nan}

    gen_id, delta_mw = max(finite_items, key=lambda item: abs(item[1]))
    return {
        "gen_id": gen_id,
        "delta_mw": delta_mw,
        "emission_factor": float(gen_emission_factors.get(gen_id, 600.0)),
    }


def fill_mode_columns(row: dict[str, Any], prefix: str, mode_data: dict[str, Any], bus_index: int) -> None:
    opf_result = mode_data["opf_result"]
    mef_result = mode_data["mef_result"]
    metrics = mode_data["metrics"]
    main_plus = mode_data["main_plus"]
    main_minus = mode_data["main_minus"]

    row[f"{prefix}_opf_success"] = opf_result.success
    row[f"{prefix}_opf_message"] = opf_result.message
    row[f"{prefix}_lmp"] = _nan_if_missing(opf_result.lmp_by_bus.get(bus_index, math.nan))
    row[f"{prefix}_total_generation_cost"] = _nan_if_missing(opf_result.total_generation_cost)
    row[f"{prefix}_total_load_mw"] = _nan_if_missing(opf_result.total_load_mw)
    row[f"{prefix}_total_generation_mw"] = _nan_if_missing(opf_result.total_generation_mw)
    row[f"{prefix}_network_loss_mw"] = _nan_if_missing(opf_result.network_loss_mw)
    row[f"{prefix}_total_emission_kg"] = _nan_if_missing(mode_data["total_emission_kg"])
    row[f"{prefix}_mef_success"] = mef_result.success
    row[f"{prefix}_mef_plus_kg_per_mwh"] = _nan_if_missing(mef_result.mef_plus_kg_per_mwh)
    row[f"{prefix}_mef_minus_kg_per_mwh"] = _nan_if_missing(mef_result.mef_minus_kg_per_mwh)
    row[f"{prefix}_main_plus_marginal_gen"] = main_plus["gen_id"]
    row[f"{prefix}_main_plus_delta_mw"] = _nan_if_missing(main_plus["delta_mw"])
    row[f"{prefix}_main_plus_ef"] = _nan_if_missing(main_plus["emission_factor"])
    row[f"{prefix}_main_minus_marginal_gen"] = main_minus["gen_id"]
    row[f"{prefix}_main_minus_delta_mw"] = _nan_if_missing(main_minus["delta_mw"])
    row[f"{prefix}_main_minus_ef"] = _nan_if_missing(main_minus["emission_factor"])
    row[f"{prefix}_min_voltage_pu"] = _nan_if_missing(metrics.min_voltage_pu)
    row[f"{prefix}_max_voltage_pu"] = _nan_if_missing(metrics.max_voltage_pu)
    row[f"{prefix}_max_line_loading_percent"] = _nan_if_missing(metrics.max_line_loading_percent)
    row[f"{prefix}_voltage_violation_count"] = metrics.voltage_violation_count
    row[f"{prefix}_line_overload_count"] = metrics.line_overload_count
    row[f"{prefix}_grid_security_penalty"] = _nan_if_missing(metrics.grid_security_penalty)


def build_diagnosis(mode_results: dict[str, dict[str, Any]]) -> str:
    messages: list[str] = []
    dc = mode_results.get("dc")
    ac = mode_results.get("ac")

    for mode, data in mode_results.items():
        if not data["opf_result"].success:
            messages.append(f"{mode.upper()} OPF failed")
        if not data["mef_result"].success:
            messages.append(f"{mode.upper()} MEF failed")

    if dc and ac:
        dc_plus = dc["main_plus"]["gen_id"]
        ac_plus = ac["main_plus"]["gen_id"]
        dc_minus = dc["main_minus"]["gen_id"]
        ac_minus = ac["main_minus"]["gen_id"]
        if dc_plus != ac_plus or dc_minus != ac_minus:
            messages.append("DC/AC marginal generator differs")
        ac_loss = _safe_float(ac["opf_result"].network_loss_mw)
        if math.isfinite(ac_loss) and abs(ac_loss) > 1e-3:
            messages.append("AC network loss is non-trivial")
    messages.append("MEF is sensitive to generator emission factors")

    return "; ".join(_unique(messages))


def serialize_opf_result(opf_result, total_emission_kg: float) -> dict[str, Any]:
    return normalize_json(
        {
            "success": opf_result.success,
            "mode": opf_result.mode,
            "message": opf_result.message,
            "total_generation_cost": opf_result.total_generation_cost,
            "total_emission_kg": total_emission_kg,
            "lmp_by_bus": opf_result.lmp_by_bus,
            "gen_power_mw": opf_result.gen_power_mw,
            "bus_voltage_pu": opf_result.bus_voltage_pu,
            "line_loading_percent": opf_result.line_loading_percent,
            "total_load_mw": opf_result.total_load_mw,
            "total_generation_mw": opf_result.total_generation_mw,
            "network_loss_mw": opf_result.network_loss_mw,
        }
    )


def serialize_mef_result(mef_result) -> dict[str, Any]:
    return normalize_json(
        {
            "success": mef_result.success,
            "mode": mef_result.mode,
            "bus_id": mef_result.bus_id,
            "delta_p_mw": mef_result.delta_p_mw,
            "mef_plus_kg_per_mwh": mef_result.mef_plus_kg_per_mwh,
            "mef_minus_kg_per_mwh": mef_result.mef_minus_kg_per_mwh,
            "base_emission_kg": mef_result.base_emission_kg,
            "plus_emission_kg": mef_result.plus_emission_kg,
            "minus_emission_kg": mef_result.minus_emission_kg,
            "base_gen_power_mw": mef_result.base_gen_power_mw,
            "plus_gen_power_mw": mef_result.plus_gen_power_mw,
            "minus_gen_power_mw": mef_result.minus_gen_power_mw,
            "delta_gen_power_plus_mw": mef_result.delta_gen_power_plus_mw,
            "delta_gen_power_minus_mw": mef_result.delta_gen_power_minus_mw,
            "base_total_generation_mw": mef_result.base_total_generation_mw,
            "plus_total_generation_mw": mef_result.plus_total_generation_mw,
            "minus_total_generation_mw": mef_result.minus_total_generation_mw,
            "base_network_loss_mw": mef_result.base_network_loss_mw,
            "plus_network_loss_mw": mef_result.plus_network_loss_mw,
            "minus_network_loss_mw": mef_result.minus_network_loss_mw,
            "message": mef_result.message,
        }
    )


def serialize_grid_metrics(metrics) -> dict[str, Any]:
    return normalize_json(
        {
            "opf_success": metrics.opf_success,
            "min_voltage_pu": metrics.min_voltage_pu,
            "max_voltage_pu": metrics.max_voltage_pu,
            "max_line_loading_percent": metrics.max_line_loading_percent,
            "voltage_violation_count": metrics.voltage_violation_count,
            "line_overload_count": metrics.line_overload_count,
            "opf_infeasible_flag": metrics.opf_infeasible_flag,
            "grid_security_penalty": metrics.grid_security_penalty,
        }
    )


def normalize_json(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): normalize_json(item) for key, item in value.items()}
    if isinstance(value, list):
        return [normalize_json(item) for item in value]
    if isinstance(value, tuple):
        return [normalize_json(item) for item in value]
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    return number if math.isfinite(number) else None


def save_summary_csv(csv_path: Path, rows: list[dict[str, Any]]) -> None:
    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in CSV_FIELDS})


def save_detail_json(json_path: Path, payload: dict[str, Any]) -> None:
    with json_path.open("w", encoding="utf-8") as file:
        json.dump(normalize_json(payload), file, indent=2, ensure_ascii=False)


def print_rankings(rows: list[dict[str, Any]]) -> None:
    print("\n[RANKINGS]")
    _print_top_rows("AC-MEF plus highest top 3", rows, "ac_mef_plus_kg_per_mwh", reverse=True)
    _print_top_rows("AC-MEF plus lowest top 3", rows, "ac_mef_plus_kg_per_mwh", reverse=False)
    _print_top_rows("AC LMP highest top 3", rows, "ac_lmp", reverse=True)
    _print_top_rows("AC max_line_loading_percent highest top 3", rows, "ac_max_line_loading_percent", reverse=True)

    failed = [
        f"IEEE {row['ieee_bus_number']} idx {row['pandapower_bus_index']}"
        for row in rows
        if row.get("dc_opf_success") is False or row.get("ac_opf_success") is False
    ]
    print(f"OPF failed buses: {', '.join(failed) if failed else 'None'}")


def _print_top_rows(title: str, rows: list[dict[str, Any]], key: str, reverse: bool) -> None:
    ranked = [row for row in rows if _is_finite(row.get(key, math.nan))]
    ranked.sort(key=lambda row: float(row[key]), reverse=reverse)
    print(f"{title}:")
    if not ranked:
        print("  None")
        return
    for row in ranked[:3]:
        print(
            "  "
            f"IEEE bus {row['ieee_bus_number']} "
            f"(idx {row['pandapower_bus_index']}): {float(row[key]):.6f}"
        )


def _base_csv_row(
    case_name: str,
    ieee_bus_number: int,
    bus_index: int,
    idc_load_mw: float,
    delta_p_mw: float,
) -> dict[str, Any]:
    row = {field: "" for field in CSV_FIELDS}
    row.update(
        {
            "case_name": case_name,
            "ieee_bus_number": ieee_bus_number,
            "pandapower_bus_index": bus_index,
            "idc_load_mw": idc_load_mw,
            "delta_p_mw": delta_p_mw,
        }
    )
    return row


def _format_verbose_row(row: dict[str, Any]) -> str:
    return (
        f"  IEEE bus {row['ieee_bus_number']} / idx {row['pandapower_bus_index']} | "
        f"DC OPF={row.get('dc_opf_success', '')} DC MEF={row.get('dc_mef_plus_kg_per_mwh', '')} | "
        f"AC OPF={row.get('ac_opf_success', '')} AC MEF={row.get('ac_mef_plus_kg_per_mwh', '')}"
    )


def _safe_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def _is_finite(value: Any) -> bool:
    number = _safe_float(value)
    return math.isfinite(number)


def _nan_if_missing(value: Any) -> float:
    number = _safe_float(value)
    return number if math.isfinite(number) else math.nan


def _unique(messages: list[str]) -> list[str]:
    seen: set[str] = set()
    unique_messages: list[str] = []
    for message in messages:
        if message not in seen:
            seen.add(message)
            unique_messages.append(message)
    return unique_messages


if __name__ == "__main__":
    main()
