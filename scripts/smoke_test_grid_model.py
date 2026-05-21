"""Smoke test for the grid_model IEEE14/OPF/MEF skeleton.

Run with:
    python -m scripts.smoke_test_grid_model
"""

from __future__ import annotations

import math
import sys
from pathlib import Path


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
    get_ieee_number_by_bus_index,
    load_ieee14_case,
    solve_opf,
)


def main() -> None:
    try:
        grid_case = load_ieee14_case()
    except ImportError as exc:
        print("IEEE14 smoke test skipped: pandapower is not available.")
        print(f"Reason: {exc}")
        return

    idc_ieee_bus_number = 9
    idc_bus_idx = get_bus_index_by_ieee_number(grid_case, idc_ieee_bus_number)
    idc_load_mw = 1.0
    delta_p_mw = 0.1
    emission_factors = build_default_gen_emission_factors(grid_case)

    print(f"Loaded {grid_case.name}: {len(grid_case.bus_ids)} buses, {len(grid_case.branch_ids)} branches")
    print(
        "IDC IEEE bus number = "
        f"{idc_ieee_bus_number}; IDC pandapower bus index = {idc_bus_idx}; "
        f"IDC load = {idc_load_mw:.3f} MW; MEF delta = {delta_p_mw:.3f} MW"
    )

    dc_result = solve_opf(grid_case, mode="dc", idc_bus_id=idc_bus_idx, idc_load_mw=idc_load_mw)
    _print_opf_summary("DC OPF", dc_result, emission_factors, grid_case)

    ac_result = solve_opf(grid_case, mode="ac", idc_bus_id=idc_bus_idx, idc_load_mw=idc_load_mw)
    _print_opf_summary("AC OPF", ac_result, emission_factors, grid_case)

    dc_mef = calculate_nodal_mef(
        grid_case,
        bus_id=idc_bus_idx,
        mode="dc",
        delta_p_mw=delta_p_mw,
        gen_emission_factors_kg_per_mwh=emission_factors,
    )
    _print_mef_summary("DC-MEF", dc_mef, emission_factors)

    ac_mef = calculate_nodal_mef(
        grid_case,
        bus_id=idc_bus_idx,
        mode="ac",
        delta_p_mw=delta_p_mw,
        gen_emission_factors_kg_per_mwh=emission_factors,
    )
    _print_mef_summary("AC-MEF", ac_mef, emission_factors)

    _print_mef_diagnostic(dc_mef, ac_mef, emission_factors)


def _print_opf_summary(label: str, result, emission_factors: dict[int, float], grid_case) -> None:
    metrics = extract_grid_metrics(result)
    total_emission, _ = compute_total_emission(result.gen_power_mw, emission_factors)

    print(f"\n[{label}]")
    print(f"success: {result.success}")
    print(f"message: {result.message}")
    if not result.success and label.startswith("AC"):
        print(f"AC OPF failed: {result.message}")
    print(f"total_generation_cost: {_fmt(result.total_generation_cost)}")
    print(f"total_load_mw: {_fmt(result.total_load_mw)}")
    print(f"total_generation_mw: {_fmt(result.total_generation_mw)}")
    print(f"network_loss_mw: {_fmt(result.network_loss_mw)}")
    print(f"lmp_sample: {_sample_lmp(result.lmp_by_bus, grid_case)}")
    print(f"min_voltage_pu: {_fmt(metrics.min_voltage_pu)}")
    print(f"max_line_loading_percent: {_fmt(metrics.max_line_loading_percent)}")
    print(f"total_emission_kg: {_fmt(total_emission)}")
    _print_generator_emission_table(label, result, emission_factors)


def _print_generator_emission_table(label: str, result, emission_factors: dict[int, float]) -> None:
    rows = build_generator_emission_table(result, emission_factors)
    print(f"\n[{label} GENERATOR EMISSION TABLE]")
    if not rows:
        print("No generator dispatch rows available.")
        return

    print("gen_id | power_mw | EF_kg_per_mwh | emission_kg")
    for row in rows:
        print(
            f"{int(row['gen_id']):>6} | "
            f"{_fmt(row['gen_power_mw']):>8} | "
            f"{_fmt(row['emission_factor_kg_per_mwh']):>13} | "
            f"{_fmt(row['gen_emission_kg']):>11}"
        )

    max_power = max(rows, key=lambda row: row["gen_power_mw"])
    max_factor = max(rows, key=lambda row: row["emission_factor_kg_per_mwh"])
    max_emission = max(rows, key=lambda row: row["gen_emission_kg"])
    print(
        "highlights: "
        f"max output gen={int(max_power['gen_id'])}, "
        f"max EF gen={int(max_factor['gen_id'])}, "
        f"max emission gen={int(max_emission['gen_id'])}"
    )


def _print_mef_summary(label: str, result, emission_factors: dict[int, float]) -> None:
    print(f"\n[{label}]")
    print(f"success: {result.success}")
    print(f"message: {result.message}")
    if not result.success and label.startswith("AC"):
        print(f"AC MEF failed: {result.message}")
    print(f"MEF_plus_kg_per_mwh: {_fmt(result.mef_plus_kg_per_mwh)}")
    print(f"MEF_minus_kg_per_mwh: {_fmt(result.mef_minus_kg_per_mwh)}")
    print(f"base_emission_kg: {_fmt(result.base_emission_kg)}")
    print(f"plus_emission_kg: {_fmt(result.plus_emission_kg)}")
    print(f"minus_emission_kg: {_fmt(result.minus_emission_kg)}")
    _print_mef_dispatch_table(label, result, emission_factors)


def _print_mef_dispatch_table(label: str, result, emission_factors: dict[int, float]) -> None:
    print(f"\n[{label} BASE/PLUS/MINUS DISPATCH]")
    gen_ids = sorted(
        set(result.base_gen_power_mw)
        | set(result.plus_gen_power_mw)
        | set(result.minus_gen_power_mw)
        | set(result.delta_gen_power_plus_mw)
        | set(result.delta_gen_power_minus_mw)
    )
    if not gen_ids:
        print("No MEF dispatch diagnostics available.")
        return

    print("gen_id | base_mw | plus_mw | minus_mw | delta_plus_mw | delta_minus_mw | EF")
    for gen_id in gen_ids:
        print(
            f"{gen_id:>6} | "
            f"{_fmt(result.base_gen_power_mw.get(gen_id, 0.0)):>7} | "
            f"{_fmt(result.plus_gen_power_mw.get(gen_id, 0.0)):>7} | "
            f"{_fmt(result.minus_gen_power_mw.get(gen_id, 0.0)):>8} | "
            f"{_fmt(result.delta_gen_power_plus_mw.get(gen_id, 0.0)):>13} | "
            f"{_fmt(result.delta_gen_power_minus_mw.get(gen_id, 0.0)):>14} | "
            f"{_fmt(emission_factors.get(gen_id, 600.0))}"
        )

    plus_gen = _main_marginal_generator(result.delta_gen_power_plus_mw)
    minus_gen = _main_marginal_generator(result.delta_gen_power_minus_mw)
    print(
        "main marginal generators: "
        f"plus={_describe_marginal(plus_gen, emission_factors)}, "
        f"minus={_describe_marginal(minus_gen, emission_factors)}"
    )


def _print_mef_diagnostic(dc_mef, ac_mef, emission_factors: dict[int, float]) -> None:
    print("\n[MEF DIAGNOSTIC]")
    print(f"DC-MEF_plus: {_fmt(dc_mef.mef_plus_kg_per_mwh)}")
    print(f"DC-MEF_minus: {_fmt(dc_mef.mef_minus_kg_per_mwh)}")
    print(f"AC-MEF_plus: {_fmt(ac_mef.mef_plus_kg_per_mwh)}")
    print(f"AC-MEF_minus: {_fmt(ac_mef.mef_minus_kg_per_mwh)}")
    print(f"DC base total emission kg: {_fmt(dc_mef.base_emission_kg)}")
    print(f"AC base total emission kg: {_fmt(ac_mef.base_emission_kg)}")
    print(f"DC base total generation MW: {_fmt(dc_mef.base_total_generation_mw)}")
    print(f"AC base total generation MW: {_fmt(ac_mef.base_total_generation_mw)}")
    print(f"DC base network_loss_mw: {_fmt(dc_mef.base_network_loss_mw)}")
    print(f"AC base network_loss_mw: {_fmt(ac_mef.base_network_loss_mw)}")

    dc_plus_gen = _main_marginal_generator(dc_mef.delta_gen_power_plus_mw)
    ac_plus_gen = _main_marginal_generator(ac_mef.delta_gen_power_plus_mw)
    dc_minus_gen = _main_marginal_generator(dc_mef.delta_gen_power_minus_mw)
    ac_minus_gen = _main_marginal_generator(ac_mef.delta_gen_power_minus_mw)
    print(f"DC main plus marginal gen: {_describe_marginal(dc_plus_gen, emission_factors)}")
    print(f"AC main plus marginal gen: {_describe_marginal(ac_plus_gen, emission_factors)}")
    print(f"DC main minus marginal gen: {_describe_marginal(dc_minus_gen, emission_factors)}")
    print(f"AC main minus marginal gen: {_describe_marginal(ac_minus_gen, emission_factors)}")

    if _gen_id(dc_plus_gen) != _gen_id(ac_plus_gen) or _gen_id(dc_minus_gen) != _gen_id(ac_minus_gen):
        print("diagnosis: DC/AC marginal generator differs.")
    if _abs_finite(ac_mef.base_network_loss_mw) > 1e-3:
        print("diagnosis: AC OPF includes network loss, which may change dispatch and MEF.")
    print("diagnosis: MEF is sensitive to generator emission factors.")


def _main_marginal_generator(delta_by_gen: dict[int, float]) -> tuple[int, float] | None:
    positive_rows = [(gen_id, float(delta)) for gen_id, delta in delta_by_gen.items() if _is_finite(delta) and delta > 1e-9]
    if not positive_rows:
        return None
    return max(positive_rows, key=lambda item: item[1])


def _describe_marginal(marginal: tuple[int, float] | None, emission_factors: dict[int, float]) -> str:
    if marginal is None:
        return "none"
    gen_id, delta = marginal
    return f"gen={gen_id}, delta_mw={_fmt(delta)}, EF={_fmt(emission_factors.get(gen_id, 600.0))}"


def _gen_id(marginal: tuple[int, float] | None) -> int | None:
    if marginal is None:
        return None
    return marginal[0]


def _fmt(value: float) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "nan"
    if not math.isfinite(number):
        return "nan"
    return f"{number:.6f}"


def _sample_lmp(values: dict[int, float], grid_case, limit: int = 3) -> dict[str, str]:
    sample: dict[str, str] = {}
    for bus_idx in sorted(values)[:limit]:
        ieee_bus = get_ieee_number_by_bus_index(grid_case, bus_idx)
        sample[f"ieee_{ieee_bus}/idx_{bus_idx}"] = _fmt(values[bus_idx])
    return sample


def _is_finite(value: float) -> bool:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return False
    return math.isfinite(number)


def _abs_finite(value: float) -> float:
    if not _is_finite(value):
        return 0.0
    return abs(float(value))


if __name__ == "__main__":
    main()
