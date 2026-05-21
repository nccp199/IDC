"""Nodal marginal emission factor calculation."""

from __future__ import annotations

import math

from .emission_model import build_default_gen_emission_factors, compute_total_emission
from .grid_case import GridCase, MEFResult
from .opf_solver import solve_opf


def calculate_nodal_mef(
    grid_case: GridCase,
    bus_id: int,
    mode: str = "dc",
    delta_p_mw: float = 0.1,
    load_scale: float = 1.0,
    base_idc_load_mw: float = 0.0,
    gen_emission_factors_kg_per_mwh: dict[int, float] | None = None,
) -> MEFResult:
    """Calculate plus/minus nodal MEF around the base OPF dispatch."""

    delta = float(delta_p_mw)
    if delta <= 0.0:
        return MEFResult(
            success=False,
            mode=mode,
            bus_id=int(bus_id),
            delta_p_mw=delta,
            message=f"delta_p_mw must be positive, got {delta_p_mw!r}.",
        )

    if gen_emission_factors_kg_per_mwh is None:
        factors = build_default_gen_emission_factors(grid_case)
    else:
        factors = gen_emission_factors_kg_per_mwh

    base_idc_load = float(base_idc_load_mw)
    base_result = solve_opf(
        grid_case,
        mode=mode,
        idc_bus_id=bus_id,
        idc_load_mw=base_idc_load,
        load_scale=load_scale,
    )
    if not base_result.success:
        return _failed_mef(mode, bus_id, delta, f"base OPF failed: {base_result.message}")
    base_gen_power = dict(base_result.gen_power_mw)
    base_emission, _ = compute_total_emission(base_result.gen_power_mw, factors)

    plus_result = solve_opf(
        grid_case,
        mode=mode,
        idc_bus_id=bus_id,
        idc_load_mw=base_idc_load + delta,
        load_scale=load_scale,
    )
    if not plus_result.success:
        return _failed_mef(
            mode,
            bus_id,
            delta,
            f"plus OPF failed: {plus_result.message}",
            base_emission_kg=base_emission,
            base_gen_power_mw=base_gen_power,
            base_total_generation_mw=base_result.total_generation_mw,
            base_network_loss_mw=base_result.network_loss_mw,
        )
    plus_gen_power = dict(plus_result.gen_power_mw)
    plus_emission, _ = compute_total_emission(plus_result.gen_power_mw, factors)
    delta_gen_power_plus = _subtract_power_dicts(plus_gen_power, base_gen_power)

    minus_result = solve_opf(
        grid_case,
        mode=mode,
        idc_bus_id=bus_id,
        idc_load_mw=base_idc_load - delta,
        load_scale=load_scale,
    )
    if not minus_result.success:
        return _failed_mef(
            mode,
            bus_id,
            delta,
            f"minus OPF failed: {minus_result.message}",
            base_emission_kg=base_emission,
            plus_emission_kg=plus_emission,
            base_gen_power_mw=base_gen_power,
            plus_gen_power_mw=plus_gen_power,
            delta_gen_power_plus_mw=delta_gen_power_plus,
            base_total_generation_mw=base_result.total_generation_mw,
            plus_total_generation_mw=plus_result.total_generation_mw,
            base_network_loss_mw=base_result.network_loss_mw,
            plus_network_loss_mw=plus_result.network_loss_mw,
        )
    minus_gen_power = dict(minus_result.gen_power_mw)
    minus_emission, _ = compute_total_emission(minus_result.gen_power_mw, factors)
    delta_gen_power_minus = _subtract_power_dicts(base_gen_power, minus_gen_power)

    mef_plus = (plus_emission - base_emission) / delta
    mef_minus = (base_emission - minus_emission) / delta

    return MEFResult(
        success=math.isfinite(mef_plus) and math.isfinite(mef_minus),
        mode=mode,
        bus_id=int(bus_id),
        delta_p_mw=delta,
        mef_plus_kg_per_mwh=mef_plus,
        mef_minus_kg_per_mwh=mef_minus,
        base_emission_kg=base_emission,
        plus_emission_kg=plus_emission,
        minus_emission_kg=minus_emission,
        base_gen_power_mw=base_gen_power,
        plus_gen_power_mw=plus_gen_power,
        minus_gen_power_mw=minus_gen_power,
        delta_gen_power_plus_mw=delta_gen_power_plus,
        delta_gen_power_minus_mw=delta_gen_power_minus,
        base_total_generation_mw=base_result.total_generation_mw,
        plus_total_generation_mw=plus_result.total_generation_mw,
        minus_total_generation_mw=minus_result.total_generation_mw,
        base_network_loss_mw=base_result.network_loss_mw,
        plus_network_loss_mw=plus_result.network_loss_mw,
        minus_network_loss_mw=minus_result.network_loss_mw,
        message="MEF calculation completed.",
    )


def _failed_mef(
    mode: str,
    bus_id: int,
    delta_p_mw: float,
    message: str,
    base_emission_kg: float = math.nan,
    plus_emission_kg: float = math.nan,
    minus_emission_kg: float = math.nan,
    base_gen_power_mw: dict[int, float] | None = None,
    plus_gen_power_mw: dict[int, float] | None = None,
    minus_gen_power_mw: dict[int, float] | None = None,
    delta_gen_power_plus_mw: dict[int, float] | None = None,
    delta_gen_power_minus_mw: dict[int, float] | None = None,
    base_total_generation_mw: float = math.nan,
    plus_total_generation_mw: float = math.nan,
    minus_total_generation_mw: float = math.nan,
    base_network_loss_mw: float = math.nan,
    plus_network_loss_mw: float = math.nan,
    minus_network_loss_mw: float = math.nan,
) -> MEFResult:
    return MEFResult(
        success=False,
        mode=mode,
        bus_id=int(bus_id),
        delta_p_mw=float(delta_p_mw),
        base_emission_kg=base_emission_kg,
        plus_emission_kg=plus_emission_kg,
        minus_emission_kg=minus_emission_kg,
        base_gen_power_mw=base_gen_power_mw or {},
        plus_gen_power_mw=plus_gen_power_mw or {},
        minus_gen_power_mw=minus_gen_power_mw or {},
        delta_gen_power_plus_mw=delta_gen_power_plus_mw or {},
        delta_gen_power_minus_mw=delta_gen_power_minus_mw or {},
        base_total_generation_mw=base_total_generation_mw,
        plus_total_generation_mw=plus_total_generation_mw,
        minus_total_generation_mw=minus_total_generation_mw,
        base_network_loss_mw=base_network_loss_mw,
        plus_network_loss_mw=plus_network_loss_mw,
        minus_network_loss_mw=minus_network_loss_mw,
        message=message,
    )


def _subtract_power_dicts(left: dict[int, float], right: dict[int, float]) -> dict[int, float]:
    all_gen_ids = sorted(set(left) | set(right))
    return {gen_id: _safe_float(left.get(gen_id, 0.0)) - _safe_float(right.get(gen_id, 0.0)) for gen_id in all_gen_ids}


def _safe_float(value: float) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.0
    if not math.isfinite(number):
        return 0.0
    return number
