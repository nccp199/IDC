"""Grid-security metric extraction from OPF results."""

from __future__ import annotations

import math

from .grid_case import GridMetricResult, OPFResult


def extract_grid_metrics(
    opf_result: OPFResult,
    voltage_min: float = 0.95,
    voltage_max: float = 1.05,
    line_loading_max: float = 100.0,
    tolerance: float = 1e-6,
) -> GridMetricResult:
    voltages = _finite_values(opf_result.bus_voltage_pu.values())
    line_loadings = _finite_values(opf_result.line_loading_percent.values())

    min_voltage = min(voltages) if voltages else math.nan
    max_voltage = max(voltages) if voltages else math.nan
    max_line_loading = max(line_loadings) if line_loadings else math.nan

    voltage_violation_count, voltage_violation_magnitude = _count_voltage_violations(
        opf_result,
        fallback_min=voltage_min,
        fallback_max=voltage_max,
        tolerance=tolerance,
    )
    line_overload_count, line_overload_magnitude = _count_line_overloads(
        opf_result,
        fallback_limit=line_loading_max,
        tolerance=tolerance,
    )
    opf_infeasible_flag = not opf_result.success

    grid_security_penalty = (
        float(voltage_violation_count)
        + 2.0 * float(line_overload_count)
        + (100.0 if opf_infeasible_flag else 0.0)
    )

    return GridMetricResult(
        opf_success=opf_result.success,
        min_voltage_pu=min_voltage,
        max_voltage_pu=max_voltage,
        max_line_loading_percent=max_line_loading,
        voltage_violation_count=voltage_violation_count,
        line_overload_count=line_overload_count,
        voltage_violation_magnitude=voltage_violation_magnitude,
        line_overload_magnitude=line_overload_magnitude,
        opf_infeasible_flag=opf_infeasible_flag,
        grid_security_penalty=grid_security_penalty,
    )


def _count_voltage_violations(
    opf_result: OPFResult,
    fallback_min: float,
    fallback_max: float,
    tolerance: float,
) -> tuple[int, float]:
    count = 0
    magnitude = 0.0
    for bus_id, value in opf_result.bus_voltage_pu.items():
        voltage = _to_finite_float(value)
        if voltage is None:
            continue

        min_limit = _limit_value(opf_result.bus_voltage_min_pu.get(bus_id), fallback_min)
        max_limit = _limit_value(opf_result.bus_voltage_max_pu.get(bus_id), fallback_max)
        low_violation = min_limit - voltage
        high_violation = voltage - max_limit

        if low_violation > tolerance:
            count += 1
            magnitude += low_violation
        elif high_violation > tolerance:
            count += 1
            magnitude += high_violation

    return count, float(magnitude)


def _count_line_overloads(
    opf_result: OPFResult,
    fallback_limit: float,
    tolerance: float,
) -> tuple[int, float]:
    count = 0
    magnitude = 0.0
    for line_id, value in opf_result.line_loading_percent.items():
        loading = _to_finite_float(value)
        if loading is None:
            continue

        limit = _limit_value(opf_result.line_loading_limit_percent.get(line_id), fallback_limit)
        overload = loading - limit
        if overload > tolerance:
            count += 1
            magnitude += overload

    return count, float(magnitude)


def _limit_value(value: object, fallback: float) -> float:
    number = _to_finite_float(value)
    if number is not None:
        return number
    fallback_number = _to_finite_float(fallback)
    return fallback_number if fallback_number is not None else 0.0


def _finite_values(values: object) -> list[float]:
    finite_values: list[float] = []
    for value in values:
        number = _to_finite_float(value)
        if number is not None:
            finite_values.append(number)
    return finite_values


def _to_finite_float(value: object) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None
