"""Grid-security metric extraction from OPF results."""

from __future__ import annotations

import math

from .grid_case import GridMetricResult, OPFResult


def extract_grid_metrics(
    opf_result: OPFResult,
    voltage_min: float = 0.95,
    voltage_max: float = 1.05,
    line_loading_max: float = 100.0,
) -> GridMetricResult:
    voltages = _finite_values(opf_result.bus_voltage_pu.values())
    line_loadings = _finite_values(opf_result.line_loading_percent.values())

    min_voltage = min(voltages) if voltages else math.nan
    max_voltage = max(voltages) if voltages else math.nan
    max_line_loading = max(line_loadings) if line_loadings else math.nan

    voltage_violation_count = sum(1 for value in voltages if value < voltage_min or value > voltage_max)
    line_overload_count = sum(1 for value in line_loadings if value > line_loading_max)
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
        opf_infeasible_flag=opf_infeasible_flag,
        grid_security_penalty=grid_security_penalty,
    )


def _finite_values(values: object) -> list[float]:
    finite_values: list[float] = []
    for value in values:
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(number):
            finite_values.append(number)
    return finite_values
