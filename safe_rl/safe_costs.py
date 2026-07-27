"""Normalized safety-cost reporting for the current Safe PPO wrapper.

This module only reads the info dictionary produced by the existing grid
wrapper. It does not mutate the grid model, the IDC environment, or PPO code.
The current active penalty is mutually exclusive: OPF cost on OPF failure,
otherwise minimum-voltage cost. Line and transformer costs are reported but
not active; LMP and MEF costs are placeholders fixed at zero in this version.
"""

from __future__ import annotations

import math
from typing import Any


EPS = 1e-9


def compute_safe_costs(info: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    """Compute active and diagnostic normalized costs from grid ``info``."""

    clip_max = _positive_float(config.get("cost_clip_max", 1.0), default=1.0)
    opf_success = _truthy(info.get("grid_opf_success", info.get("opf_success", False)))

    raw_min_v = _finite_float(
        info.get("grid_min_voltage_pu", info.get("minV", info.get("raw_minV"))),
        default=math.nan,
    )
    raw_max_v = _finite_float(
        info.get("grid_max_voltage_pu", info.get("maxV", info.get("raw_maxV"))),
        default=math.nan,
    )
    raw_max_line = _finite_float(
        info.get("grid_max_line_loading_percent", info.get("maxLine", info.get("raw_maxLine"))),
        default=math.nan,
    )
    raw_max_trafo = _finite_float(
        info.get("grid_max_trafo_loading_percent", info.get("maxTrafo", info.get("raw_maxTrafo"))),
        default=math.nan,
    )
    raw_lmp = _finite_float(
        info.get("grid_lmp", info.get("LMP_bus9", info.get("raw_lmp"))),
        default=math.nan,
    )
    raw_mef = _finite_float(
        info.get("grid_mef_plus", info.get("MEF_bus9", info.get("raw_mef"))),
        default=math.nan,
    )

    c_opf = 0.0 if opf_success else 1.0
    c_voltage = _voltage_cost(raw_min_v, opf_success=opf_success, config=config)
    c_line, line_missing = _thermal_cost(
        raw_max_line,
        config.get("line_soft_limit", 80.0),
        config.get("line_hard_limit", 100.0),
        clip_max,
    )
    c_trafo, trafo_missing = _thermal_cost(
        raw_max_trafo,
        config.get("trafo_soft_limit", 80.0),
        config.get("trafo_hard_limit", 100.0),
        clip_max,
    )

    c_lmp = 0.0
    c_mef = 0.0
    c_total = _clip(c_opf + c_voltage + c_line + c_trafo + c_lmp + c_mef, 0.0, clip_max)

    if not opf_success:
        active_costs = {"opf": c_opf}
        c_active_opf = c_opf
        c_active_voltage = 0.0
    else:
        active_costs = {"voltage": c_voltage}
        c_active_opf = 0.0
        c_active_voltage = c_voltage

    safe_costs = {
        "C_opf": float(_clip(c_opf, 0.0, clip_max)),
        "C_voltage": float(_clip(c_voltage, 0.0, clip_max)),
        "C_line": float(_clip(c_line, 0.0, clip_max)),
        "C_trafo": float(_clip(c_trafo, 0.0, clip_max)),
        "C_lmp": float(_clip(c_lmp, 0.0, clip_max)),
        "C_mef": float(_clip(c_mef, 0.0, clip_max)),
        "C_total": float(c_total),
        "C_active_opf": float(_clip(c_active_opf, 0.0, clip_max)),
        "C_active_voltage": float(_clip(c_active_voltage, 0.0, clip_max)),
        "C_active_total": float(_clip(c_active_opf + c_active_voltage, 0.0, clip_max)),
        "active_costs": active_costs,
        "raw_minV": float(raw_min_v),
        "raw_maxV": float(raw_max_v),
        "raw_maxLine": 0.0 if line_missing else float(raw_max_line),
        "raw_maxTrafo": 0.0 if trafo_missing else float(raw_max_trafo),
        "raw_lmp": float(raw_lmp),
        "raw_mef": float(raw_mef),
        "opf_success": bool(opf_success),
        "line_missing": bool(line_missing),
        "trafo_missing": bool(trafo_missing),
    }
    return safe_costs


def _voltage_cost(raw_min_v: float, opf_success: bool, config: dict[str, Any]) -> float:
    if not opf_success:
        return 1.0
    if not math.isfinite(raw_min_v):
        return 0.0
    soft = _finite_float(config.get("V_soft_min", 0.98), default=0.98)
    hard = _finite_float(config.get("V_hard_min", 0.95), default=0.95)
    denom = max(soft - hard, EPS)
    clip_max = _positive_float(config.get("cost_clip_max", 1.0), default=1.0)
    return _clip(max(0.0, soft - raw_min_v) / denom, 0.0, clip_max)


def _thermal_cost(value: float, soft_limit: Any, hard_limit: Any, clip_max: float) -> tuple[float, bool]:
    if not math.isfinite(value):
        return 0.0, True
    soft = _finite_float(soft_limit, default=80.0)
    hard = _finite_float(hard_limit, default=100.0)
    denom = max(hard - soft, EPS)
    return _clip(max(0.0, value - soft) / denom, 0.0, clip_max), False


def _finite_float(value: Any, default: float = math.nan) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if math.isfinite(number) else default


def _positive_float(value: Any, default: float) -> float:
    number = _finite_float(value, default=default)
    return number if number > 0.0 else default


def _truthy(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y"}
    return bool(value)


def _clip(value: float, low: float, high: float) -> float:
    return min(max(float(value), float(low)), float(high))
