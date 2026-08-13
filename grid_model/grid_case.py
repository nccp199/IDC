"""Shared data structures for grid modeling results."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class GridCase:
    name: str
    base_mva: float
    raw_network: Any
    bus_ids: list[int]
    gen_ids: list[int]
    branch_ids: list[int]
    ieee_bus_number_to_bus_index: dict[int, int] = field(default_factory=dict)
    bus_index_to_ieee_bus_number: dict[int, int] = field(default_factory=dict)
    default_load_scale: float = 1.0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class OPFResult:
    success: bool
    mode: str
    message: str
    total_generation_cost: float = math.nan
    total_emission_kg: float = math.nan
    lmp_by_bus: dict[int, float] = field(default_factory=dict)
    reactive_lmp_by_bus: dict[int, float] = field(default_factory=dict)
    bus_active_power_mw: dict[int, float] = field(default_factory=dict)
    bus_reactive_power_mvar: dict[int, float] = field(default_factory=dict)
    gen_power_mw: dict[int, float] = field(default_factory=dict)
    bus_voltage_pu: dict[int, float] = field(default_factory=dict)
    bus_voltage_min_pu: dict[int, float] = field(default_factory=dict)
    bus_voltage_max_pu: dict[int, float] = field(default_factory=dict)
    line_loading_percent: dict[int, float] = field(default_factory=dict)
    line_loading_limit_percent: dict[int, float] = field(default_factory=dict)
    transformer_loading_percent: dict[int, float] = field(default_factory=dict)
    transformer_loading_limit_percent: dict[int, float] = field(default_factory=dict)
    total_load_mw: float = math.nan
    total_generation_mw: float = math.nan
    network_loss_mw: float = math.nan
    raw_result: Any | None = None


@dataclass(slots=True)
class MEFResult:
    success: bool
    mode: str
    bus_id: int
    delta_p_mw: float
    mef_plus_kg_per_mwh: float = math.nan
    mef_minus_kg_per_mwh: float = math.nan
    base_emission_kg: float = math.nan
    plus_emission_kg: float = math.nan
    minus_emission_kg: float = math.nan
    base_gen_power_mw: dict[int, float] = field(default_factory=dict)
    plus_gen_power_mw: dict[int, float] = field(default_factory=dict)
    minus_gen_power_mw: dict[int, float] = field(default_factory=dict)
    delta_gen_power_plus_mw: dict[int, float] = field(default_factory=dict)
    delta_gen_power_minus_mw: dict[int, float] = field(default_factory=dict)
    base_total_generation_mw: float = math.nan
    plus_total_generation_mw: float = math.nan
    minus_total_generation_mw: float = math.nan
    base_network_loss_mw: float = math.nan
    plus_network_loss_mw: float = math.nan
    minus_network_loss_mw: float = math.nan
    message: str = ""


@dataclass(slots=True)
class GridMetricResult:
    opf_success: bool
    min_voltage_pu: float = math.nan
    max_voltage_pu: float = math.nan
    max_line_loading_percent: float = math.nan
    max_transformer_loading_percent: float = math.nan
    voltage_violation_count: int = 0
    line_overload_count: int = 0
    transformer_overload_count: int = 0
    voltage_violation_magnitude: float = 0.0
    line_overload_magnitude: float = 0.0
    transformer_overload_magnitude: float = 0.0
    opf_infeasible_flag: bool = False
    grid_security_penalty: float = 0.0
