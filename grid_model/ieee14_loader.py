"""IEEE 14-bus case loading utilities."""

from __future__ import annotations

import math
import re
from typing import Any

from .grid_case import GridCase


def load_ieee14_case() -> GridCase:
    """Load the IEEE 14-bus test case as a GridCase.

    The raw pandapower network is intentionally preserved for OPF solvers.
    """

    try:
        import pandapower.networks as pn
    except ImportError as exc:
        raise ImportError(
            "pandapower is required to load the IEEE 14-bus grid case. "
            "Install it with `pip install pandapower` in the active Python environment."
        ) from exc

    net = pn.case14()
    harmonize_generator_voltage_limits(net)

    native_ext_grid_ids = [int(idx) for idx in net.ext_grid.index.tolist()] if hasattr(net, "ext_grid") else []
    ext_grid_gen_ids = [-(idx + 1) for idx in native_ext_grid_ids]
    native_gen_ids = [int(idx) for idx in net.gen.index.tolist()] if hasattr(net, "gen") else []
    ieee_to_pp_bus, pp_bus_to_ieee = build_ieee_bus_mappings(net)

    metadata = {
        "source": "pandapower.networks.case14",
        "bus_id_type": "pandapower bus index",
        "ieee_bus_number_source": "net.bus.name, falling back to 1-based bus order",
        "ext_grid_id_encoding": "external grids are encoded as -(ext_grid_index + 1)",
        "native_ext_grid_ids": native_ext_grid_ids,
        "native_gen_ids": native_gen_ids,
    }

    return GridCase(
        name="IEEE14",
        base_mva=float(getattr(net, "sn_mva", 100.0)),
        raw_network=net,
        bus_ids=[int(idx) for idx in net.bus.index.tolist()],
        gen_ids=ext_grid_gen_ids + native_gen_ids,
        branch_ids=[int(idx) for idx in net.line.index.tolist()] if hasattr(net, "line") else [],
        ieee_bus_number_to_bus_index=ieee_to_pp_bus,
        bus_index_to_ieee_bus_number=pp_bus_to_ieee,
        default_load_scale=1.0,
        metadata=metadata,
    )


def harmonize_generator_voltage_limits(net: Any) -> None:
    """Align generator voltage setpoints with bus OPF voltage limits.

    Pandapower's IEEE14 case can contain generator voltage setpoints outside
    the corresponding bus min/max voltage limits. This only harmonizes OPF
    feasibility bounds with those generator voltage setpoints. It does not
    alter the IEEE14 topology, branches, loads, or generator placement.
    """

    gen_table = getattr(net, "gen", None)
    bus_table = getattr(net, "bus", None)
    if gen_table is None or bus_table is None or len(gen_table) == 0:
        return
    required_gen_columns = {"bus", "vm_pu"}
    required_bus_columns = {"min_vm_pu", "max_vm_pu"}
    if not required_gen_columns.issubset(gen_table.columns):
        return
    if not required_bus_columns.issubset(bus_table.columns):
        return

    for _, gen in gen_table.iterrows():
        bus_idx = int(gen["bus"])
        if bus_idx not in bus_table.index:
            continue
        gen_vm_pu = _to_float(gen["vm_pu"])
        if not math.isfinite(gen_vm_pu):
            continue

        max_vm_pu = _to_float(bus_table.at[bus_idx, "max_vm_pu"])
        min_vm_pu = _to_float(bus_table.at[bus_idx, "min_vm_pu"])

        if not math.isfinite(max_vm_pu) or gen_vm_pu > max_vm_pu:
            bus_table.at[bus_idx, "max_vm_pu"] = gen_vm_pu
        if not math.isfinite(min_vm_pu) or gen_vm_pu < min_vm_pu:
            bus_table.at[bus_idx, "min_vm_pu"] = gen_vm_pu


def build_ieee_bus_mappings(net: Any) -> tuple[dict[int, int], dict[int, int]]:
    """Build IEEE bus number <-> pandapower bus index mappings."""

    ieee_to_pp_bus: dict[int, int] = {}
    pp_bus_to_ieee: dict[int, int] = {}

    for order, bus_idx in enumerate(net.bus.index.tolist(), start=1):
        bus_index = int(bus_idx)
        ieee_number = _parse_ieee_bus_number(net.bus.at[bus_idx, "name"] if "name" in net.bus.columns else None)
        if ieee_number is None or ieee_number in ieee_to_pp_bus:
            ieee_number = order
            while ieee_number in ieee_to_pp_bus:
                ieee_number += 1

        ieee_to_pp_bus[int(ieee_number)] = bus_index
        pp_bus_to_ieee[bus_index] = int(ieee_number)

    return ieee_to_pp_bus, pp_bus_to_ieee


def get_bus_index_by_ieee_number(grid_case: GridCase, ieee_bus_number: int) -> int:
    """Return the pandapower bus index for an IEEE original bus number."""

    bus_number = int(ieee_bus_number)
    try:
        return int(grid_case.ieee_bus_number_to_bus_index[bus_number])
    except KeyError as exc:
        valid = sorted(grid_case.ieee_bus_number_to_bus_index)
        raise KeyError(f"Unknown IEEE bus number {bus_number}. Valid IEEE bus numbers: {valid}") from exc


def get_ieee_number_by_bus_index(grid_case: GridCase, bus_index: int) -> int:
    """Return the IEEE original bus number for a pandapower bus index."""

    idx = int(bus_index)
    try:
        return int(grid_case.bus_index_to_ieee_bus_number[idx])
    except KeyError as exc:
        valid = sorted(grid_case.bus_index_to_ieee_bus_number)
        raise KeyError(f"Unknown pandapower bus index {idx}. Valid bus indices: {valid}") from exc


def _parse_ieee_bus_number(value: object) -> int | None:
    if value is None:
        return None
    if isinstance(value, int):
        return int(value)
    if isinstance(value, float) and value.is_integer():
        return int(value)

    text = str(value).strip()
    if not text:
        return None
    if text.isdigit():
        return int(text)

    match = re.search(r"\d+", text)
    if match is None:
        return None
    return int(match.group(0))


def _to_float(value: object) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return math.nan
