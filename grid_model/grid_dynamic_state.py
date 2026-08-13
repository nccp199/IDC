"""Current, causal IEEE-14 bus operating state for centralized critics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

from .grid_case import GridCase, OPFResult


GRID_BUS_COUNT = 14
GRID_BUS_DYNAMIC_FEATURE_NAMES = (
    "voltage_deviation",
    "net_active_power",
    "net_reactive_power",
    "nodal_lmp",
    "incident_branch_max_loading",
)
GRID_BUS_DYNAMIC_FEATURE_DIM = len(GRID_BUS_DYNAMIC_FEATURE_NAMES)
GRID_BUS_DYNAMIC_STATE_DIM = GRID_BUS_COUNT * GRID_BUS_DYNAMIC_FEATURE_DIM

# Fixed physical references. P/Q use the IEEE-14 system base MVA, LMP reuses
# GridCoupledEnv's existing grid_lmp_ref, and branch loading is percent-rated.
VOLTAGE_CENTER_PU = 1.0
VOLTAGE_DEVIATION_REF_PU = 0.10
BRANCH_LOADING_REF_PERCENT = 100.0
FEATURE_CLIP_BOUNDS = np.asarray(
    ((-2.0, 2.0), (-5.0, 5.0), (-5.0, 5.0), (-10.0, 10.0), (0.0, 5.0)),
    dtype=np.float64,
)


@dataclass(frozen=True)
class GridBusDynamicPayload:
    normalized_state: np.ndarray
    bus_vm_pu: np.ndarray
    bus_p_mw: np.ndarray
    bus_q_mvar: np.ndarray
    bus_lmp: np.ndarray
    bus_lam_q: np.ndarray
    incident_branch_max_loading_percent: np.ndarray
    line_loading_percent: np.ndarray
    transformer_loading_percent: np.ndarray
    clip_count_by_feature: np.ndarray
    missing_value_count: int
    fallback_used: bool


def normalization_metadata(*, base_mva: float, lmp_ref: float) -> dict[str, Any]:
    return {
        "feature_order": list(GRID_BUS_DYNAMIC_FEATURE_NAMES),
        "bus_count": GRID_BUS_COUNT,
        "voltage_center_pu": VOLTAGE_CENTER_PU,
        "voltage_deviation_ref_pu": VOLTAGE_DEVIATION_REF_PU,
        "active_power_ref_mw": float(base_mva),
        "reactive_power_ref_mvar": float(base_mva),
        "lmp_ref": float(lmp_ref),
        "branch_loading_ref_percent": BRANCH_LOADING_REF_PERCENT,
        "clip_bounds": FEATURE_CLIP_BOUNDS.tolist(),
        "opf_failure_fallback": "vm_pu=1; P=Q=LMP=incident_loading=0",
    }


def _ordered_values(
    values: Mapping[int, float], ids: tuple[int, ...], fallback: float
) -> tuple[np.ndarray, int]:
    result = np.full(len(ids), float(fallback), dtype=np.float64)
    missing = 0
    for position, item_id in enumerate(ids):
        try:
            number = float(values[item_id])
        except (KeyError, TypeError, ValueError):
            missing += 1
            continue
        if np.isfinite(number):
            result[position] = number
        else:
            missing += 1
    return result, missing


def _incident_loading(
    grid_case: GridCase,
    line_loading: np.ndarray,
    transformer_loading: np.ndarray,
) -> np.ndarray:
    net = grid_case.raw_network
    incident = np.zeros(GRID_BUS_COUNT, dtype=np.float64)
    for position, (_, row) in enumerate(net.line.iterrows()):
        loading = float(max(line_loading[position], 0.0))
        for bus in (int(row.from_bus), int(row.to_bus)):
            incident[bus] = max(incident[bus], loading)
    for position, (_, row) in enumerate(net.trafo.iterrows()):
        loading = float(max(transformer_loading[position], 0.0))
        for bus in (int(row.hv_bus), int(row.lv_bus)):
            incident[bus] = max(incident[bus], loading)
    return incident


def build_grid_bus_dynamic_payload(
    grid_case: GridCase,
    opf_result: OPFResult,
    *,
    lmp_ref: float,
) -> GridBusDynamicPayload:
    """Build normalized current-step state; failed OPF maps to neutral zeros."""
    bus_ids = tuple(int(value) for value in grid_case.bus_ids)
    if bus_ids != tuple(range(GRID_BUS_COUNT)):
        raise ValueError(f"Expected canonical IEEE-14 bus ids 0..13, got {bus_ids!r}.")
    net = grid_case.raw_network
    line_ids = tuple(int(value) for value in net.line.index.tolist())
    trafo_ids = tuple(int(value) for value in net.trafo.index.tolist())

    if not bool(opf_result.success):
        vm = np.ones(GRID_BUS_COUNT, dtype=np.float64)
        p_mw = np.zeros(GRID_BUS_COUNT, dtype=np.float64)
        q_mvar = np.zeros(GRID_BUS_COUNT, dtype=np.float64)
        lmp = np.zeros(GRID_BUS_COUNT, dtype=np.float64)
        lam_q = np.zeros(GRID_BUS_COUNT, dtype=np.float64)
        line = np.zeros(len(line_ids), dtype=np.float64)
        trafo = np.zeros(len(trafo_ids), dtype=np.float64)
        missing = GRID_BUS_COUNT * 5 + len(line_ids) + len(trafo_ids)
        fallback_used = True
    else:
        vm, missing_vm = _ordered_values(opf_result.bus_voltage_pu, bus_ids, 1.0)
        p_mw, missing_p = _ordered_values(opf_result.bus_active_power_mw, bus_ids, 0.0)
        q_mvar, missing_q = _ordered_values(
            opf_result.bus_reactive_power_mvar, bus_ids, 0.0
        )
        lmp, missing_lmp = _ordered_values(opf_result.lmp_by_bus, bus_ids, 0.0)
        lam_q, missing_lam_q = _ordered_values(
            opf_result.reactive_lmp_by_bus, bus_ids, 0.0
        )
        line, missing_line = _ordered_values(
            opf_result.line_loading_percent, line_ids, 0.0
        )
        trafo, missing_trafo = _ordered_values(
            opf_result.transformer_loading_percent, trafo_ids, 0.0
        )
        missing = (
            missing_vm
            + missing_p
            + missing_q
            + missing_lmp
            + missing_lam_q
            + missing_line
            + missing_trafo
        )
        fallback_used = missing > 0

    incident = _incident_loading(grid_case, line, trafo)
    base_mva = max(float(grid_case.base_mva), 1e-9)
    lmp_scale = max(float(lmp_ref), 1e-9)
    unbounded = np.stack(
        (
            (vm - VOLTAGE_CENTER_PU) / VOLTAGE_DEVIATION_REF_PU,
            p_mw / base_mva,
            q_mvar / base_mva,
            lmp / lmp_scale,
            incident / BRANCH_LOADING_REF_PERCENT,
        ),
        axis=-1,
    )
    lower = FEATURE_CLIP_BOUNDS[:, 0]
    upper = FEATURE_CLIP_BOUNDS[:, 1]
    clipped = np.clip(unbounded, lower, upper)
    clip_counts = np.count_nonzero(np.abs(clipped - unbounded) > 1e-12, axis=0)
    if not np.isfinite(clipped).all():
        raise FloatingPointError("Normalized grid bus dynamic state contains NaN/inf.")
    return GridBusDynamicPayload(
        normalized_state=clipped.astype(np.float32).reshape(-1),
        bus_vm_pu=vm,
        bus_p_mw=p_mw,
        bus_q_mvar=q_mvar,
        bus_lmp=lmp,
        bus_lam_q=lam_q,
        incident_branch_max_loading_percent=incident,
        line_loading_percent=line,
        transformer_loading_percent=trafo,
        clip_count_by_feature=clip_counts.astype(np.int64),
        missing_value_count=int(missing),
        fallback_used=bool(fallback_used),
    )
