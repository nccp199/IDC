"""Unified DC/AC OPF solver interface for pandapower-backed grid cases."""

from __future__ import annotations

import copy
import math
from typing import Any

from .grid_case import GridCase, OPFResult


def solve_opf(
    grid_case: GridCase,
    mode: str = "dc",
    idc_bus_id: int | None = None,
    idc_load_mw: float = 0.0,
    load_scale: float = 1.0,
) -> OPFResult:
    """Solve an OPF without mutating the original network.

    `idc_bus_id` is a pandapower bus index. Convert IEEE bus numbers with
    `get_bus_index_by_ieee_number()` before calling this function.
    """

    normalized_mode = mode.lower().strip()
    if normalized_mode == "dc":
        return solve_dc_opf(grid_case, idc_bus_id=idc_bus_id, idc_load_mw=idc_load_mw, load_scale=load_scale)
    if normalized_mode == "ac":
        return solve_ac_opf(grid_case, idc_bus_id=idc_bus_id, idc_load_mw=idc_load_mw, load_scale=load_scale)

    return _failed_result(normalized_mode, f"Unsupported OPF mode: {mode!r}. Expected 'dc' or 'ac'.")


def solve_dc_opf(
    grid_case: GridCase,
    idc_bus_id: int | None = None,
    idc_load_mw: float = 0.0,
    load_scale: float = 1.0,
) -> OPFResult:
    return _solve_pandapower_opf(
        grid_case=grid_case,
        mode="dc",
        idc_bus_id=idc_bus_id,
        idc_load_mw=idc_load_mw,
        load_scale=load_scale,
    )


def solve_ac_opf(
    grid_case: GridCase,
    idc_bus_id: int | None = None,
    idc_load_mw: float = 0.0,
    load_scale: float = 1.0,
) -> OPFResult:
    return _solve_pandapower_opf(
        grid_case=grid_case,
        mode="ac",
        idc_bus_id=idc_bus_id,
        idc_load_mw=idc_load_mw,
        load_scale=load_scale,
    )


def _solve_pandapower_opf(
    grid_case: GridCase,
    mode: str,
    idc_bus_id: int | None,
    idc_load_mw: float,
    load_scale: float,
) -> OPFResult:
    try:
        import pandapower as pp
    except ImportError as exc:
        return _failed_result(
            mode,
            "pandapower is required for OPF solving. Install it with `pip install pandapower`.",
            raw_exception=exc,
        )

    try:
        net = copy.deepcopy(grid_case.raw_network)
        _scale_loads(net, load_scale)
        _apply_idc_load(pp, net, idc_bus_id, idc_load_mw)
    except Exception as exc:
        return _failed_result(mode, f"Failed to prepare pandapower network: {exc}", raw_exception=exc)

    try:
        if mode == "dc":
            _run_pp_function(pp.rundcopp, net)
        else:
            _run_pp_function(pp.runopp, net)
    except Exception as exc:
        try:
            return _extract_opf_result(net, mode, False, f"{mode.upper()} OPF failed: {exc}")
        except Exception:
            return _failed_result(mode, f"{mode.upper()} OPF failed: {exc}", raw_exception=exc)

    converged = bool(getattr(net, "OPF_converged", False))
    if converged:
        message = f"{mode.upper()} OPF converged."
    else:
        message = f"{mode.upper()} OPF finished but did not report convergence."
    return _extract_opf_result(net, mode, converged, message)


def _run_pp_function(function: Any, net: Any) -> None:
    try:
        function(net, verbose=False, numba=False)
    except TypeError as exc:
        message = str(exc)
        if "numba" in message:
            try:
                function(net, verbose=False)
            except TypeError as verbose_exc:
                if "verbose" not in str(verbose_exc):
                    raise
                function(net)
            return
        if "verbose" in message:
            try:
                function(net, numba=False)
            except TypeError as numba_exc:
                if "numba" not in str(numba_exc):
                    raise
                function(net)
            return
        raise


def _scale_loads(net: Any, load_scale: float) -> None:
    scale = float(load_scale)
    if scale < 0.0:
        raise ValueError(f"load_scale must be non-negative, got {load_scale!r}")

    load_table = getattr(net, "load", None)
    if load_table is None or len(load_table) == 0:
        return

    if "scaling" in load_table.columns:
        load_table.loc[:, "scaling"] = load_table["scaling"].astype(float) * scale
        return

    for column in ("p_mw", "q_mvar"):
        if column in load_table.columns:
            load_table.loc[:, column] = load_table[column].astype(float) * scale


def _apply_idc_load(pp: Any, net: Any, idc_bus_id: int | None, idc_load_mw: float) -> None:
    if idc_bus_id is None or abs(float(idc_load_mw)) < 1e-12:
        return

    bus_index = _resolve_bus_index(net, idc_bus_id)
    try:
        pp.create_load(
            net,
            bus=bus_index,
            p_mw=float(idc_load_mw),
            q_mvar=0.0,
            name="IDC load adjustment",
            controllable=False,
        )
    except TypeError:
        load_idx = pp.create_load(
            net,
            bus=bus_index,
            p_mw=float(idc_load_mw),
            q_mvar=0.0,
            name="IDC load adjustment",
        )
        if "controllable" in net.load.columns:
            net.load.at[load_idx, "controllable"] = False


def _resolve_bus_index(net: Any, bus_id: int) -> int:
    if int(bus_id) in set(int(idx) for idx in net.bus.index.tolist()):
        return int(bus_id)

    valid_ids = [int(idx) for idx in net.bus.index.tolist()]
    preview = valid_ids[:10]
    suffix = "..." if len(valid_ids) > 10 else ""
    raise ValueError(
        f"Unknown pandapower bus index {bus_id!r}. "
        f"Valid pandapower bus indices include {preview}{suffix}."
    )


def _extract_opf_result(net: Any, mode: str, success: bool, message: str) -> OPFResult:
    gen_power = _extract_gen_power_mw(net)
    total_generation = _sum_finite(gen_power.values())
    total_load = _extract_total_load_mw(net)
    network_loss = total_generation - total_load if math.isfinite(total_generation) and math.isfinite(total_load) else math.nan

    return OPFResult(
        success=success,
        mode=mode,
        message=message,
        total_generation_cost=_extract_res_cost(net),
        total_emission_kg=math.nan,
        lmp_by_bus=_extract_bus_column(net, "res_bus", "lam_p"),
        gen_power_mw=gen_power,
        bus_voltage_pu=_extract_bus_column(net, "res_bus", "vm_pu"),
        line_loading_percent=_extract_line_loading_percent(net),
        total_load_mw=total_load,
        total_generation_mw=total_generation,
        network_loss_mw=network_loss,
        raw_result=net,
    )


def _extract_res_cost(net: Any) -> float:
    value = getattr(net, "res_cost", math.nan)
    return _to_float(value)


def _extract_gen_power_mw(net: Any) -> dict[int, float]:
    values: dict[int, float] = {}

    res_ext_grid = getattr(net, "res_ext_grid", None)
    if _has_column(res_ext_grid, "p_mw"):
        for idx, value in res_ext_grid["p_mw"].items():
            values[-(int(idx) + 1)] = _to_float(value)

    res_gen = getattr(net, "res_gen", None)
    if _has_column(res_gen, "p_mw"):
        for idx, value in res_gen["p_mw"].items():
            values[int(idx)] = _to_float(value)

    return values


def _extract_total_load_mw(net: Any) -> float:
    res_load = getattr(net, "res_load", None)
    if _has_column(res_load, "p_mw"):
        return _sum_finite(res_load["p_mw"].tolist())

    load = getattr(net, "load", None)
    if _has_column(load, "p_mw"):
        if "scaling" in load.columns:
            return _sum_finite((load["p_mw"].astype(float) * load["scaling"].astype(float)).tolist())
        return _sum_finite(load["p_mw"].tolist())

    return math.nan


def _extract_bus_column(net: Any, table_name: str, column: str) -> dict[int, float]:
    table = getattr(net, table_name, None)
    if not _has_column(table, column):
        return {}
    return {int(idx): _to_float(value) for idx, value in table[column].items()}


def _extract_line_loading_percent(net: Any) -> dict[int, float]:
    table = getattr(net, "res_line", None)
    if not _has_column(table, "loading_percent"):
        return {}
    return {int(idx): _to_float(value) for idx, value in table["loading_percent"].items()}


def _has_column(table: Any, column: str) -> bool:
    return table is not None and hasattr(table, "columns") and column in table.columns


def _to_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def _sum_finite(values: Any) -> float:
    total = 0.0
    seen = False
    for value in values:
        number = _to_float(value)
        if math.isfinite(number):
            total += number
            seen = True
    return total if seen else math.nan


def _failed_result(mode: str, message: str, raw_exception: Exception | None = None) -> OPFResult:
    if raw_exception is not None:
        message = f"{message} ({type(raw_exception).__name__}: {raw_exception})"
    return OPFResult(success=False, mode=mode, message=message)
