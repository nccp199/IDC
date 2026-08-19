"""Small in-memory cache for grid OPF and MEF results.

The cache is intentionally environment-agnostic. Each GridCoupledEnv owns its
own instance, so SubprocVecEnv workers naturally keep independent caches.
"""

from __future__ import annotations

import math
from collections import OrderedDict
from typing import Any, Hashable

from .grid_case import MEFResult, OPFResult


DEFAULT_GRID_CACHE_CONFIG = {
    "enable_grid_cache": True,
    "cache_opf": True,
    "cache_mef": True,
    "cache_load_bin_mw": 0.1,
    "cache_mef_load_bin_mw": 0.01,
    "cache_load_scale_bin": 0.005,
    "cache_max_size": 50000,
    "cache_clear_on_reset": False,
    "cache_scope": "per_worker",
    "cache_failed_results": False,
    "cache_verbose": False,
}


class GridResultCache:
    """LRU cache for lightweight OPFResult and MEFResult dataclasses."""

    def __init__(self, cache_config: dict[str, Any] | None = None):
        supplied_config = dict(cache_config or {})
        self.config = {**DEFAULT_GRID_CACHE_CONFIG, **supplied_config}
        self.enabled = bool(self.config.get("enable_grid_cache", True))
        self.cache_opf_enabled = bool(self.config.get("cache_opf", True))
        self.cache_mef_enabled = bool(self.config.get("cache_mef", True))
        self.cache_load_bin_mw = _positive_float(self.config.get("cache_load_bin_mw", 0.1), default=0.1)
        mef_load_bin = (
            self.config["cache_mef_load_bin_mw"]
            if cache_config is None or "cache_mef_load_bin_mw" in supplied_config
            else self.cache_load_bin_mw
        )
        self.cache_mef_load_bin_mw = _positive_float(
            mef_load_bin, default=self.cache_load_bin_mw
        )
        self.config["cache_mef_load_bin_mw"] = self.cache_mef_load_bin_mw
        self.cache_load_scale_bin = _positive_float(self.config.get("cache_load_scale_bin", 0.005), default=0.005)
        self.cache_max_size = max(int(self.config.get("cache_max_size", 50000)), 0)
        self.cache_failed_results = bool(self.config.get("cache_failed_results", False))
        self.cache_verbose = bool(self.config.get("cache_verbose", False))
        self.float_digits = int(self.config.get("cache_float_digits", 6))

        self._opf_cache: OrderedDict[Hashable, OPFResult] = OrderedDict()
        self._mef_cache: OrderedDict[Hashable, MEFResult] = OrderedDict()
        self._opf_hit_count = 0
        self._opf_miss_count = 0
        self._mef_hit_count = 0
        self._mef_miss_count = 0

    def make_opf_key(
        self,
        opf_mode: str,
        idc_bus: int,
        hour: int,
        grid_load_scale: float,
        idc_load_mw: float,
    ) -> tuple[Any, ...]:
        """Build a stable OPF cache key from the grid state."""
        return (
            "opf",
            str(opf_mode).strip().lower(),
            int(idc_bus),
            int(hour),
            self._bin_load_scale(grid_load_scale),
            self._bin_idc_load_mw(idc_load_mw),
        )

    def make_mef_key(
        self,
        opf_mode: str,
        idc_bus: int,
        hour: int,
        grid_load_scale: float,
        idc_load_mw: float,
        delta_p_mw: float,
    ) -> tuple[Any, ...]:
        """Build a stable MEF cache key from the grid state and perturbation."""
        rounded_delta = self._round_float(delta_p_mw)
        return (
            "mef",
            str(opf_mode).strip().lower(),
            int(idc_bus),
            int(hour),
            self._bin_load_scale(grid_load_scale),
            self._mef_load_key(idc_load_mw, rounded_delta),
            rounded_delta,
        )

    def get_opf(self, key: Hashable) -> OPFResult | None:
        if not self.enabled or not self.cache_opf_enabled:
            return None
        result = self._opf_cache.get(key)
        if result is None:
            self._opf_miss_count += 1
            self._log("OPF miss", key)
            return None
        self._opf_hit_count += 1
        self._opf_cache.move_to_end(key)
        self._log("OPF hit", key)
        return _copy_opf_result(result)

    def put_opf(self, key: Hashable, opf_result: OPFResult) -> None:
        if not self.enabled or not self.cache_opf_enabled or self.cache_max_size <= 0:
            return
        if not self.cache_failed_results and not bool(getattr(opf_result, "success", False)):
            return
        self._opf_cache[key] = _copy_opf_result(opf_result)
        self._opf_cache.move_to_end(key)
        self._trim(self._opf_cache)
        self._log("OPF put", key)

    def get_mef(self, key: Hashable) -> MEFResult | None:
        if not self.enabled or not self.cache_mef_enabled:
            return None
        result = self._mef_cache.get(key)
        if result is None:
            self._mef_miss_count += 1
            self._log("MEF miss", key)
            return None
        self._mef_hit_count += 1
        self._mef_cache.move_to_end(key)
        self._log("MEF hit", key)
        return _copy_mef_result(result)

    def put_mef(self, key: Hashable, mef_result: MEFResult) -> None:
        if not self.enabled or not self.cache_mef_enabled or self.cache_max_size <= 0:
            return
        if not self.cache_failed_results and not bool(getattr(mef_result, "success", False)):
            return
        self._mef_cache[key] = _copy_mef_result(mef_result)
        self._mef_cache.move_to_end(key)
        self._trim(self._mef_cache)
        self._log("MEF put", key)

    def stats(self) -> dict[str, Any]:
        opf_total = self._opf_hit_count + self._opf_miss_count
        mef_total = self._mef_hit_count + self._mef_miss_count
        return {
            "enabled": bool(self.enabled),
            "opf_enabled": bool(self.enabled and self.cache_opf_enabled),
            "mef_enabled": bool(self.enabled and self.cache_mef_enabled),
            "opf_hit_count": int(self._opf_hit_count),
            "opf_miss_count": int(self._opf_miss_count),
            "opf_hit_rate": float(self._opf_hit_count / opf_total) if opf_total else 0.0,
            "mef_hit_count": int(self._mef_hit_count),
            "mef_miss_count": int(self._mef_miss_count),
            "mef_hit_rate": float(self._mef_hit_count / mef_total) if mef_total else 0.0,
            "opf_size": int(len(self._opf_cache)),
            "mef_size": int(len(self._mef_cache)),
            "cache_load_bin_mw": float(self.cache_load_bin_mw),
            "cache_mef_load_bin_mw": float(self.cache_mef_load_bin_mw),
            "cache_load_scale_bin": float(self.cache_load_scale_bin),
            "cache_max_size": int(self.cache_max_size),
            "cache_failed_results": bool(self.cache_failed_results),
        }

    def clear(self) -> None:
        self._opf_cache.clear()
        self._mef_cache.clear()

    def _bin_idc_load_mw(self, value: float) -> float:
        return self._bin_float(value, self.cache_load_bin_mw)

    def _bin_load_scale(self, value: float) -> float:
        return self._bin_float(value, self.cache_load_scale_bin)

    def _mef_load_key(self, value: float, delta_p_mw: float) -> tuple[str, float]:
        base_load = _safe_float(value, default=0.0)
        boundary_epsilon = 10.0 ** (-(self.float_digits + 2))
        if base_load <= delta_p_mw + boundary_epsilon:
            return ("exact", self._round_float(base_load))
        return (
            "binned",
            self._bin_float(base_load, self.cache_mef_load_bin_mw),
        )

    def _bin_float(self, value: float, bin_size: float) -> float:
        number = _safe_float(value, default=0.0)
        if bin_size <= 0.0:
            return self._round_float(number)
        return self._round_float(round(number / bin_size) * bin_size)

    def _round_float(self, value: float) -> float:
        return round(_safe_float(value, default=0.0), self.float_digits)

    def _trim(self, cache: OrderedDict[Hashable, Any]) -> None:
        while len(cache) > self.cache_max_size:
            cache.popitem(last=False)

    def _log(self, label: str, key: Hashable) -> None:
        if self.cache_verbose:
            print(f"[GridResultCache] {label}: {key}")


def _copy_opf_result(result: OPFResult) -> OPFResult:
    return OPFResult(
        success=bool(result.success),
        mode=str(result.mode),
        message=str(result.message),
        total_generation_cost=_safe_float(result.total_generation_cost),
        total_emission_kg=_safe_float(result.total_emission_kg),
        lmp_by_bus=_copy_float_dict(result.lmp_by_bus),
        reactive_lmp_by_bus=_copy_float_dict(result.reactive_lmp_by_bus),
        bus_active_power_mw=_copy_float_dict(result.bus_active_power_mw),
        bus_reactive_power_mvar=_copy_float_dict(result.bus_reactive_power_mvar),
        gen_power_mw=_copy_float_dict(result.gen_power_mw),
        bus_voltage_pu=_copy_float_dict(result.bus_voltage_pu),
        bus_voltage_min_pu=_copy_float_dict(result.bus_voltage_min_pu),
        bus_voltage_max_pu=_copy_float_dict(result.bus_voltage_max_pu),
        line_loading_percent=_copy_float_dict(result.line_loading_percent),
        line_loading_limit_percent=_copy_float_dict(result.line_loading_limit_percent),
        transformer_loading_percent=_copy_float_dict(
            result.transformer_loading_percent
        ),
        transformer_loading_limit_percent=_copy_float_dict(
            result.transformer_loading_limit_percent
        ),
        total_load_mw=_safe_float(result.total_load_mw),
        total_generation_mw=_safe_float(result.total_generation_mw),
        network_loss_mw=_safe_float(result.network_loss_mw),
        raw_result=None,
    )


def _copy_mef_result(result: MEFResult) -> MEFResult:
    return MEFResult(
        success=bool(result.success),
        mode=str(result.mode),
        bus_id=int(result.bus_id),
        delta_p_mw=_safe_float(result.delta_p_mw),
        mef_plus_kg_per_mwh=_safe_float(result.mef_plus_kg_per_mwh),
        mef_minus_kg_per_mwh=_safe_float(result.mef_minus_kg_per_mwh),
        base_emission_kg=_safe_float(result.base_emission_kg),
        plus_emission_kg=_safe_float(result.plus_emission_kg),
        minus_emission_kg=_safe_float(result.minus_emission_kg),
        base_gen_power_mw=_copy_float_dict(result.base_gen_power_mw),
        plus_gen_power_mw=_copy_float_dict(result.plus_gen_power_mw),
        minus_gen_power_mw=_copy_float_dict(result.minus_gen_power_mw),
        delta_gen_power_plus_mw=_copy_float_dict(result.delta_gen_power_plus_mw),
        delta_gen_power_minus_mw=_copy_float_dict(result.delta_gen_power_minus_mw),
        base_total_generation_mw=_safe_float(result.base_total_generation_mw),
        plus_total_generation_mw=_safe_float(result.plus_total_generation_mw),
        minus_total_generation_mw=_safe_float(result.minus_total_generation_mw),
        base_network_loss_mw=_safe_float(result.base_network_loss_mw),
        plus_network_loss_mw=_safe_float(result.plus_network_loss_mw),
        minus_network_loss_mw=_safe_float(result.minus_network_loss_mw),
        message=str(result.message),
    )


def _copy_float_dict(values: dict[int, float] | None) -> dict[int, float]:
    if not values:
        return {}
    return {int(key): _safe_float(value) for key, value in values.items()}


def _safe_float(value: Any, default: float = math.nan) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if math.isfinite(number) else default


def _positive_float(value: Any, default: float) -> float:
    number = _safe_float(value, default=default)
    return number if number > 0.0 else default
