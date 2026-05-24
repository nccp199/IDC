"""Gymnasium wrapper coupling IDC power demand to IEEE14 grid feedback."""

from __future__ import annotations

import math
import csv
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from grid_model import (
    MEFResult,
    build_default_gen_emission_factors,
    calculate_nodal_mef,
    compute_total_emission,
    extract_grid_metrics,
    get_bus_index_by_ieee_number,
    load_ieee14_case,
    solve_opf,
)


DEFAULT_GRID_CONFIG = {
    "enable_grid_coupling": True,
    "case_name": "ieee14",
    "idc_ieee_bus_number": 9,
    "opf_mode": "ac",
    "delta_p_mw": 0.1,
    "load_scale": 1.0,
    "enable_grid_reward": False,
    "use_mef": True,
    "enable_grid_obs": True,
    "grid_obs_dim": 8,
    "grid_lmp_ref": 100.0,
    "grid_mef_ref": 1000.0,
    "grid_voltage_ref": 0.10,
    "grid_line_loading_ref": 100.0,
    "grid_network_loss_ref": 20.0,
    "grid_security_penalty_ref": 10.0,
}

DEFAULT_GRID_REWARD_CONFIG = {
    "enable_grid_reward": False,
    "grid_reward_mode": "none",
    "grid_lmp_cost_weight": 0.0,
    "grid_mef_carbon_weight": 0.0,
    "grid_safe_violation_weight": 0.0,
    "lmp_cost_ref": 100.0,
    "mef_carbon_ref": 100.0,
    "safe_violation_ref": 1.0,
    "clip_grid_reward_penalty": True,
    "grid_reward_penalty_clip": 10.0,
}

DEFAULT_GRID_SCENARIO_CONFIG = {
    "enable_dynamic_grid_load": False,
    "grid_load_scale_path": "data/grid_scenarios/nems_singapore/processed/nems_24h_load_scale.csv",
    "grid_load_scale_column": "grid_load_scale",
    "grid_usep_column": "usep_sgd_per_mwh",
    "fallback_load_scale": 1.0,
}


class GridCoupledEnv(gym.Wrapper):
    """Wrap an IDC environment and add IEEE14 OPF/MEF diagnostics to info."""

    def __init__(
        self,
        base_env,
        grid_config: dict[str, Any] | None = None,
        grid_reward_config: dict[str, Any] | None = None,
        grid_scenario_config: dict[str, Any] | None = None,
    ):
        super().__init__(base_env)
        self.grid_config = {**DEFAULT_GRID_CONFIG, **(grid_config or {})}
        self.grid_reward_config = {**DEFAULT_GRID_REWARD_CONFIG, **(grid_reward_config or {})}
        self.grid_scenario_config = {**DEFAULT_GRID_SCENARIO_CONFIG, **(grid_scenario_config or {})}

        self.grid_enabled = bool(self.grid_config.get("enable_grid_coupling", True))
        self.case_name = str(self.grid_config.get("case_name", "ieee14"))
        if self.case_name.lower() != "ieee14":
            raise ValueError(f"Only case_name='ieee14' is supported in GridCoupledEnv, got {self.case_name!r}.")

        self.grid_case = load_ieee14_case()
        self.gen_emission_factors = build_default_gen_emission_factors(self.grid_case)
        self.idc_ieee_bus_number = int(self.grid_config.get("idc_ieee_bus_number", 9))
        self.idc_bus_idx = get_bus_index_by_ieee_number(self.grid_case, self.idc_ieee_bus_number)
        self.opf_mode = str(self.grid_config.get("opf_mode", "ac")).lower()
        self.delta_p_mw = float(self.grid_config.get("delta_p_mw", 0.1))
        self.load_scale = float(self.grid_config.get("load_scale", 1.0))
        self.enable_grid_reward = bool(
            self.grid_reward_config.get(
                "enable_grid_reward",
                self.grid_config.get("enable_grid_reward", False),
            )
        )
        self.grid_reward_mode = str(self.grid_reward_config.get("grid_reward_mode", "none")).strip().lower()
        if self.grid_reward_mode not in {"none", "lmp", "mef", "security", "lmp_mef", "full"}:
            raise ValueError(
                "grid_reward_mode must be one of none, lmp, mef, security, lmp_mef, full; "
                f"got {self.grid_reward_mode!r}."
            )
        self.use_mef = bool(self.grid_config.get("use_mef", True))
        self.enable_grid_obs = bool(self.grid_config.get("enable_grid_obs", True))
        self.grid_obs_dim = int(self.grid_config.get("grid_obs_dim", 8))
        if self.grid_obs_dim != 8:
            raise ValueError(f"GridCoupledEnv currently expects grid_obs_dim=8, got {self.grid_obs_dim}.")

        self.base_obs_dim = int(base_env.observation_space.shape[0])
        self.action_dim = int(getattr(base_env, "action_dim", base_env.action_space.shape[0]))
        self.model = getattr(base_env, "model", None)
        self.horizon = int(getattr(base_env, "horizon", 0))
        self.price_t = getattr(base_env, "price_t", None)
        self._load_grid_scenario()

        if self.enable_grid_obs:
            self.observation_space = spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(self.base_obs_dim + self.grid_obs_dim,),
                dtype=np.float32,
            )
        else:
            self.observation_space = base_env.observation_space
        self.action_space = base_env.action_space

    @property
    def current_step(self) -> int:
        return int(getattr(self.env, "current_step", 0))

    def _load_grid_scenario(self) -> None:
        self.grid_scenario_enabled = bool(self.grid_scenario_config.get("enable_dynamic_grid_load", False))
        if self.grid_scenario_enabled:
            fallback = _safe_float(self.grid_scenario_config.get("fallback_load_scale", 1.0), default=1.0)
        else:
            fallback = self.load_scale
        fallback = fallback if fallback >= 0.0 else 1.0
        horizon = max(int(self.horizon or 24), 1)
        self.grid_load_scale_t = np.full(horizon, fallback, dtype=np.float64)
        self.grid_reference_usep_t = np.full(horizon, math.nan, dtype=np.float64)
        self.grid_scenario_source = "fallback"
        self.grid_scenario_message = "Dynamic grid load disabled."

        if not self.grid_scenario_enabled:
            return

        path = Path(str(self.grid_scenario_config.get("grid_load_scale_path", "")))
        if not path.is_absolute():
            path = Path(__file__).resolve().parents[1] / path
        scale_column = str(self.grid_scenario_config.get("grid_load_scale_column", "grid_load_scale"))
        usep_column = str(self.grid_scenario_config.get("grid_usep_column", "usep_sgd_per_mwh"))

        try:
            with path.open("r", encoding="utf-8-sig", newline="") as f:
                rows = list(csv.DictReader(f))
            scales = [_safe_float(row.get(scale_column), default=math.nan) for row in rows]
            scales = [value for value in scales if math.isfinite(value) and value >= 0.0]
            if len(scales) < horizon:
                raise ValueError(f"Expected at least {horizon} scale values in {path}, got {len(scales)}.")
            self.grid_load_scale_t = np.asarray(scales[:horizon], dtype=np.float64)

            usep_values = [_safe_float(row.get(usep_column), default=math.nan) for row in rows[:horizon]]
            self.grid_reference_usep_t = np.asarray(usep_values, dtype=np.float64)
            self.grid_scenario_source = str(path)
            self.grid_scenario_message = "Loaded dynamic grid load scale."
        except Exception as exc:
            self.grid_scenario_source = f"fallback: {path}"
            self.grid_scenario_message = (
                f"Failed to load dynamic grid load scale; using fallback={fallback}: "
                f"{type(exc).__name__}: {exc}"
            )

    def _grid_scenario_values(self, hour: int) -> tuple[float, float]:
        idx = int(hour) % max(len(self.grid_load_scale_t), 1)
        load_scale = _safe_float(self.grid_load_scale_t[idx], default=self.load_scale)
        if not math.isfinite(load_scale) or load_scale < 0.0:
            load_scale = self.load_scale
        usep = _safe_float(self.grid_reference_usep_t[idx], default=math.nan)
        return float(load_scale), float(usep)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        info = dict(info)
        if self.grid_enabled:
            self._run_grid_update(info, base_reward=0.0, idc_load_mw=0.0, hour=0)
            info["grid_initial_obs_mode"] = "nominal_zero_idc_load"
        else:
            info.update(self._disabled_grid_info(base_reward=0.0))
            info["grid_initial_obs_mode"] = "grid_disabled"
        return self._augment_obs(obs, info), info

    def step(self, action):
        obs, base_reward, terminated, truncated, info = self.env.step(action)
        info = dict(info)
        base_reward = float(base_reward)

        if not self.grid_enabled:
            info.update(self._disabled_grid_info(base_reward))
            return self._augment_obs(obs, info), base_reward, terminated, truncated, info

        p_grid_kw = _safe_float(info.get("P_grid_kW", 0.0), default=0.0)
        idc_load_mw = max(p_grid_kw / 1000.0, 0.0)
        hour = int(_safe_float(info.get("hour", self.current_step), default=self.current_step))
        adjusted_reward = self._run_grid_update(
            info,
            base_reward=base_reward,
            idc_load_mw=idc_load_mw,
            hour=hour,
        )

        return self._augment_obs(obs, info), float(adjusted_reward), terminated, truncated, info

    def _run_grid_update(self, info: dict[str, Any], base_reward: float, idc_load_mw: float, hour: int) -> float:
        load_scale_t, reference_usep = self._grid_scenario_values(hour)
        opf_result = solve_opf(
            grid_case=self.grid_case,
            mode=self.opf_mode,
            idc_bus_id=self.idc_bus_idx,
            idc_load_mw=idc_load_mw,
            load_scale=load_scale_t,
        )
        grid_metrics = extract_grid_metrics(opf_result)
        total_emission_kg = self._compute_opf_emission(opf_result)

        mef_result = None
        if self.use_mef:
            try:
                mef_result = calculate_nodal_mef(
                    grid_case=self.grid_case,
                    bus_id=self.idc_bus_idx,
                    mode=self.opf_mode,
                    delta_p_mw=self.delta_p_mw,
                    load_scale=load_scale_t,
                    base_idc_load_mw=idc_load_mw,
                    clamp_minus_load=True,
                    gen_emission_factors_kg_per_mwh=self.gen_emission_factors,
                )
            except Exception as exc:
                mef_result = MEFResult(
                    success=False,
                    mode=self.opf_mode,
                    bus_id=self.idc_bus_idx,
                    delta_p_mw=self.delta_p_mw,
                    message=f"MEF calculation failed: {type(exc).__name__}: {exc}",
                )

        self._inject_grid_info(
            info=info,
            base_reward=base_reward,
            idc_load_mw=idc_load_mw,
            opf_result=opf_result,
            mef_result=mef_result,
            grid_metrics=grid_metrics,
            total_emission_kg=total_emission_kg,
            load_scale_t=load_scale_t,
            reference_usep=reference_usep,
        )
        grid_reward_penalty = self._compute_grid_reward_penalty(info, opf_result, mef_result, grid_metrics)
        adjusted_reward = base_reward - grid_reward_penalty
        info["grid_reward_penalty"] = float(grid_reward_penalty)
        info["grid_adjusted_reward"] = float(adjusted_reward)
        return float(adjusted_reward)

    def _build_grid_obs_from_info(self, info: dict[str, Any]) -> np.ndarray:
        lmp_ref = max(float(self.grid_config.get("grid_lmp_ref", 100.0)), 1e-9)
        mef_ref = max(float(self.grid_config.get("grid_mef_ref", 1000.0)), 1e-9)
        voltage_ref = max(float(self.grid_config.get("grid_voltage_ref", 0.10)), 1e-9)
        line_ref = max(float(self.grid_config.get("grid_line_loading_ref", 100.0)), 1e-9)
        loss_ref = max(float(self.grid_config.get("grid_network_loss_ref", 20.0)), 1e-9)
        security_ref = max(float(self.grid_config.get("grid_security_penalty_ref", 10.0)), 1e-9)

        values = np.array(
            [
                _finite_or_zero(info.get("grid_lmp")) / lmp_ref,
                _finite_or_zero(info.get("grid_mef_plus")) / mef_ref,
                _finite_or_zero(info.get("grid_mef_minus")) / mef_ref,
                _normalized_voltage(info.get("grid_min_voltage_pu"), voltage_ref),
                _finite_or_zero(info.get("grid_max_line_loading_percent")) / line_ref,
                _finite_or_zero(info.get("grid_network_loss_mw")) / loss_ref,
                _finite_or_zero(info.get("grid_security_penalty")) / security_ref,
                1.0 if bool(info.get("grid_opf_success", False)) else 0.0,
            ],
            dtype=np.float32,
        )
        values[:7] = np.clip(values[:7], -10.0, 10.0)
        values[7] = 1.0 if values[7] >= 0.5 else 0.0
        return values.astype(np.float32)

    def _augment_obs(self, obs, info: dict[str, Any]) -> np.ndarray:
        obs_array = np.asarray(obs, dtype=np.float32).reshape(-1)
        if not self.enable_grid_obs:
            return obs_array.astype(np.float32)
        grid_obs = self._build_grid_obs_from_info(info)
        return np.concatenate([obs_array, grid_obs]).astype(np.float32)

    def _compute_opf_emission(self, opf_result) -> float:
        if not opf_result.gen_power_mw:
            return math.nan
        total_emission, _ = compute_total_emission(opf_result.gen_power_mw, self.gen_emission_factors)
        return float(total_emission)

    def _inject_grid_info(
        self,
        info: dict[str, Any],
        base_reward: float,
        idc_load_mw: float,
        opf_result,
        mef_result,
        grid_metrics,
        total_emission_kg: float,
        load_scale_t: float,
        reference_usep: float,
    ) -> None:
        grid_lmp = _safe_float(opf_result.lmp_by_bus.get(self.idc_bus_idx, math.nan))
        mef_success = bool(mef_result.success) if mef_result is not None else False
        mef_message = mef_result.message if mef_result is not None else "MEF disabled."
        mef_plus = mef_result.mef_plus_kg_per_mwh if mef_result is not None else math.nan
        mef_minus = mef_result.mef_minus_kg_per_mwh if mef_result is not None else math.nan

        safe_violation_voltage = float(grid_metrics.voltage_violation_count)
        safe_violation_line = float(grid_metrics.line_overload_count)
        safe_violation_opf = 0.0 if opf_result.success else 1.0
        safe_violation_cost = safe_violation_voltage + safe_violation_line + safe_violation_opf

        safe_cost_voltage = safe_violation_voltage
        safe_cost_line = safe_violation_line
        safe_cost_opf = safe_violation_opf
        safe_cost_total = safe_violation_cost

        info.update(
            {
                "grid_enabled": self.grid_enabled,
                "grid_opf_mode": self.opf_mode,
                "grid_opf_success": bool(opf_result.success),
                "grid_opf_message": opf_result.message,
                "grid_case_name": self.grid_case.name,
                "grid_idc_ieee_bus_number": self.idc_ieee_bus_number,
                "grid_idc_bus_index": self.idc_bus_idx,
                "grid_idc_load_mw": float(idc_load_mw),
                "grid_load_scale": _safe_float(load_scale_t),
                "grid_scenario_enabled": bool(self.grid_scenario_enabled),
                "grid_scenario_source": self.grid_scenario_source,
                "grid_scenario_message": self.grid_scenario_message,
                "grid_reference_usep": _safe_float(reference_usep),
                "grid_lmp": grid_lmp,
                "grid_mef_plus": _safe_float(mef_plus),
                "grid_mef_minus": _safe_float(mef_minus),
                "grid_mef_success": mef_success,
                "grid_mef_message": mef_message,
                "grid_total_generation_cost": _safe_float(opf_result.total_generation_cost),
                "grid_total_emission_kg": _safe_float(total_emission_kg),
                "grid_total_load_mw": _safe_float(opf_result.total_load_mw),
                "grid_total_generation_mw": _safe_float(opf_result.total_generation_mw),
                "grid_network_loss_mw": _safe_float(opf_result.network_loss_mw),
                "grid_min_voltage_pu": _safe_float(grid_metrics.min_voltage_pu),
                "grid_max_voltage_pu": _safe_float(grid_metrics.max_voltage_pu),
                "grid_max_line_loading_percent": _safe_float(grid_metrics.max_line_loading_percent),
                "grid_voltage_violation_count": int(grid_metrics.voltage_violation_count),
                "grid_line_overload_count": int(grid_metrics.line_overload_count),
                "grid_voltage_violation_magnitude": _safe_float(grid_metrics.voltage_violation_magnitude, default=0.0),
                "grid_line_overload_magnitude": _safe_float(grid_metrics.line_overload_magnitude, default=0.0),
                "grid_security_penalty": _safe_float(grid_metrics.grid_security_penalty),
                "safe_violation_voltage": safe_violation_voltage,
                "safe_violation_line": safe_violation_line,
                "safe_violation_opf": safe_violation_opf,
                "safe_violation_cost": safe_violation_cost,
                "safe_cost_voltage": safe_cost_voltage,
                "safe_cost_line": safe_cost_line,
                "safe_cost_opf": safe_cost_opf,
                "safe_cost_total": safe_cost_total,
                "grid_reward_enabled": bool(self.enable_grid_reward),
                "grid_reward_mode": self.grid_reward_mode,
                "grid_lmp_cost": 0.0,
                "grid_lmp_cost_norm": 0.0,
                "grid_lmp_cost_penalty": 0.0,
                "grid_mef_carbon": 0.0,
                "grid_mef_carbon_norm": 0.0,
                "grid_mef_carbon_penalty": 0.0,
                "grid_safe_violation": safe_violation_cost,
                "grid_safe_violation_norm": 0.0,
                "grid_safe_violation_penalty": 0.0,
                "base_reward": float(base_reward),
                "grid_reward_penalty": 0.0,
                "grid_adjusted_reward": float(base_reward),
            }
        )

    def _compute_grid_reward_penalty(self, info, opf_result, mef_result, grid_metrics) -> float:
        grid_energy_mwh = max(_safe_float(info.get("grid_energy_kWh", 0.0), default=0.0) / 1000.0, 0.0)
        grid_lmp = _finite_or_zero(info.get("grid_lmp", math.nan))
        grid_mef_plus = _finite_or_zero(info.get("grid_mef_plus", math.nan))
        safe_violation = _finite_or_zero(info.get("safe_violation_cost", 0.0))

        lmp_cost = grid_lmp * grid_energy_mwh
        mef_carbon = grid_mef_plus * grid_energy_mwh

        lmp_cost_norm = lmp_cost / _positive_ref(self.grid_reward_config, "lmp_cost_ref", 100.0)
        mef_carbon_norm = mef_carbon / _positive_ref(self.grid_reward_config, "mef_carbon_ref", 100.0)
        safe_violation_norm = safe_violation / _positive_ref(self.grid_reward_config, "safe_violation_ref", 1.0)

        w_lmp = _finite_or_zero(self.grid_reward_config.get("grid_lmp_cost_weight", 0.0))
        w_mef = _finite_or_zero(self.grid_reward_config.get("grid_mef_carbon_weight", 0.0))
        w_safe = _finite_or_zero(self.grid_reward_config.get("grid_safe_violation_weight", 0.0))

        mode = self.grid_reward_mode
        enabled = bool(self.enable_grid_reward)
        lmp_penalty = w_lmp * lmp_cost_norm if enabled and mode in {"lmp", "lmp_mef", "full"} else 0.0
        mef_penalty = w_mef * mef_carbon_norm if enabled and mode in {"mef", "lmp_mef", "full"} else 0.0
        safe_penalty = w_safe * safe_violation_norm if enabled and mode in {"security", "full"} else 0.0

        penalty = lmp_penalty + mef_penalty + safe_penalty
        if bool(self.grid_reward_config.get("clip_grid_reward_penalty", True)):
            clip_value = _positive_ref(self.grid_reward_config, "grid_reward_penalty_clip", 10.0)
            penalty = float(np.clip(penalty, -clip_value, clip_value))

        info.update(
            {
                "grid_reward_enabled": enabled,
                "grid_reward_mode": mode,
                "grid_lmp_cost": float(lmp_cost),
                "grid_lmp_cost_norm": float(lmp_cost_norm),
                "grid_lmp_cost_penalty": float(lmp_penalty),
                "grid_mef_carbon": float(mef_carbon),
                "grid_mef_carbon_norm": float(mef_carbon_norm),
                "grid_mef_carbon_penalty": float(mef_penalty),
                "grid_safe_violation": float(safe_violation),
                "grid_safe_violation_norm": float(safe_violation_norm),
                "grid_safe_violation_penalty": float(safe_penalty),
                "grid_reward_penalty": float(penalty),
            }
        )
        return float(penalty)

    def _disabled_grid_info(self, base_reward: float) -> dict[str, Any]:
        return {
            "grid_enabled": False,
            "grid_opf_mode": self.opf_mode,
            "grid_opf_success": False,
            "grid_opf_message": "Grid coupling disabled.",
            "grid_case_name": self.grid_case.name,
            "grid_idc_ieee_bus_number": self.idc_ieee_bus_number,
            "grid_idc_bus_index": self.idc_bus_idx,
            "grid_idc_load_mw": math.nan,
            "grid_load_scale": math.nan,
            "grid_scenario_enabled": bool(self.grid_scenario_enabled),
            "grid_scenario_source": self.grid_scenario_source,
            "grid_scenario_message": self.grid_scenario_message,
            "grid_reference_usep": math.nan,
            "grid_lmp": math.nan,
            "grid_mef_plus": math.nan,
            "grid_mef_minus": math.nan,
            "grid_mef_success": False,
            "grid_mef_message": "Grid coupling disabled.",
            "grid_total_generation_cost": math.nan,
            "grid_total_emission_kg": math.nan,
            "grid_total_load_mw": math.nan,
            "grid_total_generation_mw": math.nan,
            "grid_network_loss_mw": math.nan,
            "grid_min_voltage_pu": math.nan,
            "grid_max_voltage_pu": math.nan,
            "grid_max_line_loading_percent": math.nan,
            "grid_voltage_violation_count": 0,
            "grid_line_overload_count": 0,
            "grid_voltage_violation_magnitude": 0.0,
            "grid_line_overload_magnitude": 0.0,
            "grid_security_penalty": 0.0,
            "safe_violation_voltage": 0.0,
            "safe_violation_line": 0.0,
            "safe_violation_opf": 0.0,
            "safe_violation_cost": 0.0,
            "safe_cost_voltage": 0.0,
            "safe_cost_line": 0.0,
            "safe_cost_opf": 0.0,
            "safe_cost_total": 0.0,
            "grid_reward_enabled": bool(self.enable_grid_reward),
            "grid_reward_mode": self.grid_reward_mode,
            "grid_lmp_cost": 0.0,
            "grid_lmp_cost_norm": 0.0,
            "grid_lmp_cost_penalty": 0.0,
            "grid_mef_carbon": 0.0,
            "grid_mef_carbon_norm": 0.0,
            "grid_mef_carbon_penalty": 0.0,
            "grid_safe_violation": 0.0,
            "grid_safe_violation_norm": 0.0,
            "grid_safe_violation_penalty": 0.0,
            "base_reward": float(base_reward),
            "grid_reward_penalty": 0.0,
            "grid_adjusted_reward": float(base_reward),
        }


def _safe_float(value: Any, default: float = math.nan) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if math.isfinite(number) else default


def _finite_or_zero(value: Any) -> float:
    return _safe_float(value, default=0.0)


def _positive_ref(config: dict[str, Any], key: str, default: float) -> float:
    value = _safe_float(config.get(key, default), default=default)
    return max(value, 1e-9)


def _normalized_voltage(value: Any, voltage_ref: float) -> float:
    number = _safe_float(value, default=math.nan)
    if not math.isfinite(number):
        return 0.0
    return (number - 1.0) / voltage_ref
