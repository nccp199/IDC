"""Gymnasium wrapper coupling IDC power demand to IEEE14 grid feedback."""

from __future__ import annotations

import math
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
    "grid_lmp_cost_weight": 0.0,
    "grid_mef_carbon_weight": 0.0,
    "grid_security_weight": 0.0,
    "opf_failure_penalty": 5.0,
    "lmp_cost_ref": 100.0,
    "mef_carbon_ref": 100.0,
    "security_penalty_ref": 10.0,
}


class GridCoupledEnv(gym.Wrapper):
    """Wrap an IDC environment and add IEEE14 OPF/MEF diagnostics to info."""

    def __init__(
        self,
        base_env,
        grid_config: dict[str, Any] | None = None,
        grid_reward_config: dict[str, Any] | None = None,
    ):
        super().__init__(base_env)
        self.grid_config = {**DEFAULT_GRID_CONFIG, **(grid_config or {})}
        self.grid_reward_config = {**DEFAULT_GRID_REWARD_CONFIG, **(grid_reward_config or {})}

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
        self.enable_grid_reward = bool(self.grid_config.get("enable_grid_reward", False))
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

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        info = dict(info)
        if self.grid_enabled:
            self._run_grid_update(info, base_reward=0.0, idc_load_mw=0.0)
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
        adjusted_reward = self._run_grid_update(info, base_reward=base_reward, idc_load_mw=idc_load_mw)

        return self._augment_obs(obs, info), float(adjusted_reward), terminated, truncated, info

    def _run_grid_update(self, info: dict[str, Any], base_reward: float, idc_load_mw: float) -> float:
        opf_result = solve_opf(
            grid_case=self.grid_case,
            mode=self.opf_mode,
            idc_bus_id=self.idc_bus_idx,
            idc_load_mw=idc_load_mw,
            load_scale=self.load_scale,
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
                    load_scale=self.load_scale,
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
    ) -> None:
        grid_lmp = _safe_float(opf_result.lmp_by_bus.get(self.idc_bus_idx, math.nan))
        mef_success = bool(mef_result.success) if mef_result is not None else False
        mef_message = mef_result.message if mef_result is not None else "MEF disabled."
        mef_plus = mef_result.mef_plus_kg_per_mwh if mef_result is not None else math.nan
        mef_minus = mef_result.mef_minus_kg_per_mwh if mef_result is not None else math.nan

        safe_cost_voltage = float(grid_metrics.voltage_violation_count)
        safe_cost_line = float(grid_metrics.line_overload_count)
        safe_cost_opf = 0.0 if opf_result.success else 1.0
        safe_cost_total = safe_cost_voltage + safe_cost_line + safe_cost_opf

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
                "grid_security_penalty": _safe_float(grid_metrics.grid_security_penalty),
                "safe_cost_voltage": safe_cost_voltage,
                "safe_cost_line": safe_cost_line,
                "safe_cost_opf": safe_cost_opf,
                "safe_cost_total": safe_cost_total,
                "base_reward": float(base_reward),
                "grid_reward_penalty": 0.0,
                "grid_adjusted_reward": float(base_reward),
            }
        )

    def _compute_grid_reward_penalty(self, info, opf_result, mef_result, grid_metrics) -> float:
        if not self.enable_grid_reward:
            return 0.0

        grid_energy_mwh = max(_safe_float(info.get("grid_energy_kWh", 0.0), default=0.0) / 1000.0, 0.0)
        grid_lmp = _finite_or_zero(info.get("grid_lmp", math.nan))
        grid_mef_plus = _finite_or_zero(info.get("grid_mef_plus", math.nan))
        security_penalty = _finite_or_zero(grid_metrics.grid_security_penalty)

        lmp_cost = grid_lmp * grid_energy_mwh
        mef_carbon = grid_mef_plus * grid_energy_mwh

        lmp_cost_norm = lmp_cost / max(float(self.grid_reward_config.get("lmp_cost_ref", 100.0)), 1e-9)
        mef_carbon_norm = mef_carbon / max(float(self.grid_reward_config.get("mef_carbon_ref", 100.0)), 1e-9)
        security_norm = security_penalty / max(float(self.grid_reward_config.get("security_penalty_ref", 10.0)), 1e-9)

        penalty = (
            float(self.grid_reward_config.get("grid_lmp_cost_weight", 0.0)) * lmp_cost_norm
            + float(self.grid_reward_config.get("grid_mef_carbon_weight", 0.0)) * mef_carbon_norm
            + float(self.grid_reward_config.get("grid_security_weight", 0.0)) * security_norm
        )
        if not opf_result.success:
            penalty += float(self.grid_reward_config.get("opf_failure_penalty", 5.0))
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
            "grid_security_penalty": 0.0,
            "safe_cost_voltage": 0.0,
            "safe_cost_line": 0.0,
            "safe_cost_opf": 0.0,
            "safe_cost_total": 0.0,
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


def _normalized_voltage(value: Any, voltage_ref: float) -> float:
    number = _safe_float(value, default=math.nan)
    if not math.isfinite(number):
        return 0.0
    return (number - 1.0) / voltage_ref
