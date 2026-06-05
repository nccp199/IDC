"""Stable-Baselines3 callbacks for Safe PPO monitoring and lambda updates."""

from __future__ import annotations

import math
import warnings
from typing import Any

import numpy as np

try:
    from stable_baselines3.common.callbacks import BaseCallback
except Exception:  # pragma: no cover - exercised only when SB3 is absent.
    BaseCallback = None


class SafePPOCallback(BaseCallback if BaseCallback is not None else object):
    """Collect safe costs, update Lagrange multipliers, and log safe/ metrics."""

    REQUIRED_INFO_FIELDS = [
        "safe_costs",
        "reward_base",
        "reward_safe",
        "safe_penalty",
        "safe_penalty_raw",
        "safe_penalty_applied",
        "safe_reward_active",
        "lambda_update_active",
    ]

    def __init__(
        self,
        lagrangian_manager,
        lambda_update_active: bool = True,
        safe_reward_active: bool = False,
        verbose: int = 0,
    ):
        if BaseCallback is None:
            raise ImportError("stable_baselines3 is required to use SafePPOCallback.")
        super().__init__(verbose=verbose)
        self.lagrangian_manager = lagrangian_manager
        self.lambda_update_active = bool(lambda_update_active)
        self.safe_reward_active = bool(safe_reward_active)
        self.episode_count = 0
        self._step_cost_buffer: list[dict[str, float]] = []
        self._metric_buffers: dict[str, list[float]] = {}
        self._last_metric_values: dict[str, float] = {}
        self._episode_step_count = 0
        self._episode_opf_fail_count = 0
        self._rollout_step_count = 0
        self._rollout_opf_fail_count = 0
        self._warned_missing: set[str] = set()

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        dones = self.locals.get("dones", [])

        for idx, info in enumerate(infos):
            if not isinstance(info, dict):
                continue
            self._warn_missing_fields_once(info)
            safe_costs = dict(info.get("safe_costs", {}))

            if safe_costs:
                self._step_cost_buffer.append(
                    {
                        "C_opf": _finite_or_zero(safe_costs.get("C_active_opf", safe_costs.get("C_opf"))),
                        "C_voltage": _finite_or_zero(safe_costs.get("C_active_voltage", safe_costs.get("C_voltage"))),
                    }
                )
            self._episode_step_count += 1
            self._rollout_step_count += 1
            opf_success = bool(safe_costs.get("opf_success", info.get("grid_opf_success", True)))
            if not opf_success:
                self._episode_opf_fail_count += 1
                self._rollout_opf_fail_count += 1

            self._collect_step_metrics(info, safe_costs)

            done = bool(dones[idx]) if idx < len(dones) else False
            if done:
                self.episode_count += 1
                self._maybe_update_lagrangian()
                self._episode_step_count = 0
                self._episode_opf_fail_count = 0

        return True

    def _on_rollout_end(self) -> None:
        if self._step_cost_buffer:
            if self.lambda_update_active and self.lagrangian_manager.should_update(self.episode_count):
                self._maybe_update_lagrangian(force=True)
            elif not self.lambda_update_active and self.lagrangian_manager.should_observe(self.episode_count):
                self._maybe_update_lagrangian(force=True)
        self._record_rollout_metrics()
        self._metric_buffers.clear()
        self._rollout_step_count = 0
        self._rollout_opf_fail_count = 0

    def _maybe_update_lagrangian(self, force: bool = False) -> None:
        if not self._step_cost_buffer:
            return
        if self.lambda_update_active:
            if not force and not self.lagrangian_manager.should_update(self.episode_count):
                return
        elif not force and not self.lagrangian_manager.should_observe(self.episode_count):
            return

        batch_mean = {
            "C_opf": float(np.mean([row["C_opf"] for row in self._step_cost_buffer])),
            "C_voltage": float(np.mean([row["C_voltage"] for row in self._step_cost_buffer])),
        }
        if self.lambda_update_active:
            update = self.lagrangian_manager.update(
                batch_mean_costs=batch_mean,
                episode_count=self.episode_count,
                step=self.num_timesteps,
                force=force,
            )
        else:
            update = self.lagrangian_manager.observe_costs(
                batch_mean_costs=batch_mean,
                episode_count=self.episode_count,
                step=self.num_timesteps,
                force=force,
            )
        if update is not None:
            self._record_lagrangian_metrics()
            self._step_cost_buffer.clear()

    def _collect_step_metrics(self, info: dict[str, Any], safe_costs: dict[str, Any]) -> None:
        lambdas = self.lagrangian_manager.get_lambdas()
        ema_costs = self.lagrangian_manager.get_ema_costs()
        info_lambdas = info.get("lagrangian_lambdas")
        if isinstance(info_lambdas, dict):
            lambdas = {**lambdas, **info_lambdas}
        info_ema_costs = info.get("safe_ema_costs")
        if isinstance(info_ema_costs, dict):
            ema_costs = {**ema_costs, **info_ema_costs}
        c_opf = get_first_finite(info, [], ["C_opf"])
        c_voltage = get_first_finite(info, [], ["C_voltage"])
        c_total = get_first_finite(info, [], ["C_total"])
        if c_total is None:
            c_total = _sum_optional_finite([c_opf, c_voltage])

        records = {
            "safe/C_opf": c_opf,
            "safe/C_voltage": c_voltage,
            "safe/C_total": c_total,
            "safe/lambda_opf": lambdas.get("lambda_opf", 0.0),
            "safe/lambda_voltage": lambdas.get("lambda_voltage", 0.0),
            "safe/ema_C_opf": ema_costs.get("ema_C_opf", 0.0),
            "safe/ema_C_voltage": ema_costs.get("ema_C_voltage", 0.0),
            "safe/safe_penalty_raw": get_first_finite(info, ["safe_penalty_raw"]),
            "safe/safe_penalty_applied": get_first_finite(info, ["safe_penalty_applied"]),
            "safe/safe_penalty": get_first_finite(info, ["safe_penalty"]),
            "safe/safe_reward_active": get_first_finite(info, ["safe_reward_active"]),
            "safe/lambda_update_active": get_first_finite(info, ["lambda_update_active"]),
            "safe/reward_base": get_first_finite(info, ["reward_base"]),
            "safe/reward_safe": get_first_finite(info, ["reward_safe"]),
            "safe/min_voltage": get_first_finite(
                info,
                [
                    "raw_minV",
                    "minV",
                    "min_voltage",
                    "grid_minV",
                    "grid_min_voltage",
                    "grid_min_voltage_pu",
                ],
                ["raw_minV"],
            ),
            "safe/maxLine": get_first_finite(
                info,
                ["raw_maxLine", "maxLine", "max_line", "grid_maxLine", "grid_max_line_loading_percent"],
                ["raw_maxLine"],
            ),
            "safe/maxTrafo": get_first_finite(
                info,
                ["raw_maxTrafo", "maxTrafo", "max_trafo", "grid_maxTrafo", "grid_max_trafo_loading_percent"],
                ["raw_maxTrafo"],
            ),
            "safe/LMP_bus9": get_first_finite(
                info,
                ["raw_lmp", "LMP_bus9", "lmp_bus9", "grid_lmp", "grid_lmp_bus9"],
                ["raw_lmp"],
            ),
            "safe/MEF_bus9": get_first_finite(
                info,
                ["raw_mef", "MEF_bus9", "mef_bus9", "grid_mef", "grid_mef_plus", "grid_mef_bus9"],
                ["raw_mef"],
            ),
            "pv/available_kW": get_first_finite(info, ["pv_available_kW"]),
            "pv/used_kW": get_first_finite(info, ["pv_used_kW"]),
            "pv/curtail_kW": get_first_finite(info, ["pv_curtail_kW"]),
            "pv/total_available_kWh": get_first_finite(info, ["total_pv_available_kWh"]),
            "pv/total_used_kWh": get_first_finite(info, ["total_pv_used_kWh"]),
            "pv/total_curtail_kWh": get_first_finite(info, ["total_pv_curtail_kWh"]),
            "pv/utilization_rate": get_first_finite(info, ["pv_utilization_rate"]),
            "pv/renewable_share": get_first_finite(info, ["renewable_share"]),
            "grid/P_bus_net_kW": get_first_finite(info, ["P_bus_net_kW"]),
            "grid/P_grid_kW": get_first_finite(info, ["P_grid_kW"]),
            "grid/bus_net_load_mw": get_first_finite(info, ["grid_bus_net_load_mw", "grid_idc_load_mw"]),
            "safe/opf_failure_rate": 0.0 if bool(safe_costs.get("opf_success", info.get("grid_opf_success", True))) else 1.0,
        }
        for key, value in records.items():
            append_metric(self._metric_buffers, key, value)

    def _record_rollout_metrics(self) -> None:
        expected_tags = [
            "safe/C_opf",
            "safe/C_voltage",
            "safe/C_total",
            "safe/lambda_opf",
            "safe/lambda_voltage",
            "safe/ema_C_opf",
            "safe/ema_C_voltage",
            "safe/safe_penalty_raw",
            "safe/safe_penalty_applied",
            "safe/safe_penalty",
            "safe/safe_reward_active",
            "safe/lambda_update_active",
            "safe/reward_base",
            "safe/reward_safe",
            "safe/min_voltage",
            "safe/maxLine",
            "safe/maxTrafo",
            "safe/LMP_bus9",
            "safe/MEF_bus9",
            "pv/available_kW",
            "pv/used_kW",
            "pv/curtail_kW",
            "pv/total_available_kWh",
            "pv/total_used_kWh",
            "pv/total_curtail_kWh",
            "pv/utilization_rate",
            "pv/renewable_share",
            "grid/P_bus_net_kW",
            "grid/P_grid_kW",
            "grid/bus_net_load_mw",
            "safe/opf_failure_rate",
        ]

        for key, value in self.lagrangian_manager.get_lambdas().items():
            append_metric(self._metric_buffers, f"safe/{key}", value)
        for key, value in self.lagrangian_manager.get_ema_costs().items():
            append_metric(self._metric_buffers, f"safe/{key}", value)
        if self._rollout_step_count > 0:
            append_metric(
                self._metric_buffers,
                "safe/opf_failure_rate",
                self._rollout_opf_fail_count / max(self._rollout_step_count, 1),
            )
        append_metric(self._metric_buffers, "safe/safe_reward_active", self.safe_reward_active)
        append_metric(self._metric_buffers, "safe/lambda_update_active", self.lambda_update_active)

        for tag in expected_tags:
            value = mean_finite(self._metric_buffers.get(tag, []))
            if value is None:
                value = self._last_metric_values.get(tag)
            if value is None:
                self._warn_once(tag, f"No finite value collected for TensorBoard metric: {tag}")
                continue
            self.logger.record(tag, float(value))
            self._last_metric_values[tag] = float(value)

    def _record_lagrangian_metrics(self) -> None:
        for key, value in self.lagrangian_manager.get_lambdas().items():
            self.logger.record(f"safe/{key}", float(value))
        for key, value in self.lagrangian_manager.get_ema_costs().items():
            self.logger.record(f"safe/{key}", float(value))

    def _warn_missing_fields_once(self, info: dict[str, Any]) -> None:
        for key in self.REQUIRED_INFO_FIELDS:
            if key not in info:
                self._warn_once(key, f"SafePPOCallback expected info[{key!r}] but it was missing.")

    def _warn_once(self, key: str, message: str) -> None:
        if key in self._warned_missing:
            return
        self._warned_missing.add(key)
        warnings.warn(message, RuntimeWarning, stacklevel=2)


def _finite_float(value: Any, default: float = math.nan) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if math.isfinite(number) else default


def _finite_or_zero(value: Any) -> float:
    number = _finite_float(value)
    return number if math.isfinite(number) else 0.0


def to_float_or_none(value: Any) -> float | None:
    if isinstance(value, (bool, np.bool_)):
        return 1.0 if bool(value) else 0.0
    if value is None:
        return None
    if isinstance(value, str):
        text = value.strip().lower()
        if text in {"true", "yes", "y"}:
            return 1.0
        if text in {"false", "no", "n"}:
            return 0.0
        if not text:
            return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def get_first_finite(
    info: dict[str, Any],
    keys: list[str],
    nested_safe_costs_keys: list[str] | None = None,
) -> float | None:
    safe_costs = info.get("safe_costs", {})
    if isinstance(safe_costs, dict):
        for key in nested_safe_costs_keys or []:
            number = to_float_or_none(safe_costs.get(key))
            if number is not None:
                return number
    for key in keys:
        number = to_float_or_none(info.get(key))
        if number is not None:
            return number
    return None


def append_metric(buffer: dict[str, list[float]], key: str, value: Any) -> None:
    number = to_float_or_none(value)
    if number is None:
        return
    buffer.setdefault(key, []).append(float(number))


def mean_finite(values: list[float]) -> float | None:
    finite_values = []
    for value in values:
        number = to_float_or_none(value)
        if number is not None:
            finite_values.append(float(number))
    if not finite_values:
        return None
    return float(np.mean(finite_values))


def _sum_optional_finite(values: list[float | None]) -> float | None:
    finite_values = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    if not finite_values:
        return None
    return float(sum(finite_values))
