"""Gymnasium wrapper that applies Lagrangian Safe PPO rewards."""

from __future__ import annotations

import math
from copy import deepcopy
from typing import Any

import gymnasium as gym
import numpy as np

from safe_rl.safe_costs import compute_safe_costs


class SafeRewardWrapper(gym.Wrapper):
    """Wrap an existing environment with CMDP safety-cost reporting/reward."""

    def __init__(
        self,
        env,
        cost_config: dict[str, Any],
        lagrangian_manager,
        train_config: dict[str, Any],
    ):
        super().__init__(env)
        self.cost_config = deepcopy(cost_config)
        self.lagrangian_manager = lagrangian_manager
        self.train_config = deepcopy(train_config)
        self.safe_reward_active = bool(
            self.train_config.get(
                "safe_reward_active",
                self.train_config.get("enable_safe_reward", True),
            )
        )
        self.enable_safe_reward = self.safe_reward_active
        self.report_only = bool(self.train_config.get("report_only", False))
        self.lambda_update_active = bool(self.train_config.get("lambda_update_active", self.safe_reward_active))
        self.terminate_on_opf_failure = bool(self.train_config.get("terminate_on_opf_failure", True))
        self.global_safe_penalty_scale = float(self.train_config.get("global_safe_penalty_scale", 1.0))
        self._last_valid_obs = None

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._last_valid_obs = np.asarray(obs, dtype=np.float32).copy()
        info = dict(info)
        safe_costs = compute_safe_costs(info, self.cost_config)
        self._inject_safe_info(
            info,
            reward_base=0.0,
            reward_safe=0.0,
            safe_penalty_raw=0.0,
            safe_penalty_applied=0.0,
            safe_costs=safe_costs,
        )
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        info = dict(info)
        reward_base = float(reward)
        safe_costs = compute_safe_costs(info, self.cost_config)
        lambdas = self.lagrangian_manager.get_lambdas()
        safe_penalty_raw = self._compute_safe_penalty(safe_costs, lambdas)
        safe_penalty_applied = safe_penalty_raw if self.safe_reward_active else 0.0
        reward_safe = float(reward_base - safe_penalty_applied)

        opf_success = bool(safe_costs.get("opf_success", False))
        if not opf_success:
            info["opf_success"] = False
            info["grid_failure"] = True
            info["failure_reason"] = info.get("grid_opf_message", "OPF failed.")
            if self.terminate_on_opf_failure:
                terminated = True
            obs = self._finite_or_last_obs(obs)
        else:
            info["opf_success"] = True
            info["grid_failure"] = False
            if np.all(np.isfinite(np.asarray(obs, dtype=np.float32))):
                self._last_valid_obs = np.asarray(obs, dtype=np.float32).copy()

        self._inject_safe_info(
            info,
            reward_base=reward_base,
            reward_safe=reward_safe,
            safe_penalty_raw=safe_penalty_raw,
            safe_penalty_applied=safe_penalty_applied,
            safe_costs=safe_costs,
        )
        return obs, reward_safe, terminated, truncated, info

    def _compute_safe_penalty(self, safe_costs: dict[str, Any], lambdas: dict[str, float]) -> float:
        active_costs = dict(safe_costs.get("active_costs", {}))
        penalty = 0.0
        for name, cost in active_costs.items():
            penalty += float(lambdas.get(f"lambda_{name}", 0.0)) * float(cost)
        return float(self.global_safe_penalty_scale * penalty)

    def _inject_safe_info(
        self,
        info: dict[str, Any],
        reward_base: float,
        reward_safe: float,
        safe_penalty_raw: float,
        safe_penalty_applied: float,
        safe_costs: dict[str, Any],
    ) -> None:
        compact_costs = {
            key: safe_costs[key]
            for key in [
                "C_opf",
                "C_voltage",
                "C_line",
                "C_trafo",
                "C_lmp",
                "C_mef",
                "C_total",
                "C_active_opf",
                "C_active_voltage",
                "C_active_total",
                "raw_minV",
                "raw_maxV",
                "raw_maxLine",
                "raw_maxTrafo",
                "raw_lmp",
                "raw_mef",
                "opf_success",
                "line_missing",
                "trafo_missing",
                "active_costs",
            ]
            if key in safe_costs
        }
        info["reward_base"] = float(reward_base)
        info["reward_safe"] = float(reward_safe)
        info["safe_penalty_raw"] = float(safe_penalty_raw)
        info["safe_penalty_applied"] = float(safe_penalty_applied)
        # Backward-compatible alias: this is the penalty actually applied to reward.
        info["safe_penalty"] = float(safe_penalty_applied)
        info["safe_reward_active"] = bool(self.safe_reward_active)
        info["lambda_update_active"] = bool(self.lambda_update_active)
        info["safe_costs"] = compact_costs
        info["lagrangian_lambdas"] = self.lagrangian_manager.get_lambdas()
        info["safe_ema_costs"] = self.lagrangian_manager.get_ema_costs()

    def _finite_or_last_obs(self, obs):
        arr = np.asarray(obs, dtype=np.float32)
        if arr.size > 0 and np.all(np.isfinite(arr)):
            self._last_valid_obs = arr.copy()
            return obs
        if self._last_valid_obs is not None:
            return self._last_valid_obs.copy()
        return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
