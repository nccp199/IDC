"""External Lagrangian multiplier manager for Safe PPO."""

from __future__ import annotations

import math
from copy import deepcopy
from typing import Any


class LagrangianMultiplierManager:
    """Manage CMDP multipliers with EMA batch updates."""

    def __init__(self, config: dict[str, Any]):
        self.config = deepcopy(config)
        self.enabled_constraints = [str(item) for item in self.config.get("enabled_constraints", [])]
        self.ema_beta = _clip(float(self.config.get("ema_beta", 0.95)), 0.0, 0.999999)
        self.update_interval_episodes = max(int(self.config.get("update_interval_episodes", 10)), 1)
        self.warmup_episodes = max(int(self.config.get("warmup_episodes", 0)), 0)
        self.update_tolerance = max(float(self.config.get("update_tolerance", 0.0)), 0.0)
        self.update_log: list[dict[str, Any]] = []
        self.observation_log: list[dict[str, Any]] = []
        self.update_count = 0
        self.last_update_episode = 0
        self.last_observation_episode = 0

        self.lambdas: dict[str, float] = {}
        self.ema_costs: dict[str, float] = {}
        for name in self.enabled_constraints:
            self.lambdas[name] = _finite_float(self.config.get(f"lambda_init_{name}", 0.0), default=0.0)
            self.ema_costs[name] = 0.0

    def should_update(self, episode_count: int) -> bool:
        if int(episode_count) < self.warmup_episodes:
            return False
        return int(episode_count) - int(self.last_update_episode) >= self.update_interval_episodes

    def should_observe(self, episode_count: int) -> bool:
        if int(episode_count) < self.warmup_episodes:
            return False
        return int(episode_count) - int(self.last_observation_episode) >= self.update_interval_episodes

    def observe_costs(
        self,
        batch_mean_costs: dict[str, Any],
        episode_count: int,
        step: int | None = None,
        force: bool = False,
    ) -> dict[str, Any] | None:
        """Update EMA cost traces without changing lambdas."""

        episode_count = int(episode_count)
        if not force and not self.should_observe(episode_count):
            return None

        record: dict[str, Any] = {
            "step": None if step is None else int(step),
            "episode_count": episode_count,
            "observation_index": int(len(self.observation_log) + 1),
            "constraints": {},
        }

        for name in self.enabled_constraints:
            batch_mean = self._extract_batch_mean(batch_mean_costs, name)
            old_ema = float(self.ema_costs.get(name, 0.0))
            ema = self.ema_beta * old_ema + (1.0 - self.ema_beta) * batch_mean
            self.ema_costs[name] = float(ema)
            record["constraints"][name] = {
                "batch_mean_C": float(batch_mean),
                "old_ema_C": float(old_ema),
                "ema_C": float(ema),
                "lambda": float(self.lambdas.get(name, 0.0)),
                "lambda_updated": False,
            }

        self.last_observation_episode = episode_count
        self.observation_log.append(record)
        return record

    def update(
        self,
        batch_mean_costs: dict[str, Any],
        episode_count: int,
        step: int | None = None,
        force: bool = False,
    ) -> dict[str, Any] | None:
        """Update lambdas from batch mean costs after warmup/interval checks."""

        episode_count = int(episode_count)
        if not force and not self.should_update(episode_count):
            return None

        update_record: dict[str, Any] = {
            "step": None if step is None else int(step),
            "episode_count": episode_count,
            "update_index": int(self.update_count + 1),
            "constraints": {},
        }

        for name in self.enabled_constraints:
            batch_mean = self._extract_batch_mean(batch_mean_costs, name)
            old_ema = float(self.ema_costs.get(name, 0.0))
            ema = self.ema_beta * old_ema + (1.0 - self.ema_beta) * batch_mean
            self.ema_costs[name] = float(ema)

            limit = _finite_float(self.config.get(f"cost_limit_{name}", 0.0), default=0.0)
            old_lambda = float(self.lambdas.get(name, 0.0))
            new_lambda = old_lambda
            skipped = abs(ema - limit) < self.update_tolerance

            if not skipped:
                lr = _finite_float(self.config.get(f"lambda_lr_{name}", 0.0), default=0.0)
                lambda_max = max(_finite_float(self.config.get(f"lambda_max_{name}", old_lambda), default=old_lambda), 0.0)
                new_lambda = _clip(old_lambda + lr * (ema - limit), 0.0, lambda_max)
                self.lambdas[name] = float(new_lambda)

            update_record["constraints"][name] = {
                "batch_mean_C": float(batch_mean),
                "old_ema_C": float(old_ema),
                "ema_C": float(ema),
                "cost_limit": float(limit),
                "old_lambda": float(old_lambda),
                "lambda": float(new_lambda),
                "skipped_by_tolerance": bool(skipped),
            }

        self.update_count += 1
        self.last_update_episode = episode_count
        self.update_log.append(update_record)
        return update_record

    def get_lambdas(self) -> dict[str, float]:
        return {f"lambda_{name}": float(self.lambdas.get(name, 0.0)) for name in self.enabled_constraints}

    def get_ema_costs(self) -> dict[str, float]:
        return {f"ema_C_{name}": float(self.ema_costs.get(name, 0.0)) for name in self.enabled_constraints}

    def get_update_log(self) -> list[dict[str, Any]]:
        return deepcopy(self.update_log)

    def state_dict(self) -> dict[str, Any]:
        return {
            "config": deepcopy(self.config),
            "enabled_constraints": list(self.enabled_constraints),
            "lambdas": deepcopy(self.lambdas),
            "ema_costs": deepcopy(self.ema_costs),
            "update_log": deepcopy(self.update_log),
            "observation_log": deepcopy(self.observation_log),
            "update_count": int(self.update_count),
            "last_update_episode": int(self.last_update_episode),
            "last_observation_episode": int(self.last_observation_episode),
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.lambdas.update({str(k): float(v) for k, v in dict(state.get("lambdas", {})).items()})
        self.ema_costs.update({str(k): float(v) for k, v in dict(state.get("ema_costs", {})).items()})
        self.update_log = list(state.get("update_log", []))
        self.observation_log = list(state.get("observation_log", []))
        self.update_count = int(state.get("update_count", len(self.update_log)))
        self.last_update_episode = int(state.get("last_update_episode", self.last_update_episode))
        self.last_observation_episode = int(
            state.get("last_observation_episode", self.last_observation_episode)
        )

    def _extract_batch_mean(self, batch_mean_costs: dict[str, Any], name: str) -> float:
        candidates = [name, f"C_{name}", f"C_active_{name}"]
        for key in candidates:
            if key in batch_mean_costs:
                return _finite_float(batch_mean_costs.get(key), default=0.0)
        return 0.0


def _finite_float(value: Any, default: float = math.nan) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if math.isfinite(number) else default


def _clip(value: float, low: float, high: float) -> float:
    return min(max(float(value), float(low)), float(high))
