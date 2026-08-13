"""Official-SuperSuit-compatible padding over the heterogeneous HARL bridge."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
from gymnasium import spaces

from marl.bridges.harl_bridge import HARL_UPSTREAM_COMMIT, HarlIDCGridBridge
from marl.diagnostics import BESSVirtualActionMonitor
from marl.specs.state_specs import CENTRALIZED_STATE_DIM


def _pad_to(array: np.ndarray, new_shape: tuple[int, ...], pad_value: float) -> np.ndarray:
    """Match SuperSuit 3.7.0 ``homogenize_ops.pad_to`` for Box arrays."""
    if array.shape == new_shape:
        return array
    padding = [(0, new - old) for new, old in zip(new_shape, array.shape, strict=True)]
    return np.pad(array, padding, constant_values=pad_value)


def _homogenize_box_spaces(box_spaces: Sequence[spaces.Box]) -> spaces.Box:
    """Match SuperSuit 3.7.0's Box-space homogenization rules exactly."""
    if not box_spaces:
        raise ValueError("At least one Box space is required.")
    first = box_spaces[0]
    for space in box_spaces:
        if not isinstance(space, spaces.Box):
            raise TypeError("HARL padding supports homogeneous Box spaces only.")
        if len(space.shape) != len(first.shape):
            raise ValueError("All Box spaces must have the same rank.")
        if space.dtype != first.dtype:
            raise ValueError("All Box spaces must have the same dtype.")

    max_shape = tuple(np.max(np.asarray([space.shape for space in box_spaces]), axis=0))
    padded_lows = np.stack(
        [
            _pad_to(space.low, max_shape, np.minimum(0, np.min(space.low)))
            for space in box_spaces
        ]
    )
    padded_highs = np.stack(
        [
            _pad_to(space.high, max_shape, np.maximum(1e-5, np.max(space.high)))
            for space in box_spaces
        ]
    )
    return spaces.Box(
        low=np.min(padded_lows, axis=0),
        high=np.max(padded_highs, axis=0),
        dtype=first.dtype,
    )


class HarlPaddedBridge:
    """Expose stackable HARL arrays without changing the true bridge contract.

    Observation padding and action-prefix slicing reproduce SuperSuit 3.7.0's
    ``pad_observations_v0`` and ``pad_action_space_v0`` Box behavior. No action
    clipping is added: the official dehomogenization function only slices each
    action back to its original shape.
    """

    agent_order = ("idc", "bess")
    n_agents = 2
    upstream_commit = HARL_UPSTREAM_COMMIT
    stock_on_policy_runner_compatible = True
    idc_observation_dim = 288
    bess_observation_dim = 164
    padded_observation_dim = 288
    idc_action_dim = 22
    bess_action_dim = 1
    padded_action_dim = 22

    def __init__(
        self,
        env: HarlIDCGridBridge,
        *,
        diagnostics: BESSVirtualActionMonitor | None = None,
    ) -> None:
        if not isinstance(env, HarlIDCGridBridge):
            raise TypeError("HarlPaddedBridge must wrap HarlIDCGridBridge.")
        self.env = env
        self.diagnostics = diagnostics

        padded_observation_space = _homogenize_box_spaces(env.observation_space)
        padded_action_space = _homogenize_box_spaces(env.action_space)
        self.observation_space = tuple(padded_observation_space for _ in self.agent_order)
        self.action_space = tuple(padded_action_space for _ in self.agent_order)
        self.share_observation_space = env.share_observation_space
        self._validate_spaces()

        self._episode = -1
        self._episode_step = 0
        self._global_step = 0
        self._last_true_observations: tuple[np.ndarray, np.ndarray] | None = None

    def _validate_spaces(self) -> None:
        if tuple(space.shape for space in self.observation_space) != ((288,), (288,)):
            raise ValueError("Padded observation spaces must both have shape (288,).")
        if tuple(space.shape for space in self.action_space) != ((22,), (22,)):
            raise ValueError("Padded action spaces must both have shape (22,).")
        expected = ((CENTRALIZED_STATE_DIM,), (CENTRALIZED_STATE_DIM,))
        if tuple(space.shape for space in self.share_observation_space) != expected:
            raise ValueError(f"Centralized state spaces must remain shape {expected}.")

    @property
    def last_info(self):
        return self.env.last_info

    @property
    def last_true_observations(self) -> tuple[np.ndarray, np.ndarray] | None:
        """Return copies of the unpadded bridge observations for diagnostics/tests."""
        if self._last_true_observations is None:
            return None
        return tuple(value.copy() for value in self._last_true_observations)

    def seed(self, seed: int) -> list[int]:
        result = self.env.seed(seed)
        for agent_id, space in enumerate(self.action_space):
            space.seed(int(seed) + agent_id)
        for agent_id, space in enumerate(self.observation_space):
            space.seed(int(seed) + agent_id)
        return result

    def get_avail_actions(self):
        return self.env.get_avail_actions()

    @staticmethod
    def _pad_observations(observations: Sequence[np.ndarray]) -> np.ndarray:
        if len(observations) != 2:
            raise ValueError("Expected two ordered observations.")
        idc = np.asarray(observations[0], dtype=np.float32)
        bess = np.asarray(observations[1], dtype=np.float32)
        if idc.shape != (288,) or bess.shape != (164,):
            raise ValueError(f"Unexpected true observation shapes: {idc.shape}, {bess.shape}.")
        padded_bess = np.pad(bess, (0, 124), constant_values=0).astype(np.float32, copy=False)
        result = np.stack((idc, padded_bess)).astype(np.float32, copy=False)
        if result.shape != (2, 288) or not np.isfinite(result).all():
            raise ValueError("Padded observations must be finite float32 with shape (2, 288).")
        return result

    @staticmethod
    def _dehomogenize_actions(actions: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        array = np.asarray(actions, dtype=np.float32)
        if array.shape != (2, 22):
            raise ValueError(f"Padded HARL actions must have shape (2, 22), got {array.shape}.")
        if not np.isfinite(array).all():
            raise ValueError("Padded HARL actions must be finite.")
        idc_action = array[0].copy()
        bess_full_action = array[1].copy()
        bess_effective_action = bess_full_action[:1].copy()
        return idc_action, bess_effective_action, bess_full_action

    def reset(self):
        observations, share_observations, available_actions = self.env.reset()
        self._last_true_observations = tuple(value.copy() for value in observations)
        self._episode += 1
        self._episode_step = 0
        return self._pad_observations(observations), share_observations, available_actions

    def step(self, actions):
        idc_action, bess_effective_action, bess_full_action = self._dehomogenize_actions(actions)
        result = self.env.step((idc_action, bess_effective_action))
        observations, share_observations, rewards, dones, infos, available_actions = result
        self._last_true_observations = tuple(value.copy() for value in observations)

        self._global_step += 1
        self._episode_step += 1
        if self.diagnostics is not None:
            info = infos[1]
            self.diagnostics.record_step(
                global_step=self._global_step,
                episode=self._episode,
                episode_step=self._episode_step,
                bess_action=bess_full_action,
                reward=float(rewards[1, 0]),
                info=info,
                terminated=bool(info.get("terminated", False)),
                truncated=bool(info.get("truncated", False)),
            )

        return (
            self._pad_observations(observations),
            share_observations,
            rewards,
            dones,
            infos,
            available_actions,
        )

    def render(self, *args, **kwargs):
        return self.env.render(*args, **kwargs)

    def close(self) -> None:
        if self.diagnostics is not None:
            self.diagnostics.close()
        self.env.close()
