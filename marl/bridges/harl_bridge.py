"""HARL environment protocol adapter for the generic IDC/BESS environment."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from marl.specs import AGENTS, BESS_AGENT, IDC_AGENT


HARL_UPSTREAM_COMMIT = "b1af98b0dbab72a2eee9d160751cd09aedbb8ce2"


class HarlIDCGridBridge:
    """Translate ``IDCGridMultiAgentEnv`` into HARL's raw environment protocol.

    The bridge preserves the genuinely heterogeneous local observation and action
    spaces. It intentionally does not pad either agent. At the pinned upstream
    commit, HARL's stock vector wrappers and on-policy runner still stack the
    agent axis, so a separate heterogeneous-runner adaptation is required before
    MAPPO/HAPPO training. This class implements the environment boundary only.
    """

    agent_order = AGENTS
    n_agents = len(AGENTS)
    upstream_commit = HARL_UPSTREAM_COMMIT
    stock_on_policy_runner_compatible = False

    def __init__(self, env: Any) -> None:
        if tuple(getattr(env, "agents", ())) != self.agent_order:
            raise TypeError(
                f"Expected IDCGridMultiAgentEnv agents {self.agent_order}, "
                f"got {getattr(env, 'agents', None)}."
            )
        if not hasattr(env, "observation_spaces") or not hasattr(env, "action_spaces"):
            raise TypeError("Bridge requires the generic observation_spaces/action_spaces interface.")
        if not hasattr(env, "state_space"):
            raise TypeError("Bridge requires the generic centralized state_space interface.")

        self.env = env
        self.observation_space = tuple(env.observation_spaces[agent] for agent in self.agent_order)
        self.share_observation_space = tuple(env.state_space for _ in self.agent_order)
        self.action_space = tuple(env.action_spaces[agent] for agent in self.agent_order)
        self._validate_spaces()

        self._pending_seed: int | None = None
        self._last_info: dict[str, Any] | None = None

    def _validate_spaces(self) -> None:
        obs_shapes = tuple(space.shape for space in self.observation_space)
        action_shapes = tuple(space.shape for space in self.action_space)
        state_shapes = tuple(space.shape for space in self.share_observation_space)
        if obs_shapes != ((288,), (164,)):
            raise ValueError(f"Current HARL bridge requires obs shapes ((288,), (164,)), got {obs_shapes}.")
        if action_shapes != ((22,), (1,)):
            raise ValueError(f"Current HARL bridge requires action shapes ((22,), (1,)), got {action_shapes}.")
        if state_shapes != ((294,), (294,)):
            raise ValueError(f"Current HARL bridge requires repeated state shape (294,), got {state_shapes}.")

    @property
    def last_info(self) -> dict[str, Any] | None:
        """Return the last generic reset/step info without exposing lower wrappers."""
        return None if self._last_info is None else dict(self._last_info)

    def seed(self, seed: int) -> list[int]:
        """Store a seed for the next reset, matching HARL's factory call order."""
        self._pending_seed = int(seed)
        for agent_id, space in enumerate(self.action_space):
            space.seed(self._pending_seed + agent_id)
        for agent_id, space in enumerate(self.observation_space):
            space.seed(self._pending_seed + agent_id)
        return [self._pending_seed]

    def get_avail_actions(self):
        """Continuous Box actions have no discrete availability mask in HARL."""
        return None

    def _ordered_observations(self, obs_dict: Mapping[str, Any]) -> tuple[np.ndarray, ...]:
        if set(obs_dict) != set(self.agent_order):
            raise KeyError(f"Observation dictionary must contain exactly {self.agent_order}.")
        observations = tuple(
            np.asarray(obs_dict[agent], dtype=np.float32).copy() for agent in self.agent_order
        )
        for agent_id, (obs, space) in enumerate(zip(observations, self.observation_space, strict=True)):
            if obs.shape != space.shape or not np.isfinite(obs).all():
                raise ValueError(
                    f"Agent {agent_id} observation must be finite with shape {space.shape}; got {obs.shape}."
                )
        return observations

    def _shared_observations(self, state: Any) -> np.ndarray:
        state_array = np.asarray(state, dtype=np.float32)
        expected_shape = self.share_observation_space[0].shape
        if state_array.shape != expected_shape or not np.isfinite(state_array).all():
            raise ValueError(
                f"Centralized state must be finite with shape {expected_shape}; got {state_array.shape}."
            )
        return np.repeat(state_array[np.newaxis, :], self.n_agents, axis=0)

    def _action_dict(self, actions: Any) -> dict[str, Any]:
        if isinstance(actions, (str, bytes, Mapping)):
            raise TypeError("HARL actions must be an ordered two-element sequence, not a mapping.")
        try:
            action_count = len(actions)
        except TypeError as exc:
            raise TypeError("HARL actions must be an ordered two-element sequence.") from exc
        if action_count != self.n_agents:
            raise ValueError(f"HARL actions must contain {self.n_agents} agents, got {action_count}.")
        return {
            IDC_AGENT: actions[0],
            BESS_AGENT: actions[1],
        }

    def reset(self):
        """Return heterogeneous local observations, repeated EP state, and no mask."""
        seed = self._pending_seed
        self._pending_seed = None
        obs_dict, state, info = self.env.reset(seed=seed, options=None)
        self._last_info = dict(info)
        return (
            self._ordered_observations(obs_dict),
            self._shared_observations(state),
            self.get_avail_actions(),
        )

    def step(self, actions):
        """Convert one ordered HARL action collection into the generic dictionary API."""
        (
            obs_dict,
            state,
            reward_dict,
            terminated_dict,
            truncated_dict,
            info,
        ) = self.env.step(self._action_dict(actions))
        self._last_info = dict(info)

        terminated = bool(terminated_dict["__all__"])
        truncated = bool(truncated_dict["__all__"])
        done = terminated or truncated
        rewards = np.asarray(
            [[reward_dict[agent]] for agent in self.agent_order],
            dtype=np.float32,
        )
        if rewards.shape != (self.n_agents, 1) or not np.isfinite(rewards).all():
            raise ValueError("HARL rewards must be finite with shape (n_agents, 1).")
        dones = np.full(self.n_agents, done, dtype=np.bool_)
        infos = []
        for agent in self.agent_order:
            agent_info = dict(info)
            agent_info["agent_id"] = agent
            agent_info["terminated"] = terminated
            agent_info["truncated"] = truncated
            agent_info["bad_transition"] = truncated
            infos.append(agent_info)

        return (
            self._ordered_observations(obs_dict),
            self._shared_observations(state),
            rewards,
            dones,
            infos,
            self.get_avail_actions(),
        )

    def render(self, *args, **kwargs):
        """Rendering is not defined by the generic environment in this stage."""
        raise NotImplementedError("IDCGridMultiAgentEnv does not expose rendering.")

    def close(self) -> None:
        """Close only through the generic wrapper when it exposes a lifecycle hook."""
        close_method = getattr(self.env, "close", None)
        if callable(close_method):
            close_method()
