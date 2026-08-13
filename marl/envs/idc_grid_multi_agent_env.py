"""Generic two-agent composition layer over the existing GridCoupledEnv."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
from gymnasium import spaces

from marl.adapters import FlatActionAdapter
from marl.observations import BESSObservationBuilder, GlobalStateBuilder, IDCObservationBuilder
from marl.specs import AGENTS, BESS_AGENT, IDC_AGENT, SUPPLEMENTAL_FIELDS
from marl.specs.state_specs import GRID_BUS_DYNAMIC_STATE_DIM


class IDCGridMultiAgentEnv:
    """Expose IDC and BESS agents without changing the legacy transition function.

    The wrapped object is expected to be ``GridCoupledEnv``. Environment execution
    always uses ``self.env.reset`` and ``self.env.step(flat_action)``. The only
    lower-layer read is ``self.env.env.bess_soc`` / ``bess_energy_kWh`` at reset,
    because the base reset info does not publish these two initialized values.
    Transition power values are never read ahead; they come from the info returned
    by the completed step and therefore belong to the next decision state.
    """

    agents = AGENTS

    def __init__(self, env: Any) -> None:
        self.env = env
        self._validate_wrapped_env()

        self.flat_action_dim = int(self.env.action_space.shape[0])
        self.action_adapter = FlatActionAdapter(self.flat_action_dim)
        self.idc_action_dim = self.action_adapter.idc_action_dim
        self.bess_action_dim = self.action_adapter.bess_action_dim

        self.base_obs_dim = int(self.env.base_obs_dim)
        self.grid_obs_dim = int(self.env.grid_obs_dim)
        self.wrapped_obs_dim = int(self.env.observation_space.shape[0])
        base_env = self.env.env
        self.forecast_obs_dim = int(base_env.forecast_obs_dim)
        self.global_obs_dim = int(base_env.global_obs_dim)

        self.idc_obs_builder = IDCObservationBuilder(self.wrapped_obs_dim)
        self.bess_obs_builder = BESSObservationBuilder(
            wrapped_obs_dim=self.wrapped_obs_dim,
            base_obs_dim=self.base_obs_dim,
            forecast_obs_dim=self.forecast_obs_dim,
            grid_obs_dim=self.grid_obs_dim,
            global_obs_dim=self.global_obs_dim,
            supplemental_dim=len(SUPPLEMENTAL_FIELDS),
        )
        self.state_builder = GlobalStateBuilder(
            self.wrapped_obs_dim,
            supplemental_dim=len(SUPPLEMENTAL_FIELDS),
            grid_bus_dynamic_dim=GRID_BUS_DYNAMIC_STATE_DIM,
        )

        self.action_spaces = {
            IDC_AGENT: spaces.Box(0.0, 1.0, shape=(self.idc_action_dim,), dtype=np.float32),
            BESS_AGENT: spaces.Box(0.0, 1.0, shape=(self.bess_action_dim,), dtype=np.float32),
        }
        self.observation_spaces = {
            IDC_AGENT: spaces.Box(
                -np.inf,
                np.inf,
                shape=(self.idc_obs_builder.observation_dim,),
                dtype=np.float32,
            ),
            BESS_AGENT: spaces.Box(
                -np.inf,
                np.inf,
                shape=(self.bess_obs_builder.observation_dim,),
                dtype=np.float32,
            ),
        }
        self.state_space = spaces.Box(
            -np.inf,
            np.inf,
            shape=(self.state_builder.state_dim,),
            dtype=np.float32,
        )

        self._last_info: dict[str, Any] | None = None
        self._last_raw_observation: np.ndarray | None = None
        self._last_flat_action: np.ndarray | None = None

    def _validate_wrapped_env(self) -> None:
        required = ("action_space", "observation_space", "base_obs_dim", "grid_obs_dim", "enable_grid_obs", "env")
        missing = [name for name in required if not hasattr(self.env, name)]
        if missing:
            raise TypeError(f"Expected a GridCoupledEnv-compatible object; missing {missing}.")
        if not bool(self.env.enable_grid_obs):
            raise ValueError("The first multi-agent contract requires GridCoupledEnv grid observations.")
        if tuple(self.env.action_space.shape) != (23,):
            raise ValueError(
                f"Current contract requires legacy action shape (23,), got {self.env.action_space.shape}."
            )
        if int(self.env.grid_obs_dim) != 8:
            raise ValueError(f"Current contract requires 8 grid features, got {self.env.grid_obs_dim}.")
        base_env = self.env.env
        base_required = (
            "forecast_obs_dim",
            "global_obs_dim",
            "task_pool_obs_dim",
            "server_feature_groups",
            "forecast_feature_groups",
            "horizon",
            "model",
        )
        base_missing = [name for name in base_required if not hasattr(base_env, name)]
        if base_missing:
            raise TypeError(f"GridCoupledEnv base environment is missing {base_missing}.")
        feature_contract = {
            "global_obs_dim": 6,
            "task_pool_obs_dim": 10,
            "server_feature_groups": 6,
            "forecast_feature_groups": 6,
        }
        mismatches = {
            name: (expected, int(getattr(base_env, name)))
            for name, expected in feature_contract.items()
            if int(getattr(base_env, name)) != expected
        }
        if mismatches:
            raise ValueError(f"Base observation feature contract mismatch: {mismatches}.")
        expected_base_dim = (
            int(base_env.global_obs_dim)
            + int(base_env.task_pool_obs_dim)
            + int(base_env.server_feature_groups) * int(base_env.model.N)
            + int(base_env.forecast_feature_groups) * int(base_env.horizon)
        )
        if int(self.env.base_obs_dim) != expected_base_dim:
            raise ValueError(
                "Base observation contract mismatch: expected "
                f"6 + 10 + 6*N + 6*horizon = {expected_base_dim}, got {self.env.base_obs_dim}."
            )
        expected_wrapped_dim = expected_base_dim + int(self.env.grid_obs_dim)
        if tuple(self.env.observation_space.shape) != (expected_wrapped_dim,):
            raise ValueError(
                f"Wrapped observation must have shape ({expected_wrapped_dim},), "
                f"got {self.env.observation_space.shape}."
            )

    @property
    def last_info(self) -> dict[str, Any] | None:
        """Return a shallow copy of the last completed reset/transition info."""
        return None if self._last_info is None else dict(self._last_info)

    @property
    def last_raw_observation(self) -> np.ndarray | None:
        """Return the unmodified observation most recently returned by GridCoupledEnv."""
        return None if self._last_raw_observation is None else self._last_raw_observation.copy()

    @property
    def last_flat_action(self) -> np.ndarray | None:
        """Return the validated flat action sent through GridCoupledEnv.step."""
        return None if self._last_flat_action is None else self._last_flat_action.copy()

    def compose_action(self, idc_action, bess_action) -> np.ndarray:
        return self.action_adapter.compose_action(idc_action, bess_action)

    def split_action(self, flat_action) -> tuple[np.ndarray, np.ndarray]:
        return self.action_adapter.split_action(flat_action)

    @staticmethod
    def _finite_float(value: Any, *, field: str) -> float:
        try:
            result = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Supplemental field {field!r} is not numeric.") from exc
        if not np.isfinite(result):
            raise ValueError(f"Supplemental field {field!r} is not finite.")
        return result

    def _build_supplemental_values(self, info: Mapping[str, Any], *, initial: bool) -> np.ndarray:
        if initial:
            base_env = self.env.env
            values = (
                getattr(base_env, "bess_soc", None),
                getattr(base_env, "bess_energy_kWh", None),
                0.0,
                0.0,
                0.0,
                0.0,
            )
        else:
            missing = [field for field in SUPPLEMENTAL_FIELDS if field not in info]
            if missing:
                raise KeyError(f"Transition info is missing supplemental fields: {missing}.")
            values = tuple(info[field] for field in SUPPLEMENTAL_FIELDS)
        return np.asarray(
            [
                self._finite_float(value, field=field)
                for field, value in zip(SUPPLEMENTAL_FIELDS, values, strict=True)
            ],
            dtype=np.float32,
        )

    def _build_outputs(
        self,
        raw_wrapped_obs,
        info: Mapping[str, Any],
        *,
        initial: bool,
    ) -> tuple[dict[str, np.ndarray], np.ndarray]:
        supplemental = self._build_supplemental_values(info, initial=initial)
        if "grid_bus_dynamic_state" not in info:
            raise KeyError("Grid info is missing grid_bus_dynamic_state.")
        grid_bus_dynamic_state = np.asarray(
            info["grid_bus_dynamic_state"], dtype=np.float32
        ).reshape(-1)
        obs_dict = {
            IDC_AGENT: self.idc_obs_builder.build(raw_wrapped_obs),
            BESS_AGENT: self.bess_obs_builder.build(raw_wrapped_obs, supplemental),
        }
        state = self.state_builder.build(
            raw_wrapped_obs, supplemental, grid_bus_dynamic_state
        )
        return obs_dict, state

    def reset(self, seed=None, options=None):
        """Reset the wrapped environment and publish only initial-time information."""
        raw_obs, info = self.env.reset(seed=seed, options=options)
        info = dict(info)
        self._last_info = info
        self._last_raw_observation = np.asarray(raw_obs, dtype=np.float32).copy()
        self._last_flat_action = None
        if seed is not None:
            self.action_spaces[IDC_AGENT].seed(int(seed))
            self.action_spaces[BESS_AGENT].seed(int(seed) + 1)
        obs_dict, state = self._build_outputs(raw_obs, info, initial=True)
        return obs_dict, state, info

    def step(self, action_dict):
        """Compose one flat action and return the resulting next-decision state."""
        if not isinstance(action_dict, Mapping):
            raise TypeError("action_dict must be a mapping with exactly 'idc' and 'bess' keys.")
        if set(action_dict) != set(self.agents):
            raise KeyError(f"action_dict must contain exactly {self.agents}, got {tuple(action_dict)}.")
        flat_action = self.compose_action(action_dict[IDC_AGENT], action_dict[BESS_AGENT])
        raw_obs, reward, terminated, truncated, info = self.env.step(flat_action)
        info = dict(info)

        # Update transition memory only after GridCoupledEnv completes action_t.
        self._last_info = info
        self._last_raw_observation = np.asarray(raw_obs, dtype=np.float32).copy()
        self._last_flat_action = flat_action.copy()
        obs_dict, state = self._build_outputs(raw_obs, info, initial=False)

        reward_value = float(reward)
        if not np.isfinite(reward_value):
            raise ValueError("GridCoupledEnv returned a non-finite reward.")
        terminated_value = bool(terminated)
        truncated_value = bool(truncated)
        reward_dict = {agent: reward_value for agent in self.agents}
        terminated_dict = {
            IDC_AGENT: terminated_value,
            BESS_AGENT: terminated_value,
            "__all__": terminated_value,
        }
        truncated_dict = {
            IDC_AGENT: truncated_value,
            BESS_AGENT: truncated_value,
            "__all__": truncated_value,
        }
        return obs_dict, state, reward_dict, terminated_dict, truncated_dict, info
