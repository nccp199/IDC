"""Centralized state construction."""

from __future__ import annotations

import numpy as np


class GlobalStateBuilder:
    """Append shared BESS and current per-bus state to the wrapped observation."""

    def __init__(
        self,
        wrapped_obs_dim: int,
        supplemental_dim: int = 6,
        grid_bus_dynamic_dim: int = 70,
    ) -> None:
        self.wrapped_obs_dim = int(wrapped_obs_dim)
        self.supplemental_dim = int(supplemental_dim)
        self.grid_bus_dynamic_dim = int(grid_bus_dynamic_dim)
        self.state_dim = (
            self.wrapped_obs_dim + self.supplemental_dim + self.grid_bus_dynamic_dim
        )

    def build(
        self, raw_wrapped_obs, supplemental_values, grid_bus_dynamic_state
    ) -> np.ndarray:
        raw = np.asarray(raw_wrapped_obs, dtype=np.float32)
        supplemental = np.asarray(supplemental_values, dtype=np.float32)
        grid_bus_dynamic = np.asarray(
            grid_bus_dynamic_state, dtype=np.float32
        ).reshape(-1)
        if raw.shape != (self.wrapped_obs_dim,):
            raise ValueError(f"Wrapped observation has unexpected shape {raw.shape}.")
        if supplemental.shape != (self.supplemental_dim,):
            raise ValueError(f"Supplemental state has unexpected shape {supplemental.shape}.")
        if grid_bus_dynamic.shape != (self.grid_bus_dynamic_dim,):
            raise ValueError(
                f"Grid bus dynamic state has unexpected shape {grid_bus_dynamic.shape}."
            )
        if (
            not np.isfinite(raw).all()
            or not np.isfinite(supplemental).all()
            or not np.isfinite(grid_bus_dynamic).all()
        ):
            raise ValueError("Centralized state inputs contain NaN or infinity.")
        return np.concatenate(
            (raw, supplemental, grid_bus_dynamic)
        ).astype(np.float32, copy=False)
