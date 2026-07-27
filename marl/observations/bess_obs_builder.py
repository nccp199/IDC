"""BESS-local observation construction from documented legacy slices."""

from __future__ import annotations

import numpy as np


class BESSObservationBuilder:
    """Keep global, forecast, grid, and BESS supplemental state only."""

    def __init__(
        self,
        *,
        wrapped_obs_dim: int,
        base_obs_dim: int,
        forecast_obs_dim: int,
        grid_obs_dim: int,
        global_obs_dim: int = 6,
        supplemental_dim: int = 6,
    ) -> None:
        self.wrapped_obs_dim = int(wrapped_obs_dim)
        self.base_obs_dim = int(base_obs_dim)
        self.forecast_obs_dim = int(forecast_obs_dim)
        self.grid_obs_dim = int(grid_obs_dim)
        self.global_obs_dim = int(global_obs_dim)
        self.supplemental_dim = int(supplemental_dim)
        self.forecast_start = self.base_obs_dim - self.forecast_obs_dim
        self.observation_dim = (
            self.global_obs_dim
            + self.forecast_obs_dim
            + self.grid_obs_dim
            + self.supplemental_dim
        )
        if self.forecast_start < self.global_obs_dim:
            raise ValueError("Forecast slice overlaps the current global feature slice.")
        if self.wrapped_obs_dim != self.base_obs_dim + self.grid_obs_dim:
            raise ValueError("Wrapped observation dimension does not equal base plus grid dimensions.")

    def build(self, raw_wrapped_obs, supplemental_values) -> np.ndarray:
        raw = np.asarray(raw_wrapped_obs, dtype=np.float32)
        supplemental = np.asarray(supplemental_values, dtype=np.float32)
        if raw.shape != (self.wrapped_obs_dim,):
            raise ValueError(f"Wrapped observation has unexpected shape {raw.shape}.")
        if supplemental.shape != (self.supplemental_dim,):
            raise ValueError(f"Supplemental state has unexpected shape {supplemental.shape}.")
        if not np.isfinite(raw).all() or not np.isfinite(supplemental).all():
            raise ValueError("BESS observation inputs contain NaN or infinity.")

        global_features = raw[: self.global_obs_dim]
        forecast_features = raw[self.forecast_start : self.base_obs_dim]
        grid_features = raw[self.base_obs_dim : self.wrapped_obs_dim]
        return np.concatenate(
            (global_features, forecast_features, grid_features, supplemental)
        ).astype(np.float32, copy=False)
