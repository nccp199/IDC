"""IDC-agent observation construction."""

from __future__ import annotations

import numpy as np


class IDCObservationBuilder:
    """Expose the complete wrapped observation to the IDC agent."""

    def __init__(self, wrapped_obs_dim: int) -> None:
        self.observation_dim = int(wrapped_obs_dim)

    def build(self, raw_wrapped_obs) -> np.ndarray:
        obs = np.asarray(raw_wrapped_obs, dtype=np.float32)
        if obs.shape != (self.observation_dim,):
            raise ValueError(
                f"Wrapped observation must have shape ({self.observation_dim},), got {obs.shape}."
            )
        if not np.isfinite(obs).all():
            raise ValueError("Wrapped observation contains NaN or infinity.")
        return obs.copy()
