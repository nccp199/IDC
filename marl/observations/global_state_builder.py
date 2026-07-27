"""Centralized state construction."""

from __future__ import annotations

import numpy as np


class GlobalStateBuilder:
    """Append the shared supplemental BESS state to the wrapped observation."""

    def __init__(self, wrapped_obs_dim: int, supplemental_dim: int = 6) -> None:
        self.wrapped_obs_dim = int(wrapped_obs_dim)
        self.supplemental_dim = int(supplemental_dim)
        self.state_dim = self.wrapped_obs_dim + self.supplemental_dim

    def build(self, raw_wrapped_obs, supplemental_values) -> np.ndarray:
        raw = np.asarray(raw_wrapped_obs, dtype=np.float32)
        supplemental = np.asarray(supplemental_values, dtype=np.float32)
        if raw.shape != (self.wrapped_obs_dim,):
            raise ValueError(f"Wrapped observation has unexpected shape {raw.shape}.")
        if supplemental.shape != (self.supplemental_dim,):
            raise ValueError(f"Supplemental state has unexpected shape {supplemental.shape}.")
        if not np.isfinite(raw).all() or not np.isfinite(supplemental).all():
            raise ValueError("Centralized state inputs contain NaN or infinity.")
        return np.concatenate((raw, supplemental)).astype(np.float32, copy=False)
