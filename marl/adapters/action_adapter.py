"""Validated conversion between per-agent and legacy flat actions."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


class FlatActionAdapter:
    """Split the final BESS dimension from a legacy flat action vector."""

    def __init__(self, flat_action_dim: int, bess_action_dim: int = 1) -> None:
        self.flat_action_dim = int(flat_action_dim)
        self.bess_action_dim = int(bess_action_dim)
        self.idc_action_dim = self.flat_action_dim - self.bess_action_dim
        if self.flat_action_dim <= 1 or self.bess_action_dim != 1:
            raise ValueError(
                "The first multi-agent interface requires one final BESS action "
                "and at least one IDC action."
            )

    @staticmethod
    def _validate_vector(value, *, name: str, dimension: int) -> np.ndarray:
        if isinstance(value, (str, bytes, dict)) or not isinstance(value, (np.ndarray, Sequence)):
            raise TypeError(f"{name} must be a one-dimensional numeric array or sequence.")
        try:
            array = np.asarray(value, dtype=np.float32)
        except (TypeError, ValueError) as exc:
            raise TypeError(f"{name} must contain only numeric values.") from exc
        if array.shape != (dimension,):
            raise ValueError(f"{name} must have shape ({dimension},), got {array.shape}.")
        if not np.isfinite(array).all():
            raise ValueError(f"{name} must contain only finite values.")
        if np.any(array < 0.0) or np.any(array > 1.0):
            raise ValueError(f"{name} values must be in [0, 1]; values are not clipped.")
        return array

    def compose_action(self, idc_action, bess_action) -> np.ndarray:
        """Return the unchanged IDC prefix followed by the unchanged BESS value."""
        idc = self._validate_vector(
            idc_action,
            name="idc_action",
            dimension=self.idc_action_dim,
        )
        bess = self._validate_vector(
            bess_action,
            name="bess_action",
            dimension=self.bess_action_dim,
        )
        flat = np.concatenate((idc, bess)).astype(np.float32, copy=False)
        if flat.shape != (self.flat_action_dim,):
            raise RuntimeError(f"Composed flat action has unexpected shape {flat.shape}.")
        return flat

    def split_action(self, flat_action) -> tuple[np.ndarray, np.ndarray]:
        """Split a validated legacy action for tests and debugging."""
        flat = self._validate_vector(
            flat_action,
            name="flat_action",
            dimension=self.flat_action_dim,
        )
        split = self.idc_action_dim
        return flat[:split].copy(), flat[split:].copy()
