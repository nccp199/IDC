"""Detached diagnostics for HARL's 22-dimensional padded BESS policy."""

from __future__ import annotations

from typing import Any

import numpy as np


def compute_bess_policy_diagnostics(
    *,
    actions: Any,
    old_log_prob: Any,
    new_log_prob: Any,
    advantages: Any,
    clip_param: float,
    action_mask: Any | None = None,
) -> dict[str, float]:
    """Compute per-dimension and counterfactual PPO metrics from detached arrays.

    This matches the pinned HARL MAPPO default ``action_aggregation=prod``:
    ratios are computed per dimension and multiplied. Returned values are plain
    floats and cannot participate in autograd or replace the real MAPPO loss.
    """
    action = _matrix(actions, "actions")
    old = _matrix(old_log_prob, "old_log_prob")
    new = _matrix(new_log_prob, "new_log_prob")
    if not (action.shape == old.shape == new.shape):
        raise ValueError("Policy diagnostic arrays must have identical shapes.")

    advantage = np.asarray(advantages, dtype=np.float64).reshape(-1)
    if advantage.size == 1:
        advantage = np.repeat(advantage, action.shape[0])
    if advantage.shape != (action.shape[0],) or not np.isfinite(advantage).all():
        raise ValueError("advantages must be finite and match the batch dimension.")

    mask = _action_mask(action_mask)

    # The affine-tanh distribution has no analytic entropy in this integration.
    # Use the same bounded action samples and transformed per-dimension density
    # evaluated by HARL. This remains detached diagnostic data only.
    entropy = -new
    log_ratio = new - old
    per_dim_ratio = np.exp(log_ratio)
    full_log_ratio = np.sum(log_ratio, axis=-1)
    effective_log_ratio = np.sum(log_ratio * mask, axis=-1)
    full_ratio = np.exp(full_log_ratio)
    effective_ratio = np.exp(effective_log_ratio)
    lower, upper = 1.0 - float(clip_param), 1.0 + float(clip_param)
    full_clipped = (full_ratio < lower) | (full_ratio > upper)
    effective_clipped = (effective_ratio < lower) | (effective_ratio > upper)

    full_loss = _diagnostic_policy_loss(full_ratio, advantage, lower, upper)
    effective_loss = _diagnostic_policy_loss(effective_ratio, advantage, lower, upper)
    effective_entropy = np.sum(entropy * mask, axis=-1)
    full_entropy = np.sum(entropy, axis=-1)
    virtual_entropy = np.sum(entropy * (1.0 - mask), axis=-1)
    entropy_fraction = np.divide(
        virtual_entropy,
        full_entropy,
        out=np.zeros_like(virtual_entropy),
        where=np.abs(full_entropy) > 1e-12,
    )

    return {
        "effective_action_mean": float(np.mean(action[:, 0])),
        "effective_action_std": float(np.std(action[:, 0])),
        "virtual_action_mean_abs": float(np.mean(np.abs(action[:, 1:]))),
        "virtual_action_std_mean": float(np.mean(np.std(action[:, 1:], axis=0))),
        "effective_entropy": float(np.mean(effective_entropy)),
        "virtual_entropy_sum": float(np.mean(virtual_entropy)),
        "full_entropy": float(np.mean(full_entropy)),
        "virtual_entropy_fraction": float(np.mean(entropy_fraction)),
        "effective_log_prob": float(np.mean(np.sum(new * mask, axis=-1))),
        "virtual_log_prob_sum": float(
            np.mean(np.sum(new * (1.0 - mask), axis=-1))
        ),
        "full_log_prob": float(np.mean(np.sum(new, axis=-1))),
        "full_ratio": float(np.mean(full_ratio)),
        "effective_ratio": float(np.mean(effective_ratio)),
        "ratio_gap": float(np.mean(np.abs(full_ratio - effective_ratio))),
        "full_ratio_clip_fraction": float(np.mean(full_clipped)),
        "effective_ratio_clip_fraction": float(np.mean(effective_clipped)),
        "clip_disagreement_fraction": float(np.mean(full_clipped != effective_clipped)),
        "effective_approx_kl": float(
            np.mean((effective_ratio - 1.0) - effective_log_ratio)
        ),
        "full_approx_kl": float(np.mean((full_ratio - 1.0) - full_log_ratio)),
        "diagnostic_policy_loss_full": full_loss,
        "diagnostic_policy_loss_effective_only": effective_loss,
        "diagnostic_policy_loss_delta": float(full_loss - effective_loss),
    }


def _action_mask(value: Any | None) -> np.ndarray:
    if value is None:
        result = np.zeros((1, 22), dtype=np.float64)
        result[0, 0] = 1.0
        return result
    result = np.asarray(value, dtype=np.float64)
    if result.shape == (22,):
        result = result.reshape(1, 22)
    if result.shape != (1, 22):
        raise ValueError("action_mask must have shape (22,) or (1, 22).")
    if not np.isfinite(result).all() or not np.all((result == 0.0) | (result == 1.0)):
        raise ValueError("action_mask must contain only finite zeros and ones.")
    if float(result.sum()) <= 0.0:
        raise ValueError("action_mask must enable at least one action dimension.")
    return result


def _matrix(value: Any, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.shape == (22,):
        array = array.reshape(1, 22)
    if array.ndim != 2 or array.shape[-1] != 22 or not np.isfinite(array).all():
        raise ValueError(f"{name} must be finite with shape (batch, 22).")
    return array


def _diagnostic_policy_loss(
    ratio: np.ndarray,
    advantage: np.ndarray,
    lower: float,
    upper: float,
) -> float:
    unclipped = ratio * advantage
    clipped = np.clip(ratio, lower, upper) * advantage
    return float(-np.mean(np.minimum(unclipped, clipped)))
