"""Controlled task-arrival forecasts with RNG isolation from environment truth."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import numpy as np


SUPPORTED_TASK_FORECAST_MODES: Final = ("perfect", "noisy", "none")
SYNTHETIC_FORECAST_SOURCE: Final = (
    "synthetic forecast generated from ground truth with controlled error"
)


@dataclass(frozen=True)
class TaskForecastMetrics:
    mae: float
    rmse: float
    mape_nonzero_percent: float


def validate_task_forecast_config(mode: str, error_level: float) -> tuple[str, float]:
    normalized_mode = str(mode).strip().lower()
    if normalized_mode not in SUPPORTED_TASK_FORECAST_MODES:
        raise ValueError(
            f"task_forecast_mode must be one of {SUPPORTED_TASK_FORECAST_MODES}, "
            f"got {mode!r}."
        )
    normalized_error = float(error_level)
    if not np.isfinite(normalized_error) or normalized_error < 0.0:
        raise ValueError("forecast_error_level must be finite and non-negative.")
    if normalized_mode == "noisy" and normalized_error == 0.0:
        raise ValueError(
            "forecast_error_level must be positive when task_forecast_mode='noisy'."
        )
    return normalized_mode, normalized_error


def generate_task_arrival_forecast(
    true_profile: np.ndarray,
    *,
    mode: str,
    error_level: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Generate one episode-level forecast without consuming task/server RNGs.

    ``noisy`` is deliberately a simulation oracle-corruption model, not a learned
    forecaster: independent Gaussian relative errors are applied once at reset.
    The resulting array is then held fixed for the full episode.
    """
    normalized_mode, normalized_error = validate_task_forecast_config(mode, error_level)
    truth = np.asarray(true_profile, dtype=np.float64).reshape(-1)
    if not np.all(np.isfinite(truth)) or np.any(truth < 0.0):
        raise ValueError("true task-arrival profile must be finite and non-negative.")

    if normalized_mode == "perfect":
        return truth.copy()
    if normalized_mode == "none":
        return np.zeros_like(truth)

    relative_error = rng.normal(loc=0.0, scale=normalized_error, size=truth.shape)
    return np.maximum(0.0, truth * (1.0 + relative_error))


def task_forecast_metrics(
    true_profile: np.ndarray, forecast_profile: np.ndarray
) -> TaskForecastMetrics:
    truth = np.asarray(true_profile, dtype=np.float64).reshape(-1)
    forecast = np.asarray(forecast_profile, dtype=np.float64).reshape(-1)
    if truth.shape != forecast.shape:
        raise ValueError(
            f"task forecast shape {forecast.shape} does not match truth shape {truth.shape}."
        )
    error = forecast - truth
    nonzero = np.abs(truth) > 1e-12
    mape = (
        float(np.mean(np.abs(error[nonzero]) / np.abs(truth[nonzero])) * 100.0)
        if np.any(nonzero)
        else 0.0
    )
    return TaskForecastMetrics(
        mae=float(np.mean(np.abs(error))),
        rmse=float(np.sqrt(np.mean(np.square(error)))),
        mape_nonzero_percent=mape,
    )
