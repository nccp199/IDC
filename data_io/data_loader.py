from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np


def load_time_series_csv(path: str | Path, column: str, horizon: int) -> np.ndarray:
    """
    Load one numeric time-series column from a CSV file.

    The CSV is expected to have a header row. Units are not converted here;
    price, carbon_factor, T_amb, PV and WT units are supplied by the user.
    """
    csv_path = Path(path)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    values = []
    with csv_path.open("r", newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"CSV file has no header row: {csv_path}")
        if column not in reader.fieldnames:
            raise ValueError(
                f"Column '{column}' not found in {csv_path}. "
                f"Available columns: {reader.fieldnames}"
            )

        for row_idx, row in enumerate(reader, start=2):
            raw_value = row.get(column)
            if raw_value is None or str(raw_value).strip() == "":
                raise ValueError(f"Missing value in column '{column}' at CSV row {row_idx}: {csv_path}")
            try:
                values.append(float(raw_value))
            except ValueError as exc:
                raise ValueError(
                    f"Non-numeric value in column '{column}' at CSV row {row_idx}: {raw_value!r}"
                ) from exc

    arr = np.asarray(values, dtype=np.float64)
    if arr.shape[0] != int(horizon):
        raise ValueError(
            f"Time series '{column}' length must equal horizon={horizon}, got {arr.shape[0]} "
            f"from {csv_path}."
        )
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"Time series '{column}' contains NaN or infinite values: {csv_path}")
    return arr


def load_optional_time_series_csv(
    path: Optional[str | Path],
    column: Optional[str],
    horizon: int,
    *,
    default_zero: bool = False,
) -> Optional[np.ndarray]:
    """Load an optional external series, or return None / zeros for default simulation mode."""
    if path is None:
        if default_zero:
            return np.zeros(int(horizon), dtype=np.float64)
        return None
    if column is None:
        raise ValueError(f"CSV column must be provided when path is set: {path}")
    return load_time_series_csv(path, column, horizon)


def _build_default_pv_curve(horizon: int, capacity_kw: float, scale_factor: float = 1.0) -> np.ndarray:
    """Create a simple daylight bell-shaped PV curve in kW."""
    horizon = int(horizon)
    effective_capacity_kw = max(float(capacity_kw), 0.0) * max(float(scale_factor), 0.0)
    if horizon <= 0 or effective_capacity_kw <= 0.0:
        return np.zeros(max(horizon, 0), dtype=np.float64)

    # Map arbitrary horizon length onto a 24h solar day: 06:00 sunrise, 18:00 sunset.
    solar_hour = np.arange(horizon, dtype=np.float64) * 24.0 / max(horizon, 1)
    daylight = (solar_hour >= 6.0) & (solar_hour <= 18.0)
    pv = np.zeros(horizon, dtype=np.float64)
    pv[daylight] = effective_capacity_kw * np.sin(np.pi * (solar_hour[daylight] - 6.0) / 12.0)
    return np.maximum(pv, 0.0)


def _normalize_pv_unit(values: np.ndarray, unit: str, scale_factor: float) -> np.ndarray:
    """Convert PV values to kW, apply scenario scaling, and enforce sane data."""
    unit_text = str(unit or "kW").strip().lower()
    if unit_text == "kw":
        factor = 1.0
    elif unit_text == "mw":
        factor = 1000.0
    else:
        raise ValueError(f"Unsupported pv_unit={unit!r}; expected 'kW' or 'MW'.")

    pv = np.asarray(values, dtype=np.float64).reshape(-1) * factor * float(scale_factor)
    if not np.all(np.isfinite(pv)):
        raise ValueError("PV time series contains NaN or infinite values after unit conversion.")
    if np.any(pv < -1e-9):
        raise ValueError("PV time series must be non-negative after unit conversion.")
    return np.maximum(pv, 0.0)


def load_pv_series_from_config(data_config: Dict[str, Any], horizon: int) -> tuple[np.ndarray, float]:
    """Load or synthesize pv_t in kW plus a positive normalization reference."""
    pv_unit = data_config.get("pv_unit", "kW")
    pv_scale_factor = float(data_config.get("pv_scale_factor", 1.0))
    pv_capacity_kw = float(data_config.get("pv_capacity_kw", 0.0))
    use_default_pv_curve = bool(data_config.get("use_default_pv_curve", False))

    path = data_config.get("pv_csv_path")
    column = data_config.get("pv_column")
    if path is None:
        if use_default_pv_curve:
            pv_t = _build_default_pv_curve(
                horizon=horizon,
                capacity_kw=pv_capacity_kw,
                scale_factor=pv_scale_factor,
            )
        else:
            pv_t = np.zeros(int(horizon), dtype=np.float64)
    else:
        if column is None:
            raise ValueError(f"CSV column must be provided when pv_csv_path is set: {path}")
        pv_raw = load_time_series_csv(path, column, horizon)
        pv_t = _normalize_pv_unit(pv_raw, unit=pv_unit, scale_factor=pv_scale_factor)

    if pv_t.shape[0] != int(horizon):
        raise ValueError(f"pv_t length must equal horizon={horizon}, got {pv_t.shape[0]}.")
    if not np.all(np.isfinite(pv_t)):
        raise ValueError("pv_t contains NaN or infinite values.")
    if np.any(pv_t < -1e-9):
        raise ValueError("pv_t must be non-negative.")

    pv_peak_kw = float(np.max(pv_t)) if pv_t.size else 0.0
    ref_kw = max(float(pv_capacity_kw) * max(pv_scale_factor, 0.0), pv_peak_kw, 1e-6)
    return np.maximum(pv_t, 0.0), ref_kw


def build_external_series_from_config(data_config: Dict[str, Any], horizon: int) -> Dict[str, Any]:
    """
    Convert DATA_CONFIG paths into arrays accepted by IDCPriceEnv20D.

    price/carbon/temperature return None when absent so the environment keeps its
    default curves. PV is returned in kW and may use a default daylight curve.
    WT remains an unused reserved interface and defaults to all zeros.
    """
    pv_t, pv_ref_kw = load_pv_series_from_config(data_config, horizon)
    return {
        "price_t": load_optional_time_series_csv(
            data_config.get("price_csv_path"),
            data_config.get("price_column"),
            horizon,
        ),
        "carbon_factor_t": load_optional_time_series_csv(
            data_config.get("carbon_csv_path"),
            data_config.get("carbon_column"),
            horizon,
        ),
        "T_amb": load_optional_time_series_csv(
            data_config.get("temperature_csv_path"),
            data_config.get("temperature_column"),
            horizon,
        ),
        "pv_t": pv_t,
        "pv_ref_kw": pv_ref_kw,
        "allow_pv_export": bool(data_config.get("allow_pv_export", False)),
        "wt_t": load_optional_time_series_csv(
            data_config.get("wt_csv_path"),
            data_config.get("wt_column"),
            horizon,
            default_zero=True,
        ),
    }
