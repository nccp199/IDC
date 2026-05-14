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


def build_external_series_from_config(data_config: Dict[str, Any], horizon: int) -> Dict[str, Optional[np.ndarray]]:
    """
    Convert DATA_CONFIG paths into arrays accepted by IDCPriceEnv20D.

    price/carbon/temperature return None when absent so the environment keeps its
    default curves. PV/WT are reserved for future use and default to all zeros.
    """
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
        "pv_t": load_optional_time_series_csv(
            data_config.get("pv_csv_path"),
            data_config.get("pv_column"),
            horizon,
            default_zero=True,
        ),
        "wt_t": load_optional_time_series_csv(
            data_config.get("wt_csv_path"),
            data_config.get("wt_column"),
            horizon,
            default_zero=True,
        ),
    }
