"""Generator emission factor and total emission helpers."""

from __future__ import annotations

import math

from .grid_case import GridCase, OPFResult

DEFAULT_EMISSION_FACTOR_KG_PER_MWH = 600.0


def build_default_gen_emission_factors(grid_case: GridCase) -> dict[int, float]:
    """Build deterministic first-pass generator emission factors."""

    factors: dict[int, float] = {}
    fuel_type_by_gen_id = grid_case.metadata.get("gen_fuel_type_by_id", {})

    for order, gen_id in enumerate(grid_case.gen_ids):
        fuel_factor = _factor_from_fuel_type(fuel_type_by_gen_id.get(gen_id))
        if fuel_factor is not None:
            factors[int(gen_id)] = fuel_factor
        elif order == 0:
            factors[int(gen_id)] = 850.0
        elif order == 1:
            factors[int(gen_id)] = 450.0
        else:
            factors[int(gen_id)] = 100.0

    return factors


def compute_total_emission(
    gen_power_mw: dict[int, float],
    gen_emission_factors_kg_per_mwh: dict[int, float],
    delta_t_hours: float = 1.0,
) -> tuple[float, dict[int, float]]:
    """Compute generator emissions in kgCO2 over a time interval."""

    duration = float(delta_t_hours)
    if duration < 0.0:
        raise ValueError(f"delta_t_hours must be non-negative, got {delta_t_hours!r}")

    emission_by_gen: dict[int, float] = {}
    total_emission = 0.0

    for gen_id, power_mw in gen_power_mw.items():
        generation_mw = _non_negative_finite(power_mw)
        factor = float(gen_emission_factors_kg_per_mwh.get(int(gen_id), DEFAULT_EMISSION_FACTOR_KG_PER_MWH))
        emission_kg = generation_mw * duration * factor
        emission_by_gen[int(gen_id)] = emission_kg
        total_emission += emission_kg

    return total_emission, emission_by_gen


def build_generator_emission_table(
    opf_result: OPFResult,
    gen_emission_factors_kg_per_mwh: dict[int, float],
    delta_t_hours: float = 1.0,
) -> list[dict[str, float | int]]:
    """Build a compact per-generator dispatch and emission table."""

    _, emission_by_gen = compute_total_emission(
        opf_result.gen_power_mw,
        gen_emission_factors_kg_per_mwh,
        delta_t_hours=delta_t_hours,
    )

    rows: list[dict[str, float | int]] = []
    for gen_id in sorted(opf_result.gen_power_mw):
        factor = float(gen_emission_factors_kg_per_mwh.get(gen_id, DEFAULT_EMISSION_FACTOR_KG_PER_MWH))
        rows.append(
            {
                "gen_id": int(gen_id),
                "gen_power_mw": _non_negative_finite(opf_result.gen_power_mw.get(gen_id, 0.0)),
                "emission_factor_kg_per_mwh": factor,
                "gen_emission_kg": emission_by_gen.get(gen_id, 0.0),
            }
        )
    return rows


def _factor_from_fuel_type(fuel_type: object) -> float | None:
    if fuel_type is None:
        return None

    text = str(fuel_type).lower()
    if "coal" in text:
        return 850.0
    if "gas" in text:
        return 450.0
    if any(token in text for token in ("wind", "solar", "hydro", "nuclear", "low", "clean")):
        return 100.0
    return DEFAULT_EMISSION_FACTOR_KG_PER_MWH


def _non_negative_finite(value: float) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.0
    if not math.isfinite(number):
        return 0.0
    return max(number, 0.0)
