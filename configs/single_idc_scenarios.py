"""Single-IDC scenario definitions for Safe PPO experiments.

These scenarios are configuration inputs only. They do not change IEEE14 base
loads, branch limits, transformer limits, the original PPO entry point, or the
grid model itself.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any


# Synthetic daily shape used to document the intended single-IDC stress level.
# The RL policy still controls the realized hourly load through the original
# IDC action space; the profile is used as scenario metadata and reporting
# reference, not as a hard-coded dispatch rule.
SINGLE_IDC_PROFILE_FRACTIONS = [
    0.48, 0.45, 0.43, 0.42, 0.45, 0.50,
    0.58, 0.66, 0.74, 0.82, 0.88, 0.92,
    0.90, 0.86, 0.82, 0.78, 0.84, 0.95,
    1.00, 0.98, 0.90, 0.78, 0.65, 0.55,
]

# Current small baseline uses server_group_size=100 and produces roughly
# 1.36 MW at full server action with neutral BESS on seed 2026. Keep this
# constant explicit so high-load scenarios remain reproducible and reviewable.
SERVER_GROUPS_PER_PEAK_MW = 73.0
SMALL_BASELINE_SERVER_GROUP_SIZE = 100


def _mean_profile_fraction() -> float:
    return float(sum(SINGLE_IDC_PROFILE_FRACTIONS) / len(SINGLE_IDC_PROFILE_FRACTIONS))


def _scenario(
    name: str,
    role: str,
    target_peak_mw: float,
    description: str,
    server_group_size: int | None = None,
) -> dict[str, Any]:
    peak = float(target_peak_mw)
    size = int(server_group_size if server_group_size is not None else round(peak * SERVER_GROUPS_PER_PEAK_MW))
    return {
        "scenario_name": name,
        "description": description,
        "role": role,
        "idc_bus": 9,
        "target_peak_mw": peak,
        "target_mean_mw": peak * _mean_profile_fraction(),
        "workload_profile_source": "synthetic_single_idc_24h_v1",
        "synthetic_profile_mw": [peak * float(frac) for frac in SINGLE_IDC_PROFILE_FRACTIONS],
        "synthetic_profile_fractions": list(SINGLE_IDC_PROFILE_FRACTIONS),
        "server_group_size": size,
        "task_workload_scale": float(size),
        "bess_scale_factor": float(size),
        "scale_bess_with_idc": True,
        "grid_load_scale": 1.0,
        "grid_base_load_policy": "keep_ieee14_original",
        "idc_grid_injection": "extra_pq_load_at_bus9",
        "is_default_safe_training": role == "normal",
    }


SINGLE_IDC_SCENARIOS = {
    "single_idc_small_baseline": _scenario(
        name="single_idc_small_baseline",
        role="baseline",
        target_peak_mw=3.0,
        description="Original small single-IDC baseline for old-result reproduction.",
        server_group_size=SMALL_BASELINE_SERVER_GROUP_SIZE,
    ),
    "single_idc_bus9_normal": _scenario(
        name="single_idc_bus9_normal",
        role="normal",
        target_peak_mw=100.0,
        description="Main Safe PPO training scenario: single IDC injected at IEEE14 bus 9.",
    ),
    "single_idc_bus9_strong": _scenario(
        name="single_idc_bus9_strong",
        role="strong",
        target_peak_mw=150.0,
        description="High-load evaluation scenario; not a default training setting.",
    ),
    "single_idc_bus9_stress": _scenario(
        name="single_idc_bus9_stress",
        role="stress",
        target_peak_mw=200.0,
        description="Stress-test scenario close to the known bus9 boundary; not for default training.",
    ),
}


def list_single_idc_scenarios() -> list[str]:
    return sorted(SINGLE_IDC_SCENARIOS)


def get_single_idc_scenario(name: str) -> dict[str, Any]:
    try:
        return deepcopy(SINGLE_IDC_SCENARIOS[str(name)])
    except KeyError as exc:
        valid = ", ".join(list_single_idc_scenarios())
        raise KeyError(f"Unknown single-IDC scenario {name!r}. Valid scenarios: {valid}") from exc


def build_synthetic_profile(name: str) -> list[float]:
    scenario = get_single_idc_scenario(name)
    return [float(value) for value in scenario["synthetic_profile_mw"]]


def apply_single_idc_scenario(
    env_config: dict[str, Any],
    grid_config: dict[str, Any],
    idc_scale_config: dict[str, Any],
    scenario_name: str,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Return copied configs with the requested single-IDC scenario applied."""

    scenario = get_single_idc_scenario(scenario_name)
    env_out = deepcopy(env_config)
    grid_out = deepcopy(grid_config)
    idc_scale_out = deepcopy(idc_scale_config)

    grid_out["idc_ieee_bus_number"] = int(scenario["idc_bus"])
    grid_out["load_scale"] = float(scenario["grid_load_scale"])
    grid_out["enable_grid_coupling"] = True
    grid_out["enable_grid_obs"] = True

    idc_scale_out["enable_server_group_model"] = True
    idc_scale_out["server_group_size"] = int(scenario["server_group_size"])
    idc_scale_out["task_workload_scale"] = float(scenario["task_workload_scale"])
    idc_scale_out["bess_scale_factor"] = float(scenario["bess_scale_factor"])
    idc_scale_out["scale_bess_with_idc"] = bool(scenario["scale_bess_with_idc"])

    # Keep old baseline in the original small-load regime. For high-load cases,
    # the internal environment references are scaled by IDCPriceEnv20D using the
    # idc_scale settings above, so the base reward remains numerically stable.
    return env_out, grid_out, idc_scale_out, scenario
