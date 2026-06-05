"""Multi-IDC scenario definitions for offline grid diagnostics.

These scenarios are diagnostic inputs only. They do not alter the main
environment, PPO training configuration, reward weights, or IEEE14 base load.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any


PROFILE_FRACTIONS = {
    "IDC_A": [
        0.45, 0.45, 0.45, 0.45, 0.45, 0.45,
        0.55, 0.65, 0.75, 0.75, 0.75, 0.75,
        0.75, 0.75, 0.75, 0.75, 0.85, 1.00,
        1.00, 1.00, 0.90, 0.80, 0.65, 0.55,
    ],
    "IDC_B": [
        0.35, 0.35, 0.35, 0.35, 0.35, 0.40,
        0.50, 0.60, 0.70, 0.70, 0.70, 0.70,
        0.70, 0.70, 0.75, 0.80, 0.90, 0.95,
        0.95, 0.90, 0.80, 0.70, 0.55, 0.45,
    ],
    "IDC_C": [
        0.40, 0.40, 0.40, 0.40, 0.45, 0.45,
        0.50, 0.60, 0.65, 0.65, 0.65, 0.65,
        0.65, 0.70, 0.75, 0.80, 0.85, 0.85,
        0.80, 0.75, 0.70, 0.60, 0.50, 0.45,
    ],
}


COMBO_A_BUS_BY_IDC = {
    "IDC_A": 9,
    "IDC_B": 10,
    "IDC_C": 13,
}


def _idc_config(idc_name: str, peak_mw: float, role: str) -> dict[str, Any]:
    mean_fraction = sum(PROFILE_FRACTIONS[idc_name]) / len(PROFILE_FRACTIONS[idc_name])
    target_peak_mw = float(peak_mw)
    return {
        "idc_name": idc_name,
        "ieee_bus": int(COMBO_A_BUS_BY_IDC[idc_name]),
        "target_peak_mw": target_peak_mw,
        "target_mean_mw": target_peak_mw * mean_fraction,
        "workload_profile_source": "synthetic_combo_A_24h_v1",
        "bess_capacity_mwh": target_peak_mw * 1.0,
        "bess_power_mw": target_peak_mw * 0.25,
        "role": role,
        # Keep q_mvar=0 by default to match the current single-IDC grid coupling.
        # Set this below 1.0 in future diagnostics if reactive IDC demand is needed.
        "power_factor": 1.0,
    }


MULTI_IDC_SCENARIOS = {
    "multi_idc_A_conservative": {
        "scenario_name": "multi_idc_A_conservative",
        "description": "Conservative training candidate; prioritizes OPF stability.",
        "combo": "A",
        "role": "normal",
        "idcs": [
            _idc_config("IDC_A", 40.0, "normal"),
            _idc_config("IDC_B", 20.0, "normal"),
            _idc_config("IDC_C", 20.0, "normal"),
        ],
    },
    "multi_idc_A_normal": {
        "scenario_name": "multi_idc_A_normal",
        "description": "Main training candidate; aims for visible grid feedback without frequent OPF failure.",
        "combo": "A",
        "role": "normal",
        "idcs": [
            _idc_config("IDC_A", 60.0, "normal"),
            _idc_config("IDC_B", 30.0, "normal"),
            _idc_config("IDC_C", 30.0, "normal"),
        ],
    },
    "multi_idc_A_strong": {
        "scenario_name": "multi_idc_A_strong",
        "description": "Strong feedback candidate; use only if margins remain acceptable.",
        "combo": "A",
        "role": "strong",
        "idcs": [
            _idc_config("IDC_A", 80.0, "strong"),
            _idc_config("IDC_B", 50.0, "strong"),
            _idc_config("IDC_C", 50.0, "strong"),
        ],
    },
    "multi_idc_A_stress": {
        "scenario_name": "multi_idc_A_stress",
        "description": "Stress-only scenario; not intended as a default training setting.",
        "combo": "A",
        "role": "stress",
        "idcs": [
            _idc_config("IDC_A", 120.0, "stress"),
            _idc_config("IDC_B", 80.0, "stress"),
            _idc_config("IDC_C", 80.0, "stress"),
        ],
    },
}


def list_multi_idc_scenarios() -> list[str]:
    return sorted(MULTI_IDC_SCENARIOS)


def get_multi_idc_scenario(name: str) -> dict[str, Any]:
    try:
        return deepcopy(MULTI_IDC_SCENARIOS[str(name)])
    except KeyError as exc:
        valid = ", ".join(list_multi_idc_scenarios())
        raise KeyError(f"Unknown multi-IDC scenario {name!r}. Valid scenarios: {valid}") from exc


def build_synthetic_profile(idc_name: str, target_peak_mw: float) -> list[float]:
    fractions = PROFILE_FRACTIONS[str(idc_name)]
    return [float(target_peak_mw) * float(fraction) for fraction in fractions]
