"""Experiment case management for report training and evaluation.

Cases only describe environment/reward/data configuration. They never bind a
PPO model path, so models can be trained on one machine and evaluated on
another with an explicit --ppo-model path.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict

from config_ultimate import DATA_CONFIG, ENV_CONFIG, REWARD_CONFIG


ENV_METADATA_KEYS = [
    "horizon",
    "num_tasks",
    "Q0",
    "grid_power_limit_kW",
    "peak_power_threshold_kW",
    "bess_capacity_kWh",
    "bess_charge_power_max_kW",
    "bess_discharge_power_max_kW",
    "bess_soc_init",
    "bess_soc_min",
    "bess_soc_max",
    "bess_soc_target",
]

REWARD_METADATA_KEYS = [
    "reward_done_weight",
    "reward_cost_weight",
    "reward_carbon_weight",
    "reward_sla_weight",
    "reward_queue_weight",
    "reward_final_queue_weight",
    "reward_grid_peak_weight",
    "reward_bess_degradation_weight",
]


def _canonical_case_name(case_name: str) -> str:
    name = (case_name or "main").strip().lower()
    if name == "report_main":
        return "main"
    return name


def get_experiment_case(case_name: str) -> Dict[str, Any]:
    """Return deep-copied config dictionaries for a named experiment case."""
    case = _canonical_case_name(case_name)
    env_config = deepcopy(ENV_CONFIG)
    reward_config = deepcopy(REWARD_CONFIG)
    data_config = deepcopy(DATA_CONFIG)

    if case == "main":
        pass
    elif case == "no_bess":
        env_config["bess_capacity_kWh"] = 0.0
        env_config["bess_charge_power_max_kW"] = 0.0
        env_config["bess_discharge_power_max_kW"] = 0.0
    elif case == "carbon_w0":
        reward_config["reward_carbon_weight"] = 0.0
    elif case == "carbon_w03":
        reward_config["reward_carbon_weight"] = 0.30
    elif case == "carbon_w05":
        reward_config["reward_carbon_weight"] = 0.50
    else:
        valid = "main/report_main, no_bess, carbon_w0, carbon_w03, carbon_w05"
        raise ValueError(f"Unknown experiment case '{case_name}'. Valid cases: {valid}")

    return {
        "case": case,
        "env_config": env_config,
        "reward_config": reward_config,
        "data_config": data_config,
    }


def key_env_config(env_config: Dict[str, Any]) -> Dict[str, Any]:
    return {key: env_config.get(key) for key in ENV_METADATA_KEYS if key in env_config}


def key_reward_config(reward_config: Dict[str, Any]) -> Dict[str, Any]:
    return {
        key: reward_config.get(key)
        for key in REWARD_METADATA_KEYS
        if key in reward_config
    }


def print_experiment_case(case_config: Dict[str, Any]) -> None:
    env_config = case_config["env_config"]
    reward_config = case_config["reward_config"]
    print("\n=== Experiment case ===")
    print(f"case: {case_config['case']}")
    print(f"BESS capacity kWh: {env_config.get('bess_capacity_kWh')}")
    print(f"BESS charge max kW: {env_config.get('bess_charge_power_max_kW')}")
    print(f"BESS discharge max kW: {env_config.get('bess_discharge_power_max_kW')}")
    print(f"reward_carbon_weight: {reward_config.get('reward_carbon_weight')}")
