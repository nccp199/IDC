"""Shared construction helpers for multi-agent contract tests."""

from configs.experiment_cases import get_experiment_case
from train.train_ppo_ultimate import make_unmonitored_env


def make_grid_env(seed: int):
    case = get_experiment_case("main")
    return make_unmonitored_env(
        case["env_config"],
        case["reward_config"],
        case["data_config"],
        seed=seed,
    )
