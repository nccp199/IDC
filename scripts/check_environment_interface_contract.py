"""Read-only runtime check for the active ordinary-PPO environment interface.

The check constructs the same unmonitored environment used by ordinary PPO,
resets it, and executes one sampled action. It does not train, save a model, or
write output files.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from configs.experiment_cases import get_experiment_case  # noqa: E402
from train.train_ppo_ultimate import make_unmonitored_env  # noqa: E402


REQUIRED_INFO_FIELDS = {
    "total_cost",
    "total_carbon_emission",
    "completion_rate",
    "task_completion_rate",
    "deadline_miss_rate",
    "final_backlog_work",
    "bess_soc",
    "P_IDC_kW",
    "P_grid_kW",
    "grid_lmp",
    "grid_mef_plus",
    "grid_mef_minus",
    "grid_min_voltage_pu",
    "grid_max_line_loading_percent",
    "grid_opf_success",
}


def main() -> None:
    case = get_experiment_case("main")
    env = make_unmonitored_env(
        case["env_config"],
        case["reward_config"],
        case["data_config"],
        seed=2026,
    )
    try:
        base_env = env.unwrapped
        expected_action_dim = int(base_env.model.N) + int(base_env.extra_action_dim)
        expected_base_obs_dim = (
            int(base_env.global_obs_dim)
            + int(base_env.task_pool_obs_dim)
            + int(base_env.server_feature_groups) * int(base_env.model.N)
            + int(base_env.forecast_feature_groups) * int(base_env.horizon)
        )
        expected_wrapped_obs_dim = expected_base_obs_dim + (
            int(env.grid_obs_dim) if bool(env.enable_grid_obs) else 0
        )

        assert base_env.extra_action_dim == 3
        assert base_env.action_space.shape == (expected_action_dim,)
        assert env.action_space.shape == base_env.action_space.shape
        assert base_env.observation_space.shape == (expected_base_obs_dim,)
        assert env.observation_space.shape == (expected_wrapped_obs_dim,)

        obs, reset_info = env.reset(seed=2026)
        assert isinstance(reset_info, dict)
        assert obs.shape == env.observation_space.shape
        assert np.isfinite(obs).all()

        env.action_space.seed(2026)
        action = env.action_space.sample()
        result = env.step(action)
        assert isinstance(result, tuple) and len(result) == 5
        next_obs, reward, terminated, truncated, info = result
        assert next_obs.shape == env.observation_space.shape
        assert np.isfinite(next_obs).all()
        assert np.isfinite(reward)
        assert isinstance(terminated, (bool, np.bool_))
        assert isinstance(truncated, (bool, np.bool_))
        assert REQUIRED_INFO_FIELDS.issubset(info)

        print(f"base_action_space={base_env.action_space}")
        print(f"wrapped_action_space={env.action_space}")
        print(f"base_observation_space={base_env.observation_space}")
        print(f"wrapped_observation_space={env.observation_space}")
        print(f"obs_shape={obs.shape}")
        print(f"step_obs_shape={next_obs.shape}")
        print(f"reward_finite={bool(np.isfinite(reward))}")
        print(f"grid_opf_success={info['grid_opf_success']}")
        print("environment_interface_contract=PASS")
    finally:
        env.close()


if __name__ == "__main__":
    main()
