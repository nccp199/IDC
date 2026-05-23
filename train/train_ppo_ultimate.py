"""Train PPO for IDCPriceEnv20D experiment cases.

Examples:
    python -m train.train_ppo_ultimate --case main --timesteps 10000 --run-name smoke
    python -m train.train_ppo_ultimate --case main --timesteps 500000 --run-name report
    python -m train.train_ppo_ultimate --case carbon_w0 --timesteps 500000 --run-name report
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

from configs.config_ultimate import (
    DEFAULT_EVAL_SEED,
    GRID_CONFIG,
    GRID_REWARD_CONFIG,
    PPO_CONFIG,
    resolve_output_path,
)
from data_io.data_loader import build_external_series_from_config
from configs.experiment_cases import (
    get_experiment_case,
    key_env_config,
    key_reward_config,
    print_experiment_case,
)
from envs.idc_price_env import IDCPriceEnv20D
from env_wrappers import GridCoupledEnv


def make_env(
    env_config: Dict[str, Any],
    reward_config: Dict[str, Any],
    data_config: Dict[str, Any],
    seed: Optional[int] = None,
) -> Any:
    """Create a monitored PPO environment from copied case configs."""
    from stable_baselines3.common.monitor import Monitor

    env_kwargs = {
        **env_config,
        **reward_config,
        **build_external_series_from_config(data_config, env_config["horizon"]),
        "server_seed": seed,
        "task_seed": seed,
    }
    base_env = IDCPriceEnv20D(**env_kwargs)
    env = GridCoupledEnv(base_env, GRID_CONFIG, GRID_REWARD_CONFIG)
    return Monitor(env)


def make_unmonitored_env(
    env_config: Dict[str, Any],
    reward_config: Dict[str, Any],
    data_config: Dict[str, Any],
    seed: Optional[int] = None,
) -> GridCoupledEnv:
    env_kwargs = {
        **env_config,
        **reward_config,
        **build_external_series_from_config(data_config, env_config["horizon"]),
        "server_seed": seed,
        "task_seed": seed,
    }
    base_env = IDCPriceEnv20D(**env_kwargs)
    return GridCoupledEnv(base_env, GRID_CONFIG, GRID_REWARD_CONFIG)


def sanity_check_env(
    env_config: Dict[str, Any],
    reward_config: Dict[str, Any],
    data_config: Dict[str, Any],
) -> tuple[int, int]:
    """Check the current PPO observation/action interface."""
    env = make_unmonitored_env(env_config, reward_config, data_config, seed=DEFAULT_EVAL_SEED)
    obs, info = env.reset()
    expected_obs_dim = int(env.observation_space.shape[0])

    print(">>> PPO environment sanity check")
    print(f"    obs.shape = {obs.shape}")
    print(f"    env.observation_space.shape = {env.observation_space.shape}")
    print(f"    action_space.shape = {env.action_space.shape}")
    print(f"    base_obs_dim = {env.base_obs_dim}")
    print(f"    grid_obs_dim = {env.grid_obs_dim}")
    print(f"    enable_grid_obs = {env.enable_grid_obs}")
    print(f"    obs_dim(info) = {info.get('obs_dim')}")
    print(f"    total_task_count = {info.get('total_task_count')}")

    if int(obs.shape[0]) != expected_obs_dim:
        raise RuntimeError(f"Expected obs_dim={expected_obs_dim}, got obs.shape={obs.shape}")
    if env.action_space.shape != (23,):
        raise RuntimeError(f"Expected action_space.shape=(23,), got {env.action_space.shape}")

    return int(obs.shape[0]), int(env.action_space.shape[0])


def infer_env_shape(
    env_config: Dict[str, Any],
    reward_config: Dict[str, Any],
    data_config: Dict[str, Any],
) -> tuple[int, int]:
    env = make_unmonitored_env(env_config, reward_config, data_config, seed=DEFAULT_EVAL_SEED)
    return int(env.observation_space.shape[0]), int(env.action_space.shape[0])


def save_training_metadata(
    output_dir: Path,
    case_config: Dict[str, Any],
    run_name: str,
    timesteps: int,
    obs_dim: int,
    action_dim: int,
    final_model_path: Path,
    best_model_path: Path,
) -> Path:
    metadata = {
        "case": case_config["case"],
        "run_name": run_name,
        "timesteps": int(timesteps),
        "obs_dim": int(obs_dim),
        "action_dim": int(action_dim),
        "ENV_CONFIG_key_params": key_env_config(case_config["env_config"]),
        "REWARD_CONFIG_key_params": key_reward_config(case_config["reward_config"]),
        "DATA_CONFIG": case_config["data_config"],
        "training_time": datetime.now().isoformat(timespec="seconds"),
        "model_save_path": str(final_model_path.with_suffix(".zip")),
        "best_model_path": str(best_model_path),
    }

    metadata_path = output_dir / "training_metadata.json"
    with metadata_path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    return metadata_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", type=str, default="main", help="Experiment case: main, no_bess, carbon_w0, carbon_w03, carbon_w05.")
    parser.add_argument("--timesteps", type=int, default=500_000, help="PPO training timesteps.")
    parser.add_argument("--run-name", type=str, default="ultimate", help="Run name used in output directory.")
    parser.add_argument("--no-sanity-check", action="store_true", help="Skip observation/action shape sanity check.")
    args = parser.parse_args()

    case_config = get_experiment_case(args.case)
    print_experiment_case(case_config)
    env_config = case_config["env_config"]
    reward_config = case_config["reward_config"]
    data_config = case_config["data_config"]

    if args.no_sanity_check:
        obs_dim, action_dim = infer_env_shape(env_config, reward_config, data_config)
    else:
        obs_dim, action_dim = sanity_check_env(env_config, reward_config, data_config)

    output_dir = resolve_output_path(f"ppo_outputs_{args.run_name}_{case_config['case']}")
    model_dir = output_dir / "models"
    log_dir = output_dir / "logs"
    best_model_dir = output_dir / "best_model"

    model_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    best_model_dir.mkdir(parents=True, exist_ok=True)

    try:
        from stable_baselines3 import PPO
        from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
    except Exception as exc:
        raise SystemExit(f"stable_baselines3 is required for PPO training: {exc}") from exc

    train_env = make_env(env_config, reward_config, data_config, seed=None)
    eval_env = make_env(env_config, reward_config, data_config, seed=DEFAULT_EVAL_SEED)

    checkpoint_callback = CheckpointCallback(
        save_freq=50_000,
        save_path=str(model_dir),
        name_prefix="ppo_idc_ultimate",
    )

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(best_model_dir),
        log_path=str(log_dir),
        eval_freq=20_000,
        n_eval_episodes=5,
        deterministic=True,
        render=False,
    )

    model = PPO(
        policy="MlpPolicy",
        env=train_env,
        tensorboard_log=str(log_dir),
        **PPO_CONFIG,
    )

    print("\n>>> Start PPO training")
    print(f">>> case: {case_config['case']}")
    print(f">>> timesteps: {args.timesteps}")
    print(f">>> output dir: {output_dir}")

    model.learn(
        total_timesteps=args.timesteps,
        callback=[checkpoint_callback, eval_callback],
        progress_bar=False,
    )

    final_model_path = model_dir / "ppo_idc_ultimate_final"
    model.save(str(final_model_path))
    best_model_path = best_model_dir / "best_model.zip"
    metadata_path = save_training_metadata(
        output_dir=output_dir,
        case_config=case_config,
        run_name=args.run_name,
        timesteps=args.timesteps,
        obs_dim=obs_dim,
        action_dim=action_dim,
        final_model_path=final_model_path,
        best_model_path=best_model_path,
    )

    print("\n>>> PPO training finished")
    print(f"Final model: {final_model_path}.zip")
    print(f"Best model dir: {best_model_dir}")
    print(f"Training logs: {log_dir}")
    print(f"Metadata: {metadata_path}")


if __name__ == "__main__":
    main()
