"""
PPO 训练脚本：256 维 ultimate 前瞻状态空间 + 23 维动作空间 + 任务启停 + 增强 reward 环境版本。

运行方式：
    python train_ppo_ultimate.py
    python train_ppo_ultimate.py --timesteps 500000
    python train_ppo_ultimate.py --timesteps 1000000
"""

from __future__ import annotations

import argparse
from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback

from config_ultimate import DATA_CONFIG, DEFAULT_EVAL_SEED, ENV_CONFIG, PPO_CONFIG, REWARD_CONFIG
from data_loader import build_external_series_from_config
from IDCPriceEnv20D_ultimate import IDCPriceEnv20D


def make_env(seed=None):
    """
    创建最终版任务类 PPO 环境。

    seed=None 表示训练时每个环境随机生成任务和服务器参数；
    固定 seed 用于评估，方便观察训练是否变好。
    """
    env_kwargs = {
        **ENV_CONFIG,
        **REWARD_CONFIG,
        **build_external_series_from_config(DATA_CONFIG, ENV_CONFIG["horizon"]),
        "server_seed": seed,
        "task_seed": seed,
    }
    env = IDCPriceEnv20D(**env_kwargs)
    return Monitor(env)


def sanity_check_env():
    """训练前快速检查状态维度和动作维度，避免跑很久才发现环境接错。"""
    env_kwargs = {
        **ENV_CONFIG,
        **REWARD_CONFIG,
        **build_external_series_from_config(DATA_CONFIG, ENV_CONFIG["horizon"]),
        "server_seed": DEFAULT_EVAL_SEED,
        "task_seed": DEFAULT_EVAL_SEED,
    }
    env = IDCPriceEnv20D(**env_kwargs)
    obs, info = env.reset()

    print(">>> 环境连通性检查")
    print(f"    obs.shape = {obs.shape}")
    print(f"    action_space.shape = {env.action_space.shape}")
    print(f"    obs_dim(info) = {info.get('obs_dim')}")
    print(f"    total_task_count = {info.get('total_task_count')}")

    if obs.shape != (256,):
        raise RuntimeError(f"状态维度错误：期望 (256,), 实际 {obs.shape}")
    if env.action_space.shape != (23,):
        raise RuntimeError(f"动作维度错误：期望 (23,), 实际 {env.action_space.shape}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--timesteps",
        type=int,
        default=500_000,
        help="PPO 总训练步数；4070/4070S 主机可以先跑 500000，时间够再跑 1000000。",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default="ultimate",
        help="输出目录名的一部分。",
    )
    parser.add_argument(
        "--no-sanity-check",
        action="store_true",
        help="跳过环境维度检查。",
    )
    args = parser.parse_args()

    if not args.no_sanity_check:
        sanity_check_env()

    output_dir = Path(f"ppo_outputs_{args.run_name}")
    model_dir = output_dir / "models"
    log_dir = output_dir / "logs"
    best_model_dir = output_dir / "best_model"

    model_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    best_model_dir.mkdir(parents=True, exist_ok=True)

    train_env = make_env(seed=None)
    eval_env = make_env(seed=DEFAULT_EVAL_SEED)

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

    print("\n>>> 开始 PPO 最终版环境训练")
    print(f">>> 总步数: {args.timesteps}")
    print(">>> 当前环境：256 维 ultimate 前瞻状态，23 维动作，Task 启停机制，增强 reward。")
    print(f">>> 输出目录: {output_dir}")

    model.learn(
        total_timesteps=args.timesteps,
        callback=[checkpoint_callback, eval_callback],
        progress_bar=False,
    )

    final_model_path = model_dir / "ppo_idc_ultimate_final"
    model.save(str(final_model_path))

    print("\n>>> PPO 训练完成。")
    print(f"最终模型已保存到: {final_model_path}.zip")
    print(f"最佳模型目录: {best_model_dir}")
    print(f"训练日志目录: {log_dir}")
    print("下一步：用最终版 evaluate 脚本对全0、全1、随机、PPO 进行对比评估。")


if __name__ == "__main__":
    main()
