"""
导出 PPO 调度下的任务时序排序表和任务甘特图。

默认用于当前 sin 版本：
    PPO 模型: ppo_outputs_sin/best_model/best_model.zip
    评估 seed: 4000
    输出目录: task_timeline_sin_seed4000

运行示例：
    python -m report_tools.export_task_timeline_sin
    python -m report_tools.export_task_timeline_sin --seed 4001
    python -m report_tools.export_task_timeline_sin --model ppo_outputs_sin/best_model/best_model.zip --seed 4000 --out task_timeline_sin_seed4000
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from stable_baselines3 import PPO


def _ensure_project_root_on_path() -> Path:
    for candidate in Path(__file__).resolve().parents:
        if (candidate / "config_ultimate.py").exists() or (candidate / "IDCPriceEnv20D_ultimate.py").exists():
            root = candidate
            break
    else:
        raise RuntimeError("Cannot locate project root containing config_ultimate.py or IDCPriceEnv20D_ultimate.py")

    root_str = str(root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)
    return root


PROJECT_ROOT = _ensure_project_root_on_path()

from envs.idc_price_env import IDCPriceEnv20D
from configs.config_ultimate import DATA_CONFIG, ENV_CONFIG, REWARD_CONFIG, resolve_output_path
from data_io.data_loader import build_external_series_from_config


def make_env(seed: int) -> IDCPriceEnv20D:
    return IDCPriceEnv20D(
        **ENV_CONFIG,
        **REWARD_CONFIG,
        **build_external_series_from_config(DATA_CONFIG, ENV_CONFIG["horizon"]),
        server_seed=seed,
        task_seed=seed,
    )


def safe_value(value: Any, default=""):
    if value is None:
        return default
    return value


def run_ppo_and_export(model_path: str, seed: int, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    model = PPO.load(model_path)
    env = make_env(seed)

    obs, _ = env.reset()
    total_reward = 0.0
    last_info = {}

    for _ in range(env.horizon):
        action, _state = model.predict(obs, deterministic=True)
        action = np.asarray(action, dtype=np.float32).reshape(-1)
        if action.shape[0] == env.action_dim - 1:
            action = np.concatenate([action, np.array([0.5], dtype=np.float32)])
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += float(reward)
        last_info = info
        if terminated or truncated:
            break

    task_rows = []
    timeline_rows = []

    for task in env.tasks:
        start_time = safe_value(getattr(task, "start_time", None), np.nan)
        finish_time = safe_value(getattr(task, "finish_time", None), np.nan)
        latest_finish_time = safe_value(getattr(task, "latest_finish_time", None), np.nan)

        is_finished = getattr(task, "status", "") == "finished"
        is_deadline_miss = False
        if is_finished and not pd.isna(finish_time) and not pd.isna(latest_finish_time):
            is_deadline_miss = float(finish_time) > float(latest_finish_time)
        if (not is_finished) and not pd.isna(latest_finish_time):
            is_deadline_miss = env.horizon > float(latest_finish_time)

        task_rows.append({
            "seed": seed,
            "task_id": int(getattr(task, "task_id", -1)),
            "profile_key": getattr(task, "profile_key", ""),
            "task_name": getattr(task, "name", ""),
            "arrival_time": int(getattr(task, "arrival_time", -1)),
            "start_time": start_time,
            "finish_time": finish_time,
            "duration": int(getattr(task, "duration", -1)),
            "deadline": int(getattr(task, "deadline", -1)),
            "latest_finish_time": latest_finish_time,
            "status": getattr(task, "status", ""),
            "workload": float(getattr(task, "workload", np.nan)),
            "remaining_work": float(getattr(task, "remaining_work", np.nan)),
            "priority": float(getattr(task, "priority", np.nan)),
            "interruptible": bool(getattr(task, "interruptible", False)),
            "parallelizable": bool(getattr(task, "parallelizable", False)),
            "pause_count": int(getattr(task, "pause_count", 0)),
            "resume_count": int(getattr(task, "resume_count", 0)),
            "non_interruptible_interruption_count": int(
                getattr(task, "non_interruptible_interruption_count", 0)
            ),
            "is_deadline_miss": bool(is_deadline_miss),
        })

        for item in getattr(task, "execution_log", []):
            timeline_rows.append({
                "seed": seed,
                "task_id": int(getattr(task, "task_id", -1)),
                "profile_key": getattr(task, "profile_key", ""),
                "task_name": getattr(task, "name", ""),
                "hour": int(item.get("hour", -1)),
                "work": float(item.get("work", 0.0)),
                "remaining_work": float(item.get("remaining_work", np.nan)),
            })

    task_df = pd.DataFrame(task_rows).sort_values(
        by=["arrival_time", "start_time", "latest_finish_time", "task_id"],
        na_position="last",
    )
    timeline_df = pd.DataFrame(timeline_rows).sort_values(
        by=["hour", "task_id"],
        na_position="last",
    )

    task_csv = out_dir / "task_schedule.csv"
    timeline_csv = out_dir / "task_execution_timeline.csv"

    task_df.to_csv(task_csv, index=False, encoding="utf-8-sig")
    timeline_df.to_csv(timeline_csv, index=False, encoding="utf-8-sig")

    show_cols = [
        "task_id",
        "profile_key",
        "arrival_time",
        "start_time",
        "finish_time",
        "latest_finish_time",
        "status",
        "workload",
        "remaining_work",
        "priority",
        "is_deadline_miss",
    ]

    print("\n=== PPO task schedule sorted by time ===")
    print(task_df[show_cols].to_string(index=False))

    print("\n=== episode metrics ===")
    print(f"seed={seed}")
    print(f"total_reward={total_reward:.4f}")
    print(f"completion_rate={float(last_info.get('completion_rate', np.nan)):.4f}")
    print(f"task_completion_rate={float(last_info.get('task_completion_rate', np.nan)):.4f}")
    print(f"unit_task_cost={float(last_info.get('unit_task_cost', np.nan)):.5f}")
    print(f"total_cost={float(last_info.get('total_cost', np.nan)):.2f}")
    print(f"total_carbon_emission={float(last_info.get('total_carbon_emission', np.nan)):.2f}")
    print(f"total_carbon_cost={float(last_info.get('total_carbon_cost', np.nan)):.2f}")
    print(f"carbon_per_task={float(last_info.get('carbon_per_task', np.nan)):.5f}")
    print(f"final_backlog_work={float(last_info.get('final_backlog_work', np.nan)):.2f}")

    plot_gantt(task_df, out_dir, seed)

    print(f"\nSaved task schedule CSV:       {task_csv}")
    print(f"Saved execution timeline CSV:  {timeline_csv}")
    print(f"Saved gantt figure:            {out_dir / 'task_gantt.png'}")


def plot_gantt(task_df: pd.DataFrame, out_dir: Path, seed: int) -> None:
    df = task_df.copy()
    df = df.sort_values(
        by=["arrival_time", "start_time", "latest_finish_time", "task_id"],
        na_position="last",
    ).reset_index(drop=True)

    fig_height = max(7, 0.36 * len(df) + 2)
    plt.figure(figsize=(14, fig_height))
    ax = plt.gca()

    y_positions = np.arange(len(df))

    for y, row in zip(y_positions, df.itertuples(index=False)):
        task_id = int(row.task_id)
        arrival = float(row.arrival_time)
        latest_finish = float(row.latest_finish_time) if not pd.isna(row.latest_finish_time) else np.nan
        start = float(row.start_time) if not pd.isna(row.start_time) else np.nan
        finish = float(row.finish_time) if not pd.isna(row.finish_time) else np.nan
        status = str(row.status)

        ax.scatter(arrival, y, marker="|", s=100)

        if not pd.isna(latest_finish):
            ax.scatter(latest_finish, y, marker="x", s=45)

        if not pd.isna(start):
            finish_for_plot = 24.0 if pd.isna(finish) else finish
            width = max(finish_for_plot - start, 0.15)
            ax.barh(y, width, left=start, height=0.55)
        else:
            ax.scatter(arrival, y, marker="o", s=35)

        label = f"{task_id}-{row.profile_key}"
        if status != "finished":
            label += f" ({status})"
        ax.text(24.15, y, label, va="center", fontsize=8)

    ax.set_xlim(-0.5, 27.5)
    ax.set_xticks(range(0, 25, 1))
    ax.set_xlabel("Hour of the Day")
    ax.set_ylabel("Tasks sorted by arrival / execution time")
    ax.set_title(f"PPO Task Execution Timeline / Gantt Chart (seed={seed})")
    ax.grid(axis="x", alpha=0.3)

    ax.set_yticks(y_positions)
    ax.set_yticklabels([str(int(x)) for x in df["task_id"]])

    ax.scatter([], [], marker="|", s=100, label="arrival time")
    ax.scatter([], [], marker="x", s=45, label="latest finish time")
    ax.barh([], [], label="execution interval")
    ax.legend(loc="lower right")

    plt.tight_layout()
    fig_path = out_dir / "task_gantt.png"
    plt.savefig(fig_path, dpi=300)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        default="ppo_outputs_sin/best_model/best_model.zip",
        help="PPO model path.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=4000,
        help="Evaluation seed used to generate task timeline.",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="task_timeline_sin_seed4000",
        help="Output directory.",
    )
    args = parser.parse_args()

    run_ppo_and_export(
        model_path=args.model,
        seed=args.seed,
        out_dir=resolve_output_path(args.out),
    )


if __name__ == "__main__":
    main()
