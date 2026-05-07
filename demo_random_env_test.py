"""
环境层随机动作连通性测试。

运行方式：
    python demo_random_env_test.py

作用：
    1. 检查环境是否可以 reset 和 step；
    2. 检查状态维度是否为 256、动作维度是否为 22；
    3. 用随机动作跑完 24 小时 episode，快速确认重构后环境没有断。
"""

from IDCPriceEnv20D_ultimate import IDCPriceEnv20D
from config_ultimate import ENV_CONFIG, REWARD_CONFIG


def main():
    env = IDCPriceEnv20D(
    **ENV_CONFIG,
    **REWARD_CONFIG,
    server_seed=2026,
    task_seed=2026,
    )

    obs, info = env.reset()
    print("初始状态维度:", obs.shape)
    print("动作空间维度:", env.action_space.shape)
    print("初始任务数:", info.get("total_task_count"))
    print("初始积压任务量:", f"{info.get('initial_backlog_work', 0.0):.2f}")

    print("\n开始随机动作测试：")
    print("=" * 150)

    done = False
    step_count = 0

    while not done:
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        print(
            f"step={step_count:<2} | "
            f"hour={info['hour']:<2} | "
            f"price={info['price']:.2f} | "
            f"action_mean={info['action_mean']:.3f} | "
            f"urgent={info['urgent_preference']:.2f} | "
            f"cont={info['continuity_preference']:.2f} | "
            f"planned_cap={info['planned_capacity']:.2f} | "
            f"completed={info['completed_work']:.2f} | "
            f"unused={info['unused_capacity']:.2f} | "
            f"cost={info['cost']:.2f} | "
            f"Q={info['Q']:.2f} | "
            f"miss={info['deadline_miss_count']} | "
            f"pause={info['total_pause_count']} | "
            f"resume={info['total_resume_count']} | "
            f"nonint={info['total_non_interruptible_interruption_count']} | "
            f"reward={reward:.3f}"
        )

        step_count += 1

    print("=" * 150)
    print("随机动作测试完成。")
    print(f"总购电量: {info['total_energy_kWh']:.2f} kWh")
    print(f"总用电成本: {info['total_cost']:.2f} 元")
    print(f"总完成任务量: {info['total_completed_work']:.2f}")
    print(f"工作量完成率: {info['completion_rate'] * 100:.2f}%")
    print(f"完成任务数: {info['finished_task_count']} / {info['total_task_count']}")
    print(f"任务级完成率: {info['task_completion_rate'] * 100:.2f}%")
    print(f"超时任务数: {info['deadline_miss_count']}")
    print(f"任务超时率: {info['deadline_miss_rate'] * 100:.2f}%")
    print(f"平均等待时间: {info['avg_waiting_time']:.2f} h")
    print(f"平均周转时间: {info['avg_turnaround_time']:.2f} h")
    print(f"最终积压任务量: {info['final_backlog_work']:.2f}")
    print(f"暂停次数: {info['total_pause_count']}")
    print(f"恢复次数: {info['total_resume_count']}")
    print(f"不可暂停任务中断次数: {info['total_non_interruptible_interruption_count']}")
    print(f"单位任务耗电量: {info['energy_per_task']:.4f} kWh / 任务量")
    print(f"单位任务成本: {info['unit_task_cost']:.4f} 元 / 任务量")


if __name__ == "__main__":
    main()
