"""
任务层与功耗层 demo。

运行方式：
    python -m scripts.demo_task_model

作用：
    1. 检查 Task 对象、任务参数范围化、初始积压任务是否正常；
    2. 运行一个价格感知规则调度占位方案；
    3. 输出 24 小时功耗、成本、积压和任务完成情况。
"""

import sys
from pathlib import Path

import numpy as np


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

from idc_model.task_model import IDCEnergyTaskModel


def main():
    print(">>> 阶段一任务类 + 参数范围化 + 服务器算力来源改进版：分时电价下的智算中心低成本任务调度评估启动...")

    model = IDCEnergyTaskModel(N=20)
    horizon = 24
    hours = np.arange(horizon)

    T_amb = 25 + 5 * np.sin(np.pi * (hours - 8) / 12)
    price_t = model.create_price_curve(horizon=horizon)

    tasks = model.create_demo_tasks(num_tasks=12, horizon=horizon)

    Q0_original = 50.0
    initial_backlog_task = model.create_initial_backlog_task(Q0_original)
    tasks.insert(0, initial_backlog_task)

    input_violations = model.validate_task_schedule(tasks, horizon=horizon)
    if len(input_violations) > 0:
        print("\n发现任务输入约束问题：")
        for item in input_violations:
            print(f"- 任务 {item['task_id']}：{item['type']}，{item['message']}")
    else:
        print("\n任务输入约束检查通过。")

    lambda_t = model.build_task_arrival_curve(tasks, horizon=horizon)
    planned_task_load_t = model.build_price_aware_task_load_plan(price_t)

    Q_traj, completed_work_t, raw_capacity_t, task_metrics = model.simulate_task_execution_stepwise(
        tasks=tasks,
        task_load_t=planned_task_load_t,
        horizon=horizon,
    )

    actual_task_load_t = np.clip(completed_work_t / model.C_IDC, 0.0, 1.0)

    base_load = 0.05
    L_t = np.clip(base_load + actual_task_load_t, 0.0, 1.0)
    L_matrix = model.load_balance(L_t)

    P_IDC, P_IT, PUE, COP, P_cooling = model.calc_pue_and_total_power(L_matrix, T_amb)

    metrics = model.evaluate_stage1_metrics(
        P_IDC=P_IDC,
        price_t=price_t,
        completed_work_t=completed_work_t,
        lambda_t=lambda_t,
        Q0=0.0,
        Q_traj=Q_traj,
        delta_t=1.0,
    )

    result_violations = model.validate_task_schedule(tasks, horizon=horizon)

    print("\n" + "=" * 120)
    print("服务器算力容量来源改进结果")
    print("=" * 120)
    print(f"目标总算力尺度 C_IDC_target: {model.C_IDC_target:.2f}")
    print(f"实际总算力 C_IDC = sum(C_server): {model.C_IDC:.2f}")
    print(
        f"单台服务器算力 C_server: "
        f"min={np.min(model.C_server):.2f}, "
        f"mean={np.mean(model.C_server):.2f}, "
        f"max={np.max(model.C_server):.2f}"
    )
    print(
        f"服务器算力/最大功耗 C_server/P_max: "
        f"min={np.min(model.server_compute_efficiency):.4f}, "
        f"mean={np.mean(model.server_compute_efficiency):.4f}, "
        f"max={np.max(model.server_compute_efficiency):.4f}"
    )
    print("前 5 台服务器示例：")
    print(f"{'server':<8}{'P_idle(W)':<14}{'P_max(W)':<14}{'C_server':<14}{'C/Pmax':<10}")
    print("-" * 70)
    for i in range(min(5, model.N)):
        print(
            f"{i:<8}"
            f"{model.P_idle[i]:<14.2f}"
            f"{model.P_max[i]:<14.2f}"
            f"{model.C_server[i]:<14.2f}"
            f"{model.server_compute_efficiency[i]:<10.4f}"
        )

    print("\n" + "=" * 120)
    print("任务 Profile 参数范围列表")
    print("=" * 120)
    print(
        f"{'Profile':<16}"
        f"{'任务名称':<14}"
        f"{'duration范围':<16}"
        f"{'load范围':<18}"
        f"{'workload估计范围':<22}"
        f"{'deadline范围':<18}"
        f"{'priority范围':<18}"
        f"{'可暂停':<10}"
        f"{'可并行':<10}"
    )
    print("-" * 120)

    for key, profile in model.task_profiles.items():
        print(
            f"{key:<16}"
            f"{profile['name']:<14}"
            f"{str(profile['duration_range']):<16}"
            f"{str(profile['load_range']):<18}"
            f"{str(tuple(round(v, 2) for v in profile['workload_range'])):<22}"
            f"{str(profile['deadline_range']):<18}"
            f"{str(profile['priority_range']):<18}"
            f"{str(profile['interruptible']):<10}"
            f"{str(profile['parallelizable']):<10}"
        )

    generated_non_initial_tasks = [task for task in tasks if task.task_id != 0]
    generated_workloads = np.array([task.workload for task in generated_non_initial_tasks], dtype=np.float64)
    generated_priorities = np.array([task.priority for task in generated_non_initial_tasks], dtype=np.float64)
    generated_durations = np.array([task.duration for task in generated_non_initial_tasks], dtype=np.float64)

    print("\n" + "=" * 120)
    print("本次随机生成任务参数统计（不含初始积压任务）")
    print("=" * 120)
    print(f"任务数量: {len(generated_non_initial_tasks)}")
    print(f"任务量 workload: min={np.min(generated_workloads):.2f}, mean={np.mean(generated_workloads):.2f}, max={np.max(generated_workloads):.2f}")
    print(f"持续时间 duration: min={np.min(generated_durations):.0f}, mean={np.mean(generated_durations):.2f}, max={np.max(generated_durations):.0f}")
    print(f"优先级 priority: min={np.min(generated_priorities):.2f}, mean={np.mean(generated_priorities):.2f}, max={np.max(generated_priorities):.2f}")

    print("\n" + "=" * 120)
    print("任务执行结果：Task 对象状态")
    print("=" * 160)
    print(
        f"{'任务ID':<8}"
        f"{'类型':<16}"
        f"{'任务名称':<14}"
        f"{'到达':<8}"
        f"{'开始':<8}"
        f"{'完成':<8}"
        f"{'duration':<10}"
        f"{'deadline':<10}"
        f"{'最晚完成':<10}"
        f"{'状态':<14}"
        f"{'总任务量':<12}"
        f"{'剩余任务量':<12}"
        f"{'平均负载':<10}"
        f"{'优先级':<8}"
    )
    print("-" * 160)

    for task in tasks:
        start_time_str = "-" if task.start_time is None else str(task.start_time)
        finish_time_str = "-" if task.finish_time is None else str(task.finish_time)

        print(
            f"{task.task_id:<8}"
            f"{task.profile_key:<16}"
            f"{task.name:<14}"
            f"{task.arrival_time:<8}"
            f"{start_time_str:<8}"
            f"{finish_time_str:<8}"
            f"{task.duration:<10}"
            f"{task.deadline:<10}"
            f"{task.latest_finish_time:<10}"
            f"{task.status:<14}"
            f"{task.workload:<12.2f}"
            f"{task.remaining_work:<12.2f}"
            f"{task.avg_load:<10.3f}"
            f"{task.priority:<8.2f}"
        )

    print("\n" + "=" * 150)
    print("24 小时仿真结果")
    print("=" * 150)
    print(
        f"{'小时':<6}"
        f"{'T_amb':<10}"
        f"{'lambda_t':<12}"
        f"{'plan_L':<10}"
        f"{'actual_L':<10}"
        f"{'L_t':<10}"
        f"{'capacity':<12}"
        f"{'completed':<12}"
        f"{'P_IDC(kW)':<14}"
        f"{'price':<10}"
        f"{'cost':<10}"
        f"{'PUE':<10}"
        f"{'Q':<10}"
    )
    print("-" * 150)

    for t in range(horizon):
        print(
            f"{t:<6}"
            f"{T_amb[t]:<10.2f}"
            f"{lambda_t[t]:<12.2f}"
            f"{planned_task_load_t[t]:<10.3f}"
            f"{actual_task_load_t[t]:<10.3f}"
            f"{L_t[t]:<10.3f}"
            f"{raw_capacity_t[t]:<12.2f}"
            f"{completed_work_t[t]:<12.2f}"
            f"{P_IDC[t] / 1000:<14.3f}"
            f"{price_t[t]:<10.2f}"
            f"{metrics['cost_t'][t]:<10.2f}"
            f"{PUE[t]:<10.3f}"
            f"{Q_traj[t]:<10.2f}"
        )

    if len(result_violations) > 0:
        print("\n发现任务执行约束问题：")
        for item in result_violations:
            print(f"- 任务 {item['task_id']}：{item['type']}，{item['message']}")
    else:
        print("\n任务执行约束检查通过。")

    print("\n" + "=" * 90)
    print("阶段一整体评估结果：分时电价下的 IDC 运行成本")
    print("=" * 90)
    print(f"原始初始队列任务量 Q0: {Q0_original:.2f}（已改造成 task_id=0 的初始积压任务）")
    print(f"24 小时新到任务量: {metrics['total_arrived_work']:.2f}")
    print(f"总可处理任务量: {metrics['total_available_work']:.2f}")
    print(f"实际完成任务量: {metrics['total_completed_work']:.2f}")
    print(f"工作量完成率: {metrics['completion_rate'] * 100:.2f}%")
    print(f"队列末尾积压: {metrics['final_queue']:.2f}")
    print(f"IDC 总购电量: {metrics['total_energy_kWh']:.2f} kWh")
    print(f"IDC 用电成本: {metrics['total_cost']:.2f} 元")
    print(f"单位任务耗电量: {metrics['energy_per_task']:.4f} kWh / 任务量")
    print(f"单位任务成本: {metrics['unit_task_cost']:.4f} 元 / 任务量")
    print(f"平均 PUE: {np.mean(PUE):.3f}")
    print(f"平均 COP: {np.mean(COP):.3f}")
    print(f"平均制冷功耗: {np.mean(P_cooling) / 1000:.2f} kW")
    print(f"系统峰值功耗: {np.max(P_IDC) / 1000:.2f} kW")
    print(f"平均负载率: {np.mean(L_t):.3f}")
    print(f"峰值负载率: {np.max(L_t):.3f}")

    print("\n" + "=" * 90)
    print("任务级评估结果")
    print("=" * 90)
    print(f"完成任务数: {task_metrics['finished_task_count']} / {task_metrics['total_task_count']}")
    print(f"任务级完成率: {task_metrics['task_completion_rate'] * 100:.2f}%")
    print(f"超时任务数: {task_metrics['deadline_miss_count']}")
    print(f"任务超时率: {task_metrics['deadline_miss_rate'] * 100:.2f}%")
    print(f"平均等待时间: {task_metrics['avg_waiting_time']:.2f} h")
    print(f"平均周转时间: {task_metrics['avg_turnaround_time']:.2f} h")
    print(f"任务级最终积压: {task_metrics['final_backlog_work']:.2f}")
    print("=" * 90)


if __name__ == "__main__":
    main()
