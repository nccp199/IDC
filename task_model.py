import numpy as np
from typing import Optional

from task import Task
from power_model import IDCPowerModel


class IDCEnergyTaskModel(IDCPowerModel):
    """
    智算中心低成本任务调度模型。

    拆分后的职责：
    1. IDCPowerModel 负责服务器、电价、功耗和成本评估；
    2. 本类负责 Task 对象、任务参数范围化、任务到达曲线、任务执行与调度检查；
    3. 对外仍然保留 IDCEnergyTaskModel 这个类名，方便环境层继续调用。
    """

    def __init__(
        self,
        N=20,
        P_idle_base=200.0,
        P_max_base=600.0,
        k=1.3,
        delta_P_loss=5000.0,
        alpha=0.05,
        beta=-0.1,
        gamma=3.0,
        T_target=24.0,
        P_others=2000.0,
        C_IDC=500.0,
        server_capacity_variation=0.25,
        server_seed=None,
        task_seed=None,
    ):
        super().__init__(
            N=N,
            P_idle_base=P_idle_base,
            P_max_base=P_max_base,
            k=k,
            delta_P_loss=delta_P_loss,
            alpha=alpha,
            beta=beta,
            gamma=gamma,
            T_target=T_target,
            P_others=P_others,
            C_IDC=C_IDC,
            server_capacity_variation=server_capacity_variation,
            server_seed=server_seed,
            task_seed=task_seed,
        )

        # 初始化任务 Profile。具体任务实例会在 create_task() 中从范围内采样。
        self.task_profiles = self._init_task_profiles()

    def _init_task_profiles(self) -> dict:
            """
            初始化任务 Profile。

            当前版本中，Profile 不再表示一个固定任务，
            而是表示一类任务的参数范围模板。

            每个具体任务实例的 duration、load_profile、deadline、priority
            会在 create_task() 中从对应范围内随机采样得到。
            """
            profiles = {
                "A_inference": {
                    "name": "轻量推理任务",
                    "duration_range": (1, 2),
                    "load_range": (0.08, 0.18),
                    "deadline_range": (2, 4),
                    "priority_range": (2.6, 3.4),
                    "interruptible": False,
                    "parallelizable": False,
                    "type_probability": 0.40,
                },
                "B_rl_training": {
                    "name": "强化学习训练任务",
                    "duration_range": (2, 5),
                    "load_range": (0.14, 0.30),
                    "deadline_range": (8, 16),
                    "priority_range": (1.8, 2.5),
                    "interruptible": True,
                    "parallelizable": True,
                    "type_probability": 0.25,
                },
                "C_dl_training": {
                    "name": "深度学习训练任务",
                    "duration_range": (4, 8),
                    "load_range": (0.22, 0.45),
                    "deadline_range": (14, 24),
                    "priority_range": (1.2, 2.0),
                    "interruptible": True,
                    "parallelizable": True,
                    "type_probability": 0.10,
                },
                "D_preprocess": {
                    "name": "数据预处理任务",
                    "duration_range": (1, 4),
                    "load_range": (0.08, 0.20),
                    "deadline_range": (5, 10),
                    "priority_range": (0.8, 1.6),
                    "interruptible": True,
                    "parallelizable": False,
                    "type_probability": 0.25,
                },
            }

            # 为了兼容旧函数和输出展示，给每类任务生成一个“代表值”。
            # 注意：这些代表值不再直接决定具体任务，具体任务会在 create_task() 中随机采样。
            for profile in profiles.values():
                d_min, d_max = profile["duration_range"]
                l_min, l_max = profile["load_range"]
                dl_min, dl_max = profile["deadline_range"]
                p_min, p_max = profile["priority_range"]

                representative_duration = int(round((d_min + d_max) / 2))
                representative_load = (l_min + l_max) / 2
                profile["duration"] = representative_duration
                profile["load_profile"] = np.full(representative_duration, representative_load, dtype=np.float64)
                profile["deadline"] = int(round((dl_min + dl_max) / 2))
                profile["priority"] = float((p_min + p_max) / 2)
                profile["workload"] = float(np.sum(profile["load_profile"]) * self.C_IDC)
                profile["workload_range"] = (
                    float(d_min * l_min * self.C_IDC),
                    float(d_max * l_max * self.C_IDC),
                )

            return profiles

    def _sample_task_parameters(self, profile_key: str) -> dict:
            """
            从某类任务的参数范围中随机生成一个具体任务实例参数。

            这一步是任务参数范围化的核心：
            同一类任务不再拥有完全固定的 duration、load_profile、deadline、priority。
            """
            if profile_key not in self.task_profiles:
                raise KeyError(f"未知任务类型：{profile_key}")

            profile = self.task_profiles[profile_key]

            duration_min, duration_max = profile["duration_range"]
            load_min, load_max = profile["load_range"]
            deadline_min, deadline_max = profile["deadline_range"]
            priority_min, priority_max = profile["priority_range"]

            duration = int(self.task_rng.integers(duration_min, duration_max + 1))
            load_profile = self.task_rng.uniform(load_min, load_max, size=duration)
            workload = float(np.sum(load_profile) * self.C_IDC)

            # deadline 是相对 arrival_time 的允许最大延迟。
            # 为避免生成明显不合理任务，deadline 下限至少不小于 duration。
            real_deadline_min = max(int(deadline_min), duration)
            deadline = int(self.task_rng.integers(real_deadline_min, int(deadline_max) + 1))
            priority = float(self.task_rng.uniform(priority_min, priority_max))

            return {
                "duration": duration,
                "load_profile": load_profile,
                "workload": workload,
                "deadline": deadline,
                "priority": priority,
            }

    def create_task(self, task_id: int, profile_key: str, arrival_time: int) -> Task:
            """
            根据任务 Profile 参数范围创建一个 Task 对象。

            注意：
            当前版本不再人为指定 start_time；
            同一 profile_key 下的不同任务，也会有不同 duration、workload、deadline、priority。
            """
            if profile_key not in self.task_profiles:
                raise KeyError(f"未知任务类型：{profile_key}")

            profile = self.task_profiles[profile_key]
            sampled = self._sample_task_parameters(profile_key)

            return Task(
                task_id=int(task_id),
                profile_key=profile_key,
                name=profile["name"],
                arrival_time=int(arrival_time),
                duration=sampled["duration"],
                load_profile=sampled["load_profile"],
                workload=sampled["workload"],
                deadline=sampled["deadline"],
                priority=sampled["priority"],
                interruptible=bool(profile["interruptible"]),
                parallelizable=bool(profile["parallelizable"]),
            )

    def create_random_tasks(
            self,
            num_tasks: int = 12,
            horizon: int = 24,
            arrival_start: int = 1,
            arrival_end: Optional[int] = None,
        ) -> list[Task]:
            """
            生成一组随机任务实例。

            当前用于替代上一版中 6 个固定任务的 demo tasks。
            任务类型、到达时间和任务内部参数都会随机生成。
            默认 task_seed=None，因此每次运行结果会不同；
            如果需要复现实验，可以在初始化模型时显式传入 task_seed。
            """
            if arrival_end is None:
                # 预留一部分时间给任务执行，避免所有任务都堆到最后几个小时。
                arrival_end = max(arrival_start, horizon - 6)

            arrival_start = int(max(0, arrival_start))
            arrival_end = int(min(horizon - 1, arrival_end))
            if arrival_end < arrival_start:
                raise ValueError("arrival_end 不能小于 arrival_start。")

            profile_keys = list(self.task_profiles.keys())
            probabilities = np.array(
                [self.task_profiles[key]["type_probability"] for key in profile_keys],
                dtype=np.float64
            )
            probabilities = probabilities / np.sum(probabilities)

            raw_specs = []
            for _ in range(int(num_tasks)):
                profile_key = str(self.task_rng.choice(profile_keys, p=probabilities))
                arrival_time = int(self.task_rng.integers(arrival_start, arrival_end + 1))
                raw_specs.append((arrival_time, profile_key))

            # 只按到达时间整理生成顺序，便于输出观察。
            # 这里没有使用 priority/deadline 排序，不把人工“最优调度规则”提前写入环境。
            raw_specs.sort(key=lambda item: item[0])

            tasks = []
            for task_id, (arrival_time, profile_key) in enumerate(raw_specs, start=1):
                tasks.append(self.create_task(task_id, profile_key, arrival_time))

            return tasks

    def create_demo_tasks(self, num_tasks: int = 12, horizon: int = 24) -> list[Task]:
            """
            构造一组 24 小时内到达的示例任务。

            当前任务参数范围化后：
            1. 不再使用 6 个固定参数任务；
            2. 任务类型按概率生成；
            3. arrival_time 在范围内随机生成；
            4. duration、load_profile、workload、deadline、priority 从各自范围内采样；
            5. start_time 仍然由调度执行过程自动产生。
            """
            return self.create_random_tasks(
                num_tasks=num_tasks,
                horizon=horizon,
                arrival_start=1,
                arrival_end=max(1, horizon - 6),
            )

    def create_initial_backlog_task(self, Q0: float) -> Task:
            """
            将原来的 Q0 初始队列积压改造成一个初始积压任务。

            这样后续所有未完成任务都统一由 Task 对象管理。
            """
            backlog_load = min(float(Q0) / self.C_IDC, 1.0) if self.C_IDC > 0 else 0.0

            return Task(
                task_id=0,
                profile_key="initial_backlog",
                name="初始积压任务",
                arrival_time=0,
                duration=1,
                load_profile=np.array([backlog_load], dtype=np.float64),
                workload=float(Q0),
                deadline=24,
                priority=1.0,
                interruptible=True,
                parallelizable=True,
            )

    def build_task_arrival_curve(self, tasks: list[Task], horizon: int = 24) -> np.ndarray:
            """
            根据 Task 对象列表生成任务到达曲线 lambda_t。

            lambda_t[t] 表示第 t 小时新到达的任务量。
            注意：lambda_t 现在只是统计特征，不再是任务本体。
            """
            lambda_t = np.zeros(horizon, dtype=np.float64)

            for task in tasks:
                arrival_time = int(task.arrival_time)

                if 0 <= arrival_time < horizon:
                    lambda_t[arrival_time] += float(task.workload)

            return lambda_t

    def build_price_aware_task_load_plan(
            self,
            price_t: np.ndarray,
            valley_load: float = 0.45,
            flat_load: float = 0.30,
            peak_load: float = 0.12
        ) -> np.ndarray:
            """
            构造一个简单的价格感知任务负载计划。

            当前只是任务类改进阶段的规则调度占位：
            低电价时提高处理强度；
            平段电价中等处理；
            高电价时降低处理强度。

            后续 PPO 接入后，这里的 task_load_t 将由 PPO 动作生成。
            """
            price_t = np.asarray(price_t, dtype=np.float64)
            task_load_t = np.zeros_like(price_t)

            min_price = np.min(price_t)
            max_price = np.max(price_t)

            for t, price in enumerate(price_t):
                if np.isclose(price, min_price):
                    task_load_t[t] = valley_load
                elif np.isclose(price, max_price):
                    task_load_t[t] = peak_load
                else:
                    task_load_t[t] = flat_load

            return task_load_t

    def build_load_curve_from_tasks(
            self,
            tasks: list,
            horizon: int = 24,
            base_load: float = 0.05
        ) -> tuple[np.ndarray, np.ndarray, list]:
            """
            根据任务执行记录生成 24 小时整体负载率曲线。

            兼容说明：
            - 如果输入的是 Task 对象，优先使用 task.execution_log；
            - 如果输入的是旧版 dict 且包含 start_time，则使用旧版逻辑。
            """
            task_load_t = np.zeros(horizon, dtype=np.float64)
            task_timeline = []

            for task in tasks:
                if isinstance(task, Task):
                    for item in task.execution_log:
                        t = int(item["hour"])
                        work = float(item["work"])
                        if 0 <= t < horizon:
                            load_value = work / self.C_IDC
                            task_load_t[t] += load_value
                            task_timeline.append({
                                "task_id": task.task_id,
                                "profile": task.profile_key,
                                "task_name": task.name,
                                "hour": t,
                                "work": work,
                                "load_value": float(load_value),
                            })
                elif isinstance(task, dict) and "start_time" in task:
                    profile = self.task_profiles[task["profile"]]
                    start_time = int(task["start_time"])
                    load_profile = profile["load_profile"]

                    for local_idx, load_value in enumerate(load_profile):
                        t = start_time + local_idx

                        if 0 <= t < horizon:
                            task_load_t[t] += float(load_value)
                            task_timeline.append({
                                "task_id": task["task_id"],
                                "profile": task["profile"],
                                "task_name": profile["name"],
                                "hour": t,
                                "local_step": local_idx,
                                "load_value": float(load_value),
                            })

            raw_L_t = base_load + task_load_t
            overflow_t = np.maximum(raw_L_t - 1.0, 0.0)

            if np.any(overflow_t > 0):
                print("警告：部分时刻任务负载超过 1.0，已自动裁剪。")
                print("超出负载时刻：", np.where(overflow_t > 0)[0])

            L_t = np.clip(raw_L_t, 0.0, 1.0)

            return L_t, task_load_t, task_timeline

    def validate_task_schedule(self, tasks: list[Task], horizon: int = 24) -> list:
            """
            检查任务调度是否满足基本约束。

            任务类改进后，start_time 和 finish_time 不再人为设定，
            而是执行后自动产生。因此本函数既可以在执行前检查输入，
            也可以在执行后检查结果。
            """
            violations = []

            for task in tasks:
                if task.arrival_time < 0 or task.arrival_time >= horizon:
                    violations.append({
                        "task_id": task.task_id,
                        "type": "到达时间异常",
                        "message": f"任务 {task.task_id} 到达时间 {task.arrival_time} 超出仿真周期。",
                    })

                if task.workload < 0:
                    violations.append({
                        "task_id": task.task_id,
                        "type": "任务量异常",
                        "message": f"任务 {task.task_id} 的任务量为负。",
                    })

                if getattr(task, "duration", 0) <= 0:
                    violations.append({
                        "task_id": task.task_id,
                        "type": "持续时间异常",
                        "message": f"任务 {task.task_id} 的 duration 必须大于 0。",
                    })

                if getattr(task, "deadline", 0) <= 0:
                    violations.append({
                        "task_id": task.task_id,
                        "type": "截止时间异常",
                        "message": f"任务 {task.task_id} 的 deadline 必须大于 0。",
                    })

                if task.start_time is not None and task.start_time < task.arrival_time:
                    violations.append({
                        "task_id": task.task_id,
                        "type": "提前执行",
                        "message": f"任务 {task.task_id} 在到达前就开始执行。",
                    })

                if task.finish_time is not None and task.finish_time > horizon:
                    violations.append({
                        "task_id": task.task_id,
                        "type": "超出仿真周期",
                        "message": (
                            f"任务 {task.task_id} 完成时间 {task.finish_time} "
                            f"超过仿真周期 {horizon}。"
                        ),
                    })

                if task.finish_time is not None and task.finish_time > task.latest_finish_time:
                    violations.append({
                        "task_id": task.task_id,
                        "type": "超过截止时间",
                        "message": (
                            f"任务 {task.task_id} 完成时间 {task.finish_time} "
                            f"超过最晚完成时间 {task.latest_finish_time}。"
                        ),
                    })

            return violations

    def simulate_task_queue_stepwise(
            self,
            Q0: float,
            lambda_t: np.ndarray,
            task_load_t: np.ndarray
        ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
            """
            旧版聚合队列模拟函数。

            保留该函数用于对比，不再作为任务类改进版主流程。
            """
            horizon = len(lambda_t)

            Q_traj = np.zeros(horizon, dtype=np.float64)
            completed_work_t = np.zeros(horizon, dtype=np.float64)
            raw_capacity_t = np.zeros(horizon, dtype=np.float64)

            Q_t = float(Q0)

            for t in range(horizon):
                available_work = Q_t + float(lambda_t[t])
                raw_capacity = float(task_load_t[t] * self.C_IDC)
                completed_work = min(raw_capacity, available_work)
                Q_next = available_work - completed_work

                Q_traj[t] = Q_next
                completed_work_t[t] = completed_work
                raw_capacity_t[t] = raw_capacity

                Q_t = Q_next

            return Q_traj, completed_work_t, raw_capacity_t

    def simulate_task_execution_stepwise(
            self,
            tasks: list[Task],
            task_load_t: np.ndarray,
            horizon: int = 24
        ) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
            """
            使用 Task 对象逐小时模拟任务执行。

            当前第一版逻辑：
            1. 任务到达后进入 waiting；
            2. 每小时根据 task_load_t[t] 计算可用处理能力；
            3. 不再使用 priority/deadline 人为排序规则；
            4. 按任务列表当前顺序执行，后续由 PPO 动作空间接管任务选择；
            5. 扣减 task.remaining_work；
            6. 完成任务进入 finished；
            7. 未完成任务继续保留。
            """
            task_load_t = np.asarray(task_load_t, dtype=np.float64)
            horizon = int(horizon)

            if len(task_load_t) != horizon:
                raise ValueError("task_load_t 长度必须等于 horizon。")

            Q_traj = np.zeros(horizon, dtype=np.float64)
            completed_work_t = np.zeros(horizon, dtype=np.float64)
            raw_capacity_t = np.zeros(horizon, dtype=np.float64)

            finished_tasks = []
            deadline_miss_tasks = []

            for t in range(horizon):
                # 1. 当前小时新任务到达
                for task in tasks:
                    if task.arrival_time == t and task.status == "not_arrived":
                        task.status = "waiting"

                # 2. 当前小时理论处理能力
                raw_capacity = float(task_load_t[t] * self.C_IDC)
                remaining_capacity = raw_capacity
                raw_capacity_t[t] = raw_capacity

                # 3. 取出当前可执行任务
                active_tasks = [
                    task for task in tasks
                    if task.status in ["waiting", "running"]
                    and task.remaining_work > 1e-6
                ]

                # 4. 不再加入 priority/deadline 人为排序规则。
                # 当前保持 tasks 列表中的任务顺序，避免在 PPO 接入前预设“最优顺序”。
                # 后续如果要让 PPO 学习任务选择，需要把任务选择信息加入动作空间。

                # 5. 逐个执行任务
                completed_this_hour = 0.0

                for task in active_tasks:
                    if remaining_capacity <= 1e-6:
                        break

                    actual_work = task.execute(
                        work_amount=remaining_capacity,
                        current_time=t
                    )

                    remaining_capacity -= actual_work
                    completed_this_hour += actual_work

                    if task.status == "finished" and task not in finished_tasks:
                        finished_tasks.append(task)

                completed_work_t[t] = completed_this_hour

                # 6. 检查 deadline miss：第一版只记录超时，不直接强制失败
                for task in tasks:
                    if (
                        task.status not in ["finished", "failed"]
                        and task.status != "not_arrived"
                        and t + 1 > task.latest_finish_time
                    ):
                        if task not in deadline_miss_tasks:
                            deadline_miss_tasks.append(task)

                # 7. 当前积压任务量：所有已到达但未完成任务的剩余工作量
                unfinished_work = sum(
                    task.remaining_work
                    for task in tasks
                    if task.status != "finished" and task.status != "not_arrived"
                )
                Q_traj[t] = float(unfinished_work)

            # 8. 汇总任务级统计
            arrived_tasks = [
                task for task in tasks
                if task.arrival_time < horizon
            ]

            unfinished_tasks = [
                task for task in arrived_tasks
                if task.status != "finished"
            ]

            finished_task_count = len([
                task for task in arrived_tasks
                if task.status == "finished"
            ])

            total_task_count = len(arrived_tasks)

            task_completion_rate = (
                finished_task_count / total_task_count
                if total_task_count > 0
                else 0.0
            )

            avg_waiting_time_list = []
            avg_turnaround_time_list = []

            for task in arrived_tasks:
                if task.start_time is not None:
                    avg_waiting_time_list.append(task.start_time - task.arrival_time)

                if task.finish_time is not None:
                    avg_turnaround_time_list.append(task.finish_time - task.arrival_time)

            task_metrics = {
                "finished_tasks": finished_tasks,
                "unfinished_tasks": unfinished_tasks,
                "deadline_miss_tasks": deadline_miss_tasks,
                "finished_task_count": finished_task_count,
                "total_task_count": total_task_count,
                "task_completion_rate": float(task_completion_rate),
                "deadline_miss_count": len(deadline_miss_tasks),
                "deadline_miss_rate": (
                    len(deadline_miss_tasks) / total_task_count
                    if total_task_count > 0
                    else 0.0
                ),
                "avg_waiting_time": (
                    float(np.mean(avg_waiting_time_list))
                    if len(avg_waiting_time_list) > 0
                    else 0.0
                ),
                "avg_turnaround_time": (
                    float(np.mean(avg_turnaround_time_list))
                    if len(avg_turnaround_time_list) > 0
                    else 0.0
                ),
                "final_backlog_work": float(Q_traj[-1]) if horizon > 0 else 0.0,
            }

            return Q_traj, completed_work_t, raw_capacity_t, task_metrics
