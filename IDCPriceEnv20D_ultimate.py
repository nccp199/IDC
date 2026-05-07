import numpy as np
import gymnasium as gym
from gymnasium import spaces

from task_model import IDCEnergyTaskModel


class IDCPriceEnv20D(gym.Env):
    """
    面向 PPO 的 22 维动作智算中心分时电价任务调度环境：ultimate 前瞻状态版。

    本版相对旧环境的核心变化：
    1. 接入 Task 对象，不再把任务本体压缩成聚合队列 Q；
    2. reset() 时重新生成任务，避免上一轮 episode 的任务状态污染下一轮；
    3. Q_t 只作为由 Task.remaining_work 统计得到的积压量；
    4. step() 内部按小时激活到达任务，并按 FIFO 规则执行任务；
    5. PPO 动作空间扩展为 22 维：20 台服务器任务执行强度 + 紧急任务偏好 + 连续执行偏好；
    6. 状态空间扩展为 256 维：保留 136 维当前状态，并加入 120 维全天前瞻状态；
    7. 功耗按实际完成任务量反推实际负载，并加入计划负载预留损耗，避免高计划负载完全无成本；
    8. reward 扩展为任务类综合奖励：完成量、完整任务完成、高优先级任务完成、成本、积压、紧急积压、等待、超时、未使用能力、高电价高负载、暂停/恢复和不可暂停中断；
    9. 新增任务启停跟踪：记录任务启动、暂停、恢复与不可暂停任务中断。

    当前仍未做：
    - 服务器级任务绑定；
    - 真正的任务并行拆分；
    - PPO 直接选择具体 task_id。
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        horizon: int = 24,
        base_load: float = 0.05,
        max_task_load_per_server: float = 0.80,
        Q0: float = 50.0,
        num_tasks: int = 12,
        price_ref: float = 1.50,
        lambda_ref: float = 1000.0,
        queue_ref: float = 1500.0,
        cost_ref: float = 30.0,
        planned_load_reserve_alpha: float = 0.25,
        reward_done_weight: float = 3.0,
        reward_cost_weight: float = 0.5,
        reward_queue_weight: float = 0.8,
        reward_final_queue_weight: float = 2.0,
        reward_deadline_weight: float = 0.8,
        reward_unused_capacity_weight: float = 0.10,
        reward_finished_task_weight: float = 1.0,
        reward_priority_finish_weight: float = 0.8,
        reward_urgent_backlog_weight: float = 0.8,
        reward_waiting_weight: float = 0.3,
        reward_peak_load_weight: float = 0.2,
        reward_pause_weight: float = 0.2,
        reward_resume_weight: float = 0.05,
        reward_non_interruptible_weight: float = 1.0,
        server_seed=None,
        task_seed=None,
    ):
        super().__init__()

        # 1. 底层物理模型：建议使用最新 servercapacity 版本
        self.model = IDCEnergyTaskModel(
            N=20,
            server_seed=server_seed,
            task_seed=task_seed,
        )

        # 2. 仿真参数
        self.horizon = int(horizon)
        self.base_load = float(base_load)
        self.max_task_load_per_server = float(max_task_load_per_server)
        self.initial_Q = float(Q0)
        self.num_tasks = int(num_tasks)

        # 3. 归一化参考值
        self.price_ref = float(price_ref)
        self.lambda_ref = float(lambda_ref)
        self.queue_ref = float(queue_ref)
        self.cost_ref = float(cost_ref)
        self.planned_load_reserve_alpha = float(planned_load_reserve_alpha)

        # 4. reward 权重
        self.reward_done_weight = float(reward_done_weight)
        self.reward_cost_weight = float(reward_cost_weight)
        self.reward_queue_weight = float(reward_queue_weight)
        self.reward_final_queue_weight = float(reward_final_queue_weight)
        self.reward_deadline_weight = float(reward_deadline_weight)
        self.reward_unused_capacity_weight = float(reward_unused_capacity_weight)
        self.reward_finished_task_weight = float(reward_finished_task_weight)
        self.reward_priority_finish_weight = float(reward_priority_finish_weight)
        self.reward_urgent_backlog_weight = float(reward_urgent_backlog_weight)
        self.reward_waiting_weight = float(reward_waiting_weight)
        self.reward_peak_load_weight = float(reward_peak_load_weight)
        self.reward_pause_weight = float(reward_pause_weight)
        self.reward_resume_weight = float(reward_resume_weight)
        self.reward_non_interruptible_weight = float(reward_non_interruptible_weight)

        # 5. 动作空间：22 维
        #    action[0:20]：20 台服务器任务执行强度；
        #    action[20]：紧急任务偏好，越高越偏向 deadline 近、priority 高的任务；
        #    action[21]：连续执行偏好，越高越偏向继续执行已经启动但未完成的任务。
        self.server_action_dim = self.model.N
        self.extra_action_dim = 2
        self.action_dim = self.server_action_dim + self.extra_action_dim
        self.action_space = spaces.Box(
            low=np.zeros(self.action_dim, dtype=np.float32),
            high=np.ones(self.action_dim, dtype=np.float32),
            dtype=np.float32,
        )

        # 6. 状态空间扩展为 256 维：
        #    当前状态 136 维：6 个全局状态 + 10 个任务池状态 + 6 组服务器状态 × 20 台服务器；
        #    全天前瞻状态 120 维：price、T_amb、lambda_t、time_sin、time_cos 各 24 维。
        #    这样 PPO 既能看到当前任务/服务器运行状态，也能看到全天电价、温度和任务到达趋势。
        self.global_obs_dim = 6
        self.task_pool_obs_dim = 10
        self.server_feature_groups = 6
        self.current_obs_dim = (
            self.global_obs_dim
            + self.task_pool_obs_dim
            + self.server_feature_groups * self.model.N
        )
        self.forecast_feature_groups = 5
        self.forecast_obs_dim = self.forecast_feature_groups * self.horizon
        self.obs_dim = self.current_obs_dim + self.forecast_obs_dim
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.obs_dim,),
            dtype=np.float32,
        )

        # 7. 固定 24 小时外部输入
        self.hours = np.arange(self.horizon)
        self.T_amb = 25 + 5 * np.sin(np.pi * (self.hours - 8) / 12)

        if hasattr(self.model, "create_price_curve"):
            self.price_t = self.model.create_price_curve(horizon=self.horizon)
        else:
            self.price_t = self._create_price_curve(horizon=self.horizon)

        # 8. 运行状态变量会在 reset() 中初始化
        self.current_step = 0
        self.tasks = []
        self.lambda_t = np.zeros(self.horizon, dtype=np.float64)
        self.Q_t = 0.0
        self.prev_loads = np.full(self.model.N, self.base_load, dtype=np.float32)

        # 9. episode 累计指标
        self.total_energy_kWh = 0.0
        self.total_cost = 0.0
        self.total_completed_work = 0.0
        self.deadline_miss_task_ids = set()

        # 10. 任务启停统计指标
        self.total_pause_count = 0
        self.total_resume_count = 0
        self.total_non_interruptible_interruption_count = 0

    def _create_price_curve(self, horizon: int = 24) -> np.ndarray:
        """备用分时电价曲线。"""
        price_t = np.zeros(horizon, dtype=np.float64)
        valley_price = 0.35
        flat_price = 0.65
        peak_price = 1.05

        for t in range(horizon):
            if 0 <= t < 7:
                price_t[t] = valley_price
            elif 10 <= t < 15 or 18 <= t < 21:
                price_t[t] = peak_price
            else:
                price_t[t] = flat_price

        return price_t

    def reset(self, seed=None, options=None):
        """重置环境，开始新的 24 小时 episode。"""
        super().reset(seed=seed)

        self.current_step = 0
        self.prev_loads = np.full(self.model.N, self.base_load, dtype=np.float32)

        self.total_energy_kWh = 0.0
        self.total_cost = 0.0
        self.total_completed_work = 0.0
        self.deadline_miss_task_ids = set()
        self.total_pause_count = 0
        self.total_resume_count = 0
        self.total_non_interruptible_interruption_count = 0

        # 每个 episode 重新生成任务，防止 Task.status/remaining_work 等状态残留。
        self.tasks = self.model.create_demo_tasks(
            num_tasks=self.num_tasks,
            horizon=self.horizon,
        )

        # 将原来的 Q0 改造成初始积压任务，并纳入 Task 列表。
        initial_backlog_task = self.model.create_initial_backlog_task(self.initial_Q)
        self.tasks.insert(0, initial_backlog_task)
        self._initialize_task_runtime_state()

        self.lambda_t = self.model.build_task_arrival_curve(
            tasks=self.tasks,
            horizon=self.horizon,
        )

        # reset 后先激活 t=0 已到达任务，让初始状态能看到初始积压。
        self._activate_arrivals(current_time=0)
        self.Q_t = self._compute_backlog_work()

        obs = self._get_obs()
        info = {
            "total_task_count": len(self.tasks),
            "initial_backlog_work": self.Q_t,
            "obs_dim": self.obs_dim,
        }

        return obs, info

    def step(self, action):
        """
        执行一步，也就是推进 1 小时。

        PPO 输入：
            action: shape=(22,)
            action[0:20] 表示对应服务器的任务执行强度；
            action[20] 表示紧急任务偏好；
            action[21] 表示连续执行偏好。

        本版处理逻辑：
            1. 动作先转换为计划任务负载和计划处理能力；
            2. 根据紧急任务偏好和连续执行偏好，引导任务执行顺序；
            3. 根据实际完成任务量反推实际任务负载；
            4. 用实际负载计算功耗和成本；
            5. Q_t 由 Task.remaining_work 统计得到。
        """
        t = self.current_step

        # 1. 解析 22 维动作
        action = np.asarray(action, dtype=np.float32).reshape(-1)
        if action.shape[0] != self.action_dim:
            raise ValueError(
                f"动作维度错误：期望 {self.action_dim} 维，实际 {action.shape[0]} 维。"
            )
        action = np.clip(action, 0.0, 1.0)

        server_action = action[:self.model.N]
        urgent_preference = float(action[self.model.N])
        continuity_preference = float(action[self.model.N + 1])

        # 2. PPO 计划任务负载：仅用于计算计划处理能力，不直接用于功耗
        planned_task_loads = server_action * self.max_task_load_per_server
        planned_total_loads = np.clip(self.base_load + planned_task_loads, 0.0, 1.0)

        # 按每台服务器算力计算计划处理能力
        planned_capacity = float(np.sum(planned_task_loads * self.model.C_server))

        # 3. 当前小时外部输入
        T_amb_t = float(self.T_amb[t])
        price_now = float(self.price_t[t])
        lambda_now = float(self.lambda_t[t])

        # 4. 当前小时新任务到达
        self._activate_arrivals(current_time=t)

        # 5. 根据动作中的任务偏好执行任务，并记录任务暂停/恢复事件
        (
            completed_work,
            newly_finished_count,
            newly_finished_priority_sum,
            new_deadline_miss_count,
            pause_count_this_step,
            resume_count_this_step,
            non_interruptible_interruption_this_step,
        ) = self._execute_tasks_action_guided(
            available_capacity=planned_capacity,
            current_time=t,
            urgent_preference=urgent_preference,
            continuity_preference=continuity_preference,
        )

        unused_capacity = max(planned_capacity - completed_work, 0.0)

        # 6. 按实际完成任务量反推实际任务负载
        actual_task_loads = self._actual_loads_from_completed_work(
            planned_task_loads=planned_task_loads,
            planned_capacity=planned_capacity,
            completed_work=completed_work,
        )
        actual_total_loads = np.clip(self.base_load + actual_task_loads, 0.0, 1.0)

        # 7. 用实际负载计算当前小时功耗
        L_matrix = actual_total_loads.reshape(1, -1)
        P_IDC_arr, P_IT_arr, PUE_arr = self.model.calc_pue_and_total_power(
            L_matrix=L_matrix,
            T_amb=np.array([T_amb_t], dtype=np.float64),
        )

        P_IDC_t = float(P_IDC_arr[0])
        P_IT_t = float(P_IT_arr[0])
        PUE_t = float(PUE_arr[0])

        # 8. 当前小时购电量和用电成本
        energy_kWh = P_IDC_t / 1000.0
        cost_t = energy_kWh * price_now

        # 9. 更新任务积压统计
        Q_next = self._compute_backlog_work()

        # 10. 累计统计
        self.total_energy_kWh += energy_kWh
        self.total_cost += cost_t
        self.total_completed_work += completed_work

        # 11. reward：任务类综合奖励
        # 奖励项：完成工作量、完整完成任务数、高优先级任务完成；
        # 惩罚项：成本、普通积压、紧急积压、等待压力、超时、未使用能力、高电价高负载、暂停/恢复和不可暂停任务中断。
        urgent_backlog_work, avg_waiting_pressure = self._compute_reward_task_pressure(current_time=t)

        completed_norm = completed_work / self.queue_ref
        finished_task_norm = newly_finished_count / max(len(self.tasks), 1)
        priority_finish_norm = newly_finished_priority_sum / max(5.0 * len(self.tasks), 1e-6)
        cost_norm = cost_t / self.cost_ref
        queue_norm = Q_next / self.queue_ref
        urgent_backlog_norm = urgent_backlog_work / max(self.queue_ref, 1e-6)
        waiting_norm = avg_waiting_pressure / max(self.horizon, 1)
        deadline_miss_norm = new_deadline_miss_count / max(len(self.tasks), 1)
        unused_capacity_norm = unused_capacity / max(self.queue_ref, 1e-6)
        price_min = float(np.min(self.price_t))
        price_max = float(np.max(self.price_t))
        price_pressure = (price_now - price_min) / max(price_max - price_min, 1e-6)
        peak_load_norm = price_pressure * float(np.mean(planned_task_loads))        
        pause_norm = pause_count_this_step / max(len(self.tasks), 1)
        resume_norm = resume_count_this_step / max(len(self.tasks), 1)
        non_interruptible_norm = non_interruptible_interruption_this_step / max(len(self.tasks), 1)

        r_done = self.reward_done_weight * completed_norm
        r_finished_task = self.reward_finished_task_weight * finished_task_norm
        r_priority_finish = self.reward_priority_finish_weight * priority_finish_norm
        r_cost = -self.reward_cost_weight * cost_norm
        r_queue = -self.reward_queue_weight * queue_norm
        r_urgent_backlog = -self.reward_urgent_backlog_weight * urgent_backlog_norm
        r_waiting = -self.reward_waiting_weight * waiting_norm
        r_deadline = -self.reward_deadline_weight * deadline_miss_norm
        r_unused = -self.reward_unused_capacity_weight * unused_capacity_norm
        r_peak_load = -self.reward_peak_load_weight * peak_load_norm
        r_pause = -self.reward_pause_weight * pause_norm
        r_resume = -self.reward_resume_weight * resume_norm
        r_non_interruptible = -self.reward_non_interruptible_weight * non_interruptible_norm
        r_final_queue = 0.0

        reward = (
            r_done
            + r_finished_task
            + r_priority_finish
            + r_cost
            + r_queue
            + r_urgent_backlog
            + r_waiting
            + r_deadline
            + r_unused
            + r_peak_load
            + r_pause
            + r_resume
            + r_non_interruptible
        )

        terminated = (self.current_step + 1) >= self.horizon
        truncated = False

        # 最后一小时额外惩罚最终积压，避免 PPO 一直拖任务
        if terminated:
            final_queue_norm = Q_next / self.queue_ref
            r_final_queue = -self.reward_final_queue_weight * final_queue_norm
            reward += r_final_queue

        # 12. 更新环境内部状态
        self.Q_t = Q_next
        self.prev_loads = actual_total_loads.astype(np.float32)
        self.current_step += 1

        # 13. 生成下一状态
        if terminated:
            obs = np.zeros(self.observation_space.shape, dtype=np.float32)
        else:
            obs = self._get_obs()

        # 14. 记录信息，方便训练后画图和计算指标
        task_metrics = self._compute_task_metrics()
        total_available_work = self._total_available_work()

        completion_rate = (
            self.total_completed_work / total_available_work
            if total_available_work > 0
            else 0.0
        )

        if self.total_completed_work > 0:
            unit_task_cost = self.total_cost / self.total_completed_work
            energy_per_task = self.total_energy_kWh / self.total_completed_work
        else:
            unit_task_cost = np.inf
            energy_per_task = np.inf

        info = {
            "hour": t,
            "price": price_now,
            "lambda_t": lambda_now,

            "action_mean": float(np.mean(server_action)),
            "action_min": float(np.min(server_action)),
            "action_max": float(np.max(server_action)),
            "urgent_preference": urgent_preference,
            "continuity_preference": continuity_preference,

            "planned_task_load_mean": float(np.mean(planned_task_loads)),
            "planned_total_load_mean": float(np.mean(planned_total_loads)),
            "actual_task_load_mean": float(np.mean(actual_task_loads)),
            "actual_total_load_mean": float(np.mean(actual_total_loads)),

            "planned_capacity": planned_capacity,
            "raw_capacity": planned_capacity,  # 兼容旧字段名
            "completed_work": completed_work,
            "unused_capacity": unused_capacity,
            "Q": Q_next,
            "backlog_work": Q_next,

            "newly_finished_count": newly_finished_count,
            "newly_finished_priority_sum": float(newly_finished_priority_sum),
            "urgent_backlog_work": float(urgent_backlog_work),
            "avg_waiting_pressure": float(avg_waiting_pressure),
            "new_deadline_miss_count": new_deadline_miss_count,
            "deadline_miss_count": task_metrics["deadline_miss_count"],

            "pause_count_this_step": pause_count_this_step,
            "resume_count_this_step": resume_count_this_step,
            "non_interruptible_interruption_this_step": non_interruptible_interruption_this_step,
            "total_pause_count": self.total_pause_count,
            "total_resume_count": self.total_resume_count,
            "total_non_interruptible_interruption_count": self.total_non_interruptible_interruption_count,

            # reward 分项，便于后续调参和定位问题
            "r_done": float(r_done),
            "r_finished_task": float(r_finished_task),
            "r_priority_finish": float(r_priority_finish),
            "r_cost": float(r_cost),
            "r_queue": float(r_queue),
            "r_urgent_backlog": float(r_urgent_backlog),
            "r_waiting": float(r_waiting),
            "r_deadline": float(r_deadline),
            "r_unused": float(r_unused),
            "r_peak_load": float(r_peak_load),
            "r_pause": float(r_pause),
            "r_resume": float(r_resume),
            "r_non_interruptible": float(r_non_interruptible),
            "r_final_queue": float(r_final_queue),
            "reward_total": float(reward),

            "P_IDC": P_IDC_t,
            "P_IT": P_IT_t,
            "PUE": PUE_t,
            "energy_kWh": energy_kWh,
            "cost": cost_t,

            "total_energy_kWh": self.total_energy_kWh,
            "total_cost": self.total_cost,
            "total_completed_work": self.total_completed_work,
            "completion_rate": completion_rate,
            "unit_task_cost": unit_task_cost,
            "energy_per_task": energy_per_task,

            **task_metrics,
        }

        return obs, float(reward), terminated, truncated, info

    def _activate_arrivals(self, current_time: int) -> int:
        """激活当前小时到达的任务。"""
        count = 0
        for task in self.tasks:
            if task.arrival_time == current_time and task.status == "not_arrived":
                task.status = "waiting"
                count += 1
        return count

    def _initialize_task_runtime_state(self) -> None:
        """
        初始化每个 Task 的启停统计字段。

        这些字段不要求写入任务类定义中，环境层在每个 episode
        reset 后为任务对象动态添加，方便后续记录暂停、恢复和
        不可暂停任务中断。
        """
        for task in self.tasks:
            task.pause_count = 0
            task.resume_count = 0
            task.non_interruptible_interruption_count = 0
            task.last_executed_time = None
            task.is_paused = False

    def _task_selection_score(
        self,
        task,
        current_time: int,
        urgent_preference: float,
        continuity_preference: float,
    ) -> float:
        """
        根据 PPO 给出的任务偏好，为候选任务计算选择分数。

        urgent_preference 越高：
            越偏向 deadline 近、priority 高、已经超时的任务。
        continuity_preference 越高：
            越偏向继续执行已经启动但尚未完成的任务。

        当两个偏好都接近 0 时，所有任务得分接近 0，
        后续排序会退化为 FIFO。
        """
        urgent_preference = float(np.clip(urgent_preference, 0.0, 1.0))
        continuity_preference = float(np.clip(continuity_preference, 0.0, 1.0))

        deadline_left = float(task.latest_finish_time - current_time)
        # deadline 越近，deadline_score 越高；已超时任务给满分。
        if deadline_left <= 0:
            deadline_score = 1.0
            overdue_score = 1.0
        else:
            deadline_score = 1.0 - np.clip(deadline_left / max(self.horizon, 1), 0.0, 1.0)
            overdue_score = 0.0

        priority_score = np.clip(float(getattr(task, "priority", 0.0)) / 5.0, 0.0, 1.0)
        urgency_score = (
            0.55 * deadline_score
            + 0.35 * priority_score
            + 0.10 * overdue_score
        )

        # 已经启动但未完成的任务更需要连续执行。
        has_started = task.start_time is not None
        was_executed_before = getattr(task, "last_executed_time", None) is not None
        is_paused = bool(getattr(task, "is_paused", False))
        continuity_score = 1.0 if (has_started or was_executed_before or is_paused) else 0.0

        return float(
            urgent_preference * urgency_score
            + continuity_preference * continuity_score
        )

    def _execute_tasks_action_guided(
        self,
        available_capacity: float,
        current_time: int,
        urgent_preference: float,
        continuity_preference: float,
    ):
        """
        根据动作中的任务偏好执行任务，并跟踪任务启停。

        启停机制：
        1. 任务第一次执行时，Task.execute 会自动记录 start_time；
        2. 已启动但未完成的任务，如果本小时没有继续执行，则记为 pause；
        3. 已暂停任务再次执行，则记为 resume；
        4. interruptible=False 的任务如果被暂停，记录不可暂停任务中断。

        任务选择：
        - 不再是纯 FIFO；
        - urgent_preference 控制对紧急/高优先级任务的偏向；
        - continuity_preference 控制对已启动未完成任务的连续执行偏向；
        - 当两个偏好都很低时，排序退化为 FIFO。
        """
        remaining_capacity = float(max(available_capacity, 0.0))
        completed_this_hour = 0.0
        newly_finished_count = 0
        newly_finished_priority_sum = 0.0
        resume_count_this_step = 0
        executed_task_ids = set()

        active_tasks = [
            task for task in self.tasks
            if task.status in ["waiting", "running", "paused"]
            and task.remaining_work > 1e-6
        ]

        scored_tasks = []
        for task in active_tasks:
            score = self._task_selection_score(
                task=task,
                current_time=current_time,
                urgent_preference=urgent_preference,
                continuity_preference=continuity_preference,
            )
            scored_tasks.append((score, task))

        # 得分高的优先；得分相同则按 FIFO。
        scored_tasks.sort(
            key=lambda item: (
                -item[0],
                item[1].arrival_time,
                item[1].task_id,
            )
        )
        ordered_tasks = [task for _, task in scored_tasks]

        for task in ordered_tasks:
            if remaining_capacity <= 1e-6:
                break

            before_status = task.status

            # 如果任务之前被暂停，本小时重新获得算力，则记为恢复。
            if bool(getattr(task, "is_paused", False)):
                task.resume_count = int(getattr(task, "resume_count", 0)) + 1
                self.total_resume_count += 1
                resume_count_this_step += 1
                task.is_paused = False

            actual_work = task.execute(
                work_amount=remaining_capacity,
                current_time=current_time,
            )

            if actual_work > 1e-9:
                executed_task_ids.add(task.task_id)
                task.last_executed_time = int(current_time)

            remaining_capacity -= actual_work
            completed_this_hour += actual_work

            if before_status != "finished" and task.status == "finished":
                newly_finished_count += 1
                newly_finished_priority_sum += float(getattr(task, "priority", 0.0))
                task.is_paused = False

        pause_count_this_step, non_interruptible_interruption_this_step = self._update_pause_events(
            current_time=current_time,
            executed_task_ids=executed_task_ids,
        )

        new_deadline_miss_count = self._update_deadline_miss(current_time=current_time)

        return (
            float(completed_this_hour),
            int(newly_finished_count),
            float(newly_finished_priority_sum),
            int(new_deadline_miss_count),
            int(pause_count_this_step),
            int(resume_count_this_step),
            int(non_interruptible_interruption_this_step),
        )

    def _update_pause_events(self, current_time: int, executed_task_ids: set[int]):
        """
        根据本小时实际执行任务集合，更新暂停/不可暂停中断统计。

        判断逻辑：
        - 任务已经启动过 start_time is not None；
        - 任务尚未完成；
        - 本小时没有执行；
        - 任务当前还没有处于 paused 状态。

        满足以上条件时，认为该任务在本小时被暂停。若任务不可暂停
        interruptible=False，则额外记为不可暂停任务中断。
        """
        pause_count_this_step = 0
        non_interruptible_interruption_this_step = 0

        for task in self.tasks:
            if task.status in ["not_arrived", "finished", "failed"]:
                continue
            if task.remaining_work <= 1e-6:
                continue
            if task.start_time is None:
                continue
            if task.task_id in executed_task_ids:
                continue
            if bool(getattr(task, "is_paused", False)):
                continue

            task.pause_count = int(getattr(task, "pause_count", 0)) + 1
            task.is_paused = True
            task.status = "paused"
            self.total_pause_count += 1
            pause_count_this_step += 1

            if not bool(task.interruptible):
                task.non_interruptible_interruption_count = (
                    int(getattr(task, "non_interruptible_interruption_count", 0)) + 1
                )
                self.total_non_interruptible_interruption_count += 1
                non_interruptible_interruption_this_step += 1

        return int(pause_count_this_step), int(non_interruptible_interruption_this_step)

    def _update_deadline_miss(self, current_time: int) -> int:
        """
        记录新发生的 deadline miss。

        这里不会把任务直接标记为 failed，避免过早引入失败终止逻辑；
        当前阶段只在 reward 和 info 中记录超时压力。
        """
        new_count = 0
        for task in self.tasks:
            if task.status in ["not_arrived", "finished", "failed"]:
                continue
            if current_time + 1 > task.latest_finish_time:
                if task.task_id not in self.deadline_miss_task_ids:
                    self.deadline_miss_task_ids.add(task.task_id)
                    new_count += 1
        return new_count

    def _actual_loads_from_completed_work(
        self,
        planned_task_loads: np.ndarray,
        planned_capacity: float,
        completed_work: float,
    ) -> np.ndarray:
        """
        根据实际完成任务量反推实际任务负载，并加入计划负载预留损耗。

        旧逻辑：
            actual_task_loads = planned_task_loads * usage_ratio

        新逻辑：
            实际负载 = 实际使用负载 + alpha * 未使用计划负载

        含义：
            即使计划能力没有完全转化为任务完成量，被调度到高负载的服务器
            仍然会产生一部分资源预留、空转或调度开销。
        """
        planned_task_loads = np.asarray(planned_task_loads, dtype=np.float64)

        if planned_capacity <= 1e-9:
            return np.zeros_like(planned_task_loads, dtype=np.float64)

        if completed_work <= 1e-9:
            used_task_loads = np.zeros_like(planned_task_loads, dtype=np.float64)
        else:
            usage_ratio = float(np.clip(completed_work / planned_capacity, 0.0, 1.0))
            used_task_loads = planned_task_loads * usage_ratio

        reserve_alpha = float(np.clip(self.planned_load_reserve_alpha, 0.0, 1.0))
        unused_planned_loads = np.maximum(planned_task_loads - used_task_loads, 0.0)
        actual_task_loads = used_task_loads + reserve_alpha * unused_planned_loads

        return np.clip(actual_task_loads, 0.0, self.max_task_load_per_server)

    def _compute_backlog_work(self) -> float:
        """由 Task 列表统计当前已到达但未完成任务的剩余工作量。"""
        return float(sum(
            task.remaining_work
            for task in self.tasks
            if task.status != "finished" and task.status != "not_arrived"
        ))

    def _total_available_work(self) -> float:
        """统计本 episode 内所有会到达任务的总工作量，包括初始积压任务。"""
        return float(sum(
            task.workload
            for task in self.tasks
            if task.arrival_time < self.horizon
        ))

    def _compute_reward_task_pressure(self, current_time: int) -> tuple[float, float]:
        """
        计算 reward 中使用的任务压力指标。

        urgent_backlog_work：快到 deadline 或已经超时的未完成任务量；
        avg_waiting_pressure：已到达未完成任务的平均等待/滞留时间。
        """
        urgent_window = 3
        unfinished_tasks = [
            task for task in self.tasks
            if task.status not in ["not_arrived", "finished", "failed"]
            and task.remaining_work > 1e-6
        ]

        urgent_backlog_work = 0.0
        waiting_pressure_list = []

        for task in unfinished_tasks:
            deadline_left = task.latest_finish_time - current_time
            if deadline_left <= urgent_window:
                urgent_backlog_work += float(task.remaining_work)

            # 对已经到达但尚未完成的任务，记录其滞留时间。
            # 尚未启动的任务等待时间 = 当前时间 - 到达时间；
            # 已启动但未完成的任务也计入滞留压力，避免任务长期半完成。
            waiting_pressure_list.append(max(0, current_time - task.arrival_time))

        avg_waiting_pressure = (
            float(np.mean(waiting_pressure_list))
            if len(waiting_pressure_list) > 0
            else 0.0
        )

        return float(urgent_backlog_work), float(avg_waiting_pressure)

    def _compute_task_metrics(self) -> dict:
        """计算任务级统计指标。"""
        arrived_tasks = [
            task for task in self.tasks
            if task.arrival_time < self.horizon
        ]
        finished_tasks = [
            task for task in arrived_tasks
            if task.status == "finished"
        ]
        unfinished_tasks = [
            task for task in arrived_tasks
            if task.status != "finished"
        ]

        total_task_count = len(arrived_tasks)
        finished_task_count = len(finished_tasks)
        unfinished_task_count = len(unfinished_tasks)

        task_completion_rate = (
            finished_task_count / total_task_count
            if total_task_count > 0
            else 0.0
        )

        waiting_time_list = [
            task.start_time - task.arrival_time
            for task in finished_tasks + unfinished_tasks
            if task.start_time is not None
        ]
        turnaround_time_list = [
            task.finish_time - task.arrival_time
            for task in finished_tasks
            if task.finish_time is not None
        ]

        paused_task_count = len([
            task for task in arrived_tasks
            if task.status == "paused"
        ])

        return {
            "total_task_count": total_task_count,
            "finished_task_count": finished_task_count,
            "unfinished_task_count": unfinished_task_count,
            "paused_task_count": int(paused_task_count),
            "task_completion_rate": float(task_completion_rate),
            "deadline_miss_count": int(len(self.deadline_miss_task_ids)),
            "deadline_miss_rate": (
                len(self.deadline_miss_task_ids) / total_task_count
                if total_task_count > 0
                else 0.0
            ),
            "avg_waiting_time": (
                float(np.mean(waiting_time_list))
                if len(waiting_time_list) > 0
                else 0.0
            ),
            "avg_turnaround_time": (
                float(np.mean(turnaround_time_list))
                if len(turnaround_time_list) > 0
                else 0.0
            ),
            "final_backlog_work": float(self.Q_t),
            "total_pause_count": int(self.total_pause_count),
            "total_resume_count": int(self.total_resume_count),
            "total_non_interruptible_interruption_count": int(
                self.total_non_interruptible_interruption_count
            ),
        }

    def _get_task_pool_features(self, current_time: int) -> np.ndarray:
        """
        构造任务池状态特征，共 10 维。

        这些特征来自 Task 对象，体现任务类改进的价值：
        PPO 不再只能看到总队列 Q，而是能看到任务数量、紧急任务量、
        超时任务量、平均 deadline、优先级、可暂停/可并行任务压力。
        """
        arrived_tasks = [
            task for task in self.tasks
            if task.arrival_time <= current_time
        ]
        waiting_tasks = [
            task for task in arrived_tasks
            if task.status in ["waiting", "paused"]
        ]
        running_tasks = [
            task for task in arrived_tasks
            if task.status == "running"
        ]
        finished_tasks = [
            task for task in arrived_tasks
            if task.status == "finished"
        ]
        unfinished_tasks = [
            task for task in arrived_tasks
            if task.status not in ["finished", "failed", "not_arrived"]
            and task.remaining_work > 1e-6
        ]

        total_task_ref = max(len(self.tasks), 1)
        work_ref = max(self.queue_ref, 1e-6)
        priority_ref = 5.0
        urgent_window = 3

        waiting_task_count_norm = len(waiting_tasks) / total_task_ref
        running_task_count_norm = len(running_tasks) / total_task_ref
        finished_task_count_norm = len(finished_tasks) / total_task_ref
        unfinished_task_count_norm = len(unfinished_tasks) / total_task_ref

        urgent_work = 0.0
        overdue_work = 0.0
        deadline_left_list = []
        priority_list = []
        parallelizable_work = 0.0
        interruptible_work = 0.0

        for task in unfinished_tasks:
            deadline_left = task.latest_finish_time - current_time
            deadline_left_list.append(deadline_left)
            priority_list.append(float(task.priority))

            if 0 < deadline_left <= urgent_window:
                urgent_work += float(task.remaining_work)
            if deadline_left <= 0:
                overdue_work += float(task.remaining_work)
            if bool(task.parallelizable):
                parallelizable_work += float(task.remaining_work)
            if bool(task.interruptible):
                interruptible_work += float(task.remaining_work)

        avg_deadline_left_norm = (
            np.mean(np.clip(deadline_left_list, 0, self.horizon)) / max(self.horizon, 1)
            if len(deadline_left_list) > 0
            else 0.0
        )
        avg_priority_norm = (
            np.mean(priority_list) / priority_ref
            if len(priority_list) > 0
            else 0.0
        )

        features = np.array([
            waiting_task_count_norm,
            running_task_count_norm,
            finished_task_count_norm,
            unfinished_task_count_norm,
            urgent_work / work_ref,
            overdue_work / work_ref,
            avg_deadline_left_norm,
            avg_priority_norm,
            parallelizable_work / work_ref,
            interruptible_work / work_ref,
        ], dtype=np.float32)

        return np.clip(features, 0.0, 1.5).astype(np.float32)

    def _get_server_features(self, current_time: int) -> np.ndarray:
        """
        构造服务器状态特征，共 6 * N 维。

        每台服务器包含 6 组信息：
        1. 上一小时实际总负载 prev_loads；
        2. 单台服务器算力容量 C_server；
        3. 服务器能效 C_server / P_max；
        4. 当前电价下的单位算力成本；
        5. 服务器剩余可用负载容量；
        6. 基于环境温度和负载估算的服务器温度。
        """
        eps = 1e-6
        price_now = float(self.price_t[current_time])
        prev_loads = np.clip(self.prev_loads.astype(np.float64), 0.0, 1.0)

        server_capacity = np.asarray(self.model.C_server, dtype=np.float64)
        server_capacity_norm = server_capacity / max(float(np.max(server_capacity)), eps)

        if hasattr(self.model, "server_compute_efficiency"):
            efficiency = np.asarray(self.model.server_compute_efficiency, dtype=np.float64)
        else:
            efficiency = server_capacity / np.maximum(np.asarray(self.model.P_max, dtype=np.float64), eps)
        server_efficiency_norm = efficiency / max(float(np.max(efficiency)), eps)

        # 单位算力成本：电价 × 任务相关功耗增量 / 服务器算力。
        # 这个特征能让 PPO 感知“哪台服务器在当前电价下更便宜”。
        incremental_power = np.maximum(
            np.asarray(self.model.P_max, dtype=np.float64) - np.asarray(self.model.P_idle, dtype=np.float64),
            eps,
        )
        server_unit_cost = price_now * incremental_power / np.maximum(server_capacity, eps)
        server_unit_cost_norm = server_unit_cost / max(float(np.max(server_unit_cost)), eps)

        server_available_norm = np.clip(1.0 - prev_loads, 0.0, 1.0)

        # 简化估算服务器温度：环境温度 + 负载造成的热量增量。
        # 当前不是热管理模型，只是为状态空间提供服务器局部状态特征。
        estimated_server_temp = self.T_amb[current_time] + 15.0 * prev_loads
        server_temp_norm = np.clip(estimated_server_temp / 80.0, 0.0, 1.0)

        features = np.concatenate([
            prev_loads.astype(np.float32),
            server_capacity_norm.astype(np.float32),
            server_efficiency_norm.astype(np.float32),
            server_unit_cost_norm.astype(np.float32),
            server_available_norm.astype(np.float32),
            server_temp_norm.astype(np.float32),
        ])

        return np.clip(features, 0.0, 1.5).astype(np.float32)

    def _get_forecast_features(self) -> np.ndarray:
        """
        构造全天固定 24 小时前瞻特征，共 5 * horizon 维。

        本版本不使用滚动窗口，而是每一步都提供同一组 0-23 小时全天外部时序信息：
        1. 分时电价 price_t；
        2. 环境温度 T_amb；
        3. 任务到达量 lambda_t；
        4. 小时时间编码 time_sin；
        5. 小时时间编码 time_cos。

        这些变量属于已知/可预测的外部条件，不包含未来队列、未来任务完成状态、
        未来服务器真实负载等由 PPO 动作决定的结果，避免未来信息泄露。
        """
        eps = 1e-6
        hours = np.arange(self.horizon, dtype=np.float64)

        price_24h = np.asarray(self.price_t, dtype=np.float64) / max(self.price_ref, eps)
        T_amb_24h = np.asarray(self.T_amb, dtype=np.float64) / 40.0
        lambda_24h = np.asarray(self.lambda_t, dtype=np.float64) / max(self.lambda_ref, eps)
        time_sin_24h = np.sin(2 * np.pi * hours / max(self.horizon, 1))
        time_cos_24h = np.cos(2 * np.pi * hours / max(self.horizon, 1))

        forecast_features = np.concatenate([
            price_24h,
            T_amb_24h,
            lambda_24h,
            time_sin_24h,
            time_cos_24h,
        ]).astype(np.float32)

        if forecast_features.shape[0] != self.forecast_obs_dim:
            raise RuntimeError(
                f"前瞻状态维度错误：期望 {self.forecast_obs_dim}，实际 {forecast_features.shape[0]}。"
            )

        # price/T/lambda 归一化后理论上多为正值；sin/cos 在 [-1, 1]。
        # 这里做温和裁剪，避免偶发极端任务到达量造成输入过大。
        return np.clip(forecast_features, -1.5, 1.5).astype(np.float32)

    def _get_obs(self):
        """构造当前状态向量。状态空间为 256 维：136 维当前状态 + 120 维全天前瞻状态。"""
        t = self.current_step

        T_norm = self.T_amb[t] / 40.0
        price_norm = self.price_t[t] / self.price_ref
        lambda_norm = self.lambda_t[t] / self.lambda_ref
        Q_norm = self.Q_t / self.queue_ref

        time_sin = np.sin(2 * np.pi * t / self.horizon)
        time_cos = np.cos(2 * np.pi * t / self.horizon)

        global_features = np.array(
            [T_norm, price_norm, lambda_norm, Q_norm, time_sin, time_cos],
            dtype=np.float32,
        )
        task_pool_features = self._get_task_pool_features(current_time=t)
        server_features = self._get_server_features(current_time=t)

        current_features = np.concatenate([
            global_features,
            task_pool_features,
            server_features,
        ]).astype(np.float32)

        if current_features.shape[0] != self.current_obs_dim:
            raise RuntimeError(
                f"当前状态维度错误：期望 {self.current_obs_dim}，实际 {current_features.shape[0]}。"
            )

        forecast_features = self._get_forecast_features()

        obs = np.concatenate([
            current_features,
            forecast_features,
        ]).astype(np.float32)

        if obs.shape[0] != self.obs_dim:
            raise RuntimeError(
                f"状态维度错误：期望 {self.obs_dim}，实际 {obs.shape[0]}。"
            )

        return obs
