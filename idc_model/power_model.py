import numpy as np


class IDCPowerModel:
    """
    服务器与电力侧基础模型。

    本文件只保留与服务器异构参数、算力容量、IT 功耗、PUE、
    分时电价和阶段一成本评估有关的内容。
    任务生成与任务执行逻辑放在 task_model.py。
    """

    def __init__(self,
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
                     enable_server_group_model=False,
                     server_group_size=1,
                     num_server_groups=None):
            """
            初始化 N 台具有不同物理特性的服务器。

            服务器算力来源改进说明
            ----
            旧版逻辑：先人为设定 IDC 总算力 C_IDC，再按 P_max 权重反推每台服务器算力。
            新版逻辑：先生成每台服务器自身的算力容量 C_server[i]，再求和得到 IDC 总算力 C_IDC。

            C_IDC 在新版中只作为“目标总算力尺度”，用于确定单台服务器平均算力基准，
            实际 self.C_IDC 由 sum(self.C_server) 得到。
            """
            if num_server_groups is not None:
                N = int(num_server_groups)
            self.N = int(N)
            self.server_group_model_enabled = bool(enable_server_group_model)
            self.server_group_size = (
                max(int(server_group_size), 1)
                if self.server_group_model_enabled
                else 1
            )
            self.num_server_groups = self.N
            self.effective_total_server_count = self.N * self.server_group_size

            # 1. 随机数生成器
            # 默认不固定种子，每次运行会生成不同的服务器和任务；
            # 如果后续需要复现实验，可以显式传入 server_seed 和 task_seed。
            self.server_seed = server_seed
            self.task_seed = task_seed
            self.server_rng = np.random.default_rng(server_seed)
            self.task_rng = np.random.default_rng(task_seed)

            # 2. 异构服务器功耗参数初始化
            self.single_server_P_idle = self.server_rng.uniform(P_idle_base * 0.8, P_idle_base * 1.2, self.N)
            self.single_server_P_max = self.server_rng.uniform(P_max_base * 0.8, P_max_base * 1.2, self.N)
            self.P_idle = self.single_server_P_idle * self.server_group_size
            self.P_max = self.single_server_P_max * self.server_group_size
            self.delta_P_loss = delta_P_loss
            self.alpha, self.beta, self.gamma = alpha, beta, gamma

            self.k = k  # 非线性功耗系数
            self.delta_P_loss = delta_P_loss  # 配电损耗
            self.alpha, self.beta, self.gamma = alpha, beta, gamma  # COP 系数
            self.T_target = T_target  # 目标温度
            self.P_others = P_others  # 其他基础设施功耗

            # 3. 每台服务器算力容量：先生成单台服务器算力，再求和得到总算力
            self.C_IDC_target = float(C_IDC)  # 目标总算力尺度，不再直接作为最终总算力
            self.server_capacity_variation = float(server_capacity_variation)
            self.C_IDC_target = float(C_IDC)
            self.server_capacity_variation = float(server_capacity_variation)
            C_server_base = self.C_IDC_target / self.N

            capacity_factor = self.server_rng.uniform(
                1.0 - self.server_capacity_variation,
                1.0 + self.server_capacity_variation,
                self.N
            )
            self.single_server_C_server = C_server_base * capacity_factor
            self.single_server_C_server = np.maximum(self.single_server_C_server, 1e-6)
            self.C_IDC_base = float(np.sum(self.single_server_C_server))
            self.C_server = self.single_server_C_server * self.server_group_size
            self.C_server = np.maximum(self.C_server, 1e-6)

            # 实际 IDC 总算力由各服务器算力相加得到
            self.C_IDC = float(np.sum(self.C_server))
            self.total_group_capacity = self.C_IDC
            self.total_group_idle_power_kW = float(np.sum(self.P_idle) / 1000.0)
            self.total_group_max_power_kW = float(np.sum(self.P_max) / 1000.0)

            # 每台服务器的单位功耗算力，用于后续状态空间/调度策略扩展
            self.server_compute_efficiency = self.C_server / self.P_max

            # 4. 初始化任务 Profile
            self.task_profiles = self._init_task_profiles()

    def calc_it_power(self, L: np.ndarray) -> np.ndarray:
            """
            计算 IT 功耗。

            参数
            ----
            L : np.ndarray
                服务器负载率。
                可以是 [N,] 向量，也可以是 [T, N] 矩阵。

            返回
            ----
            P_IT : float 或 np.ndarray
                IT 功耗，单位 W。
            """
            L = np.clip(L, 0.0, 1.0)

            # 单台服务器功耗模型
            power_per_server = self.P_idle + (self.P_max - self.P_idle) * (2 * L - L ** self.k)

            # 如果输入是 [T, N]，按每个小时求和
            if L.ndim == 2:
                return np.sum(power_per_server, axis=1) + self.delta_P_loss

            # 如果输入是 [N,]，直接求和
            return np.sum(power_per_server) + self.delta_P_loss

    def load_balance(self, total_L_t: np.ndarray) -> np.ndarray:
            """
            将整体负载率曲线分配到 N 台服务器。

            当前阶段采用简单规则：
            按服务器算力容量 C_server 权重分配负载，
            而不再按 P_max 权重分配。

            注意：这里仍是简化的负载均衡规则。
            后续接入 PPO 后，每台服务器负载将主要由 PPO 动作决定。
            """
            total_L_t = np.asarray(total_L_t, dtype=np.float64)

            weights = self.C_server / np.sum(self.C_server)
            L_matrix = total_L_t[:, np.newaxis] * self.N * weights

            return np.clip(L_matrix, 0.0, 1.0)

    def calc_pue_and_total_power(self, L_matrix: np.ndarray, T_amb: np.ndarray):
            """
            动态 PUE 与总功耗计算。
            """
            P_IT = self.calc_it_power(L_matrix)
            avg_L = np.mean(L_matrix, axis=1)

            COP = np.maximum(
                self.alpha * (self.T_target - T_amb) + self.beta * avg_L + self.gamma,
                0.1
            )

            P_cooling = P_IT / COP
            PUE = 1.0 + (1.0 / COP) + (self.P_others / P_IT)
            P_IDC = P_IT + P_cooling + self.P_others

            return P_IDC, P_IT, PUE, COP, P_cooling

    def create_price_curve(self, horizon: int = 24) -> np.ndarray:
            """
            构造 24 小时分时电价曲线 price_t。

            当前阶段只考虑分时电价，不考虑微电网基础负荷、新能源出力和碳排放因子。

            单位：
                元 / kWh

            设定逻辑：
                低谷时段：0:00-7:00，电价较低；
                平段时段：7:00-10:00、15:00-18:00、21:00-24:00；
                高峰时段：10:00-15:00、18:00-21:00，电价较高。

            注意：
                这里的电价是仿真设定值，不代表某一地区真实商业电价。
                后续如果能拿到真实分时电价表，可以直接替换本函数。
            """
            price_t = np.zeros(horizon, dtype=np.float64)

            valley_price = 0.35  # 低谷电价，元/kWh
            flat_price = 0.65    # 平段电价，元/kWh
            peak_price = 1.05    # 高峰电价，元/kWh

            for t in range(horizon):
                if 0 <= t < 7:
                    price_t[t] = valley_price
                elif 10 <= t < 15 or 18 <= t < 21:
                    price_t[t] = peak_price
                else:
                    price_t[t] = flat_price

            return price_t

    def evaluate_stage1_metrics(
            self,
            P_IDC: np.ndarray,
            price_t: np.ndarray,
            completed_work_t: np.ndarray,
            lambda_t: np.ndarray,
            Q0: float,
            Q_traj: np.ndarray,
            delta_t: float = 1.0
        ) -> dict:
            """
            阶段一评估函数：在分时电价条件下计算智算中心自身运行指标。

            当前阶段只评价智算中心自身：
                1. IDC 总购电量；
                2. IDC 用电成本；
                3. 总完成任务量；
                4. 单位任务成本；
                5. 队列末尾积压；
                6. 任务完成率。
            """
            P_IDC = np.asarray(P_IDC, dtype=np.float64)
            price_t = np.asarray(price_t, dtype=np.float64)
            completed_work_t = np.asarray(completed_work_t, dtype=np.float64)
            lambda_t = np.asarray(lambda_t, dtype=np.float64)
            Q_traj = np.asarray(Q_traj, dtype=np.float64)

            if len(P_IDC) != len(price_t):
                raise ValueError("P_IDC 和 price_t 长度不一致，无法计算用电成本。")

            # 1. 每小时购电量
            # P_IDC 单位是 W，除以 1000 转成 kW，再乘以小时数得到 kWh
            energy_kWh_t = P_IDC / 1000.0 * delta_t

            # 2. 每小时用电成本
            cost_t = energy_kWh_t * price_t

            # 3. 总购电量与总成本
            total_energy_kWh = float(np.sum(energy_kWh_t))
            total_cost = float(np.sum(cost_t))

            # 4. 任务完成情况
            total_arrived_work = float(np.sum(lambda_t))
            total_available_work = float(Q0 + total_arrived_work)
            total_completed_work = float(np.sum(completed_work_t))
            final_queue = float(Q_traj[-1])

            if total_available_work > 0:
                completion_rate = total_completed_work / total_available_work
            else:
                completion_rate = 0.0

            # 5. 单位任务成本：阶段一最重要指标
            if total_completed_work > 0:
                unit_task_cost = total_cost / total_completed_work
                energy_per_task = total_energy_kWh / total_completed_work
            else:
                unit_task_cost = np.inf
                energy_per_task = np.inf

            metrics = {
                "energy_kWh_t": energy_kWh_t,
                "cost_t": cost_t,
                "total_energy_kWh": total_energy_kWh,
                "total_cost": total_cost,
                "total_arrived_work": total_arrived_work,
                "total_available_work": total_available_work,
                "total_completed_work": total_completed_work,
                "unit_task_cost": float(unit_task_cost),
                "energy_per_task": float(energy_per_task),
                "final_queue": final_queue,
                "completion_rate": float(completion_rate),
            }

            return metrics
