import numpy as np
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Task:
    """
    任务类：用于替代原来的任务 dict 和聚合队列 Q。

    当前版本在上一版 Task 对象基础上，进一步加入任务参数范围化：
    同一类任务不再固定 duration、load_profile、deadline、priority，
    而是在任务生成时从对应范围内采样得到具体任务实例。
    """
    task_id: int
    profile_key: str
    name: str
    arrival_time: int
    duration: int
    load_profile: np.ndarray
    workload: float
    deadline: int
    priority: float
    interruptible: bool
    parallelizable: bool

    remaining_work: float = field(init=False)
    status: str = "not_arrived"
    start_time: Optional[int] = None
    finish_time: Optional[int] = None
    assigned_servers: list = field(default_factory=list)
    execution_log: list = field(default_factory=list)

    def __post_init__(self):
        self.duration = int(self.duration)
        self.arrival_time = int(self.arrival_time)
        self.deadline = int(self.deadline)
        self.priority = float(self.priority)
        self.load_profile = np.asarray(self.load_profile, dtype=np.float64)
        self.workload = float(self.workload)
        self.remaining_work = float(self.workload)

    @property
    def latest_finish_time(self) -> int:
        """任务最晚完成时间。"""
        return int(self.arrival_time + self.deadline)

    @property
    def is_finished(self) -> bool:
        return self.remaining_work <= 1e-6

    @property
    def avg_load(self) -> float:
        """任务负载曲线的平均负载率。"""
        if len(self.load_profile) == 0:
            return 0.0
        return float(np.mean(self.load_profile))

    def execute(self, work_amount: float, current_time: int) -> float:
        """
        执行任务的一部分工作量。

        参数
        ----
        work_amount : float
            当前小时分配给该任务的处理能力。

        current_time : int
            当前小时。

        返回
        ----
        actual_work : float
            实际完成的任务量。
        """
        if self.status in ["finished", "failed", "not_arrived"]:
            return 0.0

        if self.start_time is None:
            self.start_time = int(current_time)

        self.status = "running"

        actual_work = min(self.remaining_work, float(work_amount))
        self.remaining_work -= actual_work

        self.execution_log.append({
            "hour": int(current_time),
            "work": float(actual_work),
            "remaining_work": float(max(self.remaining_work, 0.0)),
        })

        if self.remaining_work <= 1e-6:
            self.remaining_work = 0.0
            self.status = "finished"
            self.finish_time = int(current_time + 1)
        else:
            # 当前版本允许任务跨小时继续执行。
            # 这里先统一回到 waiting，后续再严格区分 interruptible。
            self.status = "waiting"

        return float(actual_work)
