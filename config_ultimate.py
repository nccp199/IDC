"""
集中配置文件。

拆分目标：
1. 训练脚本只从这里读取环境参数和 PPO 参数；
2. 后续调实验时优先改这个文件，不再去环境层和训练层里翻大量代码；
3. 保持原 ultimate 版本的核心参数不变。
"""

ENV_CONFIG = {
    "horizon": 24,
    "base_load": 0.05,
    "max_task_load_per_server": 0.80,
    "Q0": 150.0,
    "num_tasks": 22,
    "price_ref": 1.50,
    "lambda_ref": 2000.0,
    "queue_ref": 5000.0,
    "cost_ref": 60.0,
    # 计划负载预留损耗系数：未被实际使用的计划负载中，有多少比例计入实际功耗。
    # 用于模拟资源预留、空转和调度开销，避免全一策略无成本地长期满负载。
    "planned_load_reserve_alpha": 0.40,
}

REWARD_CONFIG = {
    "reward_done_weight": 4.5,
    "reward_cost_weight": 0.7,
    "reward_queue_weight": 0.7,
    "reward_final_queue_weight": 2.4,
    "reward_deadline_weight": 0.9,
    "reward_unused_capacity_weight": 0.05,
    "reward_finished_task_weight": 1.2,
    "reward_priority_finish_weight": 0.5,
    "reward_urgent_backlog_weight": 0.7,
    "reward_waiting_weight": 0.20,
    "reward_peak_load_weight": 1.2,
    "reward_pause_weight": 0.15,
    "reward_resume_weight": 0.03,
    "reward_non_interruptible_weight": 0.8,
}

PPO_CONFIG = {
    "learning_rate": 3e-4,
    "n_steps": 768,
    "batch_size": 256,
    "n_epochs": 10,
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "clip_range": 0.2,
    "ent_coef": 0.003,
    "vf_coef": 0.5,
    "max_grad_norm": 0.5,
    "policy_kwargs": {
        "net_arch": {
            "pi": [256, 256],
            "vf": [256, 256],
        }
    },
    "verbose": 1,
    "device": "auto",
}

DEFAULT_EVAL_SEED = 2026
