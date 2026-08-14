"""
集中配置文件。

拆分目标：
1. 训练脚本只从这里读取环境参数和 PPO 参数；
2. 后续调实验时优先改这个文件，不再去环境层和训练层里翻大量代码；
3. 保持原 ultimate 版本的核心参数不变。
"""

from pathlib import Path


# All scripts write reports, CSVs, figures, and model artifacts under this root by default.
OUTPUT_ROOT = "report_outputs"


def resolve_output_path(name_or_path: str) -> Path:
    path = Path(name_or_path)
    if path.is_absolute():
        return path
    return Path(OUTPUT_ROOT) / path


ENV_CONFIG = {
    "horizon": 24,
    "base_load": 0.05,
    "max_task_load_per_server": 0.60,
    "Q0": 300.0,
    "num_tasks": 30,
    # Formal runs must remain non-oracle. "perfect" is debug/oracle-only.
    "task_forecast_mode": "noisy",
    "forecast_error_level": 0.20,
    # Independent forecast RNG seed = worker task seed + this offset.
    "task_forecast_seed_offset": 300000,
    "price_ref": 1.50,
    "lambda_ref": 2000.0,
    "queue_ref": 6000.0,
    "queue_capacity_ref": 6000.0,
    "cost_ref": 60.0,
    "carbon_ref": 15.0,
    # carbon_price is used only to report carbon_cost from grid carbon emissions.
    "carbon_price": 0.0,
    "delta_t_hours": 1.0,
    "peak_power_threshold_kW": 18.0,
    "peak_power_ref_kW": 10.0,
    "grid_power_limit_kW": 18.0,
    # Count/priority/lateness normalization; deliberately independent of workload scale.
    "sla_penalty_ref": 50.0,
    "bess_capacity_kWh": 100.0,
    "bess_soc_init": 0.50,
    "bess_soc_min": 0.10,
    "bess_soc_max": 0.90,
    "bess_soc_target": 0.50,
    "bess_soc_final_tolerance": 0.05,
    "bess_charge_power_max_kW": 20.0,
    "bess_discharge_power_max_kW": 20.0,
    "bess_charge_efficiency": 0.95,
    "bess_discharge_efficiency": 0.95,
    "bess_degradation_cost_per_kWh": 0.02,
    # None means derive once from max BESS power * degradation coefficient * timestep.
    "bess_degradation_cost_ref": None,
    # 计划负载预留损耗系数：未被实际使用的计划负载中，有多少比例计入实际功耗。
    # 用于模拟资源预留、空转和调度开销，避免全一策略无成本地长期满负载。
    "planned_load_reserve_alpha": 0.40,
}

IDC_SCALE_CONFIG = {
    # Formal 25 MW definition: facility P_IT + P_cooling + P_others at legal
    # compute action=1, planned total load=0.65, and 30 C high-temperature reference.
    "facility_rated_power_mw": 25.0,
    "enable_server_group_model": True,
    "server_group_size": 1841,
    "num_server_groups": 20,
    "task_workload_scale": 1841,
    # The current 2 MW / 10 MWh BESS remains unchanged in Phase 2.7.
    "bess_scale_factor": 100,
    "scale_bess_with_idc": True,
}

DATA_CONFIG = {
    "price_csv_path": None,
    "price_column": None,
    "carbon_csv_path": None,
    "carbon_column": None,
    "temperature_csv_path": None,
    "temperature_column": None,
    "pv_csv_path": None,
    "pv_column": None,
    "pv_unit": "kW",
    "pv_scale_factor": 1.0,
    "pv_capacity_kw": 500.0,
    "use_default_pv_curve": True,
    "allow_pv_export": False,
    "wt_csv_path": None,
    "wt_column": None,
}

REWARD_CONFIG = {
    "reward_done_weight": 5.0,
    "reward_cost_weight": 0.35,
    # Increase this weight to prefer lower-carbon grid energy without changing reward modes.
    "reward_carbon_weight": 0.30,
    "reward_sla_weight": 0.80,
    "reward_queue_weight": 0.8,
    "reward_queue_overflow_weight": 1.2,
    "reward_final_queue_weight": 3.0,
    "reward_deadline_weight": 1.2,
    "reward_unused_capacity_weight": 0.08,
    "reward_finished_task_weight": 1.5,
    "reward_priority_finish_weight": 0.6,
    "reward_urgent_backlog_weight": 0.8,
    "reward_waiting_weight": 0.25,
    "reward_peak_load_weight": 1.0,
    "reward_pause_weight": 0.15,
    "reward_resume_weight": 0.03,
    "reward_non_interruptible_weight": 0.8,
    "reward_load_smooth_weight": 0.05,
    "reward_action_smooth_weight": 0.03,
    "reward_bess_degradation_weight": 1.0,
    "reward_bess_invalid_action_weight": 0.2,
    "reward_soc_final_weight": 2.0,
    "reward_grid_peak_weight": 1.0,
}

GRID_CONFIG = {
    "enable_grid_coupling": True,
    "case_name": "ieee14",
    "idc_ieee_bus_number": 9,
    "opf_mode": "ac",
    "delta_p_mw": 0.1,
    "load_scale": 1.0,
    "enable_grid_reward": False,
    "use_mef": True,
    "enable_grid_obs": True,
    "grid_obs_dim": 8,
    "grid_lmp_ref": 100.0,
    "grid_mef_ref": 1000.0,
    "grid_voltage_ref": 0.10,
    "grid_line_loading_ref": 100.0,
    "grid_network_loss_ref": 20.0,
    "grid_security_penalty_ref": 10.0,
}

GRID_REWARD_CONFIG = {
    "enable_grid_reward": False,
    "grid_reward_mode": "none",
    "grid_lmp_cost_weight": 0.0,
    "grid_mef_carbon_weight": 0.0,
    "grid_safe_violation_weight": 0.0,
    "lmp_cost_ref": 100.0,
    "mef_carbon_ref": 100.0,
    "safe_violation_ref": 1.0,
    "clip_grid_reward_penalty": True,
    "grid_reward_penalty_clip": 10.0,
}

GRID_SCENARIO_CONFIG = {
    "enable_dynamic_grid_load": True,
    "grid_load_scale_path": "data/grid_scenarios/nems_singapore/processed/nems_24h_load_scale.csv",
    "grid_load_scale_column": "grid_load_scale",
    "grid_usep_column": "usep_sgd_per_mwh",
    "fallback_load_scale": 1.0,
}

GRID_CACHE_CONFIG = {
    "enable_grid_cache": True,
    "cache_opf": True,
    "cache_mef": True,
    "cache_load_bin_mw": 0.1,
    "cache_load_scale_bin": 0.005,
    "cache_max_size": 50000,
    "cache_clear_on_reset": False,
    "cache_scope": "per_worker",
    "cache_failed_results": False,
    "cache_verbose": False,
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
