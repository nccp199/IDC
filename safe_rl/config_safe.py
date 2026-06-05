"""Configuration for the first Safe PPO / Lagrangian PPO stage."""

from __future__ import annotations


SAFE_COST_CONFIG = {
    "V_soft_min": 0.98,
    "V_hard_min": 0.95,
    "line_soft_limit": 80.0,
    "line_hard_limit": 100.0,
    "trafo_soft_limit": 80.0,
    "trafo_hard_limit": 100.0,
    "lmp_threshold_mode": "disabled_first_version",
    "lmp_threshold": None,
    "lmp_ref": None,
    "mef_threshold_mode": "disabled_first_version",
    "mef_threshold": None,
    "mef_ref": None,
    "cost_clip_max": 1.0,
}


LAGRANGIAN_CONFIG = {
    "enabled_constraints": ["opf", "voltage"],
    "cost_limit_opf": 0.005,
    "cost_limit_voltage": 0.02,
    "lambda_init_opf": 0.0,
    "lambda_init_voltage": 0.0,
    "lambda_lr_opf": 0.01,
    "lambda_lr_voltage": 0.005,
    "lambda_max_opf": 10.0,
    "lambda_max_voltage": 5.0,
    "ema_beta": 0.95,
    "update_interval_episodes": 10,
    "update_tolerance": 0.002,
    "warmup_episodes": 10,
}


SAFE_TRAIN_CONFIG = {
    "enable_safe_reward": True,
    "report_only": False,
    "terminate_on_opf_failure": True,
    "global_safe_penalty_scale": 1.0,
}
