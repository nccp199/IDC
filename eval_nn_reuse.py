"""
eval_nn_reuse.py

最近邻历史方案迁移评估脚本。

用途：
1. 不修改原有环境、PPO、GA、PSO、eval_base 等文件；
2. 使用历史 GA / PSO 搜索得到的动作方案库；
3. 对全新 seed 场景提取场景特征；
4. 用归一化加权最近邻规则，匹配最相近的历史场景；
5. 将该历史场景的 GA / PSO 最优动作方案直接套用到新场景；
6. 同时可评估 PPO 模型在新场景上的直接推理表现。

推荐运行：
    python .\eval_nn_reuse.py

显式指定：
    python .\eval_nn_reuse.py --history-start 3000 --history-n 30 --eval-start 5000 --eval-n 30 --out eval_nn_reuse_5000

前提：
    ga_out/ga_plan_seed3000.npy ... ga_plan_seed3029.npy
    pso_out/pso_plan_seed3000.npy ... pso_plan_seed3029.npy
    ppo_outputs_mid_balance_1m/models/ppo_idc_ultimate_final.zip
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np

from config_ultimate import DATA_CONFIG, ENV_CONFIG, REWARD_CONFIG, resolve_output_path
from data_loader import build_external_series_from_config
from IDCPriceEnv20D_ultimate import IDCPriceEnv20D


# ============================================================
# 1. 最近邻规则参数区
# ============================================================

DEFAULT_HISTORY_START_SEED = 3000
DEFAULT_HISTORY_N_SEEDS = 30
DEFAULT_EVAL_START_SEED = 5000
DEFAULT_EVAL_N_SEEDS = 30

# 启用标准化归一化：z = (x - mean) / (std + eps)
USE_STANDARD_NORMALIZATION = True
NORM_EPS = 1e-8
DISTANCE_MODE = "weighted_euclidean"

# 可选特征组：
# lambda_curve       24小时任务到达曲线，反映任务何时到达
# lambda_stats       任务到达统计量，反映总任务压力和峰值压力
# server_stats       服务器算力统计，反映可用算力与异构性
# task_stats         任务池统计，反映deadline、priority、可暂停/可并行比例
# price_curve        24小时电价曲线，当前固定，默认关闭；真实电价接入后可打开
# temperature_curve  24小时温度曲线，当前固定，默认关闭；真实温度接入后可打开
ENABLED_FEATURE_GROUPS = [
    "lambda_curve",
    "lambda_stats",
    "server_stats",
    "task_stats",
    # "price_curve",
    # "temperature_curve",
]

# 权重设置单独列出。权重会自动扩展到该组每一维特征。
FEATURE_GROUP_WEIGHTS = {
    "lambda_curve": 2.0,
    "lambda_stats": 1.5,
    "server_stats": 1.0,
    "task_stats": 1.5,
    "price_curve": 1.0,
    "temperature_curve": 0.5,
}

GA_PLAN_TEMPLATE = str(resolve_output_path("ga_out/ga_plan_seed{seed}.npy"))
PSO_PLAN_TEMPLATE = str(resolve_output_path("pso_out/pso_plan_seed{seed}.npy"))
DEFAULT_PPO_MODEL = str(resolve_output_path("ppo_outputs_mid_balance_1m/models/ppo_idc_ultimate_final.zip"))


# ============================================================
# 2. 环境创建与特征提取
# ============================================================

def make_env(seed: int) -> IDCPriceEnv20D:
    """使用统一 config 创建环境。"""
    env_kwargs = {
        **ENV_CONFIG,
        **REWARD_CONFIG,
        **build_external_series_from_config(DATA_CONFIG, ENV_CONFIG["horizon"]),
        "server_seed": seed,
        "task_seed": seed,
    }
    return IDCPriceEnv20D(**env_kwargs)


def _safe_mean(values: Sequence[float], default: float = 0.0) -> float:
    return float(np.mean(values)) if len(values) > 0 else float(default)


def _safe_std(values: Sequence[float], default: float = 0.0) -> float:
    return float(np.std(values)) if len(values) > 0 else float(default)


def extract_scene_features(seed: int) -> Tuple[np.ndarray, List[str], List[str]]:
    """
    提取一个场景的特征向量。

    返回：
        features: np.ndarray, shape=(m,)
        names: 每一维特征名
        groups: 每一维所属特征组，用于扩展权重
    """
    env = make_env(seed)
    obs, info = env.reset()

    features: List[float] = []
    names: List[str] = []
    groups: List[str] = []

    def add_group(group_name: str, values: Sequence[float], value_names: Sequence[str]) -> None:
        if group_name not in ENABLED_FEATURE_GROUPS:
            return
        if len(values) != len(value_names):
            raise ValueError(f"{group_name}: values 和 value_names 长度不一致。")
        features.extend(float(v) for v in values)
        names.extend(str(n) for n in value_names)
        groups.extend([group_name] * len(values))

    # 1. 24小时任务到达曲线 lambda_t
    lambda_t = np.asarray(env.lambda_t, dtype=np.float64)
    add_group(
        "lambda_curve",
        lambda_t.tolist(),
        [f"lambda_{t:02d}" for t in range(env.horizon)],
    )

    # 2. 任务到达统计量
    total_lambda = float(np.sum(lambda_t))
    mean_lambda = float(np.mean(lambda_t))
    std_lambda = float(np.std(lambda_t))
    max_lambda = float(np.max(lambda_t))
    peak_hour = float(np.argmax(lambda_t) / max(env.horizon - 1, 1))
    early_work = float(np.sum(lambda_t[0:7]))
    morning_work = float(np.sum(lambda_t[7:12]))
    afternoon_work = float(np.sum(lambda_t[12:18]))
    evening_work = float(np.sum(lambda_t[18:24]))
    nonzero_hours = int(np.sum(lambda_t > 1e-9))

    add_group(
        "lambda_stats",
        [
            total_lambda,
            mean_lambda,
            std_lambda,
            max_lambda,
            peak_hour,
            early_work,
            morning_work,
            afternoon_work,
            evening_work,
            float(nonzero_hours),
            float(env.initial_Q),
        ],
        [
            "lambda_total",
            "lambda_mean",
            "lambda_std",
            "lambda_max",
            "lambda_peak_hour_norm",
            "lambda_early_sum",
            "lambda_morning_sum",
            "lambda_afternoon_sum",
            "lambda_evening_sum",
            "lambda_nonzero_hours",
            "initial_Q",
        ],
    )

    # 3. 服务器算力统计
    c_server = np.asarray(env.model.C_server, dtype=np.float64)
    server_eff = np.asarray(env.model.server_compute_efficiency, dtype=np.float64)
    add_group(
        "server_stats",
        [
            float(env.model.C_IDC),
            float(np.mean(c_server)),
            float(np.std(c_server)),
            float(np.min(c_server)),
            float(np.max(c_server)),
            float(np.std(c_server) / max(np.mean(c_server), NORM_EPS)),
            float(np.mean(server_eff)),
            float(np.std(server_eff)),
        ],
        [
            "C_IDC",
            "C_server_mean",
            "C_server_std",
            "C_server_min",
            "C_server_max",
            "C_server_cv",
            "server_eff_mean",
            "server_eff_std",
        ],
    )

    # 4. 任务池统计
    arrived_tasks = [task for task in env.tasks if task.arrival_time < env.horizon]
    non_initial_tasks = [task for task in arrived_tasks if int(task.task_id) != 0]

    workloads = [float(task.workload) for task in non_initial_tasks]
    deadlines = [float(task.deadline) for task in non_initial_tasks]
    priorities = [float(task.priority) for task in non_initial_tasks]
    durations = [float(task.duration) for task in non_initial_tasks]
    arrivals = [float(task.arrival_time) for task in non_initial_tasks]
    deadline_slack_values = [max(float(task.deadline - task.duration), 0.0) for task in non_initial_tasks]

    interruptible_ratio = (
        float(np.mean([1.0 if bool(task.interruptible) else 0.0 for task in non_initial_tasks]))
        if len(non_initial_tasks) > 0 else 0.0
    )
    parallelizable_ratio = (
        float(np.mean([1.0 if bool(task.parallelizable) else 0.0 for task in non_initial_tasks]))
        if len(non_initial_tasks) > 0 else 0.0
    )
    urgent_ratio = (
        float(np.mean([1.0 if task.deadline <= 4 else 0.0 for task in non_initial_tasks]))
        if len(non_initial_tasks) > 0 else 0.0
    )

    add_group(
        "task_stats",
        [
            float(len(non_initial_tasks)),
            float(np.sum(workloads)) if workloads else 0.0,
            _safe_mean(workloads),
            _safe_std(workloads),
            _safe_mean(deadlines),
            _safe_std(deadlines),
            _safe_mean(priorities),
            _safe_std(priorities),
            _safe_mean(durations),
            _safe_std(durations),
            _safe_mean(arrivals),
            _safe_std(arrivals),
            _safe_mean(deadline_slack_values),
            _safe_std(deadline_slack_values),
            urgent_ratio,
            interruptible_ratio,
            parallelizable_ratio,
        ],
        [
            "task_count",
            "task_total_work",
            "task_work_mean",
            "task_work_std",
            "task_deadline_mean",
            "task_deadline_std",
            "task_priority_mean",
            "task_priority_std",
            "task_duration_mean",
            "task_duration_std",
            "task_arrival_mean",
            "task_arrival_std",
            "task_deadline_slack_mean",
            "task_deadline_slack_std",
            "task_urgent_ratio",
            "task_interruptible_ratio",
            "task_parallelizable_ratio",
        ],
    )

    # 5. 电价曲线，可选
    price_t = np.asarray(env.price_t, dtype=np.float64)
    add_group("price_curve", price_t.tolist(), [f"price_{t:02d}" for t in range(env.horizon)])

    # 6. 温度曲线，可选
    T_amb = np.asarray(env.T_amb, dtype=np.float64)
    add_group("temperature_curve", T_amb.tolist(), [f"T_amb_{t:02d}" for t in range(env.horizon)])

    return np.asarray(features, dtype=np.float64), names, groups


def build_weight_vector(groups: Sequence[str]) -> np.ndarray:
    """根据每一维所属特征组，把组权重扩展成逐维权重向量。"""
    return np.asarray([float(FEATURE_GROUP_WEIGHTS.get(group, 1.0)) for group in groups], dtype=np.float64)


def standardize_features(X_hist: np.ndarray, X_eval: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """使用历史场景均值和标准差，对历史和评估特征做同一套标准化。"""
    mu = np.mean(X_hist, axis=0)
    sigma = np.std(X_hist, axis=0)
    sigma_safe = np.where(sigma < NORM_EPS, 1.0, sigma)

    if USE_STANDARD_NORMALIZATION:
        X_hist_z = (X_hist - mu) / sigma_safe
        X_eval_z = (X_eval - mu) / sigma_safe
    else:
        X_hist_z = X_hist.copy()
        X_eval_z = X_eval.copy()

    return X_hist_z, X_eval_z, mu, sigma_safe


def weighted_euclidean_distance(x_new: np.ndarray, X_hist: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """
    加权欧氏距离：
        d_i = sqrt( sum_k w_k * (x_new_k - x_i_k)^2 / sum_k w_k )
    """
    diff = X_hist - x_new.reshape(1, -1)
    weighted_sq = (diff ** 2) * weights.reshape(1, -1)
    denom = max(float(np.sum(weights)), NORM_EPS)
    return np.sqrt(np.sum(weighted_sq, axis=1) / denom)


def nearest_neighbor_match(
    history_seeds: Sequence[int],
    X_hist_z: np.ndarray,
    x_eval_z: np.ndarray,
    weights: np.ndarray,
) -> Tuple[int, float, int]:
    """返回最近历史 seed、最近距离、历史下标。"""
    if DISTANCE_MODE != "weighted_euclidean":
        raise ValueError(f"当前只支持 weighted_euclidean，收到：{DISTANCE_MODE}")

    distances = weighted_euclidean_distance(x_eval_z, X_hist_z, weights)
    nearest_index = int(np.argmin(distances))
    nearest_seed = int(history_seeds[nearest_index])
    nearest_distance = float(distances[nearest_index])
    return nearest_seed, nearest_distance, nearest_index


# ============================================================
# 3. 方案评估
# ============================================================

def final_metrics_from_info(total_reward: float, info: Dict[str, Any]) -> Dict[str, float]:
    """从环境 info 中抽取最终指标。"""
    def sf(x: Any, default: float = np.nan) -> float:
        try:
            if x is None:
                return default
            return float(x)
        except Exception:
            return default

    return {
        "total_reward": float(total_reward),
        "fitness": float(total_reward),
        "completion_rate": sf(info.get("completion_rate")),
        "task_completion_rate": sf(info.get("task_completion_rate")),
        "total_completed_work": sf(info.get("total_completed_work")),
        "final_backlog_work": sf(info.get("final_backlog_work", info.get("Q"))),
        "deadline_miss_rate": sf(info.get("deadline_miss_rate")),
        "deadline_miss_count": sf(info.get("deadline_miss_count")),
        "avg_waiting_time": sf(info.get("avg_waiting_time")),
        "avg_turnaround_time": sf(info.get("avg_turnaround_time")),
        "total_cost": sf(info.get("total_cost")),
        "unit_task_cost": sf(info.get("unit_task_cost")),
        "total_energy_kWh": sf(info.get("total_energy_kWh")),
        "P_grid_kW": sf(info.get("P_grid_kW")),
        "grid_energy_kWh": sf(info.get("grid_energy_kWh")),
        "idc_energy_kWh": sf(info.get("idc_energy_kWh")),
        "total_grid_energy_kWh": sf(info.get("total_grid_energy_kWh")),
        "total_idc_energy_kWh": sf(info.get("total_idc_energy_kWh")),
        "energy_per_task": sf(info.get("energy_per_task")),
        "idc_energy_per_task": sf(info.get("idc_energy_per_task")),
        "total_carbon_emission": sf(info.get("total_carbon_emission")),
        "carbon_cost": sf(info.get("carbon_cost")),
        "total_carbon_cost": sf(info.get("total_carbon_cost")),
        "carbon_per_task": sf(info.get("carbon_per_task")),
        "episode_grid_peak_power_kW": sf(info.get("episode_grid_peak_power_kW")),
        "total_grid_peak_excess_kW_hour": sf(info.get("total_grid_peak_excess_kW_hour")),
        "bess_soc": sf(info.get("bess_soc")),
        "total_bess_charge_kWh": sf(info.get("total_bess_charge_kWh")),
        "total_bess_discharge_kWh": sf(info.get("total_bess_discharge_kWh")),
        "total_bess_degradation_cost": sf(info.get("total_bess_degradation_cost")),
        "finished_task_count": sf(info.get("finished_task_count")),
        "total_task_count": sf(info.get("total_task_count")),
        "total_pause_count": sf(info.get("total_pause_count")),
        "total_resume_count": sf(info.get("total_resume_count")),
        "total_non_interruptible_interruption_count": sf(info.get("total_non_interruptible_interruption_count")),
    }


def evaluate_action_plan(action_plan: np.ndarray, env_seed: int) -> Dict[str, float]:
    """把一个历史 24x22 动作方案直接套用到新 env_seed 场景上评估。"""
    env = make_env(env_seed)
    obs, reset_info = env.reset()

    action_plan = np.asarray(action_plan, dtype=np.float32)
    expected_shape = (env.horizon, env.action_dim)
    if action_plan.shape == (env.horizon, env.action_dim - 1):
        neutral_bess = np.full((env.horizon, 1), 0.5, dtype=np.float32)
        action_plan = np.concatenate([action_plan, neutral_bess], axis=1)
    if action_plan.shape != expected_shape:
        raise ValueError(f"action_plan shape 应为 {expected_shape}，实际 {action_plan.shape}")

    total_reward = 0.0
    info: Dict[str, Any] = {}

    for t in range(env.horizon):
        action = np.clip(action_plan[t], 0.0, 1.0).astype(np.float32)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += float(reward)
        if terminated or truncated:
            break

    return final_metrics_from_info(total_reward, info)


def evaluate_ppo_model(model: Any, env_seed: int) -> Dict[str, float]:
    """PPO 在新场景上直接推理评估。"""
    env = make_env(env_seed)
    obs, reset_info = env.reset()

    total_reward = 0.0
    info: Dict[str, Any] = {}

    for _ in range(env.horizon):
        action, _state = model.predict(obs, deterministic=True)
        action = np.asarray(action, dtype=np.float32).reshape(-1)
        if action.shape[0] == env.action_dim - 1:
            action = np.concatenate([action, np.array([0.5], dtype=np.float32)])
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += float(reward)
        if terminated or truncated:
            break

    return final_metrics_from_info(total_reward, info)


def load_plan(path_template: str, seed: int) -> np.ndarray:
    path = Path(path_template.format(seed=seed))
    if not path.exists():
        raise FileNotFoundError(
            f"找不到历史动作方案：{path}\n"
            f"请确认之前运行 GA/PSO 时使用了 --save-plan，并且 seed={seed} 的 plan 文件存在。"
        )
    return np.load(path)


# ============================================================
# 4. CSV 输出与汇总
# ============================================================

SUMMARY_KEYS = [
    "total_reward",
    "completion_rate",
    "task_completion_rate",
    "total_completed_work",
    "final_backlog_work",
    "deadline_miss_rate",
    "avg_waiting_time",
    "avg_turnaround_time",
    "total_cost",
    "unit_task_cost",
    "total_energy_kWh",
    "P_grid_kW",
    "grid_energy_kWh",
    "idc_energy_kWh",
    "total_grid_energy_kWh",
    "total_idc_energy_kWh",
    "energy_per_task",
    "idc_energy_per_task",
    "total_carbon_emission",
    "carbon_cost",
    "total_carbon_cost",
    "carbon_per_task",
    "episode_grid_peak_power_kW",
    "total_grid_peak_excess_kW_hour",
    "bess_soc",
    "total_bess_charge_kWh",
    "total_bess_discharge_kWh",
    "total_bess_degradation_cost",
]


def safe_float(value: Any, default: float = np.nan) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except Exception:
        return default


def ordered_fieldnames(rows: List[Dict[str, Any]]) -> List[str]:
    preferred = [
        "algorithm", "eval_seed", "matched_seed", "matched_distance",
        "source_plan", "ppo_model_path", "fitness", "total_reward",
        "completion_rate", "task_completion_rate", "total_completed_work",
        "final_backlog_work", "deadline_miss_rate", "deadline_miss_count",
        "avg_waiting_time", "avg_turnaround_time", "total_cost", "unit_task_cost",
        "total_energy_kWh", "P_grid_kW", "grid_energy_kWh", "idc_energy_kWh",
        "total_grid_energy_kWh", "total_idc_energy_kWh",
        "energy_per_task", "idc_energy_per_task", "total_carbon_emission",
        "carbon_cost", "total_carbon_cost", "carbon_per_task",
        "episode_grid_peak_power_kW", "total_grid_peak_excess_kW_hour",
        "bess_soc", "total_bess_charge_kWh",
        "total_bess_discharge_kWh", "total_bess_degradation_cost",
        "finished_task_count", "total_task_count",
        "total_pause_count", "total_resume_count", "total_non_interruptible_interruption_count",
    ]
    keys = set()
    for row in rows:
        keys.update(row.keys())
    return [k for k in preferred if k in keys] + sorted(k for k in keys if k not in preferred)


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=ordered_fieldnames(rows), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def build_summary(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    algorithms = sorted(set(str(r["algorithm"]) for r in rows))
    summary: List[Dict[str, Any]] = []

    for alg in algorithms:
        alg_rows = [r for r in rows if str(r["algorithm"]) == alg]
        out: Dict[str, Any] = {
            "algorithm": alg,
            "n_rows": len(alg_rows),
            "n_seeds": len(set(int(r["eval_seed"]) for r in alg_rows)),
        }
        for key in SUMMARY_KEYS:
            values = np.asarray([safe_float(r.get(key)) for r in alg_rows], dtype=np.float64)
            values = values[~np.isnan(values)]
            if len(values) == 0:
                out[f"{key}_mean"] = np.nan
                out[f"{key}_std"] = np.nan
            else:
                out[f"{key}_mean"] = float(np.mean(values))
                out[f"{key}_std"] = float(np.std(values))
        summary.append(out)

    def sort_key(row: Dict[str, Any]) -> Tuple[float, float, float]:
        completion = safe_float(row.get("completion_rate_mean"), default=-np.inf)
        backlog = safe_float(row.get("final_backlog_work_mean"), default=np.inf)
        unit_cost = safe_float(row.get("unit_task_cost_mean"), default=np.inf)
        return (-completion, backlog, unit_cost)

    summary.sort(key=sort_key)
    return summary


def print_summary(summary: List[Dict[str, Any]]) -> None:
    print("\n=== Nearest-neighbor reuse summary ===")
    header = (
        f"{'algorithm':<14} {'n':>4} "
        f"{'comp':>9} {'task_comp':>10} {'backlog':>12} {'miss':>9} "
        f"{'unit_cost':>10} {'cost':>10} {'reward':>10}"
    )
    print(header)
    print("-" * len(header))

    for row in summary:
        print(
            f"{str(row['algorithm']):<14} {int(row['n_seeds']):>4} "
            f"{safe_float(row.get('completion_rate_mean')):>9.4f} "
            f"{safe_float(row.get('task_completion_rate_mean')):>10.4f} "
            f"{safe_float(row.get('final_backlog_work_mean')):>12.2f} "
            f"{safe_float(row.get('deadline_miss_rate_mean')):>9.4f} "
            f"{safe_float(row.get('unit_task_cost_mean')):>10.5f} "
            f"{safe_float(row.get('total_cost_mean')):>10.2f} "
            f"{safe_float(row.get('total_reward_mean')):>10.4f}"
        )


# ============================================================
# 5. 主流程
# ============================================================

def parse_seed_range(start: int, n: int) -> List[int]:
    return list(range(int(start), int(start) + int(n)))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--history-start", type=int, default=DEFAULT_HISTORY_START_SEED)
    parser.add_argument("--history-n", type=int, default=DEFAULT_HISTORY_N_SEEDS)
    parser.add_argument("--eval-start", type=int, default=DEFAULT_EVAL_START_SEED)
    parser.add_argument("--eval-n", type=int, default=DEFAULT_EVAL_N_SEEDS)
    parser.add_argument("--out", type=str, default="eval_nn_reuse")
    parser.add_argument("--ppo-model", type=str, default=DEFAULT_PPO_MODEL)
    parser.add_argument("--no-ppo", action="store_true")
    parser.add_argument("--no-ga", action="store_true")
    parser.add_argument("--no-pso", action="store_true")
    args = parser.parse_args()

    history_seeds = parse_seed_range(args.history_start, args.history_n)
    eval_seeds = parse_seed_range(args.eval_start, args.eval_n)
    out_dir = resolve_output_path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=== Nearest-neighbor historical-plan reuse evaluation ===")
    print(f"history seeds: {history_seeds}")
    print(f"eval seeds:    {eval_seeds}")
    print(f"enabled feature groups: {ENABLED_FEATURE_GROUPS}")
    print(f"group weights: {FEATURE_GROUP_WEIGHTS}")
    print(f"normalization: {USE_STANDARD_NORMALIZATION}")
    print(f"distance mode: {DISTANCE_MODE}")
    print(f"output dir:    {out_dir.resolve()}")

    # 1. 提取历史特征
    X_hist_list = []
    feature_names_ref: List[str] | None = None
    feature_groups_ref: List[str] | None = None

    print("\n>>> Extracting history features...")
    for seed in history_seeds:
        x, names, groups = extract_scene_features(seed)
        if feature_names_ref is None:
            feature_names_ref = names
            feature_groups_ref = groups
        elif names != feature_names_ref:
            raise RuntimeError("不同历史 seed 提取到的 feature_names 不一致。")
        X_hist_list.append(x)

    X_hist = np.vstack(X_hist_list)

    # 2. 提取新场景特征
    print(">>> Extracting eval features...")
    X_eval_list = []
    for seed in eval_seeds:
        x, names, groups = extract_scene_features(seed)
        if names != feature_names_ref:
            raise RuntimeError("eval seed 提取到的 feature_names 与 history 不一致。")
        X_eval_list.append(x)

    X_eval = np.vstack(X_eval_list)

    assert feature_names_ref is not None
    assert feature_groups_ref is not None

    weights = build_weight_vector(feature_groups_ref)
    X_hist_z, X_eval_z, mu, sigma = standardize_features(X_hist, X_eval)

    # 保存特征配置，方便写报告说明。
    feature_rows = []
    for idx, (name, group, w, mean, std) in enumerate(zip(feature_names_ref, feature_groups_ref, weights, mu, sigma)):
        feature_rows.append({
            "index": idx,
            "feature_name": name,
            "feature_group": group,
            "weight": float(w),
            "normalization_mean_from_history": float(mean),
            "normalization_std_from_history": float(std),
        })
    write_csv(out_dir / "feature_config.csv", feature_rows)

    # 3. 加载 PPO 模型
    ppo_model = None
    ppo_model_path = Path(args.ppo_model)
    if not args.no_ppo:
        if not ppo_model_path.exists():
            print(f"\n>>> PPO model not found: {ppo_model_path}; skipped PPO.")
        else:
            print(f"\n>>> Loading PPO model: {ppo_model_path}")
            from stable_baselines3 import PPO
            ppo_model = PPO.load(str(ppo_model_path))

    # 4. 最近邻匹配 + 评估
    all_rows: List[Dict[str, Any]] = []
    match_rows: List[Dict[str, Any]] = []

    print("\n>>> Evaluating reuse plans and PPO on eval seeds...")
    for idx, eval_seed in enumerate(eval_seeds):
        nearest_seed, nearest_distance, nearest_idx = nearest_neighbor_match(
            history_seeds=history_seeds,
            X_hist_z=X_hist_z,
            x_eval_z=X_eval_z[idx],
            weights=weights,
        )

        match_rows.append({
            "eval_seed": int(eval_seed),
            "matched_seed": int(nearest_seed),
            "matched_distance": float(nearest_distance),
        })

        print(f"eval_seed={eval_seed} matched_seed={nearest_seed} distance={nearest_distance:.4f}")

        if not args.no_ga:
            ga_plan_path = GA_PLAN_TEMPLATE.format(seed=nearest_seed)
            ga_plan = load_plan(GA_PLAN_TEMPLATE, nearest_seed)
            metrics = evaluate_action_plan(ga_plan, env_seed=eval_seed)
            row: Dict[str, Any] = {
                "algorithm": "GA_REUSE_NN",
                "eval_seed": int(eval_seed),
                "matched_seed": int(nearest_seed),
                "matched_distance": float(nearest_distance),
                "source_plan": ga_plan_path,
            }
            row.update(metrics)
            all_rows.append(row)

        if not args.no_pso:
            pso_plan_path = PSO_PLAN_TEMPLATE.format(seed=nearest_seed)
            pso_plan = load_plan(PSO_PLAN_TEMPLATE, nearest_seed)
            metrics = evaluate_action_plan(pso_plan, env_seed=eval_seed)
            row = {
                "algorithm": "PSO_REUSE_NN",
                "eval_seed": int(eval_seed),
                "matched_seed": int(nearest_seed),
                "matched_distance": float(nearest_distance),
                "source_plan": pso_plan_path,
            }
            row.update(metrics)
            all_rows.append(row)

        if ppo_model is not None:
            metrics = evaluate_ppo_model(ppo_model, env_seed=eval_seed)
            row = {
                "algorithm": "PPO",
                "eval_seed": int(eval_seed),
                "matched_seed": "",
                "matched_distance": "",
                "ppo_model_path": str(ppo_model_path),
            }
            row.update(metrics)
            all_rows.append(row)

    # 5. 输出结果
    write_csv(out_dir / "nearest_matches.csv", match_rows)
    write_csv(out_dir / "all_results.csv", all_rows)
    summary = build_summary(all_rows)
    write_csv(out_dir / "summary.csv", summary)
    print_summary(summary)

    print(f"\nSaved nearest matches: {out_dir / 'nearest_matches.csv'}")
    print(f"Saved all results:     {out_dir / 'all_results.csv'}")
    print(f"Saved summary:         {out_dir / 'summary.csv'}")
    print(f"Saved feature config:  {out_dir / 'feature_config.csv'}")


if __name__ == "__main__":
    main()
