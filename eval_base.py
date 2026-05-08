"""
Unified baseline evaluation for IDCPriceEnv20D_ultimate.

This script evaluates simple policies and optionally PPO, then merges existing
GA/PSO CSV results into one final comparison table.

Run examples:
    python eval_base.py --quick
    python eval_base.py --start-seed 3000 --n-seeds 30
    python eval_base.py --start-seed 3000 --n-seeds 30 --no-ppo
    python eval_base.py --ppo-model ppo_outputs_ultimate_500k/best_model/best_model.zip

Put this file in the same folder as:
    IDCPriceEnv20D_ultimate.py
    IDCEnergyTaskModel_stage1_servercapacity.py
    ga_out/ga_30_30_results.csv or ga_out/ga_results.csv
    pso_out/pso_30_30_results.csv or pso_out/pso_results.csv
    ppo_outputs_ultimate/... if you want to evaluate PPO
"""

from __future__ import annotations

import argparse
import csv
import math
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from IDCPriceEnv20D_ultimate import IDCPriceEnv20D
from config_ultimate import ENV_CONFIG, REWARD_CONFIG


# =========================
# Shared environment config
# =========================

def make_env(seed: int) -> IDCPriceEnv20D:
    """
    Create evaluation environment from config_ultimate.py.

    The seed fixes both server parameters and task generation,
    so PPO / GA / PSO / rule baselines are compared on the same scenarios.
    """
    env_kwargs = {
        **ENV_CONFIG,
        **REWARD_CONFIG,
        "server_seed": seed,
        "task_seed": seed,
    }
    return IDCPriceEnv20D(**env_kwargs)


# =========================
# Metrics utilities
# =========================

METRIC_KEYS = [
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
    "energy_per_task",
    "finished_task_count",
    "total_task_count",
    "deadline_miss_count",
    "total_pause_count",
    "total_resume_count",
    "total_non_interruptible_interruption_count",
]

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
    "energy_per_task",
]

PRIMARY_PRINT_KEYS = [
    "completion_rate",
    "task_completion_rate",
    "final_backlog_work",
    "deadline_miss_rate",
    "unit_task_cost",
    "total_cost",
    "total_reward",
]


def safe_float(value: Any, default: float = np.nan) -> float:
    if value is None:
        return default
    try:
        if isinstance(value, str):
            text = value.strip()
            if not text:
                return default
            return float(text)
        return float(value)
    except Exception:
        return default


def final_metrics_from_info(total_reward: float, info: Dict[str, Any]) -> Dict[str, float]:
    return {
        "total_reward": float(total_reward),
        "fitness": float(total_reward),
        "total_energy_kWh": safe_float(info.get("total_energy_kWh")),
        "total_cost": safe_float(info.get("total_cost")),
        "total_completed_work": safe_float(info.get("total_completed_work")),
        "completion_rate": safe_float(info.get("completion_rate")),
        "unit_task_cost": safe_float(info.get("unit_task_cost")),
        "energy_per_task": safe_float(info.get("energy_per_task")),
        "final_backlog_work": safe_float(info.get("final_backlog_work", info.get("Q"))),
        "task_completion_rate": safe_float(info.get("task_completion_rate")),
        "finished_task_count": safe_float(info.get("finished_task_count")),
        "total_task_count": safe_float(info.get("total_task_count")),
        "deadline_miss_rate": safe_float(info.get("deadline_miss_rate")),
        "deadline_miss_count": safe_float(info.get("deadline_miss_count")),
        "avg_waiting_time": safe_float(info.get("avg_waiting_time")),
        "avg_turnaround_time": safe_float(info.get("avg_turnaround_time")),
        "total_pause_count": safe_float(info.get("total_pause_count")),
        "total_resume_count": safe_float(info.get("total_resume_count")),
        "total_non_interruptible_interruption_count": safe_float(
            info.get("total_non_interruptible_interruption_count")
        ),
    }


# =========================
# Basic policy evaluation
# =========================

def action_zero(env: IDCPriceEnv20D, rng: Optional[np.random.Generator] = None) -> np.ndarray:
    return np.zeros(env.action_dim, dtype=np.float32)


def action_one(env: IDCPriceEnv20D, rng: Optional[np.random.Generator] = None) -> np.ndarray:
    return np.ones(env.action_dim, dtype=np.float32)


def action_random(env: IDCPriceEnv20D, rng: Optional[np.random.Generator] = None) -> np.ndarray:
    if rng is None:
        rng = np.random.default_rng()
    return rng.uniform(0.0, 1.0, size=env.action_dim).astype(np.float32)


def action_rule(env: IDCPriceEnv20D, rng: Optional[np.random.Generator] = None) -> np.ndarray:
    """
    Price-aware rule policy:
    - valley price: high server action
    - flat price: medium server action
    - peak price: low server action
    """
    t = int(env.current_step)
    price = np.asarray(env.price_t, dtype=np.float64)
    price_now = float(price[t])
    low = float(np.min(price))
    high = float(np.max(price))

    if np.isclose(price_now, low):
        server_level = 0.95
    elif np.isclose(price_now, high):
        server_level = 0.25
    else:
        server_level = 0.55

    action = np.zeros(env.action_dim, dtype=np.float32)
    action[: env.model.N] = server_level
    action[env.model.N] = 0.85      # urgent preference
    action[env.model.N + 1] = 0.80  # continuity preference
    return np.clip(action, 0.0, 1.0).astype(np.float32)


POLICY_FUNCS = {
    "ZERO": action_zero,
    "ONE": action_one,
    "RANDOM": action_random,
    "RULE": action_rule,
}


def evaluate_basic_policy(
    algorithm: str,
    env_seed: int,
    rng_seed: Optional[int] = None,
    random_run: int = 0,
) -> Dict[str, Any]:
    if algorithm not in POLICY_FUNCS:
        raise ValueError(f"Unknown basic policy: {algorithm}")

    env = make_env(env_seed)
    obs, reset_info = env.reset()
    rng = np.random.default_rng(rng_seed) if rng_seed is not None else None

    total_reward = 0.0
    info: Dict[str, Any] = {}

    for _ in range(env.horizon):
        action = POLICY_FUNCS[algorithm](env, rng)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += float(reward)
        if terminated or truncated:
            break

    row: Dict[str, Any] = {
        "algorithm": algorithm,
        "seed": int(env_seed),
        "source": "eval_policy",
        "run_idx": int(random_run),
    }
    row.update(final_metrics_from_info(total_reward, info))
    return row


# =========================
# PPO evaluation
# =========================

def resolve_ppo_model_path(model_arg: str) -> Optional[Path]:
    if model_arg.lower() in {"", "none", "skip"}:
        return None

    if model_arg.lower() != "auto":
        path = Path(model_arg)
        if path.exists():
            return path
        if path.with_suffix(".zip").exists():
            return path.with_suffix(".zip")
        return None

    candidates = [
        Path("ppo_outputs_ultimate/best_model/best_model.zip"),
        Path("ppo_outputs_ultimate/models/ppo_idc_ultimate_final.zip"),
        Path("ppo_outputs_ultimate/best_model.zip"),
        Path("best_model.zip"),
        Path("ppo_idc_ultimate_final.zip"),
    ]
    for path in candidates:
        if path.exists():
            return path
    return None


def evaluate_ppo(model: Any, env_seed: int, hourly_rows=None) -> Dict[str, Any]:
    env = make_env(env_seed)
    obs, reset_info = env.reset()

    total_reward = 0.0
    info: Dict[str, Any] = {}

    for step in range(env.horizon):
        action, _state = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += float(reward)

        if hourly_rows is not None:
            hourly_rows.append({
                "algorithm": "PPO",
                "seed": int(env_seed),
                "step": int(step),
                "hour": int(info.get("hour", step)),
                "price": float(info.get("price", 0.0)),
                "action_mean": float(info.get("action_mean", np.mean(action))),
                "actual_total_load_mean": float(info.get("actual_total_load_mean", np.nan)),
                "planned_task_load_mean": float(info.get("planned_task_load_mean", np.nan)),
                "planned_capacity": float(info.get("planned_capacity", np.nan)),
                "completed_work": float(info.get("completed_work", np.nan)),
                "unused_capacity": float(info.get("unused_capacity", np.nan)),
                "hourly_cost": float(info.get("hourly_cost", info.get("cost", np.nan))),
                "backlog_work": float(info.get("Q", info.get("backlog_work", np.nan))),
                "reward": float(reward),
            })

        if terminated or truncated:
            break

    row: Dict[str, Any] = {
        "algorithm": "PPO",
        "seed": int(env_seed),
        "source": "eval_ppo",
        "run_idx": 0,
    }
    row.update(final_metrics_from_info(total_reward, info))
    return row


# =========================
# GA/PSO CSV importing
# =========================

def first_existing_path(paths: Sequence[str]) -> Optional[Path]:
    for p in paths:
        path = Path(p)
        if path.exists():
            return path
    return None


def import_search_csv(
    csv_path: Path,
    algorithm_name: str,
    wanted_seeds: Optional[set[int]] = None,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with csv_path.open("r", newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for raw in reader:
            seed_value = raw.get("seed", raw.get("env_seed", ""))
            seed = int(round(safe_float(seed_value, default=np.nan))) if str(seed_value).strip() else -1
            if wanted_seeds is not None and seed not in wanted_seeds:
                continue

            row: Dict[str, Any] = {
                "algorithm": algorithm_name,
                "seed": seed,
                "source": str(csv_path),
                "run_idx": 0,
            }
            for key in ["fitness"] + METRIC_KEYS:
                if key in raw:
                    row[key] = safe_float(raw.get(key))
                else:
                    row[key] = np.nan

            # GA/PSO CSVs use fitness=total_reward by default, but keep both fields robust.
            if math.isnan(safe_float(row.get("total_reward"))) and not math.isnan(safe_float(row.get("fitness"))):
                row["total_reward"] = row["fitness"]
            if math.isnan(safe_float(row.get("fitness"))) and not math.isnan(safe_float(row.get("total_reward"))):
                row["fitness"] = row["total_reward"]

            for extra in ["eval_count", "env_seed", "algo_seed"]:
                if extra in raw:
                    row[extra] = safe_float(raw.get(extra))

            rows.append(row)
    return rows


# =========================
# CSV writing and summary
# =========================

def ordered_fieldnames(rows: List[Dict[str, Any]]) -> List[str]:
    preferred = [
        "algorithm",
        "seed",
        "source",
        "run_idx",
        "fitness",
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
        "energy_per_task",
        "finished_task_count",
        "total_task_count",
        "deadline_miss_count",
        "total_pause_count",
        "total_resume_count",
        "total_non_interruptible_interruption_count",
        "eval_count",
        "env_seed",
        "algo_seed",
    ]
    keys = set()
    for row in rows:
        keys.update(row.keys())
    return [k for k in preferred if k in keys] + sorted(k for k in keys if k not in preferred)


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fieldnames = ordered_fieldnames(rows)
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def save_hourly_csv(rows, out_path):
    if not rows:
        return

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = list(rows[0].keys())
    with out_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def save_hourly_mean_csv(rows, out_path):
    if not rows:
        return

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    keys = [
        "price",
        "action_mean",
        "actual_total_load_mean",
        "planned_task_load_mean",
        "planned_capacity",
        "completed_work",
        "unused_capacity",
        "hourly_cost",
        "backlog_work",
        "reward",
    ]

    hours = sorted(set(int(r["hour"]) for r in rows))
    mean_rows = []

    for h in hours:
        h_rows = [r for r in rows if int(r["hour"]) == h]
        out = {
            "algorithm": "PPO",
            "hour": h,
            "n": len(h_rows),
        }

        for key in keys:
            vals = []
            for r in h_rows:
                v = r.get(key)
                try:
                    vals.append(float(v))
                except Exception:
                    pass

            out[f"{key}_mean"] = sum(vals) / len(vals) if vals else ""

        mean_rows.append(out)

    fieldnames = list(mean_rows[0].keys())
    with out_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(mean_rows)


def build_summary(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    algorithms = sorted(set(str(row.get("algorithm", "")) for row in rows))
    summary: List[Dict[str, Any]] = []

    for alg in algorithms:
        alg_rows = [row for row in rows if str(row.get("algorithm", "")) == alg]
        out: Dict[str, Any] = {
            "algorithm": alg,
            "n_rows": len(alg_rows),
            "n_seeds": len(set(int(row.get("seed", -1)) for row in alg_rows)),
        }
        for key in SUMMARY_KEYS:
            values = np.array([safe_float(row.get(key)) for row in alg_rows], dtype=np.float64)
            values = values[~np.isnan(values)]
            if len(values) == 0:
                out[f"{key}_mean"] = np.nan
                out[f"{key}_std"] = np.nan
            else:
                out[f"{key}_mean"] = float(np.mean(values))
                out[f"{key}_std"] = float(np.std(values))
        summary.append(out)

    # Sort by core logic: higher completion, lower backlog, lower unit cost.
    def sort_key(row: Dict[str, Any]) -> Tuple[float, float, float]:
        completion = safe_float(row.get("completion_rate_mean"), default=-np.inf)
        backlog = safe_float(row.get("final_backlog_work_mean"), default=np.inf)
        unit_cost = safe_float(row.get("unit_task_cost_mean"), default=np.inf)
        return (-completion, backlog, unit_cost)

    summary.sort(key=sort_key)
    return summary


def print_summary(summary: List[Dict[str, Any]]) -> None:
    if not summary:
        print("No summary rows.")
        return

    print("\n=== Unified summary, sorted by completion/backlog/unit cost ===")
    header = (
        f"{'algorithm':<10} {'n':>4} "
        f"{'comp':>9} {'task_comp':>10} {'backlog':>12} {'miss':>9} "
        f"{'unit_cost':>10} {'cost':>10} {'reward':>10}"
    )
    print(header)
    print("-" * len(header))

    for row in summary:
        print(
            f"{str(row['algorithm']):<10} {int(row['n_seeds']):>4} "
            f"{safe_float(row.get('completion_rate_mean')):>9.4f} "
            f"{safe_float(row.get('task_completion_rate_mean')):>10.4f} "
            f"{safe_float(row.get('final_backlog_work_mean')):>12.2f} "
            f"{safe_float(row.get('deadline_miss_rate_mean')):>9.4f} "
            f"{safe_float(row.get('unit_task_cost_mean')):>10.5f} "
            f"{safe_float(row.get('total_cost_mean')):>10.2f} "
            f"{safe_float(row.get('total_reward_mean')):>10.4f}"
        )


# =========================
# Main
# =========================

def parse_seeds(args: argparse.Namespace) -> List[int]:
    if args.seeds.strip():
        return [int(x.strip()) for x in args.seeds.split(",") if x.strip()]
    return list(range(args.start_seed, args.start_seed + args.n_seeds))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="Fast smoke test: seed=3000 only, no PPO unless model exists.")
    parser.add_argument("--start-seed", type=int, default=3000)
    parser.add_argument("--n-seeds", type=int, default=30)
    parser.add_argument("--seeds", type=str, default="", help="Comma-separated seeds, e.g. 3000,3001,3002.")
    parser.add_argument("--out", type=str, default="eval_out", help="Output directory.")

    parser.add_argument("--random-runs", type=int, default=1, help="Random policy repetitions per seed.")
    parser.add_argument("--no-basic", action="store_true", help="Skip ZERO/ONE/RANDOM/RULE evaluation.")
    parser.add_argument("--no-ga", action="store_true", help="Skip importing GA CSV.")
    parser.add_argument("--no-pso", action="store_true", help="Skip importing PSO CSV.")
    parser.add_argument("--no-ppo", action="store_true", help="Skip PPO evaluation.")

    parser.add_argument("--ga-csv", type=str, default="auto", help="Path to GA CSV, or auto.")
    parser.add_argument("--pso-csv", type=str, default="auto", help="Path to PSO CSV, or auto.")
    parser.add_argument("--ppo-model", type=str, default="auto", help="PPO model .zip path, auto, or skip.")
    args = parser.parse_args()

    if args.quick:
        args.n_seeds = 1
        if not args.seeds.strip():
            args.start_seed = 3000

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    seeds = parse_seeds(args)
    wanted_seeds = set(seeds)

    print("=== Unified baseline evaluation ===")
    print(f"seeds: {seeds}")
    print(f"output dir: {out_dir.resolve()}")

    all_rows: List[Dict[str, Any]] = []
    start_time = time.time()

    # 1. Basic policies
    if not args.no_basic:
        print("\n>>> Evaluating basic policies: ZERO, ONE, RANDOM, RULE")
        for seed in seeds:
            for alg in ["ZERO", "ONE", "RULE"]:
                row = evaluate_basic_policy(alg, seed)
                all_rows.append(row)
                print(
                    f"{alg:<6} seed={seed} comp={row['completion_rate']:.3f} "
                    f"cost={row['total_cost']:.2f} unit={row['unit_task_cost']:.4f} "
                    f"backlog={row['final_backlog_work']:.2f} reward={row['total_reward']:.4f}"
                )

            for run_idx in range(max(args.random_runs, 1)):
                rng_seed = seed + 30000 + 1000 * run_idx
                row = evaluate_basic_policy("RANDOM", seed, rng_seed=rng_seed, random_run=run_idx)
                all_rows.append(row)
                print(
                    f"RANDOM seed={seed} run={run_idx} comp={row['completion_rate']:.3f} "
                    f"cost={row['total_cost']:.2f} unit={row['unit_task_cost']:.4f} "
                    f"backlog={row['final_backlog_work']:.2f} reward={row['total_reward']:.4f}"
                )

    # 2. Import GA CSV
    if not args.no_ga:
        if args.ga_csv.lower() == "auto":
            ga_path = first_existing_path([
                "ga_out/ga_stress_30_30_results.csv",
                "ga_out/ga_results.csv",
            ])
        else:
            ga_path = Path(args.ga_csv)
        if ga_path is not None and ga_path.exists():
            ga_rows = import_search_csv(ga_path, "GA", wanted_seeds=wanted_seeds)
            all_rows.extend(ga_rows)
            print(f"\n>>> Imported GA rows: {len(ga_rows)} from {ga_path}")
        else:
            print("\n>>> GA CSV not found; skipped GA import.")

    # 3. Import PSO CSV
    if not args.no_pso:
        if args.pso_csv.lower() == "auto":
            pso_path = first_existing_path([
                "pso_out/pso_stress_30_30_results.csv",
                "pso_out/pso_results.csv",
            ])
        else:
            pso_path = Path(args.pso_csv)
        if pso_path is not None and pso_path.exists():
            pso_rows = import_search_csv(pso_path, "PSO", wanted_seeds=wanted_seeds)
            all_rows.extend(pso_rows)
            print(f"\n>>> Imported PSO rows: {len(pso_rows)} from {pso_path}")
        else:
            print("\n>>> PSO CSV not found; skipped PSO import.")

    # 4. PPO evaluation
    if not args.no_ppo:
        model_path = resolve_ppo_model_path(args.ppo_model)
        if model_path is None:
            print("\n>>> PPO model not found; skipped PPO evaluation.")
            print("    Use --ppo-model your_model.zip, or add --no-ppo to skip intentionally.")
        else:
            print(f"\n>>> Evaluating PPO model: {model_path}")
            try:
                from stable_baselines3 import PPO
            except Exception as exc:
                print(f">>> Could not import stable_baselines3.PPO; skipped PPO. Error: {exc}")
            else:
                model = PPO.load(str(model_path))
                ppo_hourly_rows = []
                for seed in seeds:
                    row = evaluate_ppo(model, seed, ppo_hourly_rows)
                    row["ppo_model_path"] = str(model_path)
                    all_rows.append(row)
                    print(
                        f"PPO    seed={seed} comp={row['completion_rate']:.3f} "
                        f"cost={row['total_cost']:.2f} unit={row['unit_task_cost']:.4f} "
                        f"backlog={row['final_backlog_work']:.2f} reward={row['total_reward']:.4f}"
                    )
                save_hourly_csv(
                    ppo_hourly_rows,
                    Path(args.out) / "ppo_hourly_detail.csv"
                )
                save_hourly_mean_csv(
                    ppo_hourly_rows,
                    Path(args.out) / "ppo_hourly_mean.csv"
                )

    # 5. Write outputs
    all_csv = out_dir / "all_results.csv"
    summary_csv = out_dir / "summary.csv"

    write_csv(all_csv, all_rows)
    summary_rows = build_summary(all_rows)
    write_csv(summary_csv, summary_rows)
    print_summary(summary_rows)

    print(f"\nTotal wall time: {time.time() - start_time:.2f} s")
    print(f"All results CSV: {all_csv}")
    print(f"Summary CSV:     {summary_csv}")


if __name__ == "__main__":
    main()
