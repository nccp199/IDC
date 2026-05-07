"""
PSO baseline for IDCPriceEnv20D_ultimate.

Run examples:
    python pso_base.py --quick
    python pso_base.py --particles 30 --iter 30 --start-seed 3000 --n-seeds 1
    python pso_base.py --particles 30 --iter 30 --start-seed 3000 --n-seeds 30 --save-plan

Put this file in the same folder as the refactored ultimate files:
    config_ultimate.py
    IDCPriceEnv20D_ultimate.py
    task_model.py
    power_model.py
    task.py
"""

from __future__ import annotations

import argparse
import csv
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from config_ultimate import ENV_CONFIG, REWARD_CONFIG
from IDCPriceEnv20D_ultimate import IDCPriceEnv20D


@dataclass
class PSOConfig:
    horizon: int = 24
    action_dim: int = 22
    num_particles: int = 30
    iterations: int = 30
    inertia: float = 0.70
    cognitive: float = 1.50
    social: float = 1.50
    vmax: float = 0.20
    fitness_mode: str = "reward"


def make_env(seed: int) -> IDCPriceEnv20D:
    """
    Create the evaluation environment using the shared ultimate config.

    GA, PSO and PPO should use the same ENV_CONFIG and REWARD_CONFIG
    so the comparison is fair. The seed fixes both server parameters
    and task generation for repeatable baseline evaluation.
    """
    env_kwargs = {
        **ENV_CONFIG,
        **REWARD_CONFIG,
        "server_seed": seed,
        "task_seed": seed,
    }
    return IDCPriceEnv20D(**env_kwargs)

def fitness_from_info(total_reward: float, info: Dict[str, float], mode: str) -> float:
    """
    Default: use the same accumulated reward as PPO optimizes.
    A conservative hybrid mode is provided for experiments, but reward is recommended first.
    """
    if mode == "reward":
        return float(total_reward)

    if mode == "hybrid":
        completion = float(info.get("completion_rate", 0.0))
        task_completion = float(info.get("task_completion_rate", 0.0))
        backlog = float(info.get("final_backlog_work", info.get("Q", 0.0)))
        cost = float(info.get("total_cost", 0.0))
        miss = float(info.get("deadline_miss_rate", 0.0))
        return 10.0 * completion + 3.0 * task_completion - 0.001 * backlog - 0.05 * cost - 2.0 * miss

    raise ValueError(f"Unknown fitness mode: {mode}")


def evaluate_plan(action_plan: np.ndarray, env_seed: int, fitness_mode: str = "reward") -> Dict[str, float]:
    """
    Evaluate one full-day action plan.

    action_plan shape: (24, 22)
    action_plan[t] is fed to env.step() at hour t.
    """
    env = make_env(env_seed)
    obs, reset_info = env.reset()

    action_plan = np.asarray(action_plan, dtype=np.float32)
    if action_plan.shape != (env.horizon, env.action_dim):
        raise ValueError(f"action_plan shape should be {(env.horizon, env.action_dim)}, got {action_plan.shape}")

    total_reward = 0.0
    info: Dict[str, float] = {}

    for t in range(env.horizon):
        action = np.clip(action_plan[t], 0.0, 1.0).astype(np.float32)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += float(reward)
        if terminated or truncated:
            break

    metrics = {
        "fitness": float(fitness_from_info(total_reward, info, fitness_mode)),
        "total_reward": float(total_reward),
        "total_energy_kWh": float(info.get("total_energy_kWh", np.nan)),
        "total_cost": float(info.get("total_cost", np.nan)),
        "total_completed_work": float(info.get("total_completed_work", np.nan)),
        "completion_rate": float(info.get("completion_rate", np.nan)),
        "unit_task_cost": float(info.get("unit_task_cost", np.nan)),
        "energy_per_task": float(info.get("energy_per_task", np.nan)),
        "final_backlog_work": float(info.get("final_backlog_work", info.get("Q", np.nan))),
        "task_completion_rate": float(info.get("task_completion_rate", np.nan)),
        "finished_task_count": float(info.get("finished_task_count", np.nan)),
        "total_task_count": float(info.get("total_task_count", np.nan)),
        "deadline_miss_rate": float(info.get("deadline_miss_rate", np.nan)),
        "deadline_miss_count": float(info.get("deadline_miss_count", np.nan)),
        "avg_waiting_time": float(info.get("avg_waiting_time", np.nan)),
        "avg_turnaround_time": float(info.get("avg_turnaround_time", np.nan)),
        "total_pause_count": float(info.get("total_pause_count", np.nan)),
        "total_resume_count": float(info.get("total_resume_count", np.nan)),
        "total_non_interruptible_interruption_count": float(
            info.get("total_non_interruptible_interruption_count", np.nan)
        ),
    }
    return metrics


def make_price_plan(env_seed: int, cfg: PSOConfig) -> np.ndarray:
    """
    A simple price-aware plan used as one initial particle.
    Low price: higher server actions; high price: lower server actions.
    """
    env = make_env(env_seed)
    price = np.asarray(env.price_t, dtype=np.float64)
    low, high = float(np.min(price)), float(np.max(price))

    plan = np.zeros((cfg.horizon, cfg.action_dim), dtype=np.float32)
    for t in range(cfg.horizon):
        if np.isclose(price[t], low):
            server_level = 0.95
        elif np.isclose(price[t], high):
            server_level = 0.25
        else:
            server_level = 0.55

        plan[t, :20] = server_level
        plan[t, 20] = 0.85
        plan[t, 21] = 0.80

    return np.clip(plan, 0.0, 1.0)


def init_particles(rng: np.random.Generator, env_seed: int, cfg: PSOConfig) -> Tuple[np.ndarray, np.ndarray]:
    positions = rng.uniform(
        0.0,
        1.0,
        size=(cfg.num_particles, cfg.horizon, cfg.action_dim),
    ).astype(np.float32)

    # Small random initial speeds. Speeds are not server speeds; they are action-plan update amounts.
    velocities = rng.uniform(
        -0.05,
        0.05,
        size=(cfg.num_particles, cfg.horizon, cfg.action_dim),
    ).astype(np.float32)

    # Inject a few meaningful baselines to improve early search stability.
    if cfg.num_particles >= 1:
        positions[0] = make_price_plan(env_seed, cfg)
        velocities[0] = 0.0
    if cfg.num_particles >= 2:
        positions[1] = np.ones((cfg.horizon, cfg.action_dim), dtype=np.float32)
        positions[1, :, 20:] = 0.8
        velocities[1] = 0.0
    if cfg.num_particles >= 3:
        positions[2] = np.zeros((cfg.horizon, cfg.action_dim), dtype=np.float32)
        positions[2, :, 20:] = 0.5
        velocities[2] = 0.0

    return positions, velocities


def run_pso(env_seed: int, algo_seed: int, cfg: PSOConfig, verbose: bool = True) -> Tuple[np.ndarray, Dict[str, float]]:
    rng = np.random.default_rng(algo_seed)
    positions, velocities = init_particles(rng, env_seed, cfg)

    pbest_positions = positions.copy()
    pbest_scores = np.full(cfg.num_particles, -np.inf, dtype=np.float64)
    pbest_metrics: List[Dict[str, float]] = [dict(fitness=-np.inf) for _ in range(cfg.num_particles)]

    gbest_position = None
    gbest_metrics: Dict[str, float] = {"fitness": -np.inf}
    eval_count = 0

    for it in range(cfg.iterations + 1):
        scores = np.zeros(cfg.num_particles, dtype=np.float64)

        for i in range(cfg.num_particles):
            metrics = evaluate_plan(positions[i], env_seed=env_seed, fitness_mode=cfg.fitness_mode)
            score = float(metrics["fitness"])
            scores[i] = score
            eval_count += 1

            if score > pbest_scores[i]:
                pbest_scores[i] = score
                pbest_positions[i] = positions[i].copy()
                pbest_metrics[i] = dict(metrics)

            if score > float(gbest_metrics["fitness"]):
                gbest_metrics = dict(metrics)
                gbest_position = positions[i].copy()

        if verbose:
            print(
                f"seed={env_seed} iter={it:03d} "
                f"best_fit={gbest_metrics['fitness']:.4f} "
                f"reward={gbest_metrics['total_reward']:.4f} "
                f"comp={gbest_metrics['completion_rate']:.3f} "
                f"cost={gbest_metrics['total_cost']:.2f} "
                f"unit={gbest_metrics['unit_task_cost']:.4f} "
                f"backlog={gbest_metrics['final_backlog_work']:.2f}"
            )

        # Last loop only evaluates; no need to move particles again.
        if it == cfg.iterations:
            break

        if gbest_position is None:
            raise RuntimeError("PSO failed to produce a global best plan.")

        r1 = rng.random(size=positions.shape).astype(np.float32)
        r2 = rng.random(size=positions.shape).astype(np.float32)

        cognitive_term = cfg.cognitive * r1 * (pbest_positions - positions)
        social_term = cfg.social * r2 * (gbest_position[None, :, :] - positions)
        velocities = cfg.inertia * velocities + cognitive_term + social_term
        velocities = np.clip(velocities, -cfg.vmax, cfg.vmax).astype(np.float32)

        positions = positions + velocities
        positions = np.clip(positions, 0.0, 1.0).astype(np.float32)

    if gbest_position is None:
        raise RuntimeError("PSO failed to produce a best plan.")

    gbest_metrics["eval_count"] = float(eval_count)
    gbest_metrics["env_seed"] = float(env_seed)
    gbest_metrics["algo_seed"] = float(algo_seed)
    return gbest_position, gbest_metrics


def write_rows_csv(path: Path, rows: List[Dict[str, float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def summarize_rows(rows: List[Dict[str, float]]) -> None:
    if not rows:
        return

    keys = [
        "fitness",
        "total_reward",
        "completion_rate",
        "task_completion_rate",
        "total_completed_work",
        "total_cost",
        "unit_task_cost",
        "final_backlog_work",
        "deadline_miss_rate",
        "avg_waiting_time",
    ]

    print("\n=== PSO summary over seeds ===")
    for key in keys:
        values = np.array([float(r[key]) for r in rows], dtype=np.float64)
        print(f"{key:<28} mean={np.nanmean(values):.6f}  std={np.nanstd(values):.6f}")


def parse_seeds(args) -> List[int]:
    if args.seeds.strip():
        return [int(x.strip()) for x in args.seeds.split(",") if x.strip()]
    return list(range(args.start_seed, args.start_seed + args.n_seeds))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="Fast smoke test: particles=8, iter=5, n_seeds=1.")
    parser.add_argument("--particles", type=int, default=30, help="Number of particles.")
    parser.add_argument("--iter", type=int, default=30, help="Number of PSO iterations.")
    parser.add_argument("--w", type=float, default=0.70, help="Inertia weight.")
    parser.add_argument("--c1", type=float, default=1.50, help="Cognitive coefficient.")
    parser.add_argument("--c2", type=float, default=1.50, help="Social coefficient.")
    parser.add_argument("--vmax", type=float, default=0.20, help="Maximum absolute velocity per action value.")
    parser.add_argument("--fitness", type=str, default="reward", choices=["reward", "hybrid"])
    parser.add_argument("--start-seed", type=int, default=3000)
    parser.add_argument("--n-seeds", type=int, default=1)
    parser.add_argument("--seeds", type=str, default="", help="Comma-separated env seeds, e.g. 3000,3001,3002.")
    parser.add_argument("--out", type=str, default="pso_out", help="Output directory.")
    parser.add_argument("--save-plan", action="store_true", help="Save best 24x22 action plan as .npy for each seed.")
    parser.add_argument("--quiet", action="store_true", help="Do not print every iteration.")
    args = parser.parse_args()

    if args.quick:
        args.particles = 8
        args.iter = 5
        args.n_seeds = 1

    cfg = PSOConfig(
        num_particles=args.particles,
        iterations=args.iter,
        inertia=args.w,
        cognitive=args.c1,
        social=args.c2,
        vmax=args.vmax,
        fitness_mode=args.fitness,
    )

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    seeds = parse_seeds(args)

    print("=== PSO baseline config ===")
    print(cfg)
    print(f"env seeds: {seeds}")
    print(f"output dir: {out_dir.resolve()}")

    rows: List[Dict[str, float]] = []
    start_time = time.time()

    for env_seed in seeds:
        algo_seed = env_seed + 20000
        best_plan, metrics = run_pso(env_seed, algo_seed, cfg, verbose=not args.quiet)
        metrics["algorithm"] = "PSO"
        row = {"algorithm": metrics.pop("algorithm")}
        row.update(metrics)
        rows.append(row)

        if args.save_plan:
            np.save(out_dir / f"pso_plan_seed{env_seed}.npy", best_plan)

        csv_path = out_dir / "pso_results.csv"
        write_rows_csv(csv_path, rows)
        print(f"\nSaved current results to: {csv_path}")

    summarize_rows(rows)
    print(f"\nTotal wall time: {time.time() - start_time:.2f} s")
    print(f"Final CSV: {out_dir / 'pso_results.csv'}")


if __name__ == "__main__":
    main()
