"""Production HARL environment factories for the IDC/BESS two-agent task."""

from __future__ import annotations

from functools import partial
from pathlib import Path
from typing import Any

from configs.experiment_cases import get_experiment_case
from marl.bridges import HarlIDCGridBridge, HarlPaddedBridge
from marl.envs.idc_grid_multi_agent_env import IDCGridMultiAgentEnv
from train.train_ppo_ultimate import make_unmonitored_env


SUPPORTED_SCENARIO = "idc_bess_padding"
DEFAULT_WORKER_SEED_STRIDE = 1000


def derive_worker_seeds(
    base_seed: int, n_threads: int, *, stride: int = DEFAULT_WORKER_SEED_STRIDE
) -> tuple[int, ...]:
    """Return deterministic, rank-stable seeds for train/eval workers."""
    if int(n_threads) <= 0:
        raise ValueError("n_threads must be positive.")
    if int(stride) <= 0:
        raise ValueError("worker seed stride must be positive.")
    seeds = tuple(int(base_seed) + rank * int(stride) for rank in range(int(n_threads)))
    if len(set(seeds)) != len(seeds):
        raise ValueError("Derived worker seeds are not unique.")
    return seeds


def make_harl_single_env(
    *,
    seed: int,
    scenario: str = SUPPORTED_SCENARIO,
    experiment_case: str = "main",
) -> HarlPaddedBridge:
    """Build one independent IDC/BESS environment with the verified wrapper order."""
    if scenario != SUPPORTED_SCENARIO:
        raise ValueError(
            f"Unsupported HARL scenario {scenario!r}; expected {SUPPORTED_SCENARIO!r}."
        )

    grid_env: Any | None = None
    try:
        case = get_experiment_case(experiment_case)
        grid_env = make_unmonitored_env(
            case["env_config"],
            case["reward_config"],
            case["data_config"],
            seed=int(seed),
        )
        return HarlPaddedBridge(
            HarlIDCGridBridge(IDCGridMultiAgentEnv(grid_env))
        )
    except Exception as exc:
        if grid_env is not None:
            try:
                grid_env.close()
            except Exception:
                pass
        raise RuntimeError(
            "Failed to create the production IDC/BESS HARL environment "
            f"for scenario={scenario!r}, case={experiment_case!r}, seed={seed}: "
            f"{type(exc).__name__}: {exc}"
        ) from exc


def make_harl_train_env(
    *,
    seed: int,
    n_rollout_threads: int,
    scenario: str = SUPPORTED_SCENARIO,
    experiment_case: str = "main",
    worker_seed_stride: int = DEFAULT_WORKER_SEED_STRIDE,
):
    """Create the production training VecEnv for one, two, or four workers."""
    return _make_harl_vec_env(
        seed=seed,
        n_threads=n_rollout_threads,
        scenario=scenario,
        experiment_case=experiment_case,
        purpose="train",
        worker_seed_stride=worker_seed_stride,
    )


def make_harl_eval_env(
    *,
    seed: int,
    n_eval_rollout_threads: int,
    scenario: str = SUPPORTED_SCENARIO,
    experiment_case: str = "main",
    worker_seed_stride: int = DEFAULT_WORKER_SEED_STRIDE,
):
    """Create an evaluation VecEnv independent from the training environment."""
    return _make_harl_vec_env(
        seed=seed,
        n_threads=n_eval_rollout_threads,
        scenario=scenario,
        experiment_case=experiment_case,
        purpose="eval",
        worker_seed_stride=worker_seed_stride,
    )


def _make_harl_vec_env(
    *,
    seed: int,
    n_threads: int,
    scenario: str,
    experiment_case: str,
    purpose: str,
    worker_seed_stride: int,
):
    if int(n_threads) not in (1, 2, 4):
        raise ValueError(f"Formal {purpose} supports 1, 2, or 4 workers; got {n_threads}.")

    try:
        from harl.envs.env_wrappers import ShareDummyVecEnv
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "HARL could not be imported while creating the formal VecEnv. "
            "Add the verified HARL checkout to PYTHONPATH or use the formal "
            "training CLI with --harl-source."
        ) from exc

    worker_seeds = derive_worker_seeds(
        int(seed), int(n_threads), stride=int(worker_seed_stride)
    )
    env_fns = [
        partial(
            make_harl_single_env,
            seed=worker_seed,
            scenario=scenario,
            experiment_case=experiment_case,
        )
        for worker_seed in worker_seeds
    ]

    try:
        if int(n_threads) == 1:
            vec_env = ShareDummyVecEnv(env_fns)
            vec_env.vec_env_type = "ShareDummyVecEnv"
            vec_env.multiprocessing_start_method = None
            vec_env.worker_seeds = worker_seeds
            vec_env.agent_order = ("idc", "bess")
        else:
            from marl.envs.parallel_vec_env import ProjectShareSubprocVecEnv

            vec_env = ProjectShareSubprocVecEnv(
                env_fns, worker_seeds=worker_seeds
            )
    except Exception as exc:
        raise RuntimeError(
            f"Failed to create the formal {purpose} VecEnv: "
            f"{type(exc).__name__}: {exc}"
        ) from exc

    if int(getattr(vec_env, "n_agents", -1)) != 2:
        vec_env.close()
        raise RuntimeError(
            f"Formal {purpose} VecEnv must expose n_agents=2, got "
            f"{getattr(vec_env, 'n_agents', None)!r}."
        )
    return vec_env


def describe_vec_env(vec_env: Any) -> dict[str, Any]:
    """Return the dimensions needed by configuration and metadata validation."""
    return {
        "n_agents": int(getattr(vec_env, "n_agents", -1)),
        "agent_order": list(getattr(vec_env, "agent_order", ())),
        "observation_dims": [int(space.shape[0]) for space in vec_env.observation_space],
        "action_dims": [int(space.shape[0]) for space in vec_env.action_space],
        "state_dims": [
            int(space.shape[0]) for space in vec_env.share_observation_space
        ],
        "factory_file": str(Path(__file__).resolve()),
        "vec_env_type": str(getattr(vec_env, "vec_env_type", type(vec_env).__name__)),
        "multiprocessing_start_method": getattr(vec_env, "multiprocessing_start_method", None),
        "worker_seeds": list(getattr(vec_env, "worker_seeds", ())),
    }
