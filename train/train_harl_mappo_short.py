"""Formal dual-agent HARL MAPPO/HAPPO short-training implementation.

This module intentionally owns only configuration resolution, production
environment construction, startup validation, and runner orchestration. The
project runner adds only agent-specific effective-action probability masks;
GAE, buffers, actor/critic networks, and bounded Box distributions retain the
pinned HARL implementation.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from marl.specs import (
    ACTION_PADDING_STRATEGY,
    AGENTS,
    EFFECTIVE_ACTION_DIMS,
    PADDED_ACTION_DIMS,
    effective_action_mask,
)
from marl.logging import LOGGER_VERSION, METRIC_SCHEMA_VERSION, REWARD_COMPONENTS
from marl.specs.state_specs import CENTRALIZED_STATE_DIM


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "harl_mappo_short.yaml"
DEFAULT_HARL_SOURCE = PROJECT_ROOT.parent / "HARL"
EXPECTED_AGENT_ORDER = ("idc", "bess")
EXPECTED_OBSERVATION_DIMS = (288, 288)
EXPECTED_ACTION_DIMS = (22, 22)
EXPECTED_STATE_DIM = CENTRALIZED_STATE_DIM
EXPECTED_EPISODE_LENGTH = 24


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run formal IDC/BESS dual-agent HARL MAPPO training."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--algorithm", choices=("mappo", "happo"))
    parser.add_argument("--critic-type", choices=("mlp", "hgta"))
    parser.add_argument("--seed", type=int)
    parser.add_argument("--updates", type=int)
    parser.add_argument("--episode-length", type=int)
    parser.add_argument("--rollout-threads", type=int)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--scenario")
    parser.add_argument("--device", choices=("cpu", "cuda"))
    parser.add_argument(
        "--resume-checkpoint",
        type=Path,
        help="Complete training_resume .pt file (or latest.json); --updates remains the total target.",
    )
    parser.add_argument(
        "--load-model-dir",
        type=Path,
        help="Model-only HARL weights for warm-start/evaluation; not training resume.",
    )
    parser.add_argument(
        "--load-checkpoint-only",
        action="store_true",
        help="Strictly restore --resume-checkpoint, then exit without an update.",
    )
    parser.add_argument("--checkpoint-interval", type=int)
    parser.add_argument(
        "--harl-source",
        type=Path,
        default=Path(os.environ.get("HARL_SOURCE_PATH", DEFAULT_HARL_SOURCE)),
        help="Verified HARL source checkout (or set HARL_SOURCE_PATH).",
    )
    parser.add_argument(
        "--runtime-path",
        type=Path,
        default=(
            Path(os.environ["HARL_RUNTIME_PATH"])
            if os.environ.get("HARL_RUNTIME_PATH")
            else None
        ),
        help="Optional directory containing installed HARL runtime dependencies.",
    )
    return parser


def configure_import_paths(
    harl_source: str | Path, runtime_path: str | Path | None = None
) -> Path:
    """Expose the verified HARL checkout and optional real dependency directory."""
    source = Path(harl_source).resolve()
    missing = [
        str(source / relative)
        for relative in ("harl/runners", "harl/algorithms", "harl/envs")
        if not (source / relative).is_dir()
    ]
    if missing:
        raise FileNotFoundError(
            "The HARL source checkout is incomplete; missing: " + ", ".join(missing)
        )

    source_text = str(source)
    if source_text not in sys.path:
        sys.path.insert(0, source_text)

    if runtime_path is not None:
        runtime = Path(runtime_path).resolve()
        if not runtime.is_dir():
            raise FileNotFoundError(f"HARL runtime dependency path does not exist: {runtime}")
        runtime_text = str(runtime)
        if runtime_text not in sys.path:
            # Append so the active environment's numpy/torch stay authoritative.
            sys.path.append(runtime_text)

    required_apis = {
        "yaml": "safe_load",
        "tensorboardX": "SummaryWriter",
        "setproctitle": "setproctitle",
    }
    missing_dependencies: list[str] = []
    for module_name, required_attribute in required_apis.items():
        try:
            module = importlib.import_module(module_name)
        except ModuleNotFoundError:
            missing_dependencies.append(module_name)
        else:
            if not hasattr(module, required_attribute):
                missing_dependencies.append(
                    f"{module_name} (missing {required_attribute})"
                )
    if missing_dependencies:
        raise ModuleNotFoundError(
            "Formal HARL runtime dependencies are missing: "
            + ", ".join(missing_dependencies)
            + ". Install them in the selected Python environment or provide an "
            "installed dependency directory with --runtime-path/HARL_RUNTIME_PATH. "
            "The formal entry point does not create fake modules."
        )
    return source


def load_yaml_config(path: str | Path) -> dict[str, Any]:
    try:
        import yaml
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "PyYAML is required to read the formal MAPPO configuration."
        ) from exc

    config_path = Path(path).resolve()
    if not config_path.is_file():
        raise FileNotFoundError(f"MAPPO configuration file does not exist: {config_path}")
    with config_path.open("r", encoding="utf-8") as stream:
        loaded = yaml.safe_load(stream)
    if not isinstance(loaded, dict):
        raise ValueError(f"MAPPO configuration root must be a mapping: {config_path}")
    return loaded


def derive_num_env_steps(
    updates: int, episode_length: int, n_rollout_threads: int
) -> int:
    values = {
        "updates": updates,
        "episode_length": episode_length,
        "n_rollout_threads": n_rollout_threads,
    }
    for name, value in values.items():
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{name} must be a positive integer, got {value!r}.")
    num_env_steps = updates * episode_length * n_rollout_threads
    derived_updates = num_env_steps // episode_length // n_rollout_threads
    remainder = num_env_steps % (episode_length * n_rollout_threads)
    if remainder != 0 or derived_updates != updates:
        raise ValueError(
            "num_env_steps is not exactly divisible by episode_length and "
            "n_rollout_threads."
        )
    return num_env_steps


def resolve_config(
    base_config: Mapping[str, Any],
    *,
    seed: int | None = None,
    updates: int | None = None,
    episode_length: int | None = None,
    rollout_threads: int | None = None,
    output_dir: str | Path | None = None,
    scenario: str | None = None,
    device: str | None = None,
    checkpoint_interval: int | None = None,
    algorithm: str | None = None,
    critic_type: str | None = None,
) -> dict[str, Any]:
    """Apply CLI-style overrides and return the complete effective config."""
    resolved = copy.deepcopy(dict(base_config))
    required_sections = (
        "main",
        "project",
        "seed",
        "device",
        "parallel",
        "train",
        "eval",
        "render",
        "model",
        "algo",
        "logger",
        "checkpoint",
        "critic",
        "env",
    )
    absent = [name for name in required_sections if not isinstance(resolved.get(name), dict)]
    if absent:
        raise ValueError("Missing MAPPO configuration sections: " + ", ".join(absent))

    if seed is not None:
        resolved["seed"]["seed"] = seed
        resolved["seed"]["seed_specify"] = True
    if updates is not None:
        resolved["train"]["updates"] = updates
    if episode_length is not None:
        resolved["train"]["episode_length"] = episode_length
    if rollout_threads is not None:
        resolved["train"]["n_rollout_threads"] = rollout_threads
    if output_dir is not None:
        resolved["logger"]["log_dir"] = str(Path(output_dir).resolve())
    if scenario is not None:
        resolved["env"]["scenario"] = scenario
    if device is not None:
        resolved["device"]["cuda"] = device == "cuda"
    if checkpoint_interval is not None:
        resolved["checkpoint"]["interval_updates"] = checkpoint_interval
    if algorithm is not None:
        resolved["main"]["algorithm_name"] = algorithm
    if critic_type is not None:
        resolved["critic"]["type"] = critic_type

    from marl.utils.rng_isolation import (
        CRITIC_INIT_SEED_RULE,
        DEFAULT_CRITIC_INIT_SEED_OFFSET,
        RNG_ISOLATION_VERSION,
    )

    rng = resolved.setdefault("rng", {})
    rng.setdefault("rng_isolation_version", RNG_ISOLATION_VERSION)
    rng.setdefault("critic_init_seed_rule", CRITIC_INIT_SEED_RULE)
    rng.setdefault("critic_init_seed_offset", DEFAULT_CRITIC_INIT_SEED_OFFSET)
    rng.setdefault("actor_sampling_rng_isolated_from_critic_init", True)
    rng["critic_init_seed"] = int(resolved["seed"]["seed"]) + int(
        rng["critic_init_seed_offset"]
    )

    from marl.methods import derive_method

    method = derive_method(
        resolved["main"].get("algorithm_name", ""), resolved["critic"].get("type", "")
    )
    resolved["main"].update(method.as_dict())
    if method.algorithm_name != "mappo" and resolved["main"].get(
        "experiment_name"
    ) == "idc_bess_mappo_short":
        resolved["main"]["experiment_name"] = f"idc_bess_happo_short_{method.method_id}"

    updates_value = resolved["train"].get("updates")
    episode_value = resolved["train"].get("episode_length")
    threads_value = resolved["train"].get("n_rollout_threads")
    resolved["train"]["num_env_steps"] = derive_num_env_steps(
        updates_value, episode_value, threads_value
    )

    log_dir = Path(resolved["logger"]["log_dir"])
    if not log_dir.is_absolute():
        resolved["logger"]["log_dir"] = str((PROJECT_ROOT / log_dir).resolve())

    validate_resolved_config(resolved)
    return resolved


def validate_resolved_config(config: Mapping[str, Any]) -> None:
    """Reject unsupported or unsafe first-version training configurations."""
    from marl.methods import derive_method

    derive_method(config["main"].get("algorithm_name", ""), config["critic"].get("type", ""))
    from marl.utils.rng_isolation import CRITIC_INIT_SEED_RULE, RNG_ISOLATION_VERSION

    rng = config.get("rng")
    if not isinstance(rng, Mapping):
        raise ValueError("Resolved config must contain the critic RNG isolation contract.")
    expected_rng = {
        "rng_isolation_version": RNG_ISOLATION_VERSION,
        "critic_init_seed_rule": CRITIC_INIT_SEED_RULE,
        "actor_sampling_rng_isolated_from_critic_init": True,
    }
    for key, expected in expected_rng.items():
        if rng.get(key) != expected:
            raise ValueError(f"rng.{key} must be {expected!r}, got {rng.get(key)!r}.")
    offset = rng.get("critic_init_seed_offset")
    if isinstance(offset, bool) or not isinstance(offset, int) or offset <= 0:
        raise ValueError("rng.critic_init_seed_offset must be a positive integer.")
    expected_critic_seed = int(config["seed"]["seed"]) + offset
    if rng.get("critic_init_seed") != expected_critic_seed:
        raise ValueError(
            "rng.critic_init_seed must equal seed.seed + critic_init_seed_offset: "
            f"expected {expected_critic_seed}, got {rng.get('critic_init_seed')!r}."
        )
    if config["main"].get("env_name") != "gym":
        raise ValueError("The project HARL logger integration requires env_name=gym.")
    if config["env"].get("state_type") != "EP":
        raise ValueError("The verified centralized state configuration requires state_type=EP.")
    if config["critic"].get("type") == "hgta":
        hgta = config["critic"].get("hgta")
        if not isinstance(hgta, Mapping):
            raise ValueError("HAPPO_HGTA requires a complete critic.hgta configuration.")
        from marl.graphs import (
            GRAPH_BUILDER_VERSION,
            GRAPH_SCHEMA_VERSION,
            HGTA_ARCHITECTURE_VERSION,
            build_graph_schema,
        )

        expected_versions = {
            "graph_schema_version": GRAPH_SCHEMA_VERSION,
            "graph_builder_version": GRAPH_BUILDER_VERSION,
            "architecture_version": HGTA_ARCHITECTURE_VERSION,
        }
        for key, expected in expected_versions.items():
            if hgta.get(key) != expected:
                raise ValueError(
                    f"critic.hgta.{key} must be {expected!r}, got {hgta.get(key)!r}."
                )
        build_graph_schema(hgta)
    if config["train"].get("episode_length") != EXPECTED_EPISODE_LENGTH:
        raise ValueError(
            f"The verified environment requires episode_length={EXPECTED_EPISODE_LENGTH}."
        )
    if config["train"].get("n_rollout_threads") not in (1, 2, 4):
        raise ValueError("Formal on-policy training supports 1, 2, or 4 rollout workers.")
    if config["eval"].get("n_eval_rollout_threads") != 1:
        raise NotImplementedError(
            "The first formal MAPPO entry supports n_eval_rollout_threads=1 only."
        )
    if config["model"].get("use_bounded_box_actions") is not True:
        raise ValueError("use_bounded_box_actions must be explicitly true.")
    if config["algo"].get("action_aggregation") != "prod":
        raise ValueError("The verified bounded Box setup requires action_aggregation=prod.")
    if config["algo"].get("share_param") is not False:
        raise ValueError("IDC and BESS must use two independent actors (share_param=false).")
    if config["render"].get("use_render") is not False:
        raise ValueError("The formal short-training entry does not support render mode.")
    checkpoint = config["checkpoint"]
    for key in ("enabled", "save_final"):
        if not isinstance(checkpoint.get(key), bool):
            raise ValueError(f"checkpoint.{key} must be a boolean.")
    if not isinstance(checkpoint.get("interval_updates"), int) or checkpoint["interval_updates"] <= 0:
        raise ValueError("checkpoint.interval_updates must be a positive integer.")
    keep_last = checkpoint.get("keep_last")
    if keep_last is not None and (not isinstance(keep_last, int) or keep_last <= 0):
        raise ValueError("checkpoint.keep_last must be null or a positive integer.")
    if checkpoint["enabled"] and config["train"].get("use_linear_lr_decay"):
        raise ValueError(
            "Exact resume with a target changed after checkpoint is not supported with "
            "use_linear_lr_decay=true; keep the verified false setting."
        )
    if not isinstance(config["seed"].get("seed"), int):
        raise ValueError("seed.seed must be an integer.")
    if config["parallel"].get("multiprocessing_start_method") != "spawn":
        raise ValueError("parallel.multiprocessing_start_method must be 'spawn'.")
    for key in ("worker_seed_stride", "blas_threads"):
        if not isinstance(config["parallel"].get(key), int) or config["parallel"][key] <= 0:
            raise ValueError(f"parallel.{key} must be a positive integer.")
    for key in (
        "step_logging_enabled",
        "episode_logging_enabled",
        "tensorboard_enabled",
    ):
        if not isinstance(config["logger"].get(key, True), bool):
            raise ValueError(f"logger.{key} must be a boolean.")
    if config["logger"].get("episode_logging_enabled", True) is not True:
        raise ValueError("Formal short training requires episode_logging_enabled=true.")
    if config["logger"].get("tensorboard_enabled", True) is not True:
        raise ValueError("Formal short training requires tensorboard_enabled=true.")

    expected_steps = derive_num_env_steps(
        config["train"]["updates"],
        config["train"]["episode_length"],
        config["train"]["n_rollout_threads"],
    )
    if config["train"].get("num_env_steps") != expected_steps:
        raise ValueError(
            "num_env_steps must equal updates * episode_length * n_rollout_threads."
        )


def split_runner_config(
    config: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    args = {
        "algo": config["main"]["algorithm_name"],
        "critic_type": config["critic"]["type"],
        "method_id": config["main"]["method_id"],
        "env": config["main"]["env_name"],
        "exp_name": config["main"]["experiment_name"],
    }
    algo_args = {
        name: copy.deepcopy(config[name])
        for name in (
            "seed",
            "rng",
            "device",
            "train",
            "eval",
            "render",
            "model",
            "algo",
            "logger",
            "critic",
        )
    }
    env_args = copy.deepcopy(config["env"])
    return args, algo_args, env_args


def validate_environment_contract(vec_env: Any, config: Mapping[str, Any]) -> dict[str, Any]:
    """Reset a disposable probe VecEnv and validate the two-agent contract."""
    import numpy as np

    from marl.envs.harl_env_factory import describe_vec_env

    dimensions = describe_vec_env(vec_env)
    if dimensions["n_agents"] != 2:
        raise RuntimeError(f"Expected n_agents=2, got {dimensions['n_agents']!r}.")
    if tuple(dimensions["agent_order"]) != EXPECTED_AGENT_ORDER:
        raise RuntimeError(
            f"Expected agent order {EXPECTED_AGENT_ORDER}, got {dimensions['agent_order']!r}."
        )
    if tuple(dimensions["observation_dims"]) != EXPECTED_OBSERVATION_DIMS:
        raise RuntimeError(
            f"Expected observation dimensions {EXPECTED_OBSERVATION_DIMS}, got "
            f"{dimensions['observation_dims']!r}."
        )
    if tuple(dimensions["action_dims"]) != EXPECTED_ACTION_DIMS:
        raise RuntimeError(
            f"Expected action dimensions {EXPECTED_ACTION_DIMS}, got "
            f"{dimensions['action_dims']!r}."
        )
    if tuple(dimensions["state_dims"]) != (EXPECTED_STATE_DIM, EXPECTED_STATE_DIM):
        raise RuntimeError(
            f"Expected centralized state dimension {EXPECTED_STATE_DIM} for each agent, "
            f"got {dimensions['state_dims']!r}."
        )

    for agent_id, action_space in enumerate(vec_env.action_space):
        if not np.all(action_space.low == 0.0):
            raise RuntimeError(f"Agent {agent_id} action_space.low must be all zeros.")
        if not np.all(action_space.high == 1.0):
            raise RuntimeError(f"Agent {agent_id} action_space.high must be all ones.")

    obs, share_obs, _ = vec_env.reset()
    if tuple(obs.shape) != (1, 2, 288):
        raise RuntimeError(f"Unexpected reset observation shape: {obs.shape!r}.")
    if tuple(share_obs.shape) != (1, 2, EXPECTED_STATE_DIM):
        raise RuntimeError(f"Unexpected reset centralized state shape: {share_obs.shape!r}.")
    if not np.isfinite(obs).all() or not np.isfinite(share_obs).all():
        raise FloatingPointError("Environment reset produced NaN or inf.")

    validate_resolved_config(config)
    dimensions["reset_observation_shape"] = list(obs.shape)
    dimensions["reset_state_shape"] = list(share_obs.shape)
    return dimensions


def _find_grid_env(vec_env: Any) -> Any:
    """Find the existing GridCoupledEnv without changing the wrapper contract."""
    current = vec_env.envs[0]
    visited: set[int] = set()
    while id(current) not in visited:
        visited.add(id(current))
        if all(
            hasattr(current, attribute)
            for attribute in (
                "grid_enabled",
                "grid_scenario_source",
                "grid_scenario_message",
                "opf_mode",
                "use_mef",
            )
        ):
            return current
        current = getattr(current, "env", None)
        if current is None:
            break
    raise RuntimeError("GridCoupledEnv was not found in the formal environment chain.")


def validate_grid_startup(vec_env: Any) -> dict[str, Any]:
    """Reject a probe reset whose live grid state is degraded or unsuccessful."""
    grid_env = _find_grid_env(vec_env)
    probe_env = vec_env.envs[0]
    info = getattr(probe_env, "last_info", None)
    if not isinstance(info, Mapping):
        raise RuntimeError(
            "Grid startup check failed: probe reset did not expose last_info."
        )

    source = str(
        info.get("grid_scenario_source", grid_env.grid_scenario_source)
    )
    scenario_message = str(
        info.get("grid_scenario_message", grid_env.grid_scenario_message)
    )
    opf_message = str(info.get("grid_opf_message", "<missing>"))
    mef_message = str(info.get("grid_mef_message", "<missing>"))
    source_is_fallback = source.strip().lower().startswith("fallback")
    scenario_loaded = bool(info.get("grid_scenario_enabled", False)) and not source_is_fallback

    values = {
        "grid_scenario_source_not_fallback": (not source_is_fallback, source),
        "grid_scenario_csv_loaded": (scenario_loaded, scenario_message),
        "grid_enabled": (bool(grid_env.grid_enabled), grid_env.grid_enabled),
        "grid_opf_mode_ac": (
            str(grid_env.opf_mode).strip().lower() == "ac",
            grid_env.opf_mode,
        ),
        "grid_use_mef": (bool(grid_env.use_mef), grid_env.use_mef),
        "initial_opf_success": (
            info.get("grid_opf_success") is True,
            info.get("grid_opf_success"),
        ),
        "initial_mef_success": (
            info.get("grid_mef_success") is True,
            info.get("grid_mef_success"),
        ),
    }
    failures = [
        f"{name}={actual!r}"
        for name, (passed, actual) in values.items()
        if not passed
    ]
    if failures:
        raise RuntimeError(
            "Grid startup check failed: "
            + ", ".join(failures)
            + f"; grid_scenario_source={source!r}"
            + f"; grid_scenario_message={scenario_message!r}"
            + f"; opf_message={opf_message!r}"
            + f"; mef_message={mef_message!r}"
        )

    return {
        "grid_scenario_source": source,
        "grid_scenario_message": scenario_message,
        "grid_enabled": bool(grid_env.grid_enabled),
        "grid_opf_mode": str(grid_env.opf_mode).strip().lower(),
        "grid_use_mef": bool(grid_env.use_mef),
        "initial_opf_success": True,
        "initial_opf_message": opf_message,
        "initial_mef_success": True,
        "initial_mef_message": mef_message,
        "cache_opf_load_bin_mw": info.get("grid_cache_load_bin_mw"),
        "cache_mef_load_bin_mw": info.get("grid_cache_mef_load_bin_mw"),
        "task_forecast_mode": str(info["task_forecast_mode"]),
        "task_forecast_error_level": float(info["forecast_error_level"]),
        "task_forecast_seed": info.get("task_forecast_seed"),
        "task_forecast_source": str(info["task_forecast_source"]),
        "task_forecast_mae": float(info["task_forecast_mae"]),
        "task_forecast_rmse": float(info["task_forecast_rmse"]),
        "task_forecast_mape_nonzero_percent": float(
            info["task_forecast_mape_nonzero_percent"]
        ),
        "probe_true_task_arrival_profile": np.asarray(
            info["true_task_arrival_profile"], dtype=np.float64
        ).tolist(),
        "probe_task_arrival_forecast": np.asarray(
            info["task_arrival_forecast"], dtype=np.float64
        ).tolist(),
    }


def prepare_training_environment(
    config: Mapping[str, Any], *, env_factory: Any | None = None
) -> tuple[Any, dict[str, Any], dict[str, Any]]:
    """Validate a disposable probe, then return an untouched fresh train VecEnv."""
    if env_factory is None:
        from marl.envs.harl_env_factory import make_harl_train_env

        env_factory = make_harl_train_env

    kwargs = {
        "seed": config["seed"]["seed"],
        "n_rollout_threads": 1,
        "scenario": config["env"]["scenario"],
        "experiment_case": config["env"]["experiment_case"],
        "worker_seed_stride": config["parallel"]["worker_seed_stride"],
    }
    probe_envs = None
    primary_error: BaseException | None = None
    try:
        probe_envs = env_factory(**kwargs)
        dimensions = validate_environment_contract(probe_envs, config)
        grid_startup = validate_grid_startup(probe_envs)
    except BaseException as exc:
        primary_error = exc
        raise
    finally:
        if probe_envs is not None:
            try:
                probe_envs.close()
            except Exception:
                if primary_error is None:
                    raise

    train_kwargs = dict(kwargs)
    train_kwargs["n_rollout_threads"] = config["train"]["n_rollout_threads"]
    train_envs = env_factory(**train_kwargs)
    from marl.envs.harl_env_factory import describe_vec_env

    train_dimensions = describe_vec_env(train_envs)
    train_dimensions["probe_reset_observation_shape"] = dimensions["reset_observation_shape"]
    train_dimensions["probe_reset_state_shape"] = dimensions["reset_state_shape"]
    return train_envs, train_dimensions, grid_startup


def configure_parallel_runtime(config: Mapping[str, Any]) -> dict[str, str]:
    """Apply explicit child-process BLAS limits before importing numeric libraries."""
    value = str(int(config["parallel"]["blas_threads"]))
    names = (
        "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    )
    for name in names:
        os.environ[name] = value
    return {name: os.environ[name] for name in names}


def _git_head(repository: Path) -> dict[str, Any]:
    try:
        completed = subprocess.run(
            ["git", "-C", str(repository), "rev-parse", "HEAD"],
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return {"head": None, "dirty": None, "error": f"{type(exc).__name__}: {exc}"}
    if completed.returncode != 0:
        error = completed.stderr.strip() or completed.stdout.strip() or "unknown git error"
        return {"head": None, "dirty": None, "error": error}
    try:
        status = subprocess.run(
            ["git", "-C", str(repository), "status", "--porcelain"],
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return {
            "head": completed.stdout.strip(),
            "dirty": None,
            "error": f"{type(exc).__name__}: {exc}",
        }
    if status.returncode != 0:
        error = status.stderr.strip() or status.stdout.strip() or "unknown git status error"
        return {"head": completed.stdout.strip(), "dirty": None, "error": error}
    return {
        "head": completed.stdout.strip(),
        "dirty": bool(status.stdout.strip()),
        "error": None,
    }


def _sha256_file(path: str | Path) -> str | None:
    source = Path(path)
    if not source.is_file():
        return None
    digest = hashlib.sha256()
    with source.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, data: Mapping[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as stream:
        json.dump(data, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write("\n")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _jsonable(value: Any) -> Any:
    """Canonicalize config/provenance values for stable fingerprints."""
    try:
        import numpy as np
    except ModuleNotFoundError:
        np = None
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, set):
        return sorted(_jsonable(item) for item in value)
    if isinstance(value, Path):
        return str(value.resolve())
    if np is not None and isinstance(value, np.ndarray):
        return value.tolist()
    if np is not None and isinstance(value, np.generic):
        return value.item()
    return value


def _fingerprint(value: Any) -> str:
    encoded = json.dumps(
        _jsonable(value), ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def build_resume_compatibility(
    *,
    resolved: Mapping[str, Any],
    runner: Any,
    dimensions: Mapping[str, Any],
    grid_startup: Mapping[str, Any],
    project_git: Mapping[str, Any],
    harl_git: Mapping[str, Any],
) -> dict[str, Any]:
    """Build the strict invariant contract stored in each training checkpoint."""
    from configs.experiment_cases import get_experiment_case

    case = get_experiment_case(resolved["env"]["experiment_case"])
    from configs.config_ultimate import GRID_CONFIG, GRID_CACHE_CONFIG

    if hasattr(runner.envs, "envs"):
        grid_env = _find_grid_env(runner.envs)
        cache_config = dict(grid_env.grid_cache.config)
        grid_case = str(grid_env.case_name)
        cache_bins = {
            "opf_load_mw": float(grid_env.grid_cache.cache_load_bin_mw),
            "mef_load_mw": float(grid_env.grid_cache.cache_mef_load_bin_mw),
            "load_scale": float(grid_env.grid_cache.cache_load_scale_bin),
        }
    else:
        cache_config = dict(GRID_CACHE_CONFIG)
        grid_case = str(GRID_CONFIG["case_name"])
        cache_bins = {
            "opf_load_mw": float(cache_config["cache_load_bin_mw"]),
            "mef_load_mw": float(cache_config["cache_mef_load_bin_mw"]),
            "load_scale": float(cache_config["cache_load_scale_bin"]),
        }
    invariant_config = {
        "main": resolved["main"],
        "project": resolved["project"],
        "train": {
            key: resolved["train"][key]
            for key in (
                "n_rollout_threads", "episode_length", "use_valuenorm",
                "use_proper_time_limits", "use_linear_lr_decay",
            )
        },
        "model": resolved["model"],
        "algo": resolved["algo"],
        "env": resolved["env"],
        "parallel": resolved["parallel"],
        "rng": resolved["rng"],
        "experiment_env_config": case["env_config"],
        "reward_config": case["reward_config"],
        "data_config": case["data_config"],
        "grid_cache_config": cache_config,
    }
    masks = [list(effective_action_mask(agent)) for agent in AGENTS]
    compatibility = {
        "algorithm": resolved["main"]["algorithm_name"],
        "algorithm_name": resolved["main"]["algorithm_name"],
        "critic_type": resolved["critic"]["type"],
        "method_id": resolved["main"]["method_id"],
        "algorithm_implementation_version": resolved["main"]["algorithm_implementation_version"],
        "critic_interface_version": resolved["main"]["critic_interface_version"],
        "agent_update_order_policy": (
            "fixed" if resolved["algo"]["fixed_order"] else "torch_randperm"
        ),
        "agent_order": list(AGENTS),
        "dims": {
            "observation": list(dimensions["observation_dims"]),
            "state": int(EXPECTED_STATE_DIM),
            "padded_action": [PADDED_ACTION_DIMS[agent] for agent in AGENTS],
            "effective_action": [EFFECTIVE_ACTION_DIMS[agent] for agent in AGENTS],
            "virtual_action": [
                PADDED_ACTION_DIMS[agent] - EFFECTIVE_ACTION_DIMS[agent]
                for agent in AGENTS
            ],
        },
        "effective_action_mask": {
            "enabled": True,
            "strategy": ACTION_PADDING_STRATEGY,
            "masks": masks,
        },
        "episode_length": int(resolved["train"]["episode_length"]),
        "rollout_threads": int(resolved["train"]["n_rollout_threads"]),
        "vec_env_type": str(runner.envs.vec_env_type),
        "multiprocessing_start_method": runner.envs.multiprocessing_start_method,
        "worker_seeds": list(runner.envs.worker_seeds),
        "rng_isolation_version": resolved["rng"]["rng_isolation_version"],
        "critic_init_seed_rule": resolved["rng"]["critic_init_seed_rule"],
        "critic_init_seed": int(resolved["rng"]["critic_init_seed"]),
        "actor_sampling_rng_isolated_from_critic_init": bool(
            resolved["rng"]["actor_sampling_rng_isolated_from_critic_init"]
        ),
        "share_param": bool(resolved["algo"]["share_param"]),
        "action_aggregation": str(resolved["algo"]["action_aggregation"]),
        "bounded_box_actions": bool(resolved["model"]["use_bounded_box_actions"]),
        "actor_network": {
            "hidden_sizes": list(resolved["model"]["hidden_sizes"]),
            "activation": resolved["model"]["activation_func"],
            "recurrent_n": int(resolved["model"]["recurrent_n"]),
            "use_recurrent_policy": bool(resolved["model"]["use_recurrent_policy"]),
        },
        "critic_network": {
            "hidden_sizes": list(resolved["model"]["hidden_sizes"]),
            "activation": resolved["model"]["activation_func"],
            "state_type": resolved["env"]["state_type"],
        },
        "optimizers": {
            "actor_agent0": type(runner.actor[0].actor_optimizer).__name__,
            "actor_agent1": type(runner.actor[1].actor_optimizer).__name__,
            "critic": type(runner.critic.critic_optimizer).__name__,
        },
        "grid_case": grid_case,
        "experiment_case": str(resolved["env"]["experiment_case"]),
        "grid_scenario_sha256": _sha256_file(grid_startup["grid_scenario_source"]),
        "cache_bins": cache_bins,
        "reward_fingerprint": _fingerprint(case["reward_config"]),
        "data_fingerprint": _fingerprint(case["data_config"]),
        "config_fingerprint": _fingerprint(invariant_config),
        "project_git_head": project_git.get("head"),
        "harl_git_head": harl_git.get("head"),
    }
    graph_metadata = getattr(runner.critic, "graph_metadata", None)
    if graph_metadata:
        compatibility["graph"] = copy.deepcopy(dict(graph_metadata))
        compatibility["critic_network"]["hgta"] = copy.deepcopy(dict(graph_metadata))
    return compatibility


def run_training(
    config: Mapping[str, Any],
    *,
    harl_source: str | Path,
    resume_checkpoint: str | Path | None = None,
    load_model_dir: str | Path | None = None,
    load_checkpoint_only: bool = False,
) -> dict[str, Any]:
    """Run one continuous standard HARL lifecycle for the resolved configuration."""
    resolved = copy.deepcopy(dict(config))
    from marl.methods import derive_method

    method = derive_method(
        resolved["main"].get("algorithm_name", ""), resolved["critic"].get("type", "")
    )
    resolved["main"].update(method.as_dict())
    blas_thread_limits = configure_parallel_runtime(resolved)
    import gymnasium
    import numpy as np
    import pandapower
    import torch

    from marl.envs.harl_env_factory import make_harl_eval_env
    from marl.checkpointing import CHECKPOINT_SCHEMA_VERSION, TrainingCheckpointManager
    from marl.runners.idc_mappo_runner import IDCOnPolicyMARunner

    if resume_checkpoint is not None and load_model_dir is not None:
        raise ValueError("--resume-checkpoint and --load-model-dir are mutually exclusive.")
    if load_checkpoint_only and resume_checkpoint is None:
        raise ValueError("load_checkpoint_only requires resume_checkpoint.")
    if load_model_dir is not None:
        resolved["train"]["model_dir"] = str(Path(load_model_dir).resolve())
    validate_resolved_config(resolved)
    harl_path = Path(harl_source).resolve()
    project_git = _git_head(PROJECT_ROOT)
    harl_git = _git_head(harl_path)
    expected_harl_head = resolved["project"]["expected_harl_head"]
    if harl_git["head"] is not None and harl_git["head"] != expected_harl_head:
        raise RuntimeError(
            f"HARL HEAD mismatch: expected {expected_harl_head}, got {harl_git['head']}."
        )

    train_envs = None
    eval_envs = None
    runner = None
    metadata_path: Path | None = None
    metadata: dict[str, Any] | None = None
    checkpoint_manager = None
    resume_info: dict[str, Any] | None = None
    try:
        train_envs, dimensions, grid_startup = prepare_training_environment(
            resolved
        )
        if resolved["eval"]["use_eval"]:
            eval_envs = make_harl_eval_env(
                seed=resolved["eval"].get("seed", resolved["seed"]["seed"] + 100000),
                n_eval_rollout_threads=resolved["eval"]["n_eval_rollout_threads"],
                scenario=resolved["env"]["scenario"],
                experiment_case=resolved["env"]["experiment_case"],
                worker_seed_stride=resolved["parallel"]["worker_seed_stride"],
            )
            validate_environment_contract(eval_envs, resolved)
            if train_envs.envs[0] is eval_envs.envs[0]:
                raise RuntimeError("Train and evaluation factories returned the same environment.")

        args, algo_args, env_args = split_runner_config(resolved)
        runner = IDCOnPolicyMARunner(
            args,
            algo_args,
            env_args,
            train_envs=train_envs,
            eval_envs=eval_envs,
        )
        train_envs = None
        eval_envs = None

        run_dir = Path(runner.run_dir).resolve()
        compatibility = build_resume_compatibility(
            resolved=resolved,
            runner=runner,
            dimensions=dimensions,
            grid_startup=grid_startup,
            project_git=project_git,
            harl_git=harl_git,
        )
        provenance = {
            "run_id": run_dir.name,
            "run_dir": str(run_dir),
            "project_git_head": project_git["head"],
            "project_working_tree_dirty": project_git["dirty"],
            "harl_git_head": harl_git["head"],
            "harl_working_tree_dirty": harl_git["dirty"],
            "python_executable": sys.executable,
            "seed": int(resolved["seed"]["seed"]),
            "created_at_utc": _utc_now(),
            **runner.method.as_dict(),
        }
        checkpoint_manager = TrainingCheckpointManager(
            runner=runner,
            compatibility=compatibility,
            provenance=provenance,
            config=resolved["checkpoint"],
        )
        runner.set_checkpoint_manager(checkpoint_manager)
        if resume_checkpoint is not None:
            if not checkpoint_manager.enabled:
                raise ValueError("checkpoint.enabled must be true when resuming training.")
            resume_info = checkpoint_manager.load(resume_checkpoint)
            if load_checkpoint_only:
                return {
                    "status": "checkpoint_loaded",
                    "checkpoint": resume_info["checkpoint"],
                    "saved_update": resume_info["saved_update"],
                    "global_step": resume_info["global_step"],
                    "episodes_completed": resume_info["episodes_completed"],
                    "method_id": runner.method.method_id,
                    "critic_type": runner.method.critic_type,
                    "run_dir": str(Path(runner.run_dir).resolve()),
                }
            if resolved["train"]["updates"] <= resume_info["saved_update"]:
                raise ValueError(
                    "--updates is the total target and must exceed the checkpoint update; "
                    f"target={resolved['train']['updates']}, saved={resume_info['saved_update']}."
                )
        resolved_snapshot = copy.deepcopy(resolved)
        resolved_snapshot["runtime"] = {
            "project_root": str(PROJECT_ROOT),
            "project_git": project_git,
            "harl_source": str(harl_path),
            "harl_git": harl_git,
            "device": str(runner.device),
            "harl_scenario": resolved["env"]["scenario"],
            "experiment_case": resolved["env"]["experiment_case"],
            "grid_scenario_source": grid_startup["grid_scenario_source"],
            "grid_scenario_message": grid_startup["grid_scenario_message"],
            "algorithm_seed": resolved["seed"]["seed"],
            "environment_seed": resolved["seed"]["seed"],
            "server_seed": resolved["seed"]["seed"],
            "task_seed": resolved["seed"]["seed"],
            "task_forecast_mode": grid_startup["task_forecast_mode"],
            "task_forecast_error_level": grid_startup["task_forecast_error_level"],
            "task_forecast_seed": grid_startup["task_forecast_seed"],
            "task_forecast_seed_rule": "worker_seed + task_forecast_seed_offset",
            "task_forecast_source": grid_startup["task_forecast_source"],
            "task_forecast_mae": grid_startup["task_forecast_mae"],
            "task_forecast_rmse": grid_startup["task_forecast_rmse"],
            "task_forecast_mape_nonzero_percent": grid_startup[
                "task_forecast_mape_nonzero_percent"
            ],
            "probe_true_task_arrival_profile": grid_startup[
                "probe_true_task_arrival_profile"
            ],
            "probe_task_arrival_forecast": grid_startup[
                "probe_task_arrival_forecast"
            ],
            "worker_seed_strategy": "base_seed + rank * worker_seed_stride",
            "worker_seed_stride": resolved["parallel"]["worker_seed_stride"],
            "worker_seeds": list(runner.envs.worker_seeds),
            "rng_isolation_version": resolved["rng"]["rng_isolation_version"],
            "critic_init_seed_rule": resolved["rng"]["critic_init_seed_rule"],
            "critic_init_seed": int(resolved["rng"]["critic_init_seed"]),
            "actor_sampling_rng_isolated_from_critic_init": bool(
                resolved["rng"]["actor_sampling_rng_isolated_from_critic_init"]
            ),
            "vec_env_type": runner.envs.vec_env_type,
            "multiprocessing_start_method": runner.envs.multiprocessing_start_method,
            "parallel_sampling_enabled": resolved["train"]["n_rollout_threads"] > 1,
            "child_process_count": (
                resolved["train"]["n_rollout_threads"]
                if resolved["train"]["n_rollout_threads"] > 1
                else 0
            ),
            "transitions_per_update": (
                resolved["train"]["episode_length"]
                * resolved["train"]["n_rollout_threads"]
            ),
            "samples_per_update": (
                resolved["train"]["episode_length"]
                * resolved["train"]["n_rollout_threads"]
            ),
            "torch_threads": resolved["device"]["torch_threads"],
            "blas_thread_limits": blas_thread_limits,
            "action_padding_strategy": ACTION_PADDING_STRATEGY,
            "effective_action_mask_enabled": True,
            "agent_effective_action_dims": list(runner.effective_action_dims),
            "agent_padded_action_dims": list(runner.padded_action_dims),
            "agent_virtual_action_dims": [
                padded - effective
                for padded, effective in zip(
                    runner.padded_action_dims,
                    runner.effective_action_dims,
                    strict=True,
                )
            ],
            "effective_action_masks": [
                list(mask) for mask in runner.effective_action_masks
            ],
            "logger_version": LOGGER_VERSION,
            "metric_schema_version": METRIC_SCHEMA_VERSION,
            "step_logging_enabled": resolved["logger"].get("step_logging_enabled", True),
            "episode_logging_enabled": resolved["logger"].get("episode_logging_enabled", True),
            "tensorboard_enabled": resolved["logger"].get("tensorboard_enabled", True),
            "checkpoint_schema_version": CHECKPOINT_SCHEMA_VERSION,
            "checkpoint": copy.deepcopy(resolved["checkpoint"]),
            "resumed": resume_info is not None,
            "resume_from_checkpoint": None if resume_info is None else resume_info["checkpoint"],
        }
        resolved_snapshot["runtime"].update(runner.method.as_dict())
        resolved_path = run_dir / "resolved_config.json"
        _write_json(resolved_path, resolved_snapshot)

        metadata = {
            "status": "running",
            "project_repository": str(PROJECT_ROOT),
            "project_git_head": project_git["head"],
            "project_git_error": project_git["error"],
            "project_working_tree_dirty": project_git["dirty"],
            "harl_repository": str(harl_path),
            "harl_actual_git_head": harl_git["head"],
            "harl_expected_git_head": expected_harl_head,
            "harl_git_error": harl_git["error"],
            "harl_working_tree_dirty": harl_git["dirty"],
            "harl_upstream_commit": resolved["project"]["harl_upstream_commit"],
            "python_version": platform.python_version(),
            "python_executable": sys.executable,
            "pytorch_version": torch.__version__,
            "numpy_version": np.__version__,
            "gymnasium_version": gymnasium.__version__,
            "pandapower_version": pandapower.__version__,
            "device": str(runner.device),
            "seed": resolved["seed"]["seed"],
            "algorithm_seed": resolved["seed"]["seed"],
            "environment_seed": resolved["seed"]["seed"],
            "server_seed": resolved["seed"]["seed"],
            "task_seed": resolved["seed"]["seed"],
            "task_forecast_mode": grid_startup["task_forecast_mode"],
            "task_forecast_error_level": grid_startup["task_forecast_error_level"],
            "task_forecast_seed": grid_startup["task_forecast_seed"],
            "task_forecast_seed_rule": "worker_seed + task_forecast_seed_offset",
            "task_forecast_source": grid_startup["task_forecast_source"],
            "task_forecast_mae": grid_startup["task_forecast_mae"],
            "task_forecast_rmse": grid_startup["task_forecast_rmse"],
            "task_forecast_mape_nonzero_percent": grid_startup[
                "task_forecast_mape_nonzero_percent"
            ],
            "probe_true_task_arrival_profile": grid_startup[
                "probe_true_task_arrival_profile"
            ],
            "probe_task_arrival_forecast": grid_startup[
                "probe_task_arrival_forecast"
            ],
            "worker_seed_strategy": "base_seed + rank * worker_seed_stride",
            "worker_seed_stride": resolved["parallel"]["worker_seed_stride"],
            "worker_seeds": list(runner.envs.worker_seeds),
            "rng_isolation_version": resolved["rng"]["rng_isolation_version"],
            "critic_init_seed_rule": resolved["rng"]["critic_init_seed_rule"],
            "critic_init_seed": int(resolved["rng"]["critic_init_seed"]),
            "actor_sampling_rng_isolated_from_critic_init": bool(
                resolved["rng"]["actor_sampling_rng_isolated_from_critic_init"]
            ),
            "vec_env_type": runner.envs.vec_env_type,
            "multiprocessing_start_method": runner.envs.multiprocessing_start_method,
            "parallel_sampling_enabled": resolved["train"]["n_rollout_threads"] > 1,
            "child_process_count": (
                resolved["train"]["n_rollout_threads"]
                if resolved["train"]["n_rollout_threads"] > 1
                else 0
            ),
            "transitions_per_update": (
                resolved["train"]["episode_length"]
                * resolved["train"]["n_rollout_threads"]
            ),
            "samples_per_update": (
                resolved["train"]["episode_length"]
                * resolved["train"]["n_rollout_threads"]
            ),
            "torch_threads": resolved["device"]["torch_threads"],
            "blas_thread_limits": blas_thread_limits,
            "updates": resolved["train"]["updates"],
            "episode_length": resolved["train"]["episode_length"],
            "n_rollout_threads": resolved["train"]["n_rollout_threads"],
            "num_env_steps": resolved["train"]["num_env_steps"],
            "scenario": resolved["env"]["scenario"],
            "harl_scenario": resolved["env"]["scenario"],
            "experiment_case": resolved["env"]["experiment_case"],
            "grid_scenario_source": grid_startup["grid_scenario_source"],
            "grid_scenario_message": grid_startup["grid_scenario_message"],
            "grid_scenario_sha256": _sha256_file(grid_startup["grid_scenario_source"]),
            "cache_opf_load_bin_mw": grid_startup["cache_opf_load_bin_mw"],
            "cache_mef_load_bin_mw": grid_startup["cache_mef_load_bin_mw"],
            "n_agents": dimensions["n_agents"],
            "agent_order": dimensions["agent_order"],
            "observation_dims": dimensions["observation_dims"],
            "action_dims": dimensions["action_dims"],
            "state_dim": EXPECTED_STATE_DIM,
            "state_dims": dimensions["state_dims"],
            "use_bounded_box_actions": resolved["model"]["use_bounded_box_actions"],
            "action_aggregation": resolved["algo"]["action_aggregation"],
            "action_padding_strategy": ACTION_PADDING_STRATEGY,
            "effective_action_mask_enabled": True,
            "agent_effective_action_dims": list(runner.effective_action_dims),
            "agent_padded_action_dims": list(runner.padded_action_dims),
            "agent_virtual_action_dims": [
                padded - effective
                for padded, effective in zip(
                    runner.padded_action_dims,
                    runner.effective_action_dims,
                    strict=True,
                )
            ],
            "effective_action_masks": [
                list(mask) for mask in runner.effective_action_masks
            ],
            "logger_version": LOGGER_VERSION,
            "metric_schema_version": METRIC_SCHEMA_VERSION,
            "step_logging_enabled": resolved["logger"].get("step_logging_enabled", True),
            "episode_logging_enabled": resolved["logger"].get("episode_logging_enabled", True),
            "tensorboard_enabled": resolved["logger"].get("tensorboard_enabled", True),
            "reward_component_names": list(REWARD_COMPONENTS),
            "reward_aliases": {"r_grid_peak": "r_peak_load"},
            "grid_reward_enabled": False,
            "safe_rl_enabled": False,
            "metrics_directory": str((run_dir / "metrics").resolve()),
            "step_metrics_path": str(runner.metrics_logger.step_path),
            "episode_metrics_path": str(runner.metrics_logger.episode_path),
            "update_metrics_path": str(runner.metrics_logger.update_path),
            "metric_schema_path": str(runner.metrics_logger.schema_path),
            "run_summary_path": str(runner.metrics_logger.summary_path),
            "start_time_utc": _utc_now(),
            "output_directory": str(run_dir),
            "model_directory": str(Path(runner.save_dir).resolve()),
            "completed_updates": 0,
            "nan_or_inf_detected": False,
            "action_bounds_checks_passed": False,
            "checkpoint_schema_version": CHECKPOINT_SCHEMA_VERSION,
            "checkpoint_enabled": checkpoint_manager.enabled,
            "checkpoint_interval_updates": checkpoint_manager.interval_updates,
            "checkpoint_resume_boundary": "post_update_only",
            "mid_rollout_resume_supported": False,
            "resumed": resume_info is not None,
            "resume_from_checkpoint": None if resume_info is None else resume_info["checkpoint"],
            "parent_run_id": None if resume_info is None else resume_info["parent_run_id"],
            "parent_update": 0 if resume_info is None else resume_info["saved_update"],
            "starting_global_step": 0 if resume_info is None else resume_info["global_step"],
            "starting_episode_id": 0 if resume_info is None else resume_info["episodes_completed"],
            "resume_segment": 0 if resume_info is None else 1,
            "model_load_mode": (
                "training_resume" if resume_info is not None
                else "model_only" if load_model_dir is not None
                else "fresh"
            ),
        }
        metadata.update(runner.method.as_dict())
        graph_metadata = getattr(runner.critic, "graph_metadata", None)
        if graph_metadata:
            metadata["graph_metadata"] = copy.deepcopy(dict(graph_metadata))
        metadata["agent_update_order_policy"] = (
            "fixed" if runner.fixed_order else "torch_randperm"
        )
        metadata_path = run_dir / "run_metadata.json"
        _write_json(metadata_path, metadata)

        runner.run()
        expected_updates = resolved["train"]["updates"]
        if runner.completed_updates != expected_updates:
            raise RuntimeError(
                f"Runner completed {runner.completed_updates} updates; expected {expected_updates}."
            )
        metadata["effective_action_update_diagnostics"] = [
            copy.deepcopy(actor.last_update_diagnostics) for actor in runner.actor
        ]
        if any(value is None for value in metadata["effective_action_update_diagnostics"]):
            raise RuntimeError("Effective-action diagnostics were not recorded by every actor.")
        runner.save()
        model_paths = [
            Path(runner.save_dir) / "actor_agent0.pt",
            Path(runner.save_dir) / "actor_agent1.pt",
            Path(runner.save_dir) / "critic_agent.pt",
        ]
        missing_models = [str(path) for path in model_paths if not path.is_file()]
        if missing_models:
            raise RuntimeError("Final HARL model files were not saved: " + ", ".join(missing_models))

        final_checkpoint = checkpoint_manager.save_final(runner.completed_updates)
        last_checkpoint = checkpoint_manager.last_record

        metadata.update(
            {
                "status": "completed",
                "completed_updates": runner.completed_updates,
                "end_time_utc": _utc_now(),
                "model_files": [str(path.resolve()) for path in model_paths],
                "nan_or_inf_detected": False,
                "action_bounds_checks_passed": True,
                "last_checkpoint": None if last_checkpoint is None else last_checkpoint["checkpoint"],
                "last_checkpoint_sha256": None if last_checkpoint is None else last_checkpoint["sha256"],
                "final_checkpoint": None if final_checkpoint is None else final_checkpoint["checkpoint"],
                "checkpoint_save_seconds": checkpoint_manager.last_save_seconds,
                "checkpoint_load_seconds": checkpoint_manager.last_load_seconds,
                "stability_counters": copy.deepcopy(runner.stability_counters),
                "last_algorithm_update_audit": {
                    key: copy.deepcopy(value)
                    for key, value in runner.last_algorithm_update.items()
                    if key != "factor_audit"
                },
            }
        )
        run_summary = runner.metrics_logger.finalize(
            status="completed",
            termination_reason="requested_updates_completed",
            model_paths=model_paths,
            nan_or_inf_detected=False,
        )
        metadata["metrics_summary"] = {
            "episodes_completed": run_summary["episodes_completed"],
            "opf_success_rate": run_summary["opf_success_rate"],
            "mef_success_rate": run_summary["mef_success_rate"],
            "run_summary_path": str(runner.metrics_logger.summary_path),
        }
        run_summary.update(
            {
                "last_checkpoint": None if last_checkpoint is None else last_checkpoint["checkpoint"],
                "last_checkpoint_sha256": None if last_checkpoint is None else last_checkpoint["sha256"],
                "final_checkpoint": None if final_checkpoint is None else final_checkpoint["checkpoint"],
                "resumed": resume_info is not None,
                "resume_parent": None if resume_info is None else resume_info["parent_run_id"],
            }
        )
        _write_json(runner.metrics_logger.summary_path, run_summary)
        _write_json(metadata_path, metadata)
        return {
            "status": "completed",
            "updates": runner.completed_updates,
            "num_env_steps": resolved["train"]["num_env_steps"],
            "run_dir": str(run_dir),
            "resolved_config": str(resolved_path),
            "run_metadata": str(metadata_path),
            "model_files": [str(path.resolve()) for path in model_paths],
            "nan_or_inf_detected": False,
            "action_bounds_checks_passed": True,
            "last_checkpoint": None if last_checkpoint is None else last_checkpoint["checkpoint"],
            "final_checkpoint": None if final_checkpoint is None else final_checkpoint["checkpoint"],
            "resumed": resume_info is not None,
            "method_id": runner.method.method_id,
        }
    except Exception as exc:
        if runner is not None and not runner.metrics_logger.finalized:
            try:
                runner.metrics_logger.finalize(
                    status="failed",
                    termination_reason=f"{type(exc).__name__}: {exc}",
                    nan_or_inf_detected=isinstance(exc, FloatingPointError),
                )
            except Exception:
                pass
        if metadata is not None and metadata_path is not None:
            metadata.update(
                {
                    "status": "failed",
                    "end_time_utc": _utc_now(),
                    "completed_updates": (
                        int(getattr(runner, "completed_updates", 0))
                        if runner is not None
                        else 0
                    ),
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
            _write_json(metadata_path, metadata)
        raise
    finally:
        if runner is not None:
            runner.close()
        else:
            if train_envs is not None:
                train_envs.close()
            if eval_envs is not None and eval_envs is not train_envs:
                eval_envs.close()


def main(argv: Sequence[str] | None = None) -> dict[str, Any]:
    cli = build_parser().parse_args(argv)
    harl_source = configure_import_paths(cli.harl_source, cli.runtime_path)
    base_config = load_yaml_config(cli.config)
    resolved = resolve_config(
        base_config,
        seed=cli.seed,
        updates=cli.updates,
        episode_length=cli.episode_length,
        rollout_threads=cli.rollout_threads,
        output_dir=cli.output_dir,
        scenario=cli.scenario,
        device=cli.device,
        checkpoint_interval=cli.checkpoint_interval,
        algorithm=cli.algorithm,
        critic_type=cli.critic_type,
    )
    result = run_training(
        resolved,
        harl_source=harl_source,
        resume_checkpoint=cli.resume_checkpoint,
        load_model_dir=cli.load_model_dir,
        load_checkpoint_only=cli.load_checkpoint_only,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


if __name__ == "__main__":
    import multiprocessing

    multiprocessing.freeze_support()
    main()
