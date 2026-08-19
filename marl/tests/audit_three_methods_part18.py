"""Read-only Part-18 fairness and three-method artifact auditor.

The script deliberately performs no optimizer step and does not alter training
artifacts.  It reconstructs the three supported configurations, verifies actor
initialization and environment reset equality, and audits three completed
one-update runs.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping

import gymnasium as gym
import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


METHODS = (
    ("mappo", "mlp", "MAPPO_MLP"),
    ("happo", "mlp", "HAPPO_MLP"),
    ("happo", "hgta", "HAPPO_HGTA"),
)

EXPECTED_CONFIG_DIFFERENCES = {
    "critic.type",
    "logger.log_dir",
    "main.algorithm_implementation_version",
    "main.algorithm_name",
    "main.critic_type",
    "main.experiment_name",
    "main.method_id",
}


def _load_checkpoint(path: Path) -> dict[str, Any]:
    return torch.load(path, map_location="cpu", weights_only=False)


def _rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def _exact(left: Any, right: Any) -> bool:
    if torch.is_tensor(left) and torch.is_tensor(right):
        return left.dtype == right.dtype and left.shape == right.shape and torch.equal(left, right)
    if isinstance(left, np.ndarray) and isinstance(right, np.ndarray):
        return left.dtype == right.dtype and left.shape == right.shape and np.array_equal(left, right)
    if isinstance(left, Mapping) and isinstance(right, Mapping):
        return set(left) == set(right) and all(_exact(left[key], right[key]) for key in left)
    if isinstance(left, (list, tuple)) and isinstance(right, (list, tuple)):
        return len(left) == len(right) and all(_exact(a, b) for a, b in zip(left, right, strict=True))
    if isinstance(left, float) and isinstance(right, float) and math.isnan(left) and math.isnan(right):
        return True
    return left == right


def _difference_paths(left: Any, right: Any, prefix: str = "") -> list[str]:
    if isinstance(left, Mapping) and isinstance(right, Mapping):
        result: list[str] = []
        for key in sorted(set(left).union(right)):
            path = f"{prefix}.{key}" if prefix else str(key)
            if key not in left or key not in right:
                result.append(path)
            else:
                result.extend(_difference_paths(left[key], right[key], path))
        return result
    if isinstance(left, (list, tuple)) and isinstance(right, (list, tuple)):
        if len(left) != len(right):
            return [prefix]
        result = []
        for index, (a, b) in enumerate(zip(left, right, strict=True)):
            result.extend(_difference_paths(a, b, f"{prefix}[{index}]"))
        return result
    return [] if _exact(left, right) else [prefix]


def _state_hash(state: Mapping[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name, tensor in state.items():
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(str(tuple(value.shape)).encode("ascii"))
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def _config_and_actor_audit(config_path: Path, output_roots: list[Path]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    from harl.utils.envs_tools import set_seed
    from marl.algorithms import EFFECTIVE_ACTION_ALGO_REGISTRY
    from marl.methods import derive_method
    from marl.specs import AGENTS, EFFECTIVE_ACTION_DIMS
    from train.train_harl_mappo_short import load_yaml_config, resolve_config

    base = load_yaml_config(config_path)
    configs = []
    actor_states = []
    obs_space = gym.spaces.Box(-np.inf, np.inf, shape=(288,), dtype=np.float32)
    action_space = gym.spaces.Box(0.0, 1.0, shape=(22,), dtype=np.float32)
    for (algorithm, critic_type, method_id), output_root in zip(METHODS, output_roots, strict=True):
        config = resolve_config(
            base,
            algorithm=algorithm,
            critic_type=critic_type,
            seed=7110,
            updates=1,
            episode_length=24,
            rollout_threads=2,
            output_dir=output_root,
            device="cpu",
            checkpoint_interval=1,
        )
        method = derive_method(algorithm, critic_type)
        if method.method_id != method_id:
            raise AssertionError(f"Unexpected method id for {(algorithm, critic_type)}.")
        configs.append(config)
        set_seed(config["seed"])
        actor_class = EFFECTIVE_ACTION_ALGO_REGISTRY[algorithm]
        states = []
        for agent in AGENTS:
            actor = actor_class(
                {**config["model"], **config["algo"]},
                obs_space,
                action_space,
                device=torch.device("cpu"),
                effective_action_dim=EFFECTIVE_ACTION_DIMS[agent],
            )
            states.append({name: value.detach().cpu().clone() for name, value in actor.actor.state_dict().items()})
        actor_states.append(states)

    pairwise_configs = {}
    pairwise_actors = {}
    for left_index in range(len(METHODS)):
        for right_index in range(left_index + 1, len(METHODS)):
            left_id = METHODS[left_index][2]
            right_id = METHODS[right_index][2]
            key = f"{left_id}__vs__{right_id}"
            differences = _difference_paths(configs[left_index], configs[right_index])
            pairwise_configs[key] = {
                "difference_paths": differences,
                "only_expected_differences": set(differences).issubset(EXPECTED_CONFIG_DIFFERENCES),
            }
            agent_results = []
            for agent_index in range(2):
                left = actor_states[left_index][agent_index]
                right = actor_states[right_index][agent_index]
                agent_results.append(
                    {
                        "agent": AGENTS[agent_index],
                        "exact": _exact(left, right),
                        "left_sha256": _state_hash(left),
                        "right_sha256": _state_hash(right),
                    }
                )
            pairwise_actors[key] = agent_results

    rejection = None
    try:
        derive_method("mappo", "hgta")
    except ValueError as exc:
        rejection = f"{type(exc).__name__}: {exc}"
    if rejection is None:
        raise AssertionError("MAPPO_HGTA was unexpectedly accepted.")

    result = {
        "supported_methods": [method_id for _, _, method_id in METHODS],
        "mappo_hgta_rejection": rejection,
        "pairwise_config_differences": pairwise_configs,
        "pairwise_initial_actor_parameters": pairwise_actors,
        "all_initial_actor_parameters_exact": all(
            item["exact"] for pair in pairwise_actors.values() for item in pair
        ),
        "all_config_differences_expected": all(
            item["only_expected_differences"] for item in pairwise_configs.values()
        ),
    }
    return result, configs


def _rng_isolation_audit(configs: list[Mapping[str, Any]]) -> dict[str, Any]:
    """Observe, without sampling or updating, RNG consumed by model construction."""
    from harl.utils.envs_tools import set_seed
    from marl.algorithms import EFFECTIVE_ACTION_ALGO_REGISTRY
    from marl.critics import build_critic
    from marl.specs import AGENTS, EFFECTIVE_ACTION_DIMS
    from marl.utils.rng_isolation import build_with_isolated_rng

    obs_space = gym.spaces.Box(-np.inf, np.inf, shape=(288,), dtype=np.float32)
    action_space = gym.spaces.Box(0.0, 1.0, shape=(22,), dtype=np.float32)
    state_space = gym.spaces.Box(-np.inf, np.inf, shape=(294,), dtype=np.float32)
    records = {}
    for (algorithm, critic_type, method_id), config in zip(METHODS, configs, strict=True):
        set_seed(config["seed"])
        actor_class = EFFECTIVE_ACTION_ALGO_REGISTRY[algorithm]
        for agent in AGENTS:
            actor_class(
                {**config["model"], **config["algo"]},
                obs_space,
                action_space,
                device=torch.device("cpu"),
                effective_action_dim=EFFECTIVE_ACTION_DIMS[agent],
            )
        after_actors = torch.get_rng_state().clone()
        critic_args = {**config["model"], **config["algo"]}
        if critic_type == "hgta":
            critic_args["hgta"] = dict(config["critic"]["hgta"])
        build_with_isolated_rng(
            lambda: build_critic(
                critic_type, critic_args, state_space, device=torch.device("cpu")
            ),
            seed=int(config["rng"]["critic_init_seed"]),
        )
        after_critic = torch.get_rng_state().clone()
        records[method_id] = {
            "after_actor_construction_sha256": hashlib.sha256(after_actors.numpy().tobytes()).hexdigest(),
            "after_critic_construction_sha256": hashlib.sha256(after_critic.numpy().tobytes()).hexdigest(),
            "critic_construction_changed_rng": not torch.equal(after_actors, after_critic),
        }
    pairwise = {}
    for left_index in range(len(METHODS)):
        for right_index in range(left_index + 1, len(METHODS)):
            left_id = METHODS[left_index][2]
            right_id = METHODS[right_index][2]
            pairwise[f"{left_id}__vs__{right_id}"] = {
                "after_actors_rng_exact": (
                    records[left_id]["after_actor_construction_sha256"]
                    == records[right_id]["after_actor_construction_sha256"]
                ),
                "after_critics_rng_exact": (
                    records[left_id]["after_critic_construction_sha256"]
                    == records[right_id]["after_critic_construction_sha256"]
                ),
            }
    return {"per_method": records, "pairwise": pairwise}


def _environment_audit(config: Mapping[str, Any]) -> dict[str, Any]:
    from marl.checkpointing.environment_state import environment_state_dict
    from marl.envs.harl_env_factory import make_harl_train_env

    snapshots = []
    resets = []
    for _algorithm, _critic_type, method_id in METHODS:
        env = make_harl_train_env(
            seed=int(config["seed"]["seed"]),
            n_rollout_threads=int(config["train"]["n_rollout_threads"]),
            scenario=str(config["env"]["scenario"]),
            experiment_case=str(config["env"]["experiment_case"]),
            worker_seed_stride=int(config["parallel"]["worker_seed_stride"]),
        )
        try:
            before_reset = environment_state_dict(env)
            reset_output = env.reset()
            after_reset = environment_state_dict(env)
            snapshots.append({"method_id": method_id, "before_reset": before_reset, "after_reset": after_reset})
            resets.append(reset_output)
        finally:
            env.close()

    pairwise = {}
    for left_index in range(len(METHODS)):
        for right_index in range(left_index + 1, len(METHODS)):
            key = f"{METHODS[left_index][2]}__vs__{METHODS[right_index][2]}"
            pairwise[key] = {
                "pre_reset_state_exact": _exact(snapshots[left_index]["before_reset"], snapshots[right_index]["before_reset"]),
                "reset_output_exact": _exact(resets[left_index], resets[right_index]),
                "post_reset_state_exact": _exact(snapshots[left_index]["after_reset"], snapshots[right_index]["after_reset"]),
            }
    return {
        "worker_count": 2,
        "worker_seeds": [7110, 8110],
        "pairwise": pairwise,
        "all_initial_environment_states_exact": all(
            value["pre_reset_state_exact"] and value["reset_output_exact"] and value["post_reset_state_exact"]
            for value in pairwise.values()
        ),
    }


def _artifact_audit(run_dirs: list[Path]) -> dict[str, Any]:
    from marl.checkpointing.training_checkpoint import (
        CheckpointCompatibilityError,
        _validate_method_metadata,
    )
    from marl.methods import derive_method

    checkpoints = []
    configs = []
    rows = []
    per_method = {}
    for (algorithm, critic_type, method_id), run_dir in zip(METHODS, run_dirs, strict=True):
        checkpoint = _load_checkpoint(run_dir / "checkpoints" / "final.pt")
        manifest = json.loads(
            (run_dir / "checkpoints" / "final.manifest.json").read_text(encoding="utf-8")
        )
        metadata = json.loads((run_dir / "run_metadata.json").read_text(encoding="utf-8"))
        config = json.loads((run_dir / "resolved_config.json").read_text(encoding="utf-8"))
        step_rows = _rows(run_dir / "metrics" / "step_metrics.csv")
        episode_rows = _rows(run_dir / "metrics" / "episode_metrics.csv")
        update_rows = _rows(run_dir / "metrics" / "update_metrics.csv")
        checkpoints.append(checkpoint)
        configs.append(config)
        rows.append(step_rows)
        audit = checkpoint["verification_state"]["algorithm_update_audit"]
        per_method[method_id] = {
            "run_dir": str(run_dir.resolve()),
            "method_metadata_exact": (
                checkpoint["algorithm_name"], checkpoint["critic_type"], checkpoint["method_id"]
            ) == (algorithm, critic_type, method_id),
            "metric_row_counts": {
                "step": len(step_rows), "episode": len(episode_rows), "update": len(update_rows)
            },
            "checkpoint_sections_present": all(
                key in checkpoint
                for key in (
                    "model_state", "optimizer_state", "normalizer_state", "rng_state",
                    "environment_state", "runner_state", "logger_state", "verification_state",
                    "compatibility", "provenance",
                )
            ),
            "global_step": int(checkpoint["global_step"]),
            "episodes_completed": int(checkpoint["episodes_completed"]),
            "agent_update_order": list(audit["agent_update_order"]),
            "agent_update_order_policy": str(audit["agent_update_order_policy"]),
            "happo_available": bool(audit["happo_available"]),
            "graph_metadata_present": bool(checkpoint.get("graph_metadata")),
            "rng_isolation_metadata_exact": all(
                checkpoint["compatibility"].get(key) == config["rng"][key]
                and manifest.get(key) == config["rng"][key]
                and metadata.get(key) == config["rng"][key]
                for key in (
                    "rng_isolation_version",
                    "critic_init_seed_rule",
                    "critic_init_seed",
                    "actor_sampling_rng_isolated_from_critic_init",
                )
            ),
            "stability_counters": checkpoint["verification_state"]["stability_counters"],
        }

    pairwise = {}
    for left_index in range(len(METHODS)):
        for right_index in range(left_index + 1, len(METHODS)):
            left_id = METHODS[left_index][2]
            right_id = METHODS[right_index][2]
            key = f"{left_id}__vs__{right_id}"
            left_verification = checkpoints[left_index]["verification_state"]
            right_verification = checkpoints[right_index]["verification_state"]
            ignored_step_fields = {"run_id"}
            left_step_projection = [
                {field: value for field, value in row.items() if field not in ignored_step_fields}
                for row in rows[left_index]
            ]
            right_step_projection = [
                {field: value for field, value in row.items() if field not in ignored_step_fields}
                for row in rows[right_index]
            ]
            pairwise[key] = {
                "actor_actions_exact": _exact(left_verification["actor_actions"], right_verification["actor_actions"]),
                "actor_action_log_probs_exact": _exact(
                    left_verification["actor_action_log_probs"], right_verification["actor_action_log_probs"]
                ),
                "critic_rewards_exact": _exact(left_verification["critic_rewards"], right_verification["critic_rewards"]),
                "post_rollout_environment_state_exact": _exact(
                    checkpoints[left_index]["environment_state"], checkpoints[right_index]["environment_state"]
                ),
                "step_log_projection_exact": _exact(left_step_projection, right_step_projection),
                "actor_action_mismatch_count": int(sum(
                    np.count_nonzero(np.asarray(a) != np.asarray(b))
                    for a, b in zip(
                        left_verification["actor_actions"],
                        right_verification["actor_actions"],
                        strict=True,
                    )
                )),
                "actor_action_max_abs_difference": float(max(
                    np.max(np.abs(np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)))
                    for a, b in zip(
                        left_verification["actor_actions"],
                        right_verification["actor_actions"],
                        strict=True,
                    )
                )),
                "config_difference_paths": _difference_paths(configs[left_index], configs[right_index]),
            }

    compatibility_matrix = {}
    for source_index, source in enumerate(checkpoints):
        source_id = METHODS[source_index][2]
        compatibility_matrix[source_id] = {}
        for target_algorithm, target_critic, target_id in METHODS:
            runner = SimpleNamespace(
                method=derive_method(target_algorithm, target_critic), fixed_order=True
            )
            try:
                _validate_method_metadata(source, runner)
            except CheckpointCompatibilityError as exc:
                compatibility_matrix[source_id][target_id] = {
                    "accepted": False, "error": f"{type(exc).__name__}: {exc}"
                }
            else:
                compatibility_matrix[source_id][target_id] = {"accepted": True, "error": None}

    return {
        "per_method": per_method,
        "pairwise_pre_update_trajectory": pairwise,
        "all_48_step_trajectories_exact": all(
            value["actor_actions_exact"]
            and value["actor_action_log_probs_exact"]
            and value["critic_rewards_exact"]
            and value["post_rollout_environment_state_exact"]
            and value["step_log_projection_exact"]
            for value in pairwise.values()
        ),
        "checkpoint_method_compatibility_matrix": compatibility_matrix,
        "correct_checkpoint_combinations_accepted": all(
            compatibility_matrix[method_id][method_id]["accepted"] for _, _, method_id in METHODS
        ),
        "all_wrong_checkpoint_combinations_rejected": all(
            not compatibility_matrix[source_id][target_id]["accepted"]
            for _, _, source_id in METHODS
            for _, _, target_id in METHODS
            if source_id != target_id
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--harl-source", type=Path, required=True)
    parser.add_argument("--runtime-path", type=Path, required=True)
    parser.add_argument("--run", action="append", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if len(args.run) != 3:
        raise ValueError("Exactly three --run arguments are required in MAPPO_MLP, HAPPO_MLP, HAPPO_HGTA order.")

    from train.train_harl_mappo_short import configure_import_paths

    configure_import_paths(args.harl_source, args.runtime_path)
    config_audit, configs = _config_and_actor_audit(
        args.config.resolve(), [run.resolve().parents[4] for run in args.run]
    )
    result = {
        "schema_version": "part18-three-method-readonly-audit-v1",
        "config_and_actor": config_audit,
        "model_construction_rng": _rng_isolation_audit(configs),
        "environment_initialization": _environment_audit(configs[0]),
        "artifacts": _artifact_audit([run.resolve() for run in args.run]),
    }
    result["hard_fairness_gate_passed"] = bool(
        result["config_and_actor"]["all_initial_actor_parameters_exact"]
        and result["config_and_actor"]["all_config_differences_expected"]
        and result["environment_initialization"]["all_initial_environment_states_exact"]
        and result["artifacts"]["all_48_step_trajectories_exact"]
    )
    encoded = json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(encoded, encoding="utf-8")
    print(encoded, end="")


if __name__ == "__main__":
    import multiprocessing

    multiprocessing.freeze_support()
    main()
