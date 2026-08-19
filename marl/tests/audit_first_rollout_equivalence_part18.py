"""Collect, without updating parameters, the first 48 transitions of all methods."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import pickle
import sys
from pathlib import Path
from typing import Any, Mapping

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

PHYSICAL_INFO_FIELDS = (
    "P_IDC_kW",
    "grid_power_kW",
    "bess_soc",
    "actual_task_load_mean",
    "P_IT",
    "P_cooling",
    "grid_min_voltage_pu",
    "grid_max_voltage_pu",
    "grid_max_line_loading_percent",
    "grid_network_loss_mw",
    "grid_opf_success",
    "grid_mef_success",
    "grid_voltage_violation_count",
    "grid_line_overload_count",
    "grid_lmp",
    "grid_mef_plus",
    "grid_mef_minus",
    "unfinished_task_count",
    "finished_task_count",
    "backlog_work",
    "Q",
    "completed_work",
    "total_completed_work",
)


def _digest(value: Any) -> str:
    return hashlib.sha256(pickle.dumps(value, protocol=5)).hexdigest()


def _numeric_max_difference(left: Any, right: Any) -> float:
    if torch.is_tensor(left) and torch.is_tensor(right):
        if left.shape != right.shape:
            return float("inf")
        if not left.numel():
            return 0.0
        return float(torch.max(torch.abs(left.detach().cpu().double() - right.detach().cpu().double())))
    if isinstance(left, np.ndarray) and isinstance(right, np.ndarray):
        if left.shape != right.shape:
            return float("inf")
        if not left.size:
            return 0.0
        if left.dtype.kind in "biufc" and right.dtype.kind in "biufc":
            return float(np.max(np.abs(left.astype(np.float64) - right.astype(np.float64))))
        return 0.0 if np.array_equal(left, right) else float("inf")
    if isinstance(left, Mapping) and isinstance(right, Mapping):
        if set(left) != set(right):
            return float("inf")
        return max(
            (_numeric_max_difference(left[key], right[key]) for key in left),
            default=0.0,
        )
    if isinstance(left, (list, tuple)) and isinstance(right, (list, tuple)):
        if len(left) != len(right):
            return float("inf")
        return max(
            (_numeric_max_difference(a, b) for a, b in zip(left, right, strict=True)),
            default=0.0,
        )
    if isinstance(left, (bool, int, float, np.number)) and isinstance(
        right, (bool, int, float, np.number)
    ):
        if isinstance(left, (float, np.floating)) and isinstance(right, (float, np.floating)):
            if math.isnan(float(left)) and math.isnan(float(right)):
                return 0.0
        return abs(float(left) - float(right))
    return 0.0 if left == right else float("inf")


def _actor_states(runner: Any) -> list[dict[str, torch.Tensor]]:
    return [
        {
            name: value.detach().cpu().clone()
            for name, value in actor.actor.state_dict().items()
        }
        for actor in runner.actor
    ]


def _collect_method(
    *,
    algorithm: str,
    critic_type: str,
    method_id: str,
    base_config: Mapping[str, Any],
    output_root: Path,
) -> dict[str, Any]:
    from marl.checkpointing.environment_state import environment_state_dict
    from marl.envs.harl_env_factory import make_harl_train_env
    from marl.runners.idc_mappo_runner import IDCOnPolicyMARunner
    from train.train_harl_mappo_short import resolve_config, split_runner_config

    resolved = resolve_config(
        base_config,
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
    env = make_harl_train_env(
        seed=7110,
        n_rollout_threads=2,
        scenario=resolved["env"]["scenario"],
        experiment_case=resolved["env"]["experiment_case"],
        worker_seed_stride=resolved["parallel"]["worker_seed_stride"],
    )
    runner = None
    try:
        pre_reset_state = environment_state_dict(env)
        args, algo_args, env_args = split_runner_config(resolved)
        runner = IDCOnPolicyMARunner(
            args, algo_args, env_args, train_envs=env, eval_envs=None
        )
        env = None
        initial_actor_states = _actor_states(runner)
        actor_optimizer_states = [
            copy.deepcopy(actor.actor_optimizer.state_dict()) for actor in runner.actor
        ]
        critic_state = {
            name: value.detach().cpu().clone()
            for name, value in runner.critic.state_dict().items()
        }
        runner.warmup()
        reset_obs = np.stack(
            [buffer.obs[0].copy() for buffer in runner.actor_buffer], axis=1
        )
        reset_state = runner.critic_buffer.share_obs[0].copy()
        post_reset_state = environment_state_dict(runner.envs)

        runner.prep_rollout()
        actions = []
        log_probs = []
        values = []
        observations = [reset_obs.copy()]
        centralized_states = [reset_state.copy()]
        rewards = []
        dones = []
        physical_actions = []
        physical_fields = {key: [] for key in PHYSICAL_INFO_FIELDS}
        info_hashes = []
        environment_state_hashes = []
        for step in range(24):
            value, action, log_prob, rnn_states, rnn_states_critic = runner.collect(step)
            obs, share_obs, reward, done, infos, available_actions = runner.envs.step(action)
            data = (
                obs,
                share_obs,
                reward,
                done,
                infos,
                available_actions,
                value,
                action,
                log_prob,
                rnn_states,
                rnn_states_critic,
            )
            runner.insert(data)
            actions.append(action.copy())
            log_probs.append(log_prob.copy())
            values.append(value.copy())
            observations.append(obs.copy())
            centralized_states.append(share_obs[:, 0].copy())
            rewards.append(reward.copy())
            dones.append(done.copy())
            physical_actions.append((2.0 * action[:, 1, 0] - 1.0).copy())
            for key in PHYSICAL_INFO_FIELDS:
                physical_fields[key].append(
                    np.asarray([worker_infos[0].get(key) for worker_infos in infos])
                )
            info_hashes.append(_digest(infos))
            environment_state_hashes.append(_digest(environment_state_dict(runner.envs)))

        stacked_actions = np.stack(actions)
        return {
            "method_id": method_id,
            "resolved_config": resolved,
            "run_dir": str(Path(runner.run_dir).resolve()),
            "initial_actor_states": initial_actor_states,
            "actor_optimizer_states": actor_optimizer_states,
            "critic_state": critic_state,
            "pre_reset_environment_state": pre_reset_state,
            "post_reset_environment_state": post_reset_state,
            "reset_observations": reset_obs,
            "reset_centralized_state": reset_state,
            "observations": np.stack(observations),
            "centralized_states": np.stack(centralized_states),
            "actions": stacked_actions,
            "action_log_probs": np.stack(log_probs),
            "critic_values": np.stack(values),
            "rewards": np.stack(rewards),
            "dones": np.stack(dones),
            "bess_physical_actions": np.stack(physical_actions),
            "physical_fields": {
                key: np.stack(value) for key, value in physical_fields.items()
            },
            "info_hashes": info_hashes,
            "environment_state_hashes": environment_state_hashes,
            "stochastic_action_unique_rows": int(
                np.unique(stacked_actions.reshape(-1, stacked_actions.shape[-1]), axis=0).shape[0]
            ),
        }
    finally:
        if runner is not None:
            runner.close()
        elif env is not None:
            env.close()


def _pairwise(left: Mapping[str, Any], right: Mapping[str, Any]) -> dict[str, Any]:
    fields = {
        "initial_actor_parameters": left["initial_actor_states"],
        "actor_optimizer_initial_state": left["actor_optimizer_states"],
        "initial_environment_state": left["pre_reset_environment_state"],
        "reset_observations": left["reset_observations"],
        "reset_centralized_state_294": left["reset_centralized_state"],
        "post_reset_environment_state": left["post_reset_environment_state"],
        "all_observations": left["observations"],
        "all_centralized_states_294": left["centralized_states"],
        "idc_actions_22": left["actions"][:, :, 0, :],
        "bess_padded_actions_22": left["actions"][:, :, 1, :],
        "bess_physical_action_dim0": left["bess_physical_actions"],
        "action_log_probs": left["action_log_probs"],
        "rewards": left["rewards"],
        "dones": left["dones"],
        "physical_fields": left["physical_fields"],
        "all_info_fields": left["info_hashes"],
        "task_server_grid_environment_states_each_step": left[
            "environment_state_hashes"
        ],
    }
    right_fields = {
        "initial_actor_parameters": right["initial_actor_states"],
        "actor_optimizer_initial_state": right["actor_optimizer_states"],
        "initial_environment_state": right["pre_reset_environment_state"],
        "reset_observations": right["reset_observations"],
        "reset_centralized_state_294": right["reset_centralized_state"],
        "post_reset_environment_state": right["post_reset_environment_state"],
        "all_observations": right["observations"],
        "all_centralized_states_294": right["centralized_states"],
        "idc_actions_22": right["actions"][:, :, 0, :],
        "bess_padded_actions_22": right["actions"][:, :, 1, :],
        "bess_physical_action_dim0": right["bess_physical_actions"],
        "action_log_probs": right["action_log_probs"],
        "rewards": right["rewards"],
        "dones": right["dones"],
        "physical_fields": right["physical_fields"],
        "all_info_fields": right["info_hashes"],
        "task_server_grid_environment_states_each_step": right[
            "environment_state_hashes"
        ],
    }
    maximums = {
        key: _numeric_max_difference(value, right_fields[key])
        for key, value in fields.items()
    }
    return {
        "left": left["method_id"],
        "right": right["method_id"],
        "max_absolute_differences": maximums,
        "all_required_trajectory_fields_exact": all(value == 0.0 for value in maximums.values()),
        "allowed_critic_value_max_absolute_difference": _numeric_max_difference(
            left["critic_values"], right["critic_values"]
        ),
        "critic_parameter_structures_equal": list(left["critic_state"]) == list(right["critic_state"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--harl-source", type=Path, required=True)
    parser.add_argument("--runtime-path", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    from train.train_harl_mappo_short import configure_import_paths, load_yaml_config

    configure_import_paths(args.harl_source, args.runtime_path)
    base_config = load_yaml_config(args.config)
    records = [
        _collect_method(
            algorithm=algorithm,
            critic_type=critic_type,
            method_id=method_id,
            base_config=base_config,
            output_root=args.output_root.resolve(),
        )
        for algorithm, critic_type, method_id in METHODS
    ]
    comparisons = [
        _pairwise(records[left], records[right])
        for left, right in ((0, 1), (0, 2), (1, 2))
    ]
    result = {
        "schema_version": "part18-first-rollout-equivalence-after-rng-fix-v1",
        "seed": 7110,
        "rollout_threads": 2,
        "episode_length": 24,
        "transitions": 48,
        "device": "cpu",
        "updates_executed": 0,
        "methods": [
            {
                "method_id": record["method_id"],
                "run_dir": record["run_dir"],
                "rng": record["resolved_config"]["rng"],
                "actions_shape": list(record["actions"].shape),
                "stochastic_action_unique_rows": record["stochastic_action_unique_rows"],
            }
            for record in records
        ],
        "pairwise": comparisons,
        "actor_exploration_remains_stochastic": all(
            record["stochastic_action_unique_rows"] > 2 for record in records
        ),
        "all_three_methods_first_rollout_exact": all(
            comparison["all_required_trajectory_fields_exact"]
            for comparison in comparisons
        ),
        "allowed_differences": [
            "critic_values",
            "critic_parameters",
            "critic_intermediate_representations",
            "advantages_and_returns_not_computed",
        ],
    }
    encoded = json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(encoded, encoding="utf-8")
    print(encoded, end="")
    if not result["all_three_methods_first_rollout_exact"]:
        raise SystemExit(2)


if __name__ == "__main__":
    import multiprocessing

    multiprocessing.freeze_support()
    main()
