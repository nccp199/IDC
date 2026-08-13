"""Deterministic, fresh-environment evaluation for fixed scenario suites."""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from marl.evaluation.fixed_scenario_suite import load_scenario, load_suite, restore_scenario_environment
from marl.evaluation.metrics import EVALUATION_SCHEMA_VERSION, EvaluationMetricsWriter
from marl.evaluation.model_loader import LoadedModel, validate_model_suite_compatibility
from marl.logging import CanonicalMetricBuilder


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _t2n(value: torch.Tensor) -> np.ndarray:
    return value.detach().cpu().numpy()


class FixedPolicyEvaluator:
    """Run one or more actor pairs under one algorithm-neutral scenario protocol."""

    def __init__(self, *, device: str = "cpu", deterministic: bool = True) -> None:
        if device != "cpu":
            raise ValueError("Fixed evaluation requires CPU.")
        if deterministic is not True:
            raise ValueError("Formal fixed evaluation is deterministic-only.")
        self.device = torch.device("cpu")
        self.deterministic = True

    def _evaluate_episode(
        self,
        *,
        model: LoadedModel,
        scenario: Mapping[str, Any],
        evaluation_id: str,
    ) -> tuple[list[dict[str, Any]], dict[str, Any], float, dict[str, Any]]:
        env, obs, _share_obs, available_actions = restore_scenario_environment(scenario)
        builder = CanonicalMetricBuilder(
            run_id=evaluation_id,
            seed=int(scenario["scenario_seed"]),
            episode_length=int(scenario["episode_length"]),
        )
        hidden_size = int(model.resolved_config["model"]["hidden_sizes"][-1])
        recurrent_n = int(model.resolved_config["model"]["recurrent_n"])
        rnn_states = [np.zeros((1, recurrent_n, hidden_size), dtype=np.float32) for _ in model.actors]
        masks = [np.ones((1, 1), dtype=np.float32) for _ in model.actors]
        rows: list[dict[str, Any]] = []
        terminal_count = 0
        electrical_trajectory = {
            "bus_vm_pu": [],
            "bus_p_mw": [],
            "bus_q_mvar": [],
            "bus_lmp": [],
            "bus_lam_q": [],
            "incident_branch_max_loading_percent": [],
            "line_loading_percent": [],
            "transformer_loading_percent": [],
            "opf_success": [],
            "network_loss_mw": [],
        }
        started = time.perf_counter()
        try:
            for actor in model.actors:
                actor.actor.eval()
            for step in range(int(scenario["episode_length"])):
                action_list = []
                with torch.no_grad():
                    for agent_id, actor in enumerate(model.actors):
                        available = None
                        if available_actions is not None:
                            available = np.asarray(available_actions)[agent_id][None]
                        action, next_rnn = actor.act(
                            obs[agent_id][None],
                            rnn_states[agent_id],
                            masks[agent_id],
                            available,
                            deterministic=True,
                        )
                        rnn_states[agent_id] = _t2n(next_rnn)
                        action_list.append(_t2n(action)[0].astype(np.float32, copy=False))
                actions = np.stack(action_list)
                if actions.shape != (2, 22) or not np.isfinite(actions).all():
                    raise ValueError(f"Deterministic actions must be finite (2,22), got {actions.shape}.")
                if np.any(actions < 0.0) or np.any(actions > 1.0):
                    raise ValueError("Deterministic bounded Box actions are outside [0,1].")
                next_obs, _next_state, rewards, dones, infos, available_actions = env.step(actions)
                terminated = bool(infos[0].get("terminated", False))
                truncated = bool(infos[0].get("truncated", False))
                is_terminal = bool(np.all(dones))
                terminal_count += int(is_terminal)
                if is_terminal != bool(terminated or truncated):
                    raise RuntimeError("Done and info terminal semantics differ.")
                if step < int(scenario["episode_length"]) - 1 and is_terminal:
                    raise RuntimeError(f"Scenario terminated early at step {step}.")
                expected_physical = 2.0 * float(actions[1, 0]) - 1.0
                if abs(float(infos[0]["bess_raw_action"]) - expected_physical) > 1e-7:
                    raise ValueError(
                        "BESS physical action does not match the existing "
                        "signed mapping 2 * padded_dim0 - 1."
                    )
                row = builder.build_step(
                    info=infos[0],
                    actions=actions,
                    episode_step=step,
                    idc_reward=float(rewards[0, 0]),
                    bess_reward=float(rewards[1, 0]),
                    terminated=terminated,
                    truncated=truncated,
                )
                row["_idc_action"] = actions[0].copy()
                rows.append(row)
                electrical_trajectory["bus_vm_pu"].append(
                    np.asarray(infos[0]["grid_bus_vm_pu"], dtype=np.float64)
                )
                electrical_trajectory["bus_p_mw"].append(
                    np.asarray(infos[0]["grid_bus_p_mw"], dtype=np.float64)
                )
                electrical_trajectory["bus_q_mvar"].append(
                    np.asarray(infos[0]["grid_bus_q_mvar"], dtype=np.float64)
                )
                electrical_trajectory["bus_lmp"].append(
                    np.asarray(infos[0]["grid_bus_lmp"], dtype=np.float64)
                )
                electrical_trajectory["bus_lam_q"].append(
                    np.asarray(infos[0]["grid_bus_lam_q"], dtype=np.float64)
                )
                electrical_trajectory[
                    "incident_branch_max_loading_percent"
                ].append(
                    np.asarray(
                        infos[0][
                            "grid_bus_incident_branch_max_loading_percent"
                        ],
                        dtype=np.float64,
                    )
                )
                electrical_trajectory["line_loading_percent"].append(
                    np.asarray(
                        infos[0]["grid_line_loading_percent"], dtype=np.float64
                    )
                )
                electrical_trajectory["transformer_loading_percent"].append(
                    np.asarray(
                        infos[0]["grid_transformer_loading_percent"],
                        dtype=np.float64,
                    )
                )
                electrical_trajectory["opf_success"].append(
                    bool(infos[0]["grid_opf_success"])
                )
                electrical_trajectory["network_loss_mw"].append(
                    float(infos[0]["grid_network_loss_mw"])
                )
                obs = np.asarray(next_obs, dtype=np.float32)
                if is_terminal:
                    masks = [np.zeros((1, 1), dtype=np.float32) for _ in model.actors]
            runtime = time.perf_counter() - started
            if terminal_count != 1 or not bool(rows[-1]["is_terminal"]):
                raise RuntimeError(f"Expected one terminal at step 23, got {terminal_count}.")
            episode = builder.aggregate_episode(rows)
            if abs(float(episode["episode_reward"]) - sum(float(row["reward_returned"]) for row in rows)) > 1e-9:
                raise ValueError("Episode reward does not equal the 24 step team rewards.")
            artifact = {
                key: (
                    np.stack(value).tolist()
                    if key not in {"opf_success", "network_loss_mw"}
                    else list(value)
                )
                for key, value in electrical_trajectory.items()
            }
            return rows, episode, runtime, artifact
        finally:
            env.close()

    def evaluate(
        self,
        *,
        suite_dir: str | Path,
        models: Sequence[LoadedModel],
        output_dir: str | Path,
        evaluation_id: str,
        scenario_ids: Sequence[str] | None = None,
    ) -> dict[str, Any]:
        if not models or len({model.model_id for model in models}) != len(models):
            raise ValueError("Evaluation requires one or more uniquely named models.")
        suite = load_suite(suite_dir)
        selected_scenarios = list(scenario_ids or suite["scenario_ids"])
        if not selected_scenarios or len(set(selected_scenarios)) != len(selected_scenarios):
            raise ValueError("Selected scenario IDs must be non-empty and unique.")
        unknown = sorted(set(selected_scenarios).difference(suite["scenario_ids"]))
        if unknown:
            raise ValueError(f"Selected scenario IDs are not in the suite: {unknown}.")
        for model in models:
            validate_model_suite_compatibility(model, suite)
        output = Path(output_dir).resolve()
        writer = EvaluationMetricsWriter(output)
        (output / "logs").mkdir(exist_ok=False)
        manifest_path = output / "evaluation_manifest.json"
        started_at = datetime.now(timezone.utc).isoformat()
        parameter_hashes_before = {model.model_id: model.parameter_sha256() for model in models}
        manifest: dict[str, Any] = {
            "evaluation_schema_version": EVALUATION_SCHEMA_VERSION,
            "evaluation_id": evaluation_id,
            "status": "running",
            "started_at": started_at,
            "suite_id": suite["suite_id"],
            "suite_type": suite["suite_type"],
            "suite_sha256": suite["suite_sha256"],
            "scenario_ids": selected_scenarios,
            "scenario_seeds": [
                suite["scenario_seeds"][list(suite["scenario_ids"]).index(scenario_id)]
                for scenario_id in selected_scenarios
            ],
            "deterministic": True,
            "device": "cpu",
            "evaluation_worker_count": 1,
            "critic_metrics_available": False,
            "environment_config_fingerprint": suite["environment_config_fingerprint"],
            "reward_config_fingerprint": suite["reward_config_fingerprint"],
            "data_config_fingerprint": suite["data_config_fingerprint"],
            "project_git_head": suite["project_git_head"],
            "harl_git_head": suite["harl_git_head"],
            "models": [
                {
                    "model_id": model.model_id,
                    "model_source": model.source_description,
                    "source_kind": model.source_kind,
                    "model_sha256": model.model_sha256,
                    "training_run_id": model.training_run_id,
                    "training_seed": model.training_seed,
                    "training_update": model.training_update,
                    "compatibility": model.compatibility,
                }
                for model in models
            ],
        }
        _write_json(manifest_path, manifest)
        evaluation_started = time.perf_counter()
        failure_count = 0
        try:
            for scenario_id in selected_scenarios:
                load_started = time.perf_counter()
                scenario = load_scenario(suite, scenario_id)
                scenario_load_seconds = time.perf_counter() - load_started
                for model in models:
                    episode_started = time.perf_counter()
                    try:
                        steps, episode, runtime, electrical = self._evaluate_episode(
                            model=model, scenario=scenario, evaluation_id=evaluation_id
                        )
                        writer.write_success(
                            evaluation_id=evaluation_id,
                            suite=suite,
                            model=model,
                            scenario=scenario,
                            steps=steps,
                            episode=episode,
                            scenario_load_seconds=scenario_load_seconds,
                            runtime_seconds=runtime,
                        )
                        electrical_dir = output / "electrical_trajectories"
                        electrical_dir.mkdir(exist_ok=True)
                        artifact_name = (
                            f"{model.model_id}__{scenario['scenario_id']}.json"
                        )
                        _write_json(electrical_dir / artifact_name, electrical)
                    except Exception as exc:
                        failure_count += 1
                        writer.write_failure(
                            evaluation_id=evaluation_id,
                            suite=suite,
                            model=model,
                            scenario=scenario,
                            error=exc,
                            scenario_load_seconds=scenario_load_seconds,
                            runtime_seconds=time.perf_counter() - episode_started,
                        )
            writer.finalize(evaluation_id=evaluation_id, suite=suite, models=models)
            after = {model.model_id: model.parameter_sha256() for model in models}
            unchanged = after == parameter_hashes_before
            grads_absent = all(
                parameter.grad is None
                for model in models
                for actor in model.actors
                for parameter in actor.actor.parameters()
            )
            eval_modes = all(not actor.actor.training for model in models for actor in model.actors)
            if not unchanged or not grads_absent or not eval_modes:
                raise RuntimeError("Evaluation changed actor parameters/gradients or actor eval mode.")
            manifest.update(
                {
                    "status": "completed" if failure_count == 0 else "completed_with_failures",
                    "completed_at": datetime.now(timezone.utc).isoformat(),
                    "total_evaluation_seconds": time.perf_counter() - evaluation_started,
                    "failure_count": failure_count,
                    "actor_parameters_unchanged": unchanged,
                    "actor_gradients_absent": grads_absent,
                    "actors_in_eval_mode": eval_modes,
                    "step_metrics": str(writer.step_path),
                    "episode_metrics": str(writer.episode_path),
                    "aggregate_metrics_csv": str(writer.aggregate_csv_path),
                    "aggregate_metrics_json": str(writer.aggregate_json_path),
                }
            )
            _write_json(manifest_path, manifest)
            return manifest
        except BaseException as exc:
            writer.close()
            manifest.update(
                {
                    "status": "failed",
                    "completed_at": datetime.now(timezone.utc).isoformat(),
                    "failure_type": type(exc).__name__,
                    "failure_message": str(exc),
                }
            )
            _write_json(manifest_path, manifest)
            raise
