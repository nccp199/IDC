"""Read-only Part-17 checkpoint and metric auditor.

This script never constructs an environment, calls an optimizer, or mutates a
training artifact.  It derives the post-clip gradient used by Adam at update t
from consecutive first-moment states:

    g_t = (m_t - beta1 * m_(t-1)) / (1 - beta1)
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import torch


EXPECTED_HASHES = {
    "feature_schema_hash": "1944df31af4e68360173cabea4f47d2d13ca8167e576fdb64ea2f76b1ed82b23",
    "topology_hash": "0a85394f385693afc351ec03831fd01bc1b75f31019e4f68bbd2c7bf5115c9f1",
    "graph_schema_hash": "32a642d211c7421634a7e4e65afb54529232ce4da19f47ddfd9fdf52e9ae4b26",
}

MODULE_PREFIXES = {
    "type_projection": "encoder.type_projection.",
    "relation_layer_0": "encoder.relation_layers.0.",
    "relation_layer_1": "encoder.relation_layers.1.",
    "forecast_encoder": "encoder.forecast_encoder.",
    "value_head": "encoder.value_head.",
}


def _rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def _load(path: Path) -> dict[str, Any]:
    return torch.load(path, map_location="cpu", weights_only=False)


def _parameter_entries(
    model_state: Mapping[str, torch.Tensor], optimizer_state: Mapping[str, Any]
) -> list[tuple[str, torch.Tensor, int, Mapping[str, Any]]]:
    ids = [item for group in optimizer_state["param_groups"] for item in group["params"]]
    model_items = list(model_state.items())
    if len(model_items) != len(ids):
        # Actor state dicts contain two non-parameter action-bound buffers.
        model_items = [
            item
            for item in model_items
            if not item[0].endswith("action_low") and not item[0].endswith("action_high")
        ]
    if len(model_items) != len(ids):
        raise AssertionError(
            f"Cannot map {len(model_items)} model tensors to {len(ids)} optimizer parameters."
        )
    result = []
    for (name, tensor), parameter_id in zip(model_items, ids, strict=True):
        state = optimizer_state["state"][parameter_id]
        if tuple(tensor.shape) != tuple(state["exp_avg"].shape):
            raise AssertionError(f"Optimizer/model shape mismatch for {name}.")
        result.append((name, tensor, parameter_id, state))
    return result


def _derived_gradients(
    current: Mapping[str, Any], previous: Mapping[str, Any] | None
) -> dict[str, torch.Tensor]:
    entries = _parameter_entries(current["model"], current["optimizer"])
    prior = (
        {}
        if previous is None
        else {name: state for name, _, _, state in _parameter_entries(previous["model"], previous["optimizer"])}
    )
    beta1 = float(current["optimizer"]["param_groups"][0]["betas"][0])
    gradients = {}
    for name, _, _, state in entries:
        prior_moment = torch.zeros_like(state["exp_avg"]) if previous is None else prior[name]["exp_avg"]
        gradients[name] = (state["exp_avg"] - beta1 * prior_moment) / (1.0 - beta1)
    return gradients


def _first_step_delta(entry: tuple[str, torch.Tensor, int, Mapping[str, Any]], group: Mapping[str, Any]) -> torch.Tensor:
    _, tensor, _, state = entry
    step = float(state["step"])
    beta1, beta2 = (float(value) for value in group["betas"])
    step_size = float(group["lr"]) / (1.0 - beta1**step)
    denominator = state["exp_avg_sq"].sqrt() / math.sqrt(1.0 - beta2**step) + float(group["eps"])
    return step_size * state["exp_avg"] / denominator


def _module_audit(
    current: Mapping[str, Any], previous: Mapping[str, Any] | None
) -> dict[str, Any]:
    entries = _parameter_entries(current["model"], current["optimizer"])
    gradients = _derived_gradients(current, previous)
    prior_model = {} if previous is None else dict(previous["model"])
    group = current["optimizer"]["param_groups"][0]
    result = {}
    for module, prefix in MODULE_PREFIXES.items():
        selected = [entry for entry in entries if entry[0].startswith(prefix)]
        module_gradients = [gradients[name] for name, _, _, _ in selected]
        if previous is None:
            deltas = [_first_step_delta(entry, group) for entry in selected]
            delta_source = "reconstructed_from_adam_step_1"
        else:
            deltas = [tensor - prior_model[name] for name, tensor, _, _ in selected]
            delta_source = "checkpoint_parameter_difference"
        result[module] = {
            "parameter_tensors": len(selected),
            "parameters": sum(tensor.numel() for _, tensor, _, _ in selected),
            "gradient_tensors": len(module_gradients),
            "nonzero_gradient_tensors": sum(
                bool(torch.count_nonzero(gradient)) for gradient in module_gradients
            ),
            "gradient_max_abs": max(float(gradient.abs().max()) for gradient in module_gradients),
            "gradient_norm": math.sqrt(
                sum(float(torch.sum(gradient.double().square())) for gradient in module_gradients)
            ),
            "parameter_delta_norm": math.sqrt(
                sum(float(torch.sum(delta.double().square())) for delta in deltas)
            ),
            "delta_source": delta_source,
            "all_finite": all(
                bool(torch.isfinite(value).all())
                for value in [*module_gradients, *deltas]
            ),
        }
    return result


def _relation_audit(current: Mapping[str, Any], previous: Mapping[str, Any] | None) -> dict[str, Any]:
    gradients = _derived_gradients(current, previous)
    result = {}
    for layer in (0, 1):
        relation_values: dict[str, float] = {}
        score_values: dict[str, float] = {}
        base = f"encoder.relation_layers.{layer}."
        for name, gradient in gradients.items():
            if name.startswith(base + "relation_value."):
                relation_values[name.rsplit(".", 1)[-1]] = float(gradient.abs().sum())
            if name.startswith(base + "relation_key.") or name.startswith(base + "relation_bias."):
                relation = name.rsplit(".", 1)[-1]
                score_values[relation] = score_values.get(relation, 0.0) + float(gradient.abs().sum())
        result[f"layer_{layer}"] = {
            "relation_count": len(relation_values),
            "all_relation_value_gradients_nonzero": all(value > 0.0 for value in relation_values.values()),
            "nonzero_score_relation_count": sum(value > 0.0 for value in score_values.values()),
            "zero_score_relations": sorted(key for key, value in score_values.items() if value == 0.0),
            "all_finite": all(math.isfinite(value) for value in [*relation_values.values(), *score_values.values()]),
        }
    return result


def _bess_virtual_audit(
    current: Mapping[str, Any], previous: Mapping[str, Any] | None
) -> dict[str, Any]:
    gradients = _derived_gradients(current, previous)
    entries = _parameter_entries(current["model"], current["optimizer"])
    prior_model = {} if previous is None else dict(previous["model"])
    group = current["optimizer"]["param_groups"][0]
    target_names = (
        "act.action_out.log_std",
        "act.action_out.fc_mean.weight",
        "act.action_out.fc_mean.bias",
    )
    virtual_gradients = []
    effective_gradients = []
    virtual_deltas = []
    for entry in entries:
        name, tensor, _, _ = entry
        if name not in target_names:
            continue
        gradient = gradients[name]
        delta = (
            _first_step_delta(entry, group)
            if previous is None
            else tensor - prior_model[name]
        )
        effective_gradients.append(gradient[0].reshape(-1))
        virtual_gradients.append(gradient[1:].reshape(-1))
        virtual_deltas.append(delta[1:].reshape(-1))
    virtual_gradient = torch.cat(virtual_gradients)
    effective_gradient = torch.cat(effective_gradients)
    virtual_delta = torch.cat(virtual_deltas)
    return {
        "virtual_gradient_max_abs": float(virtual_gradient.abs().max()),
        "virtual_gradient_norm": float(virtual_gradient.double().norm()),
        "virtual_parameter_delta_max_abs": float(virtual_delta.abs().max()),
        "virtual_parameter_delta_norm": float(virtual_delta.double().norm()),
        "effective_dim_gradient_norm": float(effective_gradient.double().norm()),
        "all_finite": bool(
            torch.isfinite(virtual_gradient).all()
            and torch.isfinite(virtual_delta).all()
            and torch.isfinite(effective_gradient).all()
        ),
    }


def _recursive_differences(left: Any, right: Any, path: str = "") -> list[str]:
    if torch.is_tensor(left) and torch.is_tensor(right):
        return [] if left.dtype == right.dtype and left.shape == right.shape and torch.equal(left, right) else [path]
    if isinstance(left, np.ndarray) and isinstance(right, np.ndarray):
        return [] if left.dtype == right.dtype and left.shape == right.shape and np.array_equal(left, right) else [path]
    if isinstance(left, Mapping) and isinstance(right, Mapping):
        differences = []
        if set(left) != set(right):
            differences.append(f"{path}.keys")
        for key in set(left).intersection(right):
            differences.extend(_recursive_differences(left[key], right[key], f"{path}.{key}"))
        return differences
    if isinstance(left, (list, tuple)) and isinstance(right, (list, tuple)):
        differences = [] if len(left) == len(right) else [f"{path}.length"]
        for index, (a, b) in enumerate(zip(left, right)):
            differences.extend(_recursive_differences(a, b, f"{path}[{index}]"))
        return differences
    if isinstance(left, float) and isinstance(right, float) and math.isnan(left) and math.isnan(right):
        return []
    return [] if left == right else [path]


def _finite_numeric_rows(rows: Iterable[Mapping[str, str]]) -> int:
    nonfinite = 0
    for row in rows:
        for value in row.values():
            try:
                number = float(value)
            except (TypeError, ValueError):
                continue
            nonfinite += int(not math.isfinite(number))
    return nonfinite


def _number(value: str) -> float:
    normalized = value.strip().lower()
    if normalized == "true":
        return 1.0
    if normalized == "false":
        return 0.0
    return float(value)


def audit(run_dir: Path) -> dict[str, Any]:
    checkpoints = [_load(run_dir / "checkpoints" / f"update_{index:06d}.pt") for index in range(1, 6)]
    final = _load(run_dir / "checkpoints" / "final.pt")
    update_rows = _rows(run_dir / "metrics" / "update_metrics.csv")
    step_rows = _rows(run_dir / "metrics" / "step_metrics.csv")
    episode_rows = _rows(run_dir / "metrics" / "episode_metrics.csv")
    if (len(update_rows), len(step_rows), len(episode_rows)) != (5, 240, 10):
        raise AssertionError("Unexpected metric row counts.")

    checkpoint_audit = []
    module_audit = []
    relation_audit = []
    bess_virtual_audit = []
    previous_critic = None
    previous_bess_actor = None
    for index, checkpoint in enumerate(checkpoints, start=1):
        graph = checkpoint["graph_metadata"]
        required = {
            "actor_agent0_state_dict",
            "actor_agent1_state_dict",
            "critic_state_dict",
        }
        optimizer_required = {
            "actor_agent0_optimizer_state",
            "actor_agent1_optimizer_state",
            "critic_optimizer_state",
        }
        critic = {
            "model": checkpoint["model_state"]["critic_state_dict"],
            "optimizer": checkpoint["optimizer_state"]["critic_optimizer_state"],
        }
        checkpoint_audit.append(
            {
                "update": index,
                "saved_at_update": int(checkpoint["saved_at_update"]),
                "global_step": int(checkpoint["global_step"]),
                "episodes_completed": int(checkpoint["episodes_completed"]),
                "model_states_complete": required.issubset(checkpoint["model_state"]),
                "optimizer_states_complete": optimizer_required.issubset(checkpoint["optimizer_state"]),
                "rng_state_present": bool(checkpoint["rng_state"]),
                "environment_state_present": bool(checkpoint["environment_state"]),
                "graph_metadata_present": bool(graph),
                "graph_hashes_match": all(graph[key] == value for key, value in EXPECTED_HASHES.items()),
                "node_count": int(graph["node_count"]),
                "edge_count": int(graph["edge_count"]),
                "node_type_count": len(graph["node_type_order"]),
                "relation_type_count": len(graph["relation_type_order"]),
                "file_bytes": (run_dir / "checkpoints" / f"update_{index:06d}.pt").stat().st_size,
            }
        )
        module_audit.append({"update": index, **_module_audit(critic, previous_critic)})
        relation_audit.append({"update": index, **_relation_audit(critic, previous_critic)})
        bess_actor = {
            "model": checkpoint["model_state"]["actor_agent1_state_dict"],
            "optimizer": checkpoint["optimizer_state"]["actor_agent1_optimizer_state"],
        }
        bess_virtual_audit.append(
            {"update": index, **_bess_virtual_audit(bess_actor, previous_bess_actor)}
        )
        previous_critic = critic
        previous_bess_actor = bess_actor

    training_sections = (
        "model_state",
        "optimizer_state",
        "normalizer_state",
        "rng_state",
        "environment_state",
        "runner_state",
        "verification_state",
        "graph_metadata",
        "compatibility",
    )
    final_differences = {
        section: _recursive_differences(checkpoints[-1][section], final[section], section)
        for section in training_sections
    }

    update_projection = []
    for row in update_rows:
        update_projection.append(
            {
                key: row[key]
                for key in (
                    "update", "global_step", "episodes_completed", "first_updated_agent",
                    "second_updated_agent", "happo_factor_initial_min", "happo_factor_initial_mean",
                    "happo_factor_initial_max", "happo_factor_after_idc_min",
                    "happo_factor_after_idc_mean", "happo_factor_after_idc_max",
                    "happo_factor_final_min", "happo_factor_final_mean", "happo_factor_final_max",
                    "happo_factor_nonfinite_count", "happo_factor_nonpositive_count",
                    "happo_factor_after_idc_reconstruction_max_diff",
                    "happo_factor_final_reconstruction_max_diff", "actor_update_count_idc",
                    "actor_update_count_bess", "critic_update_count", "idc_actor_parameter_delta_norm",
                    "bess_actor_parameter_delta_norm", "critic_parameter_delta_norm",
                    "bess_virtual_mean_grad_norm", "bess_virtual_log_std_grad_norm",
                    "bess_virtual_entropy_optimization_contribution",
                    "bess_effective_physical_ratio_max_diff",
                    "bess_happo_virtual_factor_contribution", "value_loss", "critic_grad_norm",
                    "critic_grad_norm_postclip", "value_prediction_mean", "value_prediction_std",
                    "return_mean", "return_std", "advantage_mean", "advantage_std",
                    "explained_variance", "graph_input_nonfinite_count",
                    "node_embedding_nonfinite_count", "attention_nonfinite_count",
                    "graph_embedding_nonfinite_count", "forecast_embedding_nonfinite_count",
                    "value_prediction_nonfinite_count", "opf_success_rate", "mef_success_rate",
                    "shared_reward_max_diff", "update_time_seconds", "rollout_time_seconds",
                )
            }
        )

    bess_mapping_error = max(
        abs(float(row["bess_action_physical"]) - (2.0 * float(row["bess_action_padded_dim0"]) - 1.0))
        for row in step_rows
    )
    physical = {
        "numeric_nonfinite_count": _finite_numeric_rows([*step_rows, *episode_rows, *update_rows]),
        "reward_reconstruction_max_abs": max(abs(float(row["reward_reconstruction_error"])) for row in step_rows),
        "shared_reward_max_abs": max(abs(float(row["shared_reward_abs_diff"])) for row in step_rows),
        "opf_success_rate": sum(_number(row["opf_success"]) for row in step_rows) / len(step_rows),
        "mef_success_rate": sum(_number(row["mef_success"]) for row in step_rows) / len(step_rows),
        "bess_dim0_mapping_max_abs_error": bess_mapping_error,
        "bess_padded_dim0_min": min(float(row["bess_action_padded_dim0"]) for row in step_rows),
        "bess_padded_dim0_max": max(float(row["bess_action_padded_dim0"]) for row in step_rows),
        "voltage_violation_count": sum(int(_number(row["voltage_violation"])) for row in step_rows),
        "line_violation_count": sum(int(_number(row["line_violation"])) for row in step_rows),
        "grid_security_penalty_sum": sum(float(row["grid_security_penalty"]) for row in step_rows),
        "task_overflow_sum": sum(float(row["overflow_work"]) for row in step_rows),
        "bess_infeasible_request_count": sum(int(_number(row["bess_invalid_request"])) for row in step_rows),
        "bess_infeasible_request_max_kw": max(float(row["bess_infeasible_request_power_kW"]) for row in step_rows),
        "bess_soc_boundary_count": sum(
            int(float(row["bess_soc"]) <= 0.1 + 1e-9 or float(row["bess_soc"]) >= 0.9 - 1e-9)
            for row in step_rows
        ),
        "opf_cache_hit_rate": sum(_number(row["opf_cache_hit"]) for row in step_rows) / len(step_rows),
        "mef_cache_hit_rate": sum(_number(row["mef_cache_hit"]) for row in step_rows) / len(step_rows),
    }

    return {
        "run_dir": str(run_dir.resolve()),
        "row_counts": {"step": len(step_rows), "episode": len(episode_rows), "update": len(update_rows)},
        "checkpoint_audit": checkpoint_audit,
        "module_audit": module_audit,
        "relation_audit": relation_audit,
        "bess_virtual_parameter_audit": bess_virtual_audit,
        "updates": update_projection,
        "physical_chain": physical,
        "final_vs_update5_training_differences": final_differences,
        "final_vs_update5_all_training_sections_exact": all(not values for values in final_differences.values()),
        "unavailable_without_training_instrumentation": [
            "per-update value_prediction_min/max",
            "per-update return_min/max",
            "per-update advantage_min/max",
            "separate attention-score versus attention-weight nonfinite counts",
            "separate IDC/BESS/critic wall-clock update durations",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = audit(args.run_dir.resolve())
    encoded = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(encoded, encoding="utf-8")
    print(encoded, end="")


if __name__ == "__main__":
    main()
