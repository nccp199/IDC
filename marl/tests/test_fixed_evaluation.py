"""Acceptance tests for fixed, deterministic dual-agent evaluation."""

from __future__ import annotations

import copy
import csv
import os
import shutil
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

from marl.evaluation.evaluator import FixedPolicyEvaluator
from marl.evaluation.fixed_scenario_suite import (
    SuiteIntegrityError,
    _scenario_payload,
    load_scenario,
    load_suite,
    restore_scenario_environment,
)
from marl.evaluation.metrics import (
    AGGREGATE_COLUMNS,
    AGGREGATE_METRICS,
    EVALUATION_EPISODE_COLUMNS,
    EVALUATION_STEP_COLUMNS,
)
from marl.evaluation.model_loader import (
    ModelCompatibilityError,
    load_model,
    validate_model_suite_compatibility,
)
from marl.logging import REWARD_COMPONENTS


def _artifact(name: str) -> Path:
    value = os.environ.get(name)
    if not value:
        pytest.skip(f"{name} is required for fixed-evaluation artifact acceptance.")
    path = Path(value).resolve()
    assert path.exists(), path
    return path


@pytest.fixture(scope="module")
def artifacts():
    suite_dir = _artifact("PART11_SUITE_DIR")
    run_dir = _artifact("PART11_RUN_DIR")
    checkpoint = _artifact("PART11_CHECKPOINT")
    primary = _artifact("PART11_PRIMARY_EVAL")
    repeat = _artifact("PART11_REPEAT_EVAL")
    suite = load_suite(suite_dir)
    return {
        "suite_dir": suite_dir,
        "suite": suite,
        "run_dir": run_dir,
        "checkpoint": checkpoint,
        "primary": primary,
        "repeat": repeat,
        "run_model": load_model(run_dir=run_dir, device="cpu", model_id="run-model"),
        "model_only": load_model(model_dir=run_dir / "models", device="cpu", model_id="model-only"),
        "checkpoint_model": load_model(checkpoint=checkpoint, device="cpu", model_id="checkpoint-model"),
    }


def _max_diff(left: Any, right: Any) -> float:
    if isinstance(left, np.ndarray) and isinstance(right, np.ndarray):
        if left.shape != right.shape:
            return float("inf")
        if left.dtype.kind in "fci" and right.dtype.kind in "fci":
            return float(np.max(np.abs(left.astype(np.float64) - right.astype(np.float64)))) if left.size else 0.0
        return 0.0 if np.array_equal(left, right) else float("inf")
    if torch.is_tensor(left) and torch.is_tensor(right):
        return _max_diff(left.detach().cpu().numpy(), right.detach().cpu().numpy())
    if isinstance(left, dict) and isinstance(right, dict):
        if set(left) != set(right):
            return float("inf")
        return max((_max_diff(left[key], right[key]) for key in left), default=0.0)
    if isinstance(left, (list, tuple)) and isinstance(right, (list, tuple)):
        if len(left) != len(right):
            return float("inf")
        return max((_max_diff(a, b) for a, b in zip(left, right, strict=True)), default=0.0)
    if isinstance(left, (int, float, np.number)) and isinstance(right, (int, float, np.number)):
        return abs(float(left) - float(right))
    return 0.0 if left == right else float("inf")


def _actor_actions(model, obs):
    outputs = []
    hidden = model.resolved_config["model"]["hidden_sizes"][-1]
    recurrent = model.resolved_config["model"]["recurrent_n"]
    with torch.no_grad():
        for agent_id, actor in enumerate(model.actors):
            action, _ = actor.act(
                obs[agent_id][None],
                np.zeros((1, recurrent, hidden), np.float32),
                np.ones((1, 1), np.float32),
                None,
                deterministic=True,
            )
            outputs.append(action.detach().cpu().numpy()[0])
    return np.stack(outputs)


def _csv(path: Path):
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def test_01_same_seed_scenario_generation_is_exact():
    kwargs = dict(
        suite_id="repeatability_probe",
        scenario_id="scenario_000_seed_02026",
        seed=2026,
        experiment_case="main",
        project_git_head=None,
        harl_git_head=None,
    )
    left = _scenario_payload(**kwargs)
    right = _scenario_payload(**kwargs)
    assert left["scenario_content_sha256"] == right["scenario_content_sha256"]
    assert _max_diff(left["initial_observation"], right["initial_observation"]) == 0
    assert _max_diff(left["initial_state"], right["initial_state"]) == 0
    assert _max_diff(left["environment_state"], right["environment_state"]) == 0
    assert _max_diff(left["external_curves"], right["external_curves"]) == 0


def test_02_different_seeds_change_tasks_or_servers_not_external_curves(artifacts):
    first = load_scenario(artifacts["suite"], artifacts["suite"]["scenario_ids"][0])
    second = load_scenario(artifacts["suite"], artifacts["suite"]["scenario_ids"][1])
    state_a, state_b = first["environment_state"], second["environment_state"]
    tasks_differ = _max_diff(state_a["base_environment"]["tasks"], state_b["base_environment"]["tasks"]) > 0
    servers_differ = _max_diff(state_a["model"]["server_parameters"], state_b["model"]["server_parameters"]) > 0
    assert tasks_differ or servers_differ
    for key in ("price", "carbon_factor", "pv", "grid_load_scale", "grid_reference_usep"):
        assert _max_diff(first["external_curves"][key], second["external_curves"][key]) == 0


def test_03_snapshot_round_trip_and_next_transition_are_exact(artifacts):
    scenario = load_scenario(artifacts["suite"], artifacts["suite"]["scenario_ids"][0])
    left, left_obs, left_state, _ = restore_scenario_environment(scenario)
    right, right_obs, right_state, _ = restore_scenario_environment(scenario)
    try:
        assert _max_diff(left_obs, right_obs) == 0
        assert _max_diff(left_state, right_state) == 0
        actions = np.full((2, 22), 0.5, np.float32)
        assert _max_diff(left.step(actions), right.step(actions)) == 0
    finally:
        left.close()
        right.close()


def test_04_same_model_same_scenario_and_checkpoint_trajectory_are_exact(artifacts):
    primary = [row for row in _csv(artifacts["primary"] / "step_metrics.csv") if row["scenario_id"] == "scenario_000_seed_02026"]
    repeat = _csv(artifacts["repeat"] / "step_metrics.csv")
    assert len(primary) == len(repeat) == 24
    ignored = {"evaluation_id", "model_id", "model_source", "model_sha256"}
    for left, right in zip(primary, repeat, strict=True):
        assert {key: value for key, value in left.items() if key not in ignored} == {
            key: value for key, value in right.items() if key not in ignored
        }


def test_05_scenario_load_order_is_independent(artifacts):
    suite = artifacts["suite"]
    forward = {scenario_id: load_scenario(suite, scenario_id)["scenario_content_sha256"] for scenario_id in suite["scenario_ids"]}
    reverse = {scenario_id: load_scenario(suite, scenario_id)["scenario_content_sha256"] for scenario_id in reversed(suite["scenario_ids"])}
    assert forward == reverse


def test_06_model_load_order_is_independent(artifacts):
    scenario = load_scenario(artifacts["suite"], artifacts["suite"]["scenario_ids"][0])
    obs = scenario["initial_observation"]
    models = [artifacts["model_only"], artifacts["checkpoint_model"]]
    forward = {model.model_id: _actor_actions(model, obs) for model in models}
    reverse = {model.model_id: _actor_actions(model, obs) for model in reversed(models)}
    assert all(_max_diff(forward[key], reverse[key]) == 0 for key in forward)


def test_07_model_only_and_checkpoint_actor_weights_match(artifacts):
    scenario = load_scenario(artifacts["suite"], artifacts["suite"]["scenario_ids"][0])
    assert _max_diff(
        _actor_actions(artifacts["model_only"], scenario["initial_observation"]),
        _actor_actions(artifacts["checkpoint_model"], scenario["initial_observation"]),
    ) == 0


def test_08_formal_evaluator_rejects_stochastic_default():
    with pytest.raises(ValueError, match="deterministic-only"):
        FixedPolicyEvaluator(deterministic=False)


def test_09_no_gradient_parameter_change_and_eval_mode(artifacts):
    manifest = __import__("json").loads((artifacts["primary"] / "evaluation_manifest.json").read_text(encoding="utf-8"))
    assert manifest["actor_parameters_unchanged"] is True
    assert manifest["actor_gradients_absent"] is True
    assert manifest["actors_in_eval_mode"] is True
    assert manifest["deterministic"] is True
    assert manifest["critic_metrics_available"] is False


def test_10_bess_signed_mapping_and_virtual_invariance(artifacts):
    for row in _csv(artifacts["primary"] / "step_metrics.csv"):
        assert float(row["bess_action_physical"]) == pytest.approx(
            2.0 * float(row["bess_action_padded_dim0"]) - 1.0, abs=1e-7
        )
    scenario = load_scenario(artifacts["suite"], artifacts["suite"]["scenario_ids"][0])
    left, *_ = restore_scenario_environment(scenario)
    right, *_ = restore_scenario_environment(scenario)
    try:
        idc = np.full(22, 0.5, np.float32)
        bess_a = np.zeros(22, np.float32); bess_a[0] = 0.7
        bess_b = np.ones(22, np.float32); bess_b[0] = 0.7
        assert _max_diff(left.step(np.stack((idc, bess_a))), right.step(np.stack((idc, bess_b)))) == 0
    finally:
        left.close(); right.close()


def test_11_reward_reconstruction_and_episode_sum(artifacts):
    steps = _csv(artifacts["primary"] / "step_metrics.csv")
    episodes = _csv(artifacts["primary"] / "episode_metrics.csv")
    for row in steps:
        assert sum(float(row[name]) for name in REWARD_COMPONENTS) == pytest.approx(float(row["reward_total"]), abs=1e-6)
        assert float(row["reward_returned"]) == pytest.approx(float(row["grid_adjusted_reward"]), abs=1e-6)
    for episode in episodes:
        selected = [row for row in steps if row["scenario_id"] == episode["scenario_id"]]
        assert sum(float(row["reward_returned"]) for row in selected) == pytest.approx(float(episode["episode_reward"]), abs=1e-9)


def test_12_each_model_scenario_uses_fresh_environment_state(artifacts):
    primary = [row for row in _csv(artifacts["primary"] / "step_metrics.csv") if row["scenario_id"] == "scenario_000_seed_02026"]
    repeat = _csv(artifacts["repeat"] / "step_metrics.csv")
    fields = ("bess_soc", "opf_cache_size", "mef_cache_size", "backlog_work", "finished_tasks_total")
    assert {field: primary[0][field] for field in fields} == {field: repeat[0][field] for field in fields}
    assert {field: primary[-1][field] for field in fields} == {field: repeat[-1][field] for field in fields}


@pytest.mark.parametrize(
    ("field", "bad"),
    (
        ("algorithm", "ppo"),
        ("agent_order", ["bess", "idc"]),
        ("observation_dims", [287, 288]),
        ("padded_action_dims", [22, 1]),
        ("effective_action_dims", [22, 22]),
        ("environment_config_fingerprint", "bad-fingerprint"),
    ),
)
def test_13_incompatible_model_is_rejected(artifacts, field, bad):
    model = replace(artifacts["run_model"], compatibility=copy.deepcopy(artifacts["run_model"].compatibility))
    model.compatibility[field] = bad
    with pytest.raises(ModelCompatibilityError):
        validate_model_suite_compatibility(model, artifacts["suite"])


def test_14_corrupted_scenario_file_is_rejected(artifacts, tmp_path):
    damaged = tmp_path / "suite"
    shutil.copytree(artifacts["suite_dir"], damaged)
    suite = load_suite(damaged)
    scenario_path = damaged / suite["scenario_files"][0]
    with scenario_path.open("r+b") as stream:
        stream.seek(max(scenario_path.stat().st_size // 2, 1))
        original = stream.read(1)
        stream.seek(-1, 1)
        stream.write(bytes([original[0] ^ 0xFF]))
    with pytest.raises(SuiteIntegrityError, match="hash differs"):
        load_scenario(suite, suite["scenario_ids"][0])


def test_15_output_schema_counts_and_finite_critical_fields(artifacts):
    with (artifacts["primary"] / "step_metrics.csv").open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream); steps = list(reader); assert tuple(reader.fieldnames) == EVALUATION_STEP_COLUMNS
    with (artifacts["primary"] / "episode_metrics.csv").open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream); episodes = list(reader); assert tuple(reader.fieldnames) == EVALUATION_EPISODE_COLUMNS
    with (artifacts["primary"] / "aggregate_metrics.csv").open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream); aggregates = list(reader); assert tuple(reader.fieldnames) == AGGREGATE_COLUMNS
    assert len(steps) == 72 and len(episodes) == 3 and len(aggregates) == len(AGGREGATE_METRICS)
    assert all(sum(row["is_terminal"].lower() == "true" for row in steps if row["scenario_id"] == scenario) == 1 for scenario in artifacts["suite"]["scenario_ids"])
    for row in episodes:
        for key in ("episode_reward", "final_completion_rate", "final_backlog", "cost_sum", "carbon_kg_sum", "final_bess_soc", "opf_success_rate", "mef_success_rate"):
            assert np.isfinite(float(row[key]))


def test_16_formal_evaluation_is_isolated_from_legacy_single_agent_scripts():
    root = Path(__file__).resolve().parents[2]
    formal = "\n".join(
        (root / path).read_text(encoding="utf-8")
        for path in (
            "eval/eval_harl_mappo_fixed.py",
            "marl/evaluation/fixed_scenario_suite.py",
            "marl/evaluation/model_loader.py",
            "marl/evaluation/evaluator.py",
        )
    )
    for forbidden in ("eval_base", "eval_nn_reuse", "stable_baselines3", "legacy.nn_reuse"):
        assert forbidden not in formal

