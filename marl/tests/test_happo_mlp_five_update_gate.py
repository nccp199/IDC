"""Artifact-backed Part-14 HAPPO+MLP five-update acceptance gate."""

from __future__ import annotations

import csv
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[2]
FORMAL_ROOT = PROJECT_ROOT / "runs" / "part14_happo_mlp_gate"
RESUME_ROOT = PROJECT_ROOT / "runs" / "part14_happo_mlp_resume_gate"
PART13_DEFAULT = (
    PROJECT_ROOT
    / "runs/part13_happo_mlp_smoke/gym/idc_bess_padding/happo/"
    "idc_bess_happo_short_HAPPO_MLP/seed-07110-2026-07-29-12-09-07"
)
EVAL_DEFAULT = PROJECT_ROOT / "evaluations" / "part14_happo_mlp_gate"
TIMING_FIELDS = {
    "run_id",
    "wall_time_seconds",
    "rollout_time_seconds",
    "update_time_seconds",
    "steps_per_second",
}


def _completed_run(root: Path) -> Path:
    candidates = []
    for metadata in root.glob("**/run_metadata.json"):
        try:
            if json.loads(metadata.read_text(encoding="utf-8"))["status"] == "completed":
                candidates.append(metadata.parent)
        except (KeyError, json.JSONDecodeError):
            continue
    if not candidates:
        pytest.skip(f"No completed Part-14 artifact under {root}")
    return max(candidates, key=lambda path: path.stat().st_mtime)


def _path_from_env(name: str, fallback: Path | None = None) -> Path:
    value = os.environ.get(name)
    path = Path(value).resolve() if value else fallback
    if path is None or not path.exists():
        pytest.skip(f"Artifact path is unavailable: {name}")
    return path


@pytest.fixture(scope="module")
def formal() -> Path:
    return _path_from_env("PART14_FORMAL_RUN", _completed_run(FORMAL_ROOT))


@pytest.fixture(scope="module")
def resume() -> Path:
    return _path_from_env("PART14_RESUME_RUN", _completed_run(RESUME_ROOT))


@pytest.fixture(scope="module")
def part13() -> Path:
    return _path_from_env("PART13_HAPPO_RUN", PART13_DEFAULT)


@pytest.fixture(scope="module")
def evaluation() -> Path:
    return _path_from_env("PART14_EVAL_DIR", EVAL_DEFAULT)


def _csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def _load(path: Path) -> dict[str, Any]:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _assert_common_rows_equal(
    left: list[dict[str, str]],
    right: list[dict[str, str]],
    *,
    excluded: set[str] = frozenset(),
) -> None:
    assert len(left) == len(right)
    common = set(left[0]).intersection(right[0]).difference(excluded)
    for left_row, right_row in zip(left, right, strict=True):
        assert {key: left_row[key] for key in common} == {
            key: right_row[key] for key in common
        }


def _assert_equal(left: Any, right: Any, path: str = "root") -> None:
    if isinstance(left, dict) and isinstance(right, dict):
        common = set(left).intersection(right).difference(TIMING_FIELDS)
        for key in common:
            _assert_equal(left[key], right[key], f"{path}.{key}")
        return
    if torch.is_tensor(left) and torch.is_tensor(right):
        assert torch.equal(left, right), path
        return
    if isinstance(left, np.ndarray) and isinstance(right, np.ndarray):
        assert left.shape == right.shape and left.dtype == right.dtype, path
        assert np.array_equal(left, right, equal_nan=True), path
        return
    if isinstance(left, (list, tuple)) and isinstance(right, (list, tuple)):
        assert len(left) == len(right), path
        for index, (left_value, right_value) in enumerate(zip(left, right, strict=True)):
            _assert_equal(left_value, right_value, f"{path}[{index}]")
        return
    if isinstance(left, float) and isinstance(right, float):
        if np.isnan(left) and np.isnan(right):
            return
    assert left == right, path


def test_five_updates_have_complete_rows_and_fixed_sequential_updates(formal: Path) -> None:
    steps = _csv(formal / "metrics/step_metrics.csv")
    episodes = _csv(formal / "metrics/episode_metrics.csv")
    updates = _csv(formal / "metrics/update_metrics.csv")
    assert (len(steps), len(episodes), len(updates)) == (240, 10, 5)
    assert [int(row["global_step"]) for row in steps] == list(range(1, 241))
    assert [int(row["update"]) for row in updates] == [1, 2, 3, 4, 5]
    for update in range(1, 6):
        for worker in (0, 1):
            assert sum(
                int(row["update"]) == update and int(row["worker_id"]) == worker
                for row in steps
            ) == 24
    for index, row in enumerate(updates, start=1):
        assert row["agent_update_order"] == "idc,bess"
        assert row["first_updated_agent"] == "idc"
        assert row["second_updated_agent"] == "bess"
        assert (row["actor_update_count_idc"], row["actor_update_count_bess"], row["critic_update_count"]) == ("1", "1", "1")
        assert (row["idc_actor_optimizer_step_count"], row["bess_actor_optimizer_step_count"], row["critic_optimizer_step_count"]) == (str(index), str(index), str(index))


def test_factor_is_reset_positive_finite_and_exactly_reconstructable(formal: Path) -> None:
    updates = _csv(formal / "metrics/update_metrics.csv")
    for index, row in enumerate(updates, start=1):
        assert row["happo_factor_initial_exactly_one_fraction"] == "1.0"
        assert row["happo_factor_after_idc_exactly_one_fraction"] == "0.0"
        assert row["happo_factor_nonfinite_count"] == "0"
        assert row["happo_factor_nonpositive_count"] == "0"
        assert float(row["happo_factor_after_idc_min"]) > 0.0
        assert float(row["happo_factor_final_min"]) > 0.0
        assert row["happo_factor_after_idc_reconstruction_max_diff"] == "0.0"
        assert row["happo_factor_final_reconstruction_max_diff"] == "0.0"
        payload = _load(formal / f"checkpoints/update_{index:06d}.pt")
        audit = payload["verification_state"]["algorithm_update_audit"]["factor_audit"]
        assert audit["initial"].shape == (24, 2, 1)
        assert np.array_equal(audit["initial"], np.ones((24, 2, 1), np.float32))
        assert np.array_equal(
            audit["initial"] * audit["idc_effective_ratio"], audit["after_idc"]
        )
        assert np.array_equal(
            audit["after_idc"] * audit["bess_physical_ratio"], audit["final"]
        )


def test_bess_virtual_contributions_and_buffer_mutations_are_zero(formal: Path) -> None:
    zero_fields = {
        "bess_effective_physical_ratio_max_diff",
        "bess_virtual_mean_grad_norm",
        "bess_virtual_log_std_grad_norm",
        "bess_virtual_entropy_optimization_contribution",
        "bess_happo_factor_effective_physical_max_diff",
        "bess_happo_virtual_factor_contribution",
        "idc_action_buffer_mutation_max_diff",
        "bess_action_buffer_mutation_max_diff",
        "idc_log_prob_buffer_mutation_max_diff",
        "bess_log_prob_buffer_mutation_max_diff",
        "critic_return_buffer_mutation_max_diff",
        "advantage_mutation_max_diff",
    }
    for row in _csv(formal / "metrics/update_metrics.csv"):
        assert all(float(row[field]) == 0.0 for field in zero_fields)
        assert int(row["bess_effective_action_dim"]) == 1
        assert int(row["bess_padded_action_dim"]) == 22


def test_actor_critic_updates_and_environment_stability(formal: Path) -> None:
    updates = _csv(formal / "metrics/update_metrics.csv")
    assert all(
        float(row[field]) > 0.0
        for row in updates
        for field in (
            "idc_actor_parameter_delta_norm",
            "bess_actor_parameter_delta_norm",
            "critic_parameter_delta_norm",
        )
    )
    steps = _csv(formal / "metrics/step_metrics.csv")
    assert max(abs(float(row["reward_reconstruction_error"])) for row in steps) == 0.0
    assert all(row["opf_success"] == "True" and row["mef_success"] == "True" for row in steps)
    metadata = json.loads((formal / "run_metadata.json").read_text(encoding="utf-8"))
    assert metadata["status"] == "completed" and metadata["completed_updates"] == 5
    assert set(metadata["stability_counters"].values()) == {0}


def test_update_one_is_an_exact_part13_regression(formal: Path, part13: Path) -> None:
    for name, excluded in (
        ("step_metrics.csv", {"run_id"}),
        ("episode_metrics.csv", {"run_id"}),
        ("update_metrics.csv", TIMING_FIELDS),
    ):
        _assert_common_rows_equal(
            _csv(part13 / "metrics" / name),
            _csv(formal / "metrics" / name)[: 48 if name == "step_metrics.csv" else 2 if name == "episode_metrics.csv" else 1],
            excluded=excluded,
        )
    old = _load(part13 / "checkpoints/update_000001.pt")
    new = _load(formal / "checkpoints/update_000001.pt")
    for key in ("model_state", "optimizer_state", "normalizer_state", "rng_state", "logger_state", "verification_state"):
        _assert_equal(old[key], new[key], key)


def test_update_three_resume_is_exact_through_update_five(formal: Path, resume: Path) -> None:
    for name, excluded in (
        ("step_metrics.csv", {"run_id"}),
        ("episode_metrics.csv", {"run_id"}),
        ("update_metrics.csv", TIMING_FIELDS),
    ):
        continuous = [row for row in _csv(formal / "metrics" / name) if int(row["update"]) >= 4]
        _assert_common_rows_equal(continuous, _csv(resume / "metrics" / name), excluded=excluded)
    continuous = _load(formal / "checkpoints/update_000005.pt")
    restored = _load(resume / "checkpoints/update_000005.pt")
    for key in ("model_state", "optimizer_state", "normalizer_state", "rng_state", "environment_state", "runner_state", "logger_state", "verification_state"):
        _assert_equal(continuous[key], restored[key], key)


def test_all_checkpoints_are_complete_and_final_equals_update_five(formal: Path) -> None:
    for index in range(1, 6):
        checkpoint = formal / f"checkpoints/update_{index:06d}.pt"
        manifest = json.loads(checkpoint.with_suffix(".manifest.json").read_text())
        assert hashlib.sha256(checkpoint.read_bytes()).hexdigest() == manifest["sha256"]
        assert (manifest["algorithm_name"], manifest["critic_type"], manifest["method_id"]) == ("happo", "mlp", "HAPPO_MLP")
        payload = _load(checkpoint)
        assert (payload["global_step"], payload["episodes_completed"]) == (48 * index, 2 * index)
        assert set(payload["optimizer_state"]) == {"actor_agent0_optimizer_state", "actor_agent1_optimizer_state", "critic_optimizer_state"}
    assert (formal / "checkpoints/final.pt").read_bytes() == (
        formal / "checkpoints/update_000005.pt"
    ).read_bytes()
    assert not list(formal.rglob("*.tmp"))


def test_fixed_evaluations_load_and_repeat_exactly(evaluation: Path) -> None:
    manifest = json.loads((evaluation / "evaluation_manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "completed" and manifest["failure_count"] == 0
    assert manifest["deterministic"] and manifest["evaluation_worker_count"] == 1
    assert manifest["actor_parameters_unchanged"] and manifest["actor_gradients_absent"]
    assert manifest["actors_in_eval_mode"]
    for name, excluded in (
        ("step_metrics.csv", {"model_id"}),
        ("episode_metrics.csv", {"model_id", "runtime_seconds"}),
    ):
        rows = _csv(evaluation / name)
        for update in (1, 3, 5):
            selected = [row for row in rows if int(row["training_update"]) == update]
            model_ids = sorted({row["model_id"] for row in selected})
            first = [row for row in selected if row["model_id"] == model_ids[0]]
            second = [row for row in selected if row["model_id"] == model_ids[1]]
            _assert_common_rows_equal(first, second, excluded=excluded)


def test_no_child_worker_process_is_left_behind() -> None:
    assert multiprocessing.active_children() == []

