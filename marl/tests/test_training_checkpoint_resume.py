"""Acceptance checks for versioned post-update MAPPO training checkpoints.

The real equivalence tests consume the three bounded Part-9 run artifacts via
environment variables.  They never start additional MAPPO updates.
"""

from __future__ import annotations

import copy
import csv
import json
import math
import os
import random
import shutil
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import pytest
import torch

from harl.algorithms.critics.v_critic import VCritic
from marl.algorithms import EffectiveActionMAPPO
from marl.checkpointing.training_checkpoint import (
    CheckpointIntegrityError,
    TrainingCheckpointManager,
    _compatibility_differences,
)
from marl.specs.state_specs import CENTRALIZED_STATE_DIM


def _artifact(name: str) -> Path:
    value = os.environ.get(name)
    if not value:
        pytest.skip(f"{name} is required for the real Part-9 artifact acceptance.")
    path = Path(value).resolve()
    if not path.exists():
        pytest.fail(f"Acceptance artifact does not exist: {path}")
    return path


@pytest.fixture(scope="module")
def checkpoints() -> dict[str, Any]:
    paths = {
        "continuous": _artifact("PART9_CONTINUOUS_CHECKPOINT"),
        "resumed": _artifact("PART9_RESUMED_CHECKPOINT"),
        "parent": _artifact("PART9_PARENT_CHECKPOINT"),
    }
    return {
        "paths": paths,
        **{
            name: torch.load(path, map_location="cpu", weights_only=False)
            for name, path in paths.items()
        },
    }


def _max_difference(left: Any, right: Any, *, ignored: set[str] | None = None) -> float:
    ignored = ignored or set()
    if torch.is_tensor(left) and torch.is_tensor(right):
        if left.shape != right.shape:
            return float("inf")
        if not left.numel():
            return 0.0
        return float(torch.max(torch.abs(left.detach().cpu() - right.detach().cpu())).item())
    if isinstance(left, np.ndarray) and isinstance(right, np.ndarray):
        if left.shape != right.shape:
            return float("inf")
        if not left.size:
            return 0.0
        if left.dtype.kind in "fci" and right.dtype.kind in "fci":
            return float(np.max(np.abs(left.astype(np.float64) - right.astype(np.float64))))
        return 0.0 if np.array_equal(left, right) else float("inf")
    if isinstance(left, dict) and isinstance(right, dict):
        keys = (set(left) | set(right)) - ignored
        if not keys.issubset(left) or not keys.issubset(right):
            return float("inf")
        return max((_max_difference(left[key], right[key], ignored=ignored) for key in keys), default=0.0)
    if isinstance(left, (list, tuple)) and isinstance(right, (list, tuple)):
        if len(left) != len(right):
            return float("inf")
        return max((_max_difference(a, b, ignored=ignored) for a, b in zip(left, right, strict=True)), default=0.0)
    if isinstance(left, (float, int, np.number)) and isinstance(right, (float, int, np.number)):
        if isinstance(left, (float, np.floating)) and isinstance(right, (float, np.floating)) and math.isnan(float(left)) and math.isnan(float(right)):
            return 0.0
        return abs(float(left) - float(right))
    return 0.0 if left == right else float("inf")


def _build_models(config: dict[str, Any]):
    args = {**config["model"], **config["algo"]}
    obs_space = gym.spaces.Box(-np.inf, np.inf, shape=(288,), dtype=np.float32)
    action_space = gym.spaces.Box(0.0, 1.0, shape=(22,), dtype=np.float32)
    state_space = gym.spaces.Box(
        -np.inf, np.inf, shape=(CENTRALIZED_STATE_DIM,), dtype=np.float32
    )
    actors = [
        EffectiveActionMAPPO(args, obs_space, action_space, effective_action_dim=22),
        EffectiveActionMAPPO(args, obs_space, action_space, effective_action_dim=1),
    ]
    return actors, VCritic(args, state_space)


def _fixed_outputs(models, checkpoint):
    actors, critic = models
    state = checkpoint["model_state"]
    actors[0].actor.load_state_dict(state["actor_agent0_state_dict"], strict=True)
    actors[1].actor.load_state_dict(state["actor_agent1_state_dict"], strict=True)
    critic.critic.load_state_dict(state["critic_state_dict"], strict=True)
    obs = np.linspace(-0.25, 0.25, 288, dtype=np.float32)[None]
    cent_obs = np.linspace(
        -0.5, 0.5, CENTRALIZED_STATE_DIM, dtype=np.float32
    )[None]
    rnn = np.zeros((1, 1, 32), dtype=np.float32)
    masks = np.ones((1, 1), dtype=np.float32)
    with torch.no_grad():
        actions = [actor.act(obs, rnn, masks, deterministic=True)[0] for actor in actors]
        value = critic.get_values(cent_obs, rnn, masks)[0]
    return actions, value


def test_01_model_weight_round_trip_and_fixed_outputs(checkpoints):
    config_path = checkpoints["paths"]["continuous"].parents[1] / "resolved_config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    left = _fixed_outputs(_build_models(config), checkpoints["continuous"])
    right = _fixed_outputs(_build_models(config), checkpoints["resumed"])
    assert _max_difference(left, right) == 0.0
    assert _max_difference(
        checkpoints["continuous"]["model_state"], checkpoints["resumed"]["model_state"]
    ) == 0.0


def test_02_bess_effective_action_mask_is_preserved(checkpoints):
    compatibility = checkpoints["resumed"]["compatibility"]
    assert compatibility["dims"]["padded_action"] == [22, 22]
    assert compatibility["dims"]["effective_action"] == [22, 1]
    assert compatibility["dims"]["virtual_action"] == [0, 21]
    assert compatibility["effective_action_mask"]["masks"][1] == [1.0] + [0.0] * 21
    tampered = copy.deepcopy(compatibility)
    tampered["dims"]["effective_action"][1] = 22
    assert _compatibility_differences(compatibility, tampered)


def test_03_optimizer_round_trip_is_exact(checkpoints):
    assert _max_difference(
        checkpoints["continuous"]["optimizer_state"], checkpoints["resumed"]["optimizer_state"]
    ) == 0.0
    for state in checkpoints["resumed"]["optimizer_state"].values():
        assert state["param_groups"]
        assert state["state"]
        assert all("lr" in group for group in state["param_groups"])


def test_04_python_numpy_torch_rng_round_trip(checkpoints):
    state = checkpoints["parent"]["rng_state"]
    random.setstate(state["python"])
    np.random.set_state(state["numpy_global"])
    torch.set_rng_state(state["torch_cpu"])
    expected = (random.random(), np.random.random(4), torch.rand(4))
    random.setstate(state["python"])
    np.random.set_state(state["numpy_global"])
    torch.set_rng_state(state["torch_cpu"])
    actual = (random.random(), np.random.random(4), torch.rand(4))
    assert _max_difference(expected, actual) == 0.0


def test_05_environment_and_next_episode_state_are_exact(checkpoints):
    assert _max_difference(
        checkpoints["continuous"]["environment_state"],
        checkpoints["resumed"]["environment_state"],
        ignored={"action_space_rng", "observation_space_rng", "np_random"},
    ) == 0.0
    assert _max_difference(
        checkpoints["continuous"]["runner_state"]["rollout_state"],
        checkpoints["resumed"]["runner_state"]["rollout_state"],
    ) == 0.0


def test_06_update2_actions_rewards_and_transition_outputs_are_exact(checkpoints):
    left = checkpoints["continuous"]["verification_state"]
    right = checkpoints["resumed"]["verification_state"]
    assert _max_difference(left["actor_actions"], right["actor_actions"]) == 0.0
    assert _max_difference(left["critic_rewards"], right["critic_rewards"]) == 0.0
    assert _max_difference(left["actor_action_log_probs"], right["actor_action_log_probs"]) == 0.0


def test_07_exact_training_continuation_equivalence(checkpoints):
    left = checkpoints["continuous"]
    right = checkpoints["resumed"]
    for key in ("model_state", "optimizer_state", "normalizer_state", "runner_state"):
        assert _max_difference(left[key], right[key]) == 0.0, key
    ignored = {"run_id", "wall_time_seconds", "rollout_time_seconds", "update_time_seconds", "steps_per_second"}
    assert _max_difference(
        left["verification_state"]["last_update_metrics"],
        right["verification_state"]["last_update_metrics"],
        ignored=ignored,
    ) == 0.0
    assert _max_difference(left["verification_state"]["actor_diagnostics"], right["verification_state"]["actor_diagnostics"]) == 0.0


def test_08_resumed_csv_numbering_continues(checkpoints):
    run_dir = checkpoints["paths"]["resumed"].parents[1]
    with (run_dir / "metrics" / "step_metrics.csv").open(encoding="utf-8", newline="") as stream:
        first_step = next(csv.DictReader(stream))
    with (run_dir / "metrics" / "update_metrics.csv").open(encoding="utf-8", newline="") as stream:
        update = next(csv.DictReader(stream))
    assert (int(first_step["update"]), int(first_step["global_step"]), int(first_step["episode_id"])) == (2, 25, 1)
    assert (int(update["update"]), int(update["global_step"]), int(update["episodes_completed"])) == (2, 48, 2)


def test_09_model_only_file_is_rejected_as_resume(checkpoints):
    model_file = checkpoints["paths"]["continuous"].parents[1] / "models" / "actor_agent0.pt"
    checkpoint, manifest = TrainingCheckpointManager._resolve_source(model_file)
    with pytest.raises(CheckpointIntegrityError, match="model-only"):
        TrainingCheckpointManager._verify_manifest(checkpoint, manifest)


def test_10_corrupt_checkpoint_fails_integrity_check(checkpoints, tmp_path):
    source = checkpoints["paths"]["continuous"]
    checkpoint = tmp_path / "copy.pt"
    manifest = tmp_path / "copy.manifest.json"
    shutil.copyfile(source, checkpoint)
    shutil.copyfile(source.with_name(source.stem + ".manifest.json"), manifest)
    with checkpoint.open("ab") as stream:
        stream.write(b"corruption")
    with pytest.raises(CheckpointIntegrityError, match="size mismatch"):
        TrainingCheckpointManager._verify_manifest(checkpoint, manifest)


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("dims", "effective_action", 1), 22),
        (("episode_length",), 12),
        (("agent_order",), ["bess", "idc"]),
        (("algorithm",), "happo"),
    ],
)
def test_11_incompatible_resume_contract_fails(checkpoints, path, value):
    saved = checkpoints["parent"]["compatibility"]
    current = copy.deepcopy(saved)
    target = current
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    assert _compatibility_differences(saved, current)


def test_12_legacy_model_files_remain_strictly_compatible(checkpoints):
    model_dir = checkpoints["paths"]["continuous"].parents[1] / "models"
    checkpoint_state = checkpoints["continuous"]["model_state"]
    pairs = (
        ("actor_agent0.pt", "actor_agent0_state_dict"),
        ("actor_agent1.pt", "actor_agent1_state_dict"),
        ("critic_agent.pt", "critic_state_dict"),
    )
    for filename, key in pairs:
        legacy = torch.load(model_dir / filename, map_location="cpu", weights_only=False)
        assert _max_difference(legacy, checkpoint_state[key]) == 0.0
