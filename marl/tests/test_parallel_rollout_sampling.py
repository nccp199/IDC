"""Correctness and artifact acceptance for formal parallel MAPPO sampling."""

from __future__ import annotations

import csv
import functools
import json
import os
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import pytest
import torch

from marl.checkpointing.training_checkpoint import _compatibility_differences
from marl.envs.harl_env_factory import make_harl_train_env
from marl.envs.parallel_vec_env import ProjectShareSubprocVecEnv, RemoteWorkerError


def _artifact(name: str) -> Path:
    value = os.environ.get(name)
    if not value:
        pytest.skip(f"{name} is required for the parallel artifact acceptance.")
    path = Path(value).resolve()
    assert path.exists(), path
    return path


def _numeric_diff(left: Any, right: Any, ignored: set[str] | None = None) -> float:
    ignored = ignored or set()
    if torch.is_tensor(left):
        return float((left.detach().cpu() - right.detach().cpu()).abs().max()) if left.numel() else 0.0
    if isinstance(left, np.ndarray):
        if left.dtype.kind in "fci":
            values = np.abs(left.astype(np.float64) - right.astype(np.float64))
            return 0.0 if not values.size or np.isnan(values).all() else float(np.nanmax(values))
        return 0.0 if np.array_equal(left, right) else float("inf")
    if isinstance(left, dict):
        keys = (set(left) | set(right)) - ignored
        if keys.difference(left) or keys.difference(right):
            return float("inf")
        return max((_numeric_diff(left[key], right[key], ignored) for key in keys), default=0.0)
    if isinstance(left, (list, tuple)):
        if len(left) != len(right):
            return float("inf")
        return max((_numeric_diff(a, b, ignored) for a, b in zip(left, right, strict=True)), default=0.0)
    if isinstance(left, (int, float, np.number)) and isinstance(right, (int, float, np.number)):
        if isinstance(left, (float, np.floating)) and np.isnan(left) and np.isnan(right):
            return 0.0
        return abs(float(left) - float(right))
    return 0.0 if left == right else float("inf")


def _task_signature(worker_state: dict[str, Any]):
    return tuple(
        (
            task["task_id"], task["arrival_time"], task["duration"],
            task["workload"], task["deadline"], task["priority"],
        )
        for task in worker_state["base_environment"]["tasks"]
    )


def _server_signature(worker_state: dict[str, Any]):
    values = worker_state["model"]["server_parameters"]
    return np.concatenate((values["P_idle"], values["P_max"], values["C_server"]))


@pytest.fixture(scope="module")
def real_parallel_snapshots():
    results = {}
    for workers in (2, 4):
        env = make_harl_train_env(seed=7110, n_rollout_threads=workers)
        pids = tuple(env.process_ids)
        try:
            obs, share_obs, available = env.reset()
            initial_states = env.get_env_states()
            actions = np.full((workers, 2, 22), 0.5, dtype=np.float32)
            step = env.step(actions)
            results[workers] = {
                "reset": (obs, share_obs, available),
                "step": step,
                "initial_states": initial_states,
                "worker_seeds": tuple(env.worker_seeds),
                "pids": pids,
            }
        finally:
            env.close()
        assert not any(env.alive)
    return results


def test_01_two_worker_reset_step_shapes(real_parallel_snapshots):
    reset = real_parallel_snapshots[2]["reset"]
    step = real_parallel_snapshots[2]["step"]
    assert reset[0].shape == (2, 2, 288)
    assert reset[1].shape == (2, 2, 294)
    assert step[0].shape == (2, 2, 288)
    assert step[1].shape == (2, 2, 294)
    assert step[2].shape == (2, 2, 1)
    assert step[3].shape == (2, 2)
    assert len(step[4]) == 2 and all(len(info) == 2 for info in step[4])


def test_02_four_worker_reset_step_shapes(real_parallel_snapshots):
    reset = real_parallel_snapshots[4]["reset"]
    step = real_parallel_snapshots[4]["step"]
    assert reset[0].shape == (4, 2, 288)
    assert reset[1].shape == (4, 2, 294)
    assert step[2].shape == (4, 2, 1)
    assert step[3].shape == (4, 2)
    assert len(step[4]) == 4


def test_03_worker_seeds_are_unique_and_traceable(real_parallel_snapshots):
    assert real_parallel_snapshots[2]["worker_seeds"] == (7110, 8110)
    assert real_parallel_snapshots[4]["worker_seeds"] == (7110, 8110, 9110, 10110)


def test_04_worker_tasks_and_servers_differ(real_parallel_snapshots):
    states = real_parallel_snapshots[4]["initial_states"]
    assert len({_task_signature(state) for state in states}) == 4
    for left in range(4):
        for right in range(left + 1, 4):
            assert not np.array_equal(_server_signature(states[left]), _server_signature(states[right]))


def test_05_probe_and_worker_zero_are_isolated(real_parallel_snapshots):
    probe = make_harl_train_env(seed=7110, n_rollout_threads=1)
    try:
        probe.reset()
        probe_state = probe.envs[0]
        from marl.checkpointing.environment_state import single_environment_state_dict

        state = single_environment_state_dict(probe_state)
    finally:
        probe.close()
    worker_zero = real_parallel_snapshots[2]["initial_states"][0]
    assert _task_signature(state) == _task_signature(worker_zero)
    np.testing.assert_array_equal(_server_signature(state), _server_signature(worker_zero))
    assert state["grid"]["cache"] is not worker_zero["grid"]["cache"]


def test_06_cache_and_generator_state_are_per_worker(real_parallel_snapshots):
    states = real_parallel_snapshots[4]["initial_states"]
    assert len({json.dumps(state["model"]["task_rng"], sort_keys=True) for state in states}) == 4
    assert len({json.dumps(state["model"]["server_rng"], sort_keys=True) for state in states}) == 4
    cache_snapshots = [state["grid"]["cache"] for state in states]
    assert len(cache_snapshots) == 4
    assert all(cache is not cache_snapshots[0] for cache in cache_snapshots[1:])


def test_07_shared_reward_is_once_per_worker(real_parallel_snapshots):
    rewards = real_parallel_snapshots[4]["step"][2]
    np.testing.assert_array_equal(rewards[:, 0], rewards[:, 1])
    assert np.unique(rewards[:, 0, 0]).size > 1


def test_08_agent_order_is_stable_for_every_worker(real_parallel_snapshots):
    for workers in (2, 4):
        infos = real_parallel_snapshots[workers]["step"][4]
        assert all(tuple(info[index]["agent_id"] for index in range(2)) == ("idc", "bess") for info in infos)


class _MockEnv:
    agent_order = ("idc", "bess")
    n_agents = 2

    def __init__(self, *, rank: int, fail_stage: str | None = None):
        self.rank = rank
        self.fail_stage = fail_stage
        self.steps = 0
        self.observation_space = tuple(
            gym.spaces.Box(-1.0, 1.0, shape=(288,), dtype=np.float32) for _ in range(2)
        )
        self.share_observation_space = tuple(
            gym.spaces.Box(-1.0, 1.0, shape=(294,), dtype=np.float32) for _ in range(2)
        )
        self.action_space = tuple(
            gym.spaces.Box(0.0, 1.0, shape=(22,), dtype=np.float32) for _ in range(2)
        )
        if fail_stage == "construct":
            raise RuntimeError(f"construct failure rank={rank}")

    def reset(self):
        if self.fail_stage == "reset":
            raise RuntimeError(f"reset failure rank={self.rank}")
        self.steps = 0
        return np.full((2, 288), self.rank, np.float32), np.full((2, 294), self.rank, np.float32), None

    def step(self, actions):
        if self.fail_stage == "step":
            raise RuntimeError(f"step failure rank={self.rank}")
        self.steps += 1
        done = np.full(2, self.steps >= self.rank + 1, dtype=np.bool_)
        infos = [{"agent_id": name} for name in self.agent_order]
        return (
            np.full((2, 288), self.steps, np.float32),
            np.full((2, 294), self.steps, np.float32),
            np.full((2, 1), self.rank + 0.25, np.float32),
            done,
            infos,
            None,
        )

    def close(self):
        return None


def _make_mock(rank: int, fail_stage: str | None = None):
    return _MockEnv(rank=rank, fail_stage=fail_stage)


def test_09_terminal_and_auto_reset_are_worker_local():
    env = ProjectShareSubprocVecEnv(
        [functools.partial(_make_mock, 0), functools.partial(_make_mock, 1)],
        worker_seeds=(1, 2),
    )
    try:
        env.reset()
        actions = np.zeros((2, 2, 22), np.float32)
        first = env.step(actions)
        assert first[3][0].all() and not first[3][1].any()
        assert "original_obs" in first[4][0][0]
        assert "original_obs" not in first[4][1][0]
        second = env.step(actions)
        assert second[3][0].all() and second[3][1].all()
        assert "original_obs" in second[4][0][0]
        assert "original_obs" in second[4][1][0]
    finally:
        env.close()


@pytest.mark.parametrize("stage", ("construct", "reset", "step"))
def test_10_child_exception_propagates_with_traceback_and_cleans(stage):
    env = None
    with pytest.raises((RemoteWorkerError, RuntimeError), match=f"{stage} failure rank=1"):
        env = ProjectShareSubprocVecEnv(
            [functools.partial(_make_mock, 0), functools.partial(_make_mock, 1, stage)],
            worker_seeds=(1, 2),
        )
        if stage == "reset":
            env.reset()
        elif stage == "step":
            env.reset()
            env.step(np.zeros((2, 2, 22), np.float32))
    if env is not None:
        env.close(force=True)
        assert not any(env.alive)


def test_10b_training_failure_writes_failed_summary_without_final_checkpoint(
    tmp_path, monkeypatch
):
    from marl.runners.idc_mappo_runner import IDCOnPolicyMARunner
    from train.train_harl_mappo_short import DEFAULT_CONFIG_PATH, main

    captured = {}

    def fail_run(runner):
        captured["runner"] = runner
        raise RemoteWorkerError(
            "Subprocess environment failure:\n"
            "worker=1 command=step RuntimeError: step failure rank=1\n"
            "synthetic child traceback"
        )

    monkeypatch.setattr(IDCOnPolicyMARunner, "run", fail_run)
    with pytest.raises(RemoteWorkerError, match="step failure rank=1"):
        main(
            [
                "--config",
                str(DEFAULT_CONFIG_PATH),
                "--updates",
                "1",
                "--rollout-threads",
                "2",
                "--output-dir",
                str(tmp_path),
                "--device",
                "cpu",
            ]
        )

    metadata_path = next(tmp_path.rglob("run_metadata.json"))
    summary_path = next(tmp_path.rglob("run_summary.json"))
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert metadata["status"] == "failed"
    assert "step failure rank=1" in metadata["error"]
    assert summary["status"] == "failed"
    assert "step failure rank=1" in summary["termination_reason"]
    assert not list(tmp_path.rglob("final.pt"))
    assert not any(captured["runner"].envs.alive)


def _load_checkpoint(name: str):
    return torch.load(_artifact(name), map_location="cpu", weights_only=False)


def test_11_two_worker_logging_counts_and_numbering():
    run_dir = _artifact("PART10_TWO_WORKER_RUN")
    with (run_dir / "metrics/step_metrics.csv").open(encoding="utf-8", newline="") as stream:
        steps = list(csv.DictReader(stream))
    with (run_dir / "metrics/episode_metrics.csv").open(encoding="utf-8", newline="") as stream:
        episodes = list(csv.DictReader(stream))
    with (run_dir / "metrics/update_metrics.csv").open(encoding="utf-8", newline="") as stream:
        updates = list(csv.DictReader(stream))
    first_update_steps = [row for row in steps if int(row["update"]) == 1]
    assert len(first_update_steps) == 48
    assert len([row for row in episodes if int(row["update"]) == 1]) == 2
    assert len(updates) == 2
    assert {int(row["worker_id"]) for row in first_update_steps} == {0, 1}
    assert all({int(row["episode_step"]) for row in first_update_steps if int(row["worker_id"]) == worker} == set(range(24)) for worker in (0, 1))
    first_episodes = [row for row in episodes if int(row["update"]) == 1]
    for episode in first_episodes:
        worker = int(episode["worker_id"])
        reward_sum = sum(
            float(row["reward_returned"])
            for row in first_update_steps
            if int(row["worker_id"]) == worker
        )
        assert reward_sum == pytest.approx(float(episode["episode_reward"]), abs=1e-9)
        terminal = [row for row in first_update_steps if int(row["worker_id"]) == worker and row["is_terminal"] == "True"]
        assert len(terminal) == 1 and int(terminal[0]["episode_step"]) == 23
    assert float(updates[0]["rollout_episode_reward_mean"]) == pytest.approx(
        np.mean([float(row["episode_reward"]) for row in first_episodes]), abs=1e-9
    )
    assert int(updates[0]["global_step"]) == 48


def test_12_four_worker_logging_counts_and_numbering():
    run_dir = _artifact("PART10_FOUR_WORKER_RUN")
    with (run_dir / "metrics/step_metrics.csv").open(encoding="utf-8", newline="") as stream:
        steps = list(csv.DictReader(stream))
    with (run_dir / "metrics/episode_metrics.csv").open(encoding="utf-8", newline="") as stream:
        episodes = list(csv.DictReader(stream))
    with (run_dir / "metrics/update_metrics.csv").open(encoding="utf-8", newline="") as stream:
        updates = list(csv.DictReader(stream))
    first = [row for row in steps if int(row["update"]) == 1]
    assert len(first) == 96
    assert len([row for row in episodes if int(row["update"]) == 1]) == 4
    assert {int(row["worker_id"]) for row in first} == {0, 1, 2, 3}
    assert max(int(row["global_step"]) for row in first) == 96
    assert int(updates[0]["global_step"]) == 96
    assert int(updates[0]["episodes_completed"]) == 4


def test_13_two_worker_same_seed_reproduces_exactly():
    left = _load_checkpoint("PART10_TWO_WORKER_UPDATE1_A")
    right = _load_checkpoint("PART10_TWO_WORKER_UPDATE1_B")
    ignored = {
        "run_id", "wall_time_seconds", "rollout_time_seconds",
        "update_time_seconds", "steps_per_second", "total_updates_target",
    }
    for key in ("model_state", "optimizer_state", "runner_state", "environment_state"):
        assert _numeric_diff(left[key], right[key], ignored) == 0.0, key
    assert _numeric_diff(left["verification_state"], right["verification_state"], ignored) == 0.0


def test_14_two_worker_checkpoint_resume_is_exact():
    left = _load_checkpoint("PART10_TWO_WORKER_UPDATE2_CONTINUOUS")
    right = _load_checkpoint("PART10_TWO_WORKER_UPDATE2_RESUMED")
    ignored = {"run_id", "wall_time_seconds", "rollout_time_seconds", "update_time_seconds", "steps_per_second"}
    for key in ("model_state", "optimizer_state", "runner_state", "environment_state"):
        assert _numeric_diff(left[key], right[key], ignored) == 0.0, key
    assert _numeric_diff(left["verification_state"], right["verification_state"], ignored) == 0.0


def test_15_worker_count_checkpoint_incompatibility_is_detected():
    saved = _load_checkpoint("PART10_TWO_WORKER_UPDATE1_A")["compatibility"]
    current = dict(saved)
    current["rollout_threads"] = 1
    current["worker_seeds"] = [7110]
    current["vec_env_type"] = "ShareDummyVecEnv"
    assert _compatibility_differences(saved, current)


def test_16_bess_mask_is_effective_for_parallel_update():
    state = _load_checkpoint("PART10_TWO_WORKER_UPDATE2_CONTINUOUS")
    compatibility = state["compatibility"]
    diagnostics = state["verification_state"]["actor_diagnostics"][1]
    assert compatibility["dims"]["effective_action"] == [22, 1]
    assert diagnostics["effective_action_dim"] == 1
    assert diagnostics["padded_action_dim"] == 22
    assert diagnostics["effective_physical_ratio_max_diff"] == 0.0
    assert diagnostics["virtual_mean_rows_grad_norm"] == 0.0
    assert diagnostics["virtual_log_std_grad_norm"] == 0.0
    assert diagnostics["virtual_entropy_optimization_contribution"] == 0.0
