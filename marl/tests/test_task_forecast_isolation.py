"""Acceptance tests for task-truth isolation and controlled forecasts."""

from __future__ import annotations

import copy

import numpy as np
import pytest

from envs.idc_price_env import IDCPriceEnv20D


def _base_env(*, seed: int, mode: str = "noisy", forecast_seed: int | None = None):
    return IDCPriceEnv20D(
        server_seed=seed,
        task_seed=seed,
        forecast_seed=forecast_seed if forecast_seed is not None else seed + 300000,
        task_forecast_mode=mode,
        forecast_error_level=0.20,
    )


def _task_truth(env: IDCPriceEnv20D) -> tuple[tuple[object, ...], ...]:
    return tuple(
        (
            task.task_id,
            task.arrival_time,
            task.workload,
            task.deadline,
            task.priority,
            task.interruptible,
            task.parallelizable,
        )
        for task in env.tasks
    )


def _mutate_unarrived_truth(env: IDCPriceEnv20D) -> None:
    changed = 0
    for task in env.tasks:
        if task.arrival_time <= env.current_step:
            continue
        task.workload = float(task.workload * 1.7 + 13.0)
        task.remaining_work = float(task.workload)
        task.deadline = int(task.deadline + 4)
        task.priority = float(task.priority + 2.0)
        task.interruptible = not bool(task.interruptible)
        task.parallelizable = not bool(task.parallelizable)
        changed += 1
    assert changed > 0
    env.true_task_arrival_profile = env.model.build_task_arrival_curve(
        env.tasks, horizon=env.horizon
    )
    env.lambda_t = env.true_task_arrival_profile


def test_noisy_forecast_is_separate_and_is_the_only_task_forecast_in_observation():
    env = _base_env(seed=7101)
    try:
        observation, _ = env.reset()
        assert not np.array_equal(
            env.true_task_arrival_profile, env.task_arrival_forecast
        )
        task_forecast_slice = observation[136 + 48 : 136 + 72]
        np.testing.assert_allclose(
            task_forecast_slice,
            env.task_arrival_forecast / env.lambda_ref,
            rtol=0.0,
            atol=1e-7,
        )
    finally:
        env.close()


@pytest.mark.parametrize("mode", ("perfect", "none"))
def test_oracle_and_disabled_modes_are_explicit(mode: str):
    env = _base_env(seed=7102, mode=mode)
    try:
        env.reset()
        expected = (
            env.true_task_arrival_profile
            if mode == "perfect"
            else np.zeros(env.horizon, dtype=np.float64)
        )
        np.testing.assert_array_equal(env.task_arrival_forecast, expected)
    finally:
        env.close()


def test_future_task_truth_mutation_cannot_change_current_observation_when_forecast_fixed():
    env = _base_env(seed=7103)
    try:
        before, _ = env.reset()
        fixed_forecast = env.task_arrival_forecast.copy()
        _mutate_unarrived_truth(env)
        assert not np.array_equal(env.true_task_arrival_profile, fixed_forecast)
        np.testing.assert_array_equal(env.task_arrival_forecast, fixed_forecast)
        after = env._get_obs()
        np.testing.assert_array_equal(after, before)
    finally:
        env.close()


def test_forecast_rng_is_reproducible_and_does_not_consume_task_rng():
    first = _base_env(seed=7104, forecast_seed=310104)
    replay = _base_env(seed=7104, forecast_seed=310104)
    changed_forecast_rng = _base_env(seed=7104, forecast_seed=310105)
    changed_all_rngs = _base_env(seed=7105, forecast_seed=310106)
    try:
        for env in (first, replay, changed_forecast_rng, changed_all_rngs):
            env.reset()
        assert _task_truth(first) == _task_truth(replay)
        assert _task_truth(first) == _task_truth(changed_forecast_rng)
        np.testing.assert_array_equal(
            first.true_task_arrival_profile, replay.true_task_arrival_profile
        )
        np.testing.assert_array_equal(
            first.task_arrival_forecast, replay.task_arrival_forecast
        )
        assert not np.array_equal(
            first.task_arrival_forecast,
            changed_forecast_rng.task_arrival_forecast,
        )
        assert _task_truth(first) != _task_truth(changed_all_rngs)
    finally:
        for env in (first, replay, changed_forecast_rng, changed_all_rngs):
            env.close()


def test_three_method_labels_receive_identical_environment_truth_and_forecast():
    snapshots = {}
    for method in ("mappo_mlp", "happo_mlp", "happo_hgta"):
        env = _base_env(seed=7106)
        try:
            env.reset()
            snapshots[method] = (
                _task_truth(env),
                env.true_task_arrival_profile.copy(),
                env.task_arrival_forecast.copy(),
                env.price_t.copy(),
                env.pv_t.copy(),
                env.T_amb.copy(),
            )
        finally:
            env.close()
    reference = snapshots["mappo_mlp"]
    for snapshot in snapshots.values():
        assert snapshot[0] == reference[0]
        for actual, expected in zip(snapshot[1:], reference[1:], strict=True):
            np.testing.assert_array_equal(actual, expected)


def test_formal_actor_critic_views_are_causal_under_fixed_forecast():
    pytest.importorskip("harl")
    from marl.envs.harl_env_factory import make_harl_single_env

    env = make_harl_single_env(seed=7107)
    try:
        actor_before, state_before, _ = env.reset()
        bridge = env.env
        multi = bridge.env
        base = multi.env.env
        fixed_forecast = base.task_arrival_forecast.copy()
        _mutate_unarrived_truth(base)
        np.testing.assert_array_equal(base.task_arrival_forecast, fixed_forecast)

        raw_after = multi.env._augment_obs(base._get_obs(), multi.last_info)
        views_after, state_after = multi._build_outputs(
            raw_after, multi.last_info, initial=True
        )
        np.testing.assert_array_equal(actor_before[0], views_after["idc"])
        np.testing.assert_array_equal(actor_before[1, :164], views_after["bess"])
        np.testing.assert_array_equal(state_before[0], state_after)
    finally:
        env.close()


def test_checkpoint_restores_current_forecast_and_next_episode_rng_sequence():
    pytest.importorskip("harl")
    from marl.checkpointing.environment_state import (
        load_single_environment_state_dict,
        single_environment_state_dict,
    )
    from marl.envs.harl_env_factory import make_harl_single_env

    uninterrupted = make_harl_single_env(seed=7108)
    resumed = make_harl_single_env(seed=7108)
    try:
        uninterrupted.reset()
        saved = copy.deepcopy(single_environment_state_dict(uninterrupted))
        load_single_environment_state_dict(resumed, saved)
        base_a = uninterrupted.env.env.env.env
        base_b = resumed.env.env.env.env
        np.testing.assert_array_equal(
            base_a.task_arrival_forecast, base_b.task_arrival_forecast
        )
        np.testing.assert_array_equal(
            base_a.true_task_arrival_profile, base_b.true_task_arrival_profile
        )

        uninterrupted.reset()
        resumed.reset()
        np.testing.assert_array_equal(
            base_a.true_task_arrival_profile, base_b.true_task_arrival_profile
        )
        np.testing.assert_array_equal(
            base_a.task_arrival_forecast, base_b.task_arrival_forecast
        )
    finally:
        uninterrupted.close()
        resumed.close()


def test_formal_two_agent_environment_completes_one_episode_with_opf():
    pytest.importorskip("harl")
    from marl.envs.harl_env_factory import make_harl_single_env

    env = make_harl_single_env(seed=7109)
    try:
        observations, states, _ = env.reset()
        assert observations.shape == (2, 288)
        assert states.shape == (2, 294)
        actions = np.full((2, 22), 0.5, dtype=np.float32)
        for _ in range(24):
            observations, states, rewards, dones, infos, _ = env.step(actions)
            assert np.isfinite(observations).all()
            assert np.isfinite(states).all()
            assert np.isfinite(rewards).all()
            assert infos[0]["grid_opf_success"] is True
            assert infos[1]["grid_opf_success"] is True
        assert bool(np.all(dones))
        assert "true_task_arrival_profile" in infos[0]
        assert "task_arrival_forecast" in infos[0]
    finally:
        env.close()
