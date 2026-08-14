"""Phase 2.7 acceptance tests for the formal 25 MW input/reward semantics."""

from __future__ import annotations

import numpy as np
import pytest

from configs.config_ultimate import IDC_SCALE_CONFIG
from diagnostics.idc_25mw_acopf_probe import _make_site_env
from marl import IDCGridMultiAgentEnv
from marl.checkpointing import CHECKPOINT_SCHEMA_VERSION
from marl.envs.harl_env_factory import make_harl_single_env
from marl.evaluation.model_loader import ModelCompatibilityError, _validate_resolved_config
from marl.graphs import HGTAGraphBuilder
from marl.specs import INPUT_SEMANTICS_VERSION, SUPPLEMENTAL_FIELDS
from marl.tests.helpers import make_grid_env
from train.train_harl_mappo_short import DEFAULT_CONFIG_PATH


def _rated_power_mw(seed: int = 2026) -> float:
    env = _make_site_env(int(IDC_SCALE_CONFIG["server_group_size"]), seed)
    try:
        base = env.env
        loads = np.full(
            (1, base.model.N),
            base.base_load + base.max_task_load_per_server,
            dtype=np.float64,
        )
        p_idc, *_ = base.model.calc_pue_and_total_power(
            loads, np.asarray([30.0], dtype=np.float64)
        )
        return float(p_idc[0]) / 1e6
    finally:
        env.close()


def _one_step(group_size: int):
    env = _make_site_env(group_size, seed=2026)
    try:
        env.reset(seed=2026)
        action = np.full(env.action_space.shape, 0.5, dtype=np.float32)
        action[-1] = 0.0
        _, _, _, _, info = env.step(action)
        return dict(info)
    finally:
        env.close()


def _synthetic_sla(group_size: int):
    env = _make_site_env(group_size, seed=2026)
    try:
        env.reset(seed=2026)
        base = env.env
        for task in base.tasks:
            task.status = "waiting"
            task.remaining_work = task.workload
        metrics = base._compute_sla_metrics(current_time=30)
        return metrics, metrics["sla_penalty"] / base.sla_penalty_ref, base.sla_penalty_ref
    finally:
        env.close()


def test_formal_25mw_config_and_rated_definition():
    assert IDC_SCALE_CONFIG["facility_rated_power_mw"] == 25.0
    assert IDC_SCALE_CONFIG["server_group_size"] == 1841
    assert IDC_SCALE_CONFIG["task_workload_scale"] == 1841
    assert IDC_SCALE_CONFIG["num_server_groups"] == 20
    assert _rated_power_mw() == pytest.approx(25.0, abs=0.01)


def test_reward_normalization_is_independent_of_fleet_scale():
    small_sla, small_norm, small_ref = _synthetic_sla(100)
    formal_sla, formal_norm, formal_ref = _synthetic_sla(1841)
    assert formal_sla["sla_violation_count"] == small_sla["sla_violation_count"]
    assert formal_sla["sla_penalty"] == pytest.approx(small_sla["sla_penalty"])
    assert formal_ref == small_ref == 50.0
    assert formal_norm == pytest.approx(small_norm)

    small = _one_step(100)
    formal = _one_step(1841)
    assert formal["bess_degradation_cost_ref"] == small["bess_degradation_cost_ref"] == 40.0
    assert formal["bess_degradation_cost"] == pytest.approx(small["bess_degradation_cost"])
    assert formal["r_bess_degradation"] == pytest.approx(small["r_bess_degradation"])
    for field in ("r_done", "r_queue", "r_urgent_backlog", "r_unused"):
        assert formal[field] == pytest.approx(small[field], rel=0.0, abs=1e-9)


def test_actor_state_and_action_dimensions_are_unchanged():
    env = make_harl_single_env(seed=2026)
    try:
        assert [space.shape for space in env.observation_space] == [(288,), (288,)]
        assert [space.shape for space in env.action_space] == [(22,), (22,)]
        assert [space.shape for space in env.share_observation_space] == [(364,), (364,)]
        assert env.env.env.input_semantics_version == INPUT_SEMANTICS_VERSION
    finally:
        env.close()


def test_supplemental_features_are_normalized_once_for_actor_state_and_hgta(hgta_schema):
    grid_env = make_grid_env(seed=2026)
    env = IDCGridMultiAgentEnv(grid_env)
    try:
        obs, state, info = env.reset(seed=2026)
        action = {
            "idc": np.full(22, 0.5, dtype=np.float32),
            "bess": np.asarray([0.0], dtype=np.float32),
        }
        obs, state, _, _, _, info = env.step(action)
        normalized = np.asarray(info["supplemental_features_normalized"], dtype=np.float32)
        assert normalized.shape == (6,)
        np.testing.assert_allclose(state[288:294], normalized, rtol=0.0, atol=1e-7)
        np.testing.assert_allclose(obs["bess"][-6:], normalized, rtol=0.0, atol=1e-7)
        assert list(info["supplemental_feature_references"]) == list(SUPPLEMENTAL_FIELDS)
        assert np.isfinite(normalized).all()
        assert np.max(np.abs(normalized)) <= 1.1

        graph = HGTAGraphBuilder(hgta_schema).build(state)
        np.testing.assert_allclose(
            graph.node_features["idc"][0, 0].numpy(), normalized[2:4], atol=1e-7
        )
        np.testing.assert_allclose(
            graph.node_features["bess"][0, 0].numpy(),
            normalized[[0, 1, 4, 5]],
            atol=1e-7,
        )
    finally:
        grid_env.close()


def test_checkpoint_and_model_loading_reject_legacy_input_semantics():
    assert CHECKPOINT_SCHEMA_VERSION == "idc-mappo-training-resume-v2-idc25-semantics"
    assert f"input_semantics_version: {INPUT_SEMANTICS_VERSION}" in DEFAULT_CONFIG_PATH.read_text(
        encoding="utf-8"
    )
    legacy = {
        "main": {"algorithm_name": "mappo"},
        "model": {},
        "algo": {},
        "env": {},
        "train": {},
        "seed": {},
        "critic": {"type": "mlp"},
    }
    with pytest.raises(ModelCompatibilityError, match="legacy/unknown input semantics"):
        _validate_resolved_config(legacy)
