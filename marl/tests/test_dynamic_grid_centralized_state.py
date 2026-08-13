"""Phase-2 acceptance tests for causal per-bus centralized grid state."""

from __future__ import annotations

import numpy as np

from grid_model.grid_case import OPFResult
from grid_model.grid_dynamic_state import build_grid_bus_dynamic_payload
from grid_model.ieee14_loader import load_ieee14_case
from grid_model.opf_solver import solve_ac_opf
from marl.envs.idc_grid_multi_agent_env import IDCGridMultiAgentEnv
from marl.tests.helpers import make_grid_env


def _distinct_result(case) -> OPFResult:
    buses = tuple(case.bus_ids)
    line_ids = tuple(int(value) for value in case.raw_network.line.index)
    trafo_ids = tuple(int(value) for value in case.raw_network.trafo.index)
    return OPFResult(
        success=True,
        mode="ac",
        message="synthetic",
        bus_voltage_pu={bus: 0.94 + 0.005 * bus for bus in buses},
        bus_active_power_mw={bus: -65.0 + 10.0 * bus for bus in buses},
        bus_reactive_power_mvar={bus: -13.0 + 2.0 * bus for bus in buses},
        lmp_by_bus={bus: 30.0 + bus for bus in buses},
        reactive_lmp_by_bus={bus: -2.0 + 0.25 * bus for bus in buses},
        line_loading_percent={line: 5.0 + 3.0 * line for line in line_ids},
        transformer_loading_percent={trafo: 70.0 + 2.0 * trafo for trafo in trafo_ids},
    )


def _expected_incident(case, result: OPFResult) -> np.ndarray:
    expected = np.zeros(14, dtype=np.float64)
    for line, row in case.raw_network.line.iterrows():
        value = result.line_loading_percent[int(line)]
        expected[int(row.from_bus)] = max(expected[int(row.from_bus)], value)
        expected[int(row.to_bus)] = max(expected[int(row.to_bus)], value)
    for trafo, row in case.raw_network.trafo.iterrows():
        value = result.transformer_loading_percent[int(trafo)]
        expected[int(row.hv_bus)] = max(expected[int(row.hv_bus)], value)
        expected[int(row.lv_bus)] = max(expected[int(row.lv_bus)], value)
    return expected


def test_bus_feature_and_real_topology_mapping_are_exact():
    case = load_ieee14_case()
    result = _distinct_result(case)
    payload = build_grid_bus_dynamic_payload(case, result, lmp_ref=100.0)
    dynamic = payload.normalized_state.reshape(14, 5)

    np.testing.assert_allclose(dynamic[:, 0], (payload.bus_vm_pu - 1.0) / 0.10)
    np.testing.assert_allclose(dynamic[:, 1], payload.bus_p_mw / case.base_mva)
    np.testing.assert_allclose(dynamic[:, 2], payload.bus_q_mvar / case.base_mva)
    np.testing.assert_allclose(dynamic[:, 3], payload.bus_lmp / 100.0)
    expected_incident = _expected_incident(case, result)
    np.testing.assert_allclose(payload.incident_branch_max_loading_percent, expected_incident)
    np.testing.assert_allclose(dynamic[:, 4], expected_incident / 100.0)
    assert len(np.unique(expected_incident)) > 1  # not a copied whole-grid maximum


def test_failed_opf_fallback_is_finite_deterministic_and_neutral():
    case = load_ieee14_case()
    failed = OPFResult(success=False, mode="ac", message="forced failure")
    first = build_grid_bus_dynamic_payload(case, failed, lmp_ref=100.0)
    second = build_grid_bus_dynamic_payload(case, failed, lmp_ref=100.0)
    assert first.normalized_state.shape == (70,)
    assert np.isfinite(first.normalized_state).all()
    np.testing.assert_array_equal(first.normalized_state, second.normalized_state)
    np.testing.assert_array_equal(first.normalized_state, np.zeros(70, np.float32))
    assert first.fallback_used is True


def test_real_ac_opf_dynamic_features_respond_to_current_load():
    case = load_ieee14_case()
    low = solve_ac_opf(case, idc_bus_id=8, idc_load_mw=0.25, load_scale=0.90)
    high = solve_ac_opf(case, idc_bus_id=8, idc_load_mw=3.00, load_scale=1.10)
    assert low.success and high.success
    low_payload = build_grid_bus_dynamic_payload(case, low, lmp_ref=100.0)
    high_payload = build_grid_bus_dynamic_payload(case, high, lmp_ref=100.0)
    for left, right in (
        (low_payload.bus_vm_pu, high_payload.bus_vm_pu),
        (low_payload.bus_p_mw, high_payload.bus_p_mw),
        (low_payload.bus_q_mvar, high_payload.bus_q_mvar),
        (low_payload.bus_lmp, high_payload.bus_lmp),
        (
            low_payload.incident_branch_max_loading_percent,
            high_payload.incident_branch_max_loading_percent,
        ),
    ):
        assert np.max(np.abs(left - right)) > 1e-8


def test_dynamic_state_changes_only_centralized_view_not_actor_views():
    grid_env = make_grid_env(seed=7120)
    env = IDCGridMultiAgentEnv(grid_env)
    try:
        actor_before, state_before, info = env.reset(seed=7120)
        changed_info = dict(info)
        changed_info["grid_bus_dynamic_state"] = (
            np.asarray(info["grid_bus_dynamic_state"], dtype=np.float32) + 0.25
        )
        actor_after, state_after = env._build_outputs(
            env.last_raw_observation, changed_info, initial=True
        )
        np.testing.assert_array_equal(actor_before["idc"], actor_after["idc"])
        np.testing.assert_array_equal(actor_before["bess"], actor_after["bess"])
        np.testing.assert_array_equal(state_before[:294], state_after[:294])
        assert not np.array_equal(state_before[294:364], state_after[294:364])
        assert env.observation_spaces["idc"].shape == (288,)
        assert env.observation_spaces["bess"].shape == (164,)
        assert env.state_space.shape == (364,)
    finally:
        grid_env.close()
