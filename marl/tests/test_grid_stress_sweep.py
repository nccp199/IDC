"""Regression tests for the Phase 2.5 offline grid-stress diagnostic."""

from __future__ import annotations

from copy import deepcopy

import pytest

from configs.config_ultimate import (
    DATA_CONFIG,
    ENV_CONFIG,
    GRID_CONFIG,
    GRID_SCENARIO_CONFIG,
    REWARD_CONFIG,
)
from diagnostics.grid_stress_sweep import (
    _validate_no_nonfinite,
    audit_capacity_definitions,
    collect_fixed_physics_trace,
    run_stress_case,
)


@pytest.fixture(scope="module")
def physics_trace():
    return collect_fixed_physics_trace(seed=2026)


def test_1_diagnostic_does_not_mutate_formal_defaults(physics_trace):
    before = deepcopy((ENV_CONFIG, REWARD_CONFIG, DATA_CONFIG, GRID_CONFIG, GRID_SCENARIO_CONFIG))
    audit_capacity_definitions()
    run_stress_case(physics_trace[:1], 1.25, 1.5)
    after = (ENV_CONFIG, REWARD_CONFIG, DATA_CONFIG, GRID_CONFIG, GRID_SCENARIO_CONFIG)
    assert before == after


def test_3_same_seed_and_multipliers_are_deterministic(physics_trace):
    repeated_trace = collect_fixed_physics_trace(seed=2026)
    assert repeated_trace == physics_trace
    first_summary, first_hours = run_stress_case(physics_trace[:2], 1.25, 1.5)
    second_summary, second_hours = run_stress_case(physics_trace[:2], 1.25, 1.5)
    assert first_summary == second_summary
    assert first_hours == second_hours


def test_4_one_one_reproduces_phase2_normal_scale(physics_trace):
    summary, _ = run_stress_case(physics_trace, 1.0, 1.0)
    assert summary["opf_success_count"] == 24
    assert 1.0 <= summary["min_bus_voltage_pu"] <= 1.03
    assert 1.08 <= summary["max_bus_voltage_pu"] <= 1.10
    assert 1.0 <= summary["max_line_loading_percent"] <= 1.5
    assert 0.3 <= summary["max_transformer_loading_percent"] <= 0.6


def test_5_success_and_failure_payloads_have_no_nan_or_inf(physics_trace):
    successful, successful_hours = run_stress_case(physics_trace[:1], 1.0, 1.0)
    failed, failed_hours = run_stress_case(physics_trace[:1], 100.0, 1.0)
    _validate_no_nonfinite({"summary": successful, "hourly": successful_hours})
    _validate_no_nonfinite({"summary": failed, "hourly": failed_hours})


def test_6_failed_case_is_recorded_without_aborting(physics_trace):
    summary, hourly = run_stress_case(physics_trace[:1], 100.0, 1.0)
    assert len(hourly) == 1
    assert hourly[0]["opf_success"] is False
    assert summary["opf_failure_count"] == 1
    assert summary["opf_success_rate"] == 0.0
    assert "failed" in hourly[0]["opf_message"].lower()
