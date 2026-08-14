"""Boundary tests for the rated-25-MW controllability diagnostic."""

from __future__ import annotations

from copy import deepcopy

import numpy as np
import pytest

from configs.config_ultimate import (
    DATA_CONFIG,
    ENV_CONFIG,
    GRID_CONFIG,
    GRID_REWARD_CONFIG,
    GRID_SCENARIO_CONFIG,
    IDC_SCALE_CONFIG,
    REWARD_CONFIG,
)
from diagnostics.idc_25mw_acopf_probe import _make_site_env, run_opf_case, validate_no_nonfinite
from diagnostics.idc_25mw_safety_controllability import (
    CASES,
    calibrate_rated_group_size,
    collect_control_trace,
)
from grid_model.grid_case import OPFResult


@pytest.fixture(scope="module")
def calibration():
    return calibrate_rated_group_size(seed=2026)


def test_rated_definition_uses_maximum_legal_action(calibration):
    assert calibration["chosen_group_size"] == 1841
    assert calibration["effective_server_count"] == 36820
    assert calibration["maximum_legal_server_action"] == 1.0
    assert calibration["maximum_planned_total_load"] == pytest.approx(0.65)
    assert calibration["rated_power"]["p_idc_mw"] == pytest.approx(25.0, abs=0.01)
    assert calibration["old_group_size_rated_power"]["p_idc_mw"] > 32.0


def test_five_site_action_cases_are_seed_reproducible_and_finite(calibration):
    group_size = calibration["chosen_group_size"]
    assert len(CASES) == 5
    for case_name, server_action, bess_action in CASES:
        first_site, first_control, first_static = collect_control_trace(
            case_name, group_size, server_action, bess_action, seed=2026
        )
        replay_site, replay_control, replay_static = collect_control_trace(
            case_name, group_size, server_action, bess_action, seed=2026
        )
        assert first_site == replay_site
        assert first_control == replay_control
        assert first_static == replay_static
        assert first_static["action_shape"] == [23]
        validate_no_nonfinite(
            {
                "site": first_site,
                "control": first_control,
                "static": first_static,
            }
        )


def test_candidate_scale_preserves_forecast_semantics_and_normalized_values(calibration):
    formal = _make_site_env(int(IDC_SCALE_CONFIG["server_group_size"]), seed=2026)
    candidate = _make_site_env(calibration["chosen_group_size"], seed=2026)
    try:
        formal.reset(seed=2026)
        candidate.reset(seed=2026)
        a = formal.env
        b = candidate.env
        assert a.task_forecast_mode == b.task_forecast_mode == "noisy"
        assert a.forecast_error_level == b.forecast_error_level == pytest.approx(0.20)
        np.testing.assert_allclose(
            a.task_arrival_forecast / a.lambda_ref,
            b.task_arrival_forecast / b.lambda_ref,
            rtol=0.0,
            atol=1e-12,
        )
        np.testing.assert_allclose(
            a.true_task_arrival_profile / a.lambda_ref,
            b.true_task_arrival_profile / b.lambda_ref,
            rtol=0.0,
            atol=1e-12,
        )
        assert not np.array_equal(a.task_arrival_forecast, a.true_task_arrival_profile)
    finally:
        formal.close()
        candidate.close()


def test_diagnostic_does_not_mutate_formal_config(calibration):
    before = deepcopy(
        (
            ENV_CONFIG,
            REWARD_CONFIG,
            DATA_CONFIG,
            IDC_SCALE_CONFIG,
            GRID_CONFIG,
            GRID_REWARD_CONFIG,
            GRID_SCENARIO_CONFIG,
        )
    )
    collect_control_trace("mutation_check", calibration["chosen_group_size"], 1.0, 0.5, 2026)
    assert before == (
        ENV_CONFIG,
        REWARD_CONFIG,
        DATA_CONFIG,
        IDC_SCALE_CONFIG,
        GRID_CONFIG,
        GRID_REWARD_CONFIG,
        GRID_SCENARIO_CONFIG,
    )


def test_failed_opf_is_recorded_without_crashing(calibration):
    site, _, _ = collect_control_trace(
        "failure_source", calibration["chosen_group_size"], 1.0, 0.5, seed=2026
    )

    def failed_solver(*args, **kwargs):
        return OPFResult(success=False, mode="ac", message="synthetic solver failure")

    summary, hours, generators = run_opf_case(
        "failure_handling_only", site[:1], solver=failed_solver
    )
    assert summary["opf_success_count"] == 0
    assert summary["opf_failure_count"] == 1
    assert summary["opf_failure_hours"] == [0]
    assert hours[0]["opf_message"] == "synthetic solver failure"
    assert generators == []
    validate_no_nonfinite({"summary": summary, "hours": hours})
