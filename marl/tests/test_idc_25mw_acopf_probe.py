"""Boundary and reproducibility tests for the two-case 25 MW diagnostic."""

from __future__ import annotations

from copy import deepcopy

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
from diagnostics.idc_25mw_acopf_probe import (
    calibrate_group_size,
    collect_site_trace,
    run_opf_case,
    validate_no_nonfinite,
)
from grid_model.grid_case import OPFResult


@pytest.fixture(scope="module")
def two_cases():
    original_size = int(IDC_SCALE_CONFIG["server_group_size"])
    original, original_static = collect_site_trace(original_size, seed=2026)
    calibration = calibrate_group_size(
        original,
        original_size,
        fixed_it_loss_mw=original_static["fixed_it_loss_mw"],
    )
    scaled, scaled_static = collect_site_trace(
        calibration["chosen_integer_group_size"], seed=2026
    )
    original_summary, original_hours, original_generators = run_opf_case(
        "original_approx_1mw", original
    )
    scaled_summary, scaled_hours, scaled_generators = run_opf_case(
        "idc_25mw_peak", scaled
    )
    return {
        "original": original,
        "original_static": original_static,
        "scaled": scaled,
        "scaled_static": scaled_static,
        "calibration": calibration,
        "original_summary": original_summary,
        "scaled_summary": scaled_summary,
        "original_hours": original_hours,
        "scaled_hours": scaled_hours,
        "generators": original_generators + scaled_generators,
    }


def test_diagnostic_does_not_mutate_formal_configuration():
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
    collect_site_trace(int(IDC_SCALE_CONFIG["server_group_size"]), seed=2026)
    after = (
        ENV_CONFIG,
        REWARD_CONFIG,
        DATA_CONFIG,
        IDC_SCALE_CONFIG,
        GRID_CONFIG,
        GRID_REWARD_CONFIG,
        GRID_SCENARIO_CONFIG,
    )
    assert before == after


def test_seed_reproduces_original_and_25mw_site_traces(two_cases):
    original_again, _ = collect_site_trace(
        int(IDC_SCALE_CONFIG["server_group_size"]), seed=2026
    )
    scaled_again, _ = collect_site_trace(
        two_cases["calibration"]["chosen_integer_group_size"], seed=2026
    )
    assert original_again == two_cases["original"]
    assert scaled_again == two_cases["scaled"]


def test_original_scale_reproduces_phase25_baseline(two_cases):
    summary = two_cases["original_summary"]
    assert summary["opf_success_count"] == 24
    assert summary["p_idc_mw_max"] == pytest.approx(1.0521329712, abs=1e-9)
    assert summary["min_voltage_pu"] == pytest.approx(1.0145734858, abs=1e-9)
    assert summary["max_line_loading_percent"] == pytest.approx(1.2530746133, abs=1e-9)


def test_25mw_peak_and_opf_are_reproducible(two_cases):
    calibration = two_cases["calibration"]
    summary = two_cases["scaled_summary"]
    assert calibration["chosen_integer_group_size"] == 2395
    assert max(hour.p_idc_mw for hour in two_cases["scaled"]) == pytest.approx(
        25.0, abs=0.01
    )
    assert summary["opf_success_count"] == 24
    repeated, repeated_hours, repeated_generators = run_opf_case(
        "idc_25mw_peak", two_cases["scaled"]
    )
    assert repeated == summary
    assert repeated_hours == two_cases["scaled_hours"]
    assert repeated_generators == two_cases["generators"][120:]


def test_success_payload_has_no_nan_or_inf(two_cases):
    validate_no_nonfinite(two_cases)


def test_failed_opf_is_recorded_without_crashing(two_cases):
    def failed_solver(*args, **kwargs):
        return OPFResult(success=False, mode="ac", message="synthetic solver failure")

    summary, hours, generators = run_opf_case(
        "failure_handling_only", two_cases["scaled"][:1], solver=failed_solver
    )
    assert summary["opf_success_count"] == 0
    assert summary["opf_failure_count"] == 1
    assert summary["opf_failure_hours"] == [0]
    assert hours[0]["opf_message"] == "synthetic solver failure"
    assert generators == []
    validate_no_nonfinite({"summary": summary, "hours": hours})
