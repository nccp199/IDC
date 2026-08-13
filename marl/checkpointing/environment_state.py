"""Explicit, versioned state serialization for the formal IDC/BESS env chain.

The serializer deliberately stores data fields rather than pickling the live
environment object.  It is scoped to the single-worker formal HARL chain.
"""

from __future__ import annotations

import copy
from collections import OrderedDict
from dataclasses import fields
from typing import Any

import numpy as np

from grid_model.grid_case import MEFResult, OPFResult
from idc_model.task import Task


ENVIRONMENT_STATE_VERSION = "idc-bess-vec-env-state-v3"
WORKER_ENVIRONMENT_STATE_VERSION = "idc-bess-env-state-v2"
LEGACY_ENVIRONMENT_STATE_VERSION = "idc-bess-env-state-v1"

_BASE_DYNAMIC_FIELDS = (
    "current_step", "true_task_arrival_profile", "task_arrival_forecast",
    "lambda_t", "Q_t", "prev_loads", "prev_action",
    "bess_soc", "bess_energy_kWh", "total_energy_kWh",
    "total_idc_energy_kWh", "total_grid_energy_kWh", "total_cost",
    "total_carbon_emission", "total_carbon_cost", "episode_peak_power_kW",
    "total_peak_excess_kW_hour", "episode_grid_peak_power_kW",
    "total_grid_peak_excess_kW_hour", "total_completed_work",
    "total_bess_charge_kWh", "total_bess_discharge_kWh",
    "total_bess_degradation_cost", "total_pv_available_kWh",
    "total_pv_used_kWh", "total_pv_curtail_kWh",
    "deadline_miss_task_ids", "total_pause_count", "total_resume_count",
    "total_non_interruptible_interruption_count",
)

_SERVER_FIELDS = (
    "single_server_P_idle", "single_server_P_max", "P_idle", "P_max",
    "single_server_C_server", "C_server", "C_IDC_base", "C_IDC",
    "total_group_capacity", "total_group_idle_power_kW",
    "total_group_max_power_kW", "server_compute_efficiency",
)


def _chain(padded: Any) -> tuple[Any, Any, Any, Any, Any]:
    bridge = padded.env
    multi = bridge.env
    grid = multi.env
    base = grid.env
    names = tuple(type(value).__name__ for value in (padded, bridge, multi, grid, base))
    expected = (
        "HarlPaddedBridge", "HarlIDCGridBridge", "IDCGridMultiAgentEnv",
        "GridCoupledEnv", "IDCPriceEnv20D",
    )
    if names != expected:
        raise TypeError(f"Unsupported checkpoint environment chain: {names!r}.")
    return padded, bridge, multi, grid, base


def _generator_state(generator: Any) -> dict[str, Any]:
    return copy.deepcopy(generator.bit_generator.state)


def _task_state(task: Task) -> dict[str, Any]:
    return {key: copy.deepcopy(value) for key, value in task.__dict__.items()}


def _restore_task(state: dict[str, Any]) -> Task:
    required = {
        "task_id", "profile_key", "name", "arrival_time", "duration",
        "load_profile", "workload", "deadline", "priority", "interruptible",
        "parallelizable",
    }
    missing = sorted(required.difference(state))
    if missing:
        raise ValueError(f"Task checkpoint is missing fields: {missing}.")
    task = Task(**{key: copy.deepcopy(state[key]) for key in required})
    for key, value in state.items():
        setattr(task, key, copy.deepcopy(value))
    return task


def _dataclass_state(value: OPFResult | MEFResult) -> dict[str, Any]:
    return {field.name: copy.deepcopy(getattr(value, field.name)) for field in fields(value)}


def _cache_state(cache: Any) -> dict[str, Any]:
    return {
        "config": copy.deepcopy(cache.config),
        "opf_entries": [
            (copy.deepcopy(key), _dataclass_state(value))
            for key, value in cache._opf_cache.items()
        ],
        "mef_entries": [
            (copy.deepcopy(key), _dataclass_state(value))
            for key, value in cache._mef_cache.items()
        ],
        "opf_hit_count": int(cache._opf_hit_count),
        "opf_miss_count": int(cache._opf_miss_count),
        "mef_hit_count": int(cache._mef_hit_count),
        "mef_miss_count": int(cache._mef_miss_count),
    }


def _restore_cache(cache: Any, state: dict[str, Any]) -> None:
    if dict(cache.config) != dict(state["config"]):
        raise ValueError("Grid cache configuration differs from the checkpoint.")
    cache._opf_cache = OrderedDict(
        (copy.deepcopy(key), OPFResult(**copy.deepcopy(value)))
        for key, value in state["opf_entries"]
    )
    cache._mef_cache = OrderedDict(
        (copy.deepcopy(key), MEFResult(**copy.deepcopy(value)))
        for key, value in state["mef_entries"]
    )
    cache._opf_hit_count = int(state["opf_hit_count"])
    cache._opf_miss_count = int(state["opf_miss_count"])
    cache._mef_hit_count = int(state["mef_hit_count"])
    cache._mef_miss_count = int(state["mef_miss_count"])


def single_environment_state_dict(padded_env: Any) -> dict[str, Any]:
    """Return one worker's mutable state at a post-update boundary."""
    padded, bridge, multi, grid, base = _chain(padded_env)
    model = base.model
    return {
        "version": WORKER_ENVIRONMENT_STATE_VERSION,
        "padded_bridge": {
            "episode": int(padded._episode),
            "episode_step": int(padded._episode_step),
            "global_step": int(padded._global_step),
            "last_true_observations": copy.deepcopy(padded._last_true_observations),
        },
        "bridge": {
            "pending_seed": bridge._pending_seed,
            "last_info": copy.deepcopy(bridge._last_info),
        },
        "multi_agent": {
            "last_info": copy.deepcopy(multi._last_info),
            "last_raw_observation": copy.deepcopy(multi._last_raw_observation),
            "last_flat_action": copy.deepcopy(multi._last_flat_action),
        },
        "base_environment": {
            "dynamic": {
                key: copy.deepcopy(getattr(base, key)) for key in _BASE_DYNAMIC_FIELDS
            },
            "tasks": [_task_state(task) for task in base.tasks],
        },
        "model": {
            "task_rng": _generator_state(model.task_rng),
            "server_rng": _generator_state(model.server_rng),
            "server_parameters": {
                key: copy.deepcopy(getattr(model, key)) for key in _SERVER_FIELDS
            },
        },
        "task_forecast": {
            "mode": str(base.task_forecast_mode),
            "error_level": float(base.forecast_error_level),
            "seed": base.forecast_seed,
            "rng": _generator_state(base.forecast_rng),
        },
        "grid": {
            "scenario_source": str(grid.grid_scenario_source),
            "scenario_load_scale": grid.grid_load_scale_t.copy(),
            "scenario_reference_usep": grid.grid_reference_usep_t.copy(),
            "cache": _cache_state(grid.grid_cache),
        },
    }


def environment_state_dict(vec_env: Any) -> dict[str, Any]:
    """Return all workers' state without requiring subprocess object access."""
    if hasattr(vec_env, "get_env_states"):
        workers = vec_env.get_env_states()
    else:
        workers = [single_environment_state_dict(env) for env in vec_env.envs]
    worker_seeds = list(getattr(vec_env, "worker_seeds", ()))
    return {
        "version": ENVIRONMENT_STATE_VERSION,
        "worker_count": len(workers),
        "worker_seeds": worker_seeds,
        "vec_env_type": str(getattr(vec_env, "vec_env_type", type(vec_env).__name__)),
        "multiprocessing_start_method": getattr(vec_env, "multiprocessing_start_method", None),
        "workers": workers,
    }


def load_single_environment_state_dict(padded_env: Any, state: dict[str, Any]) -> None:
    """Restore one worker state without calling reset()."""
    if state.get("version") != WORKER_ENVIRONMENT_STATE_VERSION:
        raise ValueError(f"Unsupported worker environment state: {state.get('version')!r}.")
    padded, bridge, multi, grid, base = _chain(padded_env)
    pstate = state["padded_bridge"]
    padded._episode = int(pstate["episode"])
    padded._episode_step = int(pstate["episode_step"])
    padded._global_step = int(pstate["global_step"])
    padded._last_true_observations = copy.deepcopy(pstate["last_true_observations"])

    bridge._pending_seed = state["bridge"]["pending_seed"]
    bridge._last_info = copy.deepcopy(state["bridge"]["last_info"])
    multi._last_info = copy.deepcopy(state["multi_agent"]["last_info"])
    multi._last_raw_observation = copy.deepcopy(state["multi_agent"]["last_raw_observation"])
    multi._last_flat_action = copy.deepcopy(state["multi_agent"]["last_flat_action"])
    for key, value in state["base_environment"]["dynamic"].items():
        setattr(base, key, copy.deepcopy(value))
    # Preserve the compatibility alias without allowing two truth arrays to drift.
    base.lambda_t = base.true_task_arrival_profile
    base.tasks = [_restore_task(value) for value in state["base_environment"]["tasks"]]
    model = base.model
    model.task_rng.bit_generator.state = copy.deepcopy(state["model"]["task_rng"])
    model.server_rng.bit_generator.state = copy.deepcopy(state["model"]["server_rng"])
    for key, value in state["model"]["server_parameters"].items():
        setattr(model, key, copy.deepcopy(value))

    forecast_state = state["task_forecast"]
    expected_forecast = (
        str(base.task_forecast_mode),
        float(base.forecast_error_level),
        base.forecast_seed,
    )
    saved_forecast = (
        str(forecast_state["mode"]),
        float(forecast_state["error_level"]),
        forecast_state["seed"],
    )
    if saved_forecast != expected_forecast:
        raise ValueError(
            "Task forecast configuration differs from the checkpoint: "
            f"checkpoint={saved_forecast!r}, current={expected_forecast!r}."
        )
    base.forecast_rng.bit_generator.state = copy.deepcopy(forecast_state["rng"])

    gstate = state["grid"]
    if str(grid.grid_scenario_source) != str(gstate["scenario_source"]):
        raise ValueError("Grid scenario source differs from the checkpoint.")
    np.testing.assert_array_equal(grid.grid_load_scale_t, gstate["scenario_load_scale"])
    np.testing.assert_array_equal(grid.grid_reference_usep_t, gstate["scenario_reference_usep"])
    _restore_cache(grid.grid_cache, gstate["cache"])


def load_environment_state_dict(vec_env: Any, state: dict[str, Any]) -> None:
    """Restore all workers, supporting Part-9's legacy one-worker payload."""
    version = state.get("version")
    if version == LEGACY_ENVIRONMENT_STATE_VERSION:
        raise ValueError(
            "Legacy environment state v1 has no separated task forecast and cannot "
            "be resumed under the non-oracle forecast contract."
        )
    if version != ENVIRONMENT_STATE_VERSION:
        raise ValueError(f"Unsupported environment state version: {version!r}.")
    workers = state["workers"]
    expected = int(getattr(vec_env, "num_envs", len(getattr(vec_env, "envs", ()))))
    if int(state["worker_count"]) != expected or len(workers) != expected:
        raise ValueError(
            f"Checkpoint worker count differs: checkpoint={len(workers)}, current={expected}."
        )
    current_seeds = list(getattr(vec_env, "worker_seeds", ()))
    if list(state.get("worker_seeds", ())) != current_seeds:
        raise ValueError(
            f"Checkpoint worker seeds differ: checkpoint={state.get('worker_seeds')}, current={current_seeds}."
        )
    if hasattr(vec_env, "set_env_states"):
        vec_env.set_env_states(workers)
    else:
        for env, worker_state in zip(vec_env.envs, workers, strict=True):
            load_single_environment_state_dict(env, worker_state)
