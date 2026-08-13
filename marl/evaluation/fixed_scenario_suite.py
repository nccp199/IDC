"""Materialized, integrity-checked fixed evaluation scenario suites."""

from __future__ import annotations

import copy
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from configs.config_ultimate import (
    GRID_CACHE_CONFIG,
    GRID_CONFIG,
    GRID_REWARD_CONFIG,
    GRID_SCENARIO_CONFIG,
)
from configs.experiment_cases import get_experiment_case
from marl.checkpointing.environment_state import (
    load_single_environment_state_dict,
    single_environment_state_dict,
)
from marl.envs.harl_env_factory import make_harl_single_env
from marl.specs import ACTION_PADDING_STRATEGY, AGENTS, EFFECTIVE_ACTION_DIMS, PADDED_ACTION_DIMS


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SUITE_SCHEMA_VERSION = "idc-bess-fixed-suite-v1"
SCENARIO_SCHEMA_VERSION = "idc-bess-fixed-scenario-v2"
EPISODE_LENGTH = 24


class SuiteIntegrityError(RuntimeError):
    """A materialized suite or scenario failed an integrity check."""


class SuiteCompatibilityError(RuntimeError):
    """A suite does not match the current formal environment contract."""


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return {"dtype": str(value.dtype), "shape": list(value.shape), "data": value.tolist()}
    if isinstance(value, np.generic):
        return value.item()
    if torch.is_tensor(value):
        array = value.detach().cpu().numpy()
        return {"dtype": str(array.dtype), "shape": list(array.shape), "data": array.tolist()}
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, set):
        return sorted(_jsonable(item) for item in value)
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f"Unsupported canonical hash value: {type(value).__name__}.")


def canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        _jsonable(value), ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_head(repository: Path) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "-C", str(repository), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(_jsonable(value), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def environment_fingerprints(experiment_case: str) -> dict[str, str]:
    case = get_experiment_case(experiment_case)
    environment_contract = {
        "experiment_env": case["env_config"],
        "grid": GRID_CONFIG,
        "grid_reward": GRID_REWARD_CONFIG,
        "grid_scenario": GRID_SCENARIO_CONFIG,
        "grid_cache": GRID_CACHE_CONFIG,
        "agent_order": list(AGENTS),
        "observation_dims": [288, 288],
        "state_dim": 294,
        "padded_action_dims": [PADDED_ACTION_DIMS[agent] for agent in AGENTS],
        "effective_action_dims": [EFFECTIVE_ACTION_DIMS[agent] for agent in AGENTS],
        "action_padding_strategy": ACTION_PADDING_STRATEGY,
        "episode_length": EPISODE_LENGTH,
    }
    return {
        "environment_config_fingerprint": canonical_sha256(environment_contract),
        "reward_config_fingerprint": canonical_sha256(case["reward_config"]),
        "data_config_fingerprint": canonical_sha256(case["data_config"]),
    }


def _scenario_payload(
    *,
    suite_id: str,
    scenario_id: str,
    seed: int,
    experiment_case: str,
    project_git_head: str | None,
    harl_git_head: str | None,
) -> dict[str, Any]:
    env = make_harl_single_env(seed=int(seed), experiment_case=experiment_case)
    try:
        obs, state, available_actions = env.reset()
        info = env.last_info or {}
        base_env = env.env.env.env.env
        grid_source = Path(str(info["grid_scenario_source"])).resolve()
        fingerprints = environment_fingerprints(experiment_case)
        payload = {
            "scenario_schema_version": SCENARIO_SCHEMA_VERSION,
            "suite_id": suite_id,
            "scenario_id": scenario_id,
            "scenario_seed": int(seed),
            "experiment_case": experiment_case,
            "episode_length": EPISODE_LENGTH,
            "agent_order": list(AGENTS),
            "observation_dims": [288, 288],
            "state_dim": 294,
            "padded_action_dims": [PADDED_ACTION_DIMS[agent] for agent in AGENTS],
            "effective_action_dims": [EFFECTIVE_ACTION_DIMS[agent] for agent in AGENTS],
            "action_padding_strategy": ACTION_PADDING_STRATEGY,
            "initial_observation": np.asarray(obs, dtype=np.float32).copy(),
            "initial_state": np.asarray(state, dtype=np.float32).copy(),
            "initial_available_actions": copy.deepcopy(available_actions),
            "environment_state": single_environment_state_dict(env),
            "task_forecast": {
                "mode": str(base_env.task_forecast_mode),
                "error_level": float(base_env.forecast_error_level),
                "seed": base_env.forecast_seed,
                "source": str(info["task_forecast_source"]),
                "mae": float(info["task_forecast_mae"]),
                "rmse": float(info["task_forecast_rmse"]),
                "mape_nonzero_percent": float(
                    info["task_forecast_mape_nonzero_percent"]
                ),
            },
            "external_curves": {
                "task_arrival_lambda": np.asarray(
                    base_env.true_task_arrival_profile, dtype=np.float64
                ).copy(),
                "true_task_arrival_profile": np.asarray(
                    base_env.true_task_arrival_profile, dtype=np.float64
                ).copy(),
                "task_arrival_forecast": np.asarray(
                    base_env.task_arrival_forecast, dtype=np.float64
                ).copy(),
                "price": np.asarray(base_env.price_t, dtype=np.float64).copy(),
                "carbon_factor": np.asarray(base_env.carbon_factor_t, dtype=np.float64).copy(),
                "pv": np.asarray(base_env.pv_t, dtype=np.float64).copy(),
                "temperature": None,
                "temperature_available": False,
                "grid_load_scale": np.asarray(
                    env.env.env.env.grid_load_scale_t, dtype=np.float64
                ).copy(),
                "grid_reference_usep": np.asarray(
                    env.env.env.env.grid_reference_usep_t, dtype=np.float64
                ).copy(),
            },
            "grid_csv_source": str(grid_source),
            "grid_csv_sha256": file_sha256(grid_source),
            "grid_case": str(GRID_CONFIG["case_name"]),
            "cache_bins": {
                "opf_load_mw": float(info["grid_cache_load_bin_mw"]),
                "mef_load_mw": float(info["grid_cache_mef_load_bin_mw"]),
                "load_scale": float(GRID_CACHE_CONFIG["cache_load_scale_bin"]),
            },
            **fingerprints,
            "project_git_head": project_git_head,
            "harl_git_head": harl_git_head,
        }
        payload["scenario_content_sha256"] = canonical_sha256(payload)
        return payload
    finally:
        env.close()


def generate_suite(
    *,
    suite_id: str,
    suite_type: str,
    scenario_seeds: Sequence[int],
    output_dir: str | Path,
    experiment_case: str = "main",
    harl_source: str | Path | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Materialize one suite; no live environment object is serialized."""
    if suite_type not in {"smoke", "validation", "test"}:
        raise ValueError("suite_type must be smoke, validation, or test.")
    seeds = [int(seed) for seed in scenario_seeds]
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("A suite requires a non-empty list of unique scenario seeds.")
    directory = Path(output_dir).resolve()
    if directory.exists() and any(directory.iterdir()) and not overwrite:
        raise FileExistsError(f"Suite directory is not empty: {directory}")
    directory.mkdir(parents=True, exist_ok=True)
    if overwrite:
        for existing in directory.iterdir():
            if existing.is_file():
                existing.unlink()
            else:
                raise ValueError(f"Refusing to overwrite nested suite directory: {existing}")

    project_head = _git_head(PROJECT_ROOT)
    harl_path = Path(harl_source).resolve() if harl_source else PROJECT_ROOT.parent / "HARL"
    harl_head = _git_head(harl_path)
    scenarios = []
    for index, seed in enumerate(seeds):
        scenario_id = f"scenario_{index:03d}_seed_{seed:05d}"
        filename = f"{scenario_id}.pt"
        scenario_path = directory / filename
        payload = _scenario_payload(
            suite_id=suite_id,
            scenario_id=scenario_id,
            seed=seed,
            experiment_case=experiment_case,
            project_git_head=project_head,
            harl_git_head=harl_head,
        )
        torch.save(payload, scenario_path)
        snapshot_digest = file_sha256(scenario_path)
        scenario_manifest = {
            "scenario_schema_version": SCENARIO_SCHEMA_VERSION,
            "suite_id": suite_id,
            "scenario_id": scenario_id,
            "scenario_seed": seed,
            "scenario_file": filename,
            "scenario_size_bytes": scenario_path.stat().st_size,
            "scenario_file_sha256": snapshot_digest,
            "scenario_content_sha256": payload["scenario_content_sha256"],
            "environment_config_fingerprint": payload["environment_config_fingerprint"],
            "reward_config_fingerprint": payload["reward_config_fingerprint"],
            "data_config_fingerprint": payload["data_config_fingerprint"],
            "grid_csv_sha256": payload["grid_csv_sha256"],
        }
        manifest_name = f"{scenario_id}.manifest.json"
        _write_json(directory / manifest_name, scenario_manifest)
        scenarios.append({**scenario_manifest, "scenario_manifest": manifest_name})

    fingerprints = environment_fingerprints(experiment_case)
    suite_stable = {
        "suite_schema_version": SUITE_SCHEMA_VERSION,
        "suite_id": suite_id,
        "suite_type": suite_type,
        "scenario_ids": [item["scenario_id"] for item in scenarios],
        "scenario_seeds": seeds,
        "scenario_content_sha256": [item["scenario_content_sha256"] for item in scenarios],
        "experiment_case": experiment_case,
        **fingerprints,
        "grid_csv_sha256": scenarios[0]["grid_csv_sha256"],
        "grid_case": GRID_CONFIG["case_name"],
        "episode_length": EPISODE_LENGTH,
        "agent_order": list(AGENTS),
        "observation_dims": [288, 288],
        "state_dim": 294,
        "padded_action_dims": [22, 22],
        "effective_action_dims": [22, 1],
        "cache_bins": scenarios[0] and torch.load(
            directory / scenarios[0]["scenario_file"], map_location="cpu", weights_only=False
        )["cache_bins"],
        "project_git_head": project_head,
        "harl_git_head": harl_head,
    }
    suite_manifest = {
        **suite_stable,
        "scenario_count": len(scenarios),
        "scenario_files": [item["scenario_file"] for item in scenarios],
        "scenario_manifests": [item["scenario_manifest"] for item in scenarios],
        "scenario_file_sha256": [item["scenario_file_sha256"] for item in scenarios],
        "scenario_size_bytes": [item["scenario_size_bytes"] for item in scenarios],
        "suite_sha256": canonical_sha256(suite_stable),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    _write_json(directory / "suite_manifest.json", suite_manifest)
    return suite_manifest


def load_suite(directory: str | Path, *, verify_compatibility: bool = True) -> dict[str, Any]:
    suite_dir = Path(directory).resolve()
    manifest_path = suite_dir / "suite_manifest.json"
    if not manifest_path.is_file():
        raise SuiteIntegrityError(f"Suite manifest is missing: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("suite_schema_version") != SUITE_SCHEMA_VERSION:
        raise SuiteIntegrityError(f"Unsupported suite schema: {manifest.get('suite_schema_version')!r}.")
    if int(manifest.get("scenario_count", -1)) != len(manifest.get("scenario_files", [])):
        raise SuiteIntegrityError("Suite scenario_count does not match scenario_files.")
    stable_keys = {
        key: manifest[key]
        for key in (
            "suite_schema_version", "suite_id", "suite_type", "scenario_ids",
            "scenario_seeds", "scenario_content_sha256", "experiment_case",
            "environment_config_fingerprint", "reward_config_fingerprint",
            "data_config_fingerprint", "grid_csv_sha256", "grid_case",
            "episode_length", "agent_order", "observation_dims", "state_dim",
            "padded_action_dims", "effective_action_dims", "cache_bins",
            "project_git_head", "harl_git_head",
        )
    }
    if canonical_sha256(stable_keys) != manifest.get("suite_sha256"):
        raise SuiteIntegrityError("Suite content hash is invalid.")
    if verify_compatibility:
        expected = environment_fingerprints(str(manifest["experiment_case"]))
        for key, value in expected.items():
            if manifest.get(key) != value:
                raise SuiteCompatibilityError(f"Suite {key} differs from current configuration.")
        expected_contract = {
            "episode_length": EPISODE_LENGTH,
            "agent_order": list(AGENTS),
            "observation_dims": [288, 288],
            "state_dim": 294,
            "padded_action_dims": [22, 22],
            "effective_action_dims": [22, 1],
        }
        for key, value in expected_contract.items():
            if manifest.get(key) != value:
                raise SuiteCompatibilityError(f"Suite {key} is incompatible: {manifest.get(key)!r}.")
        configured_grid = Path(str(GRID_SCENARIO_CONFIG["grid_load_scale_path"]))
        if not configured_grid.is_absolute():
            configured_grid = PROJECT_ROOT / configured_grid
        if not configured_grid.is_file() or file_sha256(configured_grid) != manifest["grid_csv_sha256"]:
            raise SuiteCompatibilityError("Suite Grid CSV hash differs from the configured data file.")
        current_harl_head = _git_head(PROJECT_ROOT.parent / "HARL")
        if manifest.get("harl_git_head") and current_harl_head != manifest["harl_git_head"]:
            raise SuiteCompatibilityError(
                f"Suite HARL HEAD differs: suite={manifest['harl_git_head']}, current={current_harl_head}."
            )
    manifest["directory"] = str(suite_dir)
    return manifest


def load_scenario(suite: Mapping[str, Any], scenario_id: str) -> dict[str, Any]:
    try:
        index = list(suite["scenario_ids"]).index(str(scenario_id))
    except ValueError as exc:
        raise KeyError(f"Unknown scenario ID {scenario_id!r}.") from exc
    directory = Path(str(suite["directory"]))
    scenario_path = directory / suite["scenario_files"][index]
    manifest_path = directory / suite["scenario_manifests"][index]
    if not scenario_path.is_file() or not manifest_path.is_file():
        raise SuiteIntegrityError(f"Scenario files are missing for {scenario_id}.")
    scenario_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected_size = int(suite["scenario_size_bytes"][index])
    if int(scenario_manifest["scenario_size_bytes"]) != expected_size or scenario_path.stat().st_size != expected_size:
        raise SuiteIntegrityError(f"Scenario size differs for {scenario_id}.")
    expected_file_hash = suite["scenario_file_sha256"][index]
    if scenario_manifest["scenario_file_sha256"] != expected_file_hash or file_sha256(scenario_path) != expected_file_hash:
        raise SuiteIntegrityError(f"Scenario file hash differs for {scenario_id}.")
    payload = torch.load(scenario_path, map_location="cpu", weights_only=False)
    if payload.get("scenario_schema_version") != SCENARIO_SCHEMA_VERSION:
        raise SuiteIntegrityError(f"Unsupported scenario schema for {scenario_id}.")
    digest = payload.pop("scenario_content_sha256", None)
    computed = canonical_sha256(payload)
    payload["scenario_content_sha256"] = digest
    expected_content_hash = suite["scenario_content_sha256"][index]
    if (
        digest != computed
        or digest != scenario_manifest["scenario_content_sha256"]
        or digest != expected_content_hash
    ):
        raise SuiteIntegrityError(f"Scenario content hash differs for {scenario_id}.")
    if payload["suite_id"] != suite["suite_id"] or payload["scenario_id"] != scenario_id:
        raise SuiteIntegrityError(f"Scenario identity differs for {scenario_id}.")
    return payload


def restore_scenario_environment(payload: Mapping[str, Any]):
    """Create a fresh single environment and restore without an evaluation reset."""
    env = make_harl_single_env(
        seed=int(payload["scenario_seed"]),
        experiment_case=str(payload["experiment_case"]),
    )
    try:
        load_single_environment_state_dict(env, copy.deepcopy(payload["environment_state"]))
    except BaseException:
        env.close()
        raise
    return (
        env,
        np.asarray(payload["initial_observation"], dtype=np.float32).copy(),
        np.asarray(payload["initial_state"], dtype=np.float32).copy(),
        copy.deepcopy(payload["initial_available_actions"]),
    )
