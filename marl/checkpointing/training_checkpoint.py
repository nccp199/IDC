"""Atomic, integrity-checked post-update MAPPO training checkpoints."""

from __future__ import annotations

import copy
import hashlib
import json
import os
import random
import re
import shutil
import time
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

from .environment_state import environment_state_dict, load_environment_state_dict


CHECKPOINT_SCHEMA_VERSION = "idc-mappo-training-resume-v1"
CHECKPOINT_TYPE = "training_resume"


class CheckpointIntegrityError(RuntimeError):
    """A checkpoint or its manifest is absent, truncated, or hash-mismatched."""


class CheckpointCompatibilityError(RuntimeError):
    """The current runner/config cannot safely continue a checkpoint."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _atomic_torch_save(path: Path, value: Any) -> None:
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as stream:
        torch.save(value, stream)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _atomic_copy(source: Path, target: Path) -> None:
    temporary = target.with_name(target.name + ".tmp")
    with source.open("rb") as input_stream, temporary.open("wb") as output_stream:
        shutil.copyfileobj(input_stream, output_stream, 1024 * 1024)
        output_stream.flush()
        os.fsync(output_stream.fileno())
    os.replace(temporary, target)


def _torch_load(path: Path, *, map_location: Any) -> Any:
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def _rng_state() -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy_global": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def _load_rng_state(state: Mapping[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy_global"])
    torch.set_rng_state(state["torch_cpu"].cpu())
    if state.get("torch_cuda") is not None:
        if not torch.cuda.is_available():
            raise CheckpointCompatibilityError(
                "Checkpoint contains CUDA RNG state but CUDA is unavailable."
            )
        torch.cuda.set_rng_state_all(state["torch_cuda"])


def _rollout_state(runner: Any) -> dict[str, Any]:
    actors = []
    for buffer in runner.actor_buffer:
        actors.append(
            {
                "obs": buffer.obs[0].copy(),
                "rnn_states": buffer.rnn_states[0].copy(),
                "masks": buffer.masks[0].copy(),
                "active_masks": buffer.active_masks[0].copy(),
                "available_actions": (
                    None
                    if buffer.available_actions is None
                    else buffer.available_actions[0].copy()
                ),
            }
        )
    critic = runner.critic_buffer
    return {
        "actor_buffers": actors,
        "critic_buffer": {
            "share_obs": critic.share_obs[0].copy(),
            "rnn_states_critic": critic.rnn_states_critic[0].copy(),
            "masks": critic.masks[0].copy(),
            "bad_masks": critic.bad_masks[0].copy(),
        },
    }


def _load_rollout_state(runner: Any, state: Mapping[str, Any]) -> None:
    if len(state["actor_buffers"]) != len(runner.actor_buffer):
        raise CheckpointCompatibilityError("Actor-buffer count differs from checkpoint.")
    for buffer, saved in zip(runner.actor_buffer, state["actor_buffers"], strict=True):
        buffer.obs[0] = saved["obs"].copy()
        buffer.rnn_states[0] = saved["rnn_states"].copy()
        buffer.masks[0] = saved["masks"].copy()
        buffer.active_masks[0] = saved["active_masks"].copy()
        if (buffer.available_actions is None) != (saved["available_actions"] is None):
            raise CheckpointCompatibilityError("Available-action buffer semantics differ.")
        if buffer.available_actions is not None:
            buffer.available_actions[0] = saved["available_actions"].copy()
    critic = runner.critic_buffer
    saved_critic = state["critic_buffer"]
    critic.share_obs[0] = saved_critic["share_obs"].copy()
    critic.rnn_states_critic[0] = saved_critic["rnn_states_critic"].copy()
    critic.masks[0] = saved_critic["masks"].copy()
    critic.bad_masks[0] = saved_critic["bad_masks"].copy()


def _verification_state(runner: Any) -> dict[str, Any]:
    return {
        "actor_actions": [buffer.actions.copy() for buffer in runner.actor_buffer],
        "actor_action_log_probs": [
            buffer.action_log_probs.copy() for buffer in runner.actor_buffer
        ],
        "critic_rewards": runner.critic_buffer.rewards.copy(),
        "last_update_metrics": copy.deepcopy(
            getattr(runner.metrics_logger, "last_update_metrics", None)
        ),
        "last_episode_metrics": copy.deepcopy(
            getattr(runner.metrics_logger, "last_episode_metrics", None)
        ),
        "actor_diagnostics": [
            copy.deepcopy(getattr(actor, "last_update_diagnostics", None))
            for actor in runner.actor
        ],
        "algorithm_update_audit": copy.deepcopy(
            getattr(runner, "last_algorithm_update", None)
        ),
        "stability_counters": copy.deepcopy(
            getattr(runner, "stability_counters", None)
        ),
    }


def _compatibility_differences(expected: Any, actual: Any, prefix: str = "") -> list[str]:
    if isinstance(expected, Mapping) and isinstance(actual, Mapping):
        differences: list[str] = []
        for key in sorted(set(expected).union(actual)):
            path = f"{prefix}.{key}" if prefix else str(key)
            if key not in expected:
                differences.append(f"{path}: unexpected current value {actual[key]!r}")
            elif key not in actual:
                differences.append(f"{path}: missing current value; checkpoint={expected[key]!r}")
            else:
                differences.extend(_compatibility_differences(expected[key], actual[key], path))
        return differences
    if isinstance(expected, (list, tuple)) and isinstance(actual, (list, tuple)):
        if len(expected) != len(actual):
            return [f"{prefix}: length checkpoint={len(expected)}, current={len(actual)}"]
        differences = []
        for index, (left, right) in enumerate(zip(expected, actual, strict=True)):
            differences.extend(_compatibility_differences(left, right, f"{prefix}[{index}]"))
        return differences
    return [] if expected == actual else [f"{prefix}: checkpoint={expected!r}, current={actual!r}"]


def _method_fields(runner: Any) -> dict[str, Any]:
    return {
        "algorithm_name": runner.method.algorithm_name,
        "critic_type": runner.method.critic_type,
        "method_id": runner.method.method_id,
        "agent_update_order_policy": "fixed" if runner.fixed_order else "torch_randperm",
    }


def _graph_metadata(runner: Any) -> dict[str, Any]:
    value = getattr(runner.critic, "graph_metadata", None)
    return {} if not value else copy.deepcopy(dict(value))


def _validate_method_metadata(payload: Mapping[str, Any], runner: Any) -> None:
    """Reject cross-algorithm, cross-critic, or cross-order checkpoint restores."""
    legacy_mappo = payload.get("algorithm") == "mappo" and "method_id" not in payload
    checkpoint_method = {
        "algorithm_name": payload.get("algorithm_name", payload.get("algorithm")),
        "critic_type": payload.get("critic_type", "mlp" if legacy_mappo else None),
        "method_id": payload.get("method_id", "MAPPO_MLP" if legacy_mappo else None),
        "agent_update_order_policy": payload.get(
            "agent_update_order_policy", "fixed" if legacy_mappo else None
        ),
    }
    expected_fields = _method_fields(runner)
    if checkpoint_method != expected_fields:
        raise CheckpointCompatibilityError(
            f"Checkpoint method compatibility differs: checkpoint={checkpoint_method!r}, "
            f"current={expected_fields!r}."
        )


class TrainingCheckpointManager:
    """Own checkpoint files for one run and restore only complete resume payloads."""

    def __init__(
        self,
        *,
        runner: Any,
        compatibility: Mapping[str, Any],
        provenance: Mapping[str, Any],
        config: Mapping[str, Any],
    ) -> None:
        self.runner = runner
        self.compatibility = copy.deepcopy(dict(compatibility))
        self.provenance = copy.deepcopy(dict(provenance))
        self.enabled = bool(config.get("enabled", True))
        self.interval_updates = int(config.get("interval_updates", 5))
        self.save_final_enabled = bool(config.get("save_final", True))
        keep_last = config.get("keep_last")
        self.keep_last = None if keep_last is None else int(keep_last)
        if self.interval_updates <= 0:
            raise ValueError("checkpoint.interval_updates must be positive.")
        if self.keep_last is not None and self.keep_last <= 0:
            raise ValueError("checkpoint.keep_last must be null or positive.")
        self.directory = Path(runner.run_dir).resolve() / "checkpoints"
        if self.enabled:
            self.directory.mkdir(parents=False, exist_ok=False)
        self.last_record: dict[str, Any] | None = None
        self.final_record: dict[str, Any] | None = None
        self.last_save_seconds: float | None = None
        self.last_load_seconds: float | None = None

    def _payload(self, update: int) -> dict[str, Any]:
        logger_state = self.runner.metrics_logger.state_dict()
        return {
            "checkpoint_schema_version": CHECKPOINT_SCHEMA_VERSION,
            "checkpoint_type": CHECKPOINT_TYPE,
            "algorithm": self.runner.method.algorithm_name,
            "algorithm_name": self.runner.method.algorithm_name,
            "critic_type": self.runner.method.critic_type,
            "method_id": self.runner.method.method_id,
            "graph_metadata": _graph_metadata(self.runner),
            "algorithm_implementation_version": (
                self.runner.method.algorithm_implementation_version
            ),
            "critic_interface_version": self.runner.method.critic_interface_version,
            "agent_update_order_policy": (
                "fixed" if self.runner.fixed_order else "torch_randperm"
            ),
            "resume_boundary": "post_update_only",
            "mid_rollout_resume_supported": False,
            "saved_at_update": int(update),
            "global_step": int(logger_state["global_step"]),
            "episodes_completed": int(logger_state["episodes_completed"]),
            "model_state": {
                "agent_order": ["idc", "bess"],
                "actor_agent0_state_dict": self.runner.actor[0].actor.state_dict(),
                "actor_agent1_state_dict": self.runner.actor[1].actor.state_dict(),
                "critic_state_dict": self.runner.critic.critic.state_dict(),
            },
            "optimizer_state": {
                "actor_agent0_optimizer_state": self.runner.actor[0].actor_optimizer.state_dict(),
                "actor_agent1_optimizer_state": self.runner.actor[1].actor_optimizer.state_dict(),
                "critic_optimizer_state": self.runner.critic.critic_optimizer.state_dict(),
            },
            "normalizer_state": {
                "use_valuenorm": self.runner.value_normalizer is not None,
                "state_dict": (
                    None
                    if self.runner.value_normalizer is None
                    else self.runner.value_normalizer.state_dict()
                ),
            },
            "rng_state": _rng_state(),
            "environment_state": environment_state_dict(self.runner.envs),
            "runner_state": {
                "current_update": int(update),
                "next_update": int(update) + 1,
                "global_step": int(logger_state["global_step"]),
                "episodes_completed": int(logger_state["episodes_completed"]),
                "rollout_threads": int(self.runner.algo_args["train"]["n_rollout_threads"]),
                "episode_length": int(self.runner.algo_args["train"]["episode_length"]),
                "total_updates_target": int(self.runner.total_updates_target),
                "stability_counters": copy.deepcopy(
                    getattr(self.runner, "stability_counters", {})
                ),
                "rollout_state": _rollout_state(self.runner),
            },
            "logger_state": logger_state,
            "verification_state": _verification_state(self.runner),
            "compatibility": copy.deepcopy(self.compatibility),
            "provenance": copy.deepcopy(self.provenance),
        }

    def maybe_save(self, update: int) -> dict[str, Any] | None:
        if not self.enabled or int(update) % self.interval_updates:
            return None
        return self.save(update)

    def save(self, update: int) -> dict[str, Any]:
        if not self.enabled:
            raise RuntimeError("Checkpoint saving is disabled.")
        started = time.perf_counter()
        checkpoint_path = self.directory / f"update_{int(update):06d}.pt"
        manifest_path = self.directory / f"update_{int(update):06d}.manifest.json"
        if checkpoint_path.exists() or manifest_path.exists():
            raise FileExistsError(f"Checkpoint for update {update} already exists in this run.")
        payload = self._payload(update)
        _atomic_torch_save(checkpoint_path, payload)
        digest = _sha256(checkpoint_path)
        manifest = self._manifest(payload, checkpoint_path, digest)
        _atomic_json(manifest_path, manifest)
        record = {
            "checkpoint": str(checkpoint_path),
            "manifest": str(manifest_path),
            "sha256": digest,
            "size_bytes": checkpoint_path.stat().st_size,
            "saved_update": int(update),
        }
        _atomic_json(
            self.directory / "latest.json",
            {
                "checkpoint": checkpoint_path.name,
                "manifest": manifest_path.name,
                "sha256": digest,
                "saved_update": int(update),
            },
        )
        self.last_record = record
        self.last_save_seconds = time.perf_counter() - started
        self._prune()
        return record

    def save_final(self, update: int) -> dict[str, Any] | None:
        if not self.enabled or not self.save_final_enabled:
            return None
        if self.last_record is None or self.last_record["saved_update"] != int(update):
            self.save(update)
        source = Path(self.last_record["checkpoint"])
        source_manifest = json.loads(Path(self.last_record["manifest"]).read_text(encoding="utf-8"))
        final_path = self.directory / "final.pt"
        final_manifest_path = self.directory / "final.manifest.json"
        if final_path.exists() or final_manifest_path.exists():
            raise FileExistsError("Final checkpoint already exists in this run.")
        _atomic_copy(source, final_path)
        digest = _sha256(final_path)
        final_manifest = dict(source_manifest)
        final_manifest.update(
            {"file": final_path.name, "size_bytes": final_path.stat().st_size, "sha256": digest}
        )
        _atomic_json(final_manifest_path, final_manifest)
        self.final_record = {
            "checkpoint": str(final_path),
            "manifest": str(final_manifest_path),
            "sha256": digest,
            "size_bytes": final_path.stat().st_size,
            "saved_update": int(update),
        }
        return self.final_record

    def load(self, source: str | Path) -> dict[str, Any]:
        started = time.perf_counter()
        checkpoint_path, manifest_path = self._resolve_source(Path(source).resolve())
        manifest = self._verify_manifest(checkpoint_path, manifest_path)
        payload = _torch_load(checkpoint_path, map_location=self.runner.device)
        if not isinstance(payload, dict) or payload.get("checkpoint_type") != CHECKPOINT_TYPE:
            raise CheckpointIntegrityError(
                "--resume-checkpoint requires a complete training_resume checkpoint; "
                "model-only weights are not resumable."
            )
        if payload.get("checkpoint_schema_version") != CHECKPOINT_SCHEMA_VERSION:
            raise CheckpointCompatibilityError(
                f"Unsupported checkpoint schema {payload.get('checkpoint_schema_version')!r}."
            )
        _validate_method_metadata(payload, self.runner)
        saved_compatibility = copy.deepcopy(payload.get("compatibility", {}))
        if payload.get("environment_state", {}).get("version") == "idc-bess-env-state-v1":
            provenance = payload.get("provenance", {})
            legacy_seed = provenance.get("seed")
            if legacy_seed is None:
                match = re.match(r"^seed-(\d+)-", str(provenance.get("run_id", "")))
                legacy_seed = None if match is None else int(match.group(1))
            saved_compatibility.update(
                {
                    "vec_env_type": "ShareDummyVecEnv",
                    "multiprocessing_start_method": None,
                    "worker_seeds": [] if legacy_seed is None else [legacy_seed],
                    # Part-9 predates only the non-trajectory parallel defaults.
                    "config_fingerprint": self.compatibility["config_fingerprint"],
                }
            )
        differences = _compatibility_differences(saved_compatibility, self.compatibility)
        if differences:
            raise CheckpointCompatibilityError(
                "Resume compatibility check failed:\n- " + "\n- ".join(differences)
            )
        provenance = payload.get("provenance", {})
        saved_seed = provenance.get("seed")
        if saved_seed is None:
            match = re.match(r"^seed-(\d+)-", str(provenance.get("run_id", "")))
            if match is None:
                raise CheckpointCompatibilityError("Checkpoint does not identify its training seed.")
            saved_seed = int(match.group(1))
        current_seed = int(self.runner.metrics_logger.seed)
        if int(saved_seed) != current_seed:
            raise CheckpointCompatibilityError(
                f"Training seed differs: checkpoint={saved_seed}, current={current_seed}."
            )

        models = payload["model_state"]
        if models.get("agent_order") != ["idc", "bess"]:
            raise CheckpointCompatibilityError("Checkpoint actor order is not ['idc', 'bess'].")
        self.runner.actor[0].actor.load_state_dict(
            models["actor_agent0_state_dict"], strict=True
        )
        self.runner.actor[1].actor.load_state_dict(
            models["actor_agent1_state_dict"], strict=True
        )
        self.runner.critic.critic.load_state_dict(models["critic_state_dict"], strict=True)
        optimizers = payload["optimizer_state"]
        self.runner.actor[0].actor_optimizer.load_state_dict(
            optimizers["actor_agent0_optimizer_state"]
        )
        self.runner.actor[1].actor_optimizer.load_state_dict(
            optimizers["actor_agent1_optimizer_state"]
        )
        self.runner.critic.critic_optimizer.load_state_dict(
            optimizers["critic_optimizer_state"]
        )
        normalizer = payload["normalizer_state"]
        if bool(normalizer["use_valuenorm"]) != (self.runner.value_normalizer is not None):
            raise CheckpointCompatibilityError("ValueNorm enablement differs from checkpoint.")
        if self.runner.value_normalizer is not None:
            self.runner.value_normalizer.load_state_dict(normalizer["state_dict"], strict=True)

        load_environment_state_dict(self.runner.envs, payload["environment_state"])
        _load_rollout_state(self.runner, payload["runner_state"]["rollout_state"])
        self.runner.metrics_logger.load_state_dict(
            payload["logger_state"],
            resume_parent=payload.get("provenance", {}).get("run_id"),
        )
        saved_update = int(payload["runner_state"]["current_update"])
        if "stability_counters" in payload["runner_state"]:
            self.runner.stability_counters = copy.deepcopy(
                payload["runner_state"]["stability_counters"]
            )
        self.runner.completed_updates = saved_update
        self.runner.resume_start_update = saved_update
        self.runner.resumed = True
        _load_rng_state(payload["rng_state"])
        self.last_load_seconds = time.perf_counter() - started
        return {
            "checkpoint": str(checkpoint_path),
            "manifest": str(manifest_path),
            "sha256": manifest["sha256"],
            "saved_update": saved_update,
            "global_step": int(payload["global_step"]),
            "episodes_completed": int(payload["episodes_completed"]),
            "parent_run_id": payload.get("provenance", {}).get("run_id"),
            "provenance": copy.deepcopy(payload.get("provenance", {})),
        }

    def _manifest(self, payload: Mapping[str, Any], path: Path, digest: str) -> dict[str, Any]:
        compatibility = payload["compatibility"]
        provenance = payload["provenance"]
        return {
            "file": path.name,
            "size_bytes": path.stat().st_size,
            "sha256": digest,
            "schema_version": CHECKPOINT_SCHEMA_VERSION,
            "checkpoint_type": CHECKPOINT_TYPE,
            "algorithm": payload["algorithm_name"],
            "algorithm_name": payload["algorithm_name"],
            "critic_type": payload["critic_type"],
            "method_id": payload["method_id"],
            "graph_metadata": copy.deepcopy(payload.get("graph_metadata", {})),
            "algorithm_implementation_version": payload[
                "algorithm_implementation_version"
            ],
            "critic_interface_version": payload["critic_interface_version"],
            "agent_update_order_policy": payload["agent_update_order_policy"],
            "rng_isolation_version": compatibility["rng_isolation_version"],
            "critic_init_seed_rule": compatibility["critic_init_seed_rule"],
            "critic_init_seed": compatibility["critic_init_seed"],
            "actor_sampling_rng_isolated_from_critic_init": compatibility[
                "actor_sampling_rng_isolated_from_critic_init"
            ],
            "saved_update": int(payload["saved_at_update"]),
            "global_step": int(payload["global_step"]),
            "episodes_completed": int(payload["episodes_completed"]),
            "agent_order": compatibility["agent_order"],
            "dims": compatibility["dims"],
            "vec_env_type": compatibility["vec_env_type"],
            "multiprocessing_start_method": compatibility["multiprocessing_start_method"],
            "worker_seeds": compatibility["worker_seeds"],
            "effective_action_mask": compatibility["effective_action_mask"],
            "project_git_head": provenance.get("project_git_head"),
            "harl_git_head": provenance.get("harl_git_head"),
            "seed": provenance.get("seed"),
            "config_fingerprint": compatibility["config_fingerprint"],
            "data_fingerprint": compatibility["data_fingerprint"],
            "resume_boundary": "post_update_only",
            "mid_rollout_resume_supported": False,
        }

    @staticmethod
    def _resolve_source(source: Path) -> tuple[Path, Path]:
        if source.suffix.lower() == ".json" and source.name == "latest.json":
            if not source.is_file():
                raise CheckpointIntegrityError(f"Checkpoint index does not exist: {source}")
            pointer = json.loads(source.read_text(encoding="utf-8"))
            return source.parent / pointer["checkpoint"], source.parent / pointer["manifest"]
        if source.suffix.lower() != ".pt":
            raise CheckpointIntegrityError("Resume source must be a .pt checkpoint or latest.json.")
        manifest_name = (
            "final.manifest.json" if source.name == "final.pt" else source.stem + ".manifest.json"
        )
        return source, source.with_name(manifest_name)

    @staticmethod
    def _verify_manifest(checkpoint: Path, manifest_path: Path) -> dict[str, Any]:
        if not checkpoint.is_file():
            raise CheckpointIntegrityError(f"Checkpoint file does not exist: {checkpoint}")
        if not manifest_path.is_file():
            raise CheckpointIntegrityError(
                "Training-resume manifest is missing; model-only files cannot be resumed: "
                f"{manifest_path}"
            )
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("checkpoint_type") != CHECKPOINT_TYPE:
            raise CheckpointIntegrityError("Manifest is not a training_resume manifest.")
        actual_size = checkpoint.stat().st_size
        if actual_size != int(manifest.get("size_bytes", -1)):
            raise CheckpointIntegrityError(
                f"Checkpoint size mismatch: manifest={manifest.get('size_bytes')}, actual={actual_size}."
            )
        actual_hash = _sha256(checkpoint)
        if actual_hash != manifest.get("sha256"):
            raise CheckpointIntegrityError(
                f"Checkpoint SHA-256 mismatch: manifest={manifest.get('sha256')}, actual={actual_hash}."
            )
        return manifest

    def _prune(self) -> None:
        if self.keep_last is None:
            return
        manifests = sorted(self.directory.glob("update_*.manifest.json"))
        for manifest in manifests[:-self.keep_last]:
            checkpoint = manifest.with_name(manifest.name.replace(".manifest.json", ".pt"))
            if checkpoint.exists():
                checkpoint.unlink()
            manifest.unlink()
