"""Strict actor-only loaders for fixed dual-agent evaluation."""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import gymnasium as gym
import numpy as np
import torch

from marl.algorithms import EFFECTIVE_ACTION_ALGO_REGISTRY
from marl.evaluation.fixed_scenario_suite import canonical_sha256, environment_fingerprints, file_sha256
from marl.methods import derive_method
from marl.specs import ACTION_PADDING_STRATEGY, AGENTS, EFFECTIVE_ACTION_DIMS, PADDED_ACTION_DIMS
from marl.specs.state_specs import CENTRALIZED_STATE_DIM


MODEL_EVALUATION_SCHEMA_VERSION = "idc-bess-actor-evaluation-v1"


class ModelCompatibilityError(RuntimeError):
    """Actor weights/configuration are incompatible with fixed evaluation."""


@dataclass
class LoadedModel:
    model_id: str
    source_kind: str
    source_path: Path
    source_description: str
    model_sha256: str
    actors: list[Any]
    resolved_config: dict[str, Any]
    training_run_id: str | None
    training_seed: int | None
    training_update: int | None
    compatibility: dict[str, Any]

    def parameter_sha256(self) -> str:
        return canonical_sha256(
            [
                {key: tensor.detach().cpu() for key, tensor in actor.actor.state_dict().items()}
                for actor in self.actors
            ]
        )


def _load_json(path: Path, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"{label} is missing: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ModelCompatibilityError(f"{label} must contain a JSON object.")
    return value


def _resolve_config_path(source: Path, explicit: str | Path | None, run_dir: Path | None) -> Path:
    if explicit is not None:
        return Path(explicit).resolve()
    candidates = []
    if run_dir is not None:
        candidates.append(run_dir / "resolved_config.json")
    candidates.extend((source / "resolved_config.json", source.parent / "resolved_config.json"))
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise ModelCompatibilityError(
        "Actor network configuration is unavailable. Supply --model-config; "
        "the evaluator never guesses architecture from current defaults."
    )


def _validate_resolved_config(config: Mapping[str, Any]) -> dict[str, Any]:
    config = dict(config)
    required_sections = {"main", "model", "algo", "env", "train", "seed"}
    missing = sorted(required_sections.difference(config))
    if missing:
        raise ModelCompatibilityError(f"Resolved model config is missing sections: {missing}.")
    if "critic" not in config:
        if config["main"].get("algorithm_name") != "mappo":
            raise ModelCompatibilityError("Resolved HAPPO config must explicitly record critic.type.")
        config["critic"] = {"type": "mlp", "legacy_schema_migration": True}
    method = derive_method(
        config["main"].get("algorithm_name", ""), config["critic"].get("type", "")
    )
    expected = {
        "episode_length": (config["train"].get("episode_length"), 24),
        "bounded_box_actions": (config["model"].get("use_bounded_box_actions"), True),
        "share_param": (config["algo"].get("share_param"), False),
        "action_aggregation": (config["algo"].get("action_aggregation"), "prod"),
        "state_type": (config["env"].get("state_type"), "EP"),
    }
    differences = [f"{name}={actual!r}, expected {wanted!r}" for name, (actual, wanted) in expected.items() if actual != wanted]
    if differences:
        raise ModelCompatibilityError("Incompatible resolved model config: " + "; ".join(differences))
    config["main"] = {**config["main"], **method.as_dict()}
    return config


def _compatibility(
    config: Mapping[str, Any],
    metadata: Mapping[str, Any] | None,
    graph_metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    method = derive_method(config["main"]["algorithm_name"], config["critic"]["type"])
    experiment_case = str(config["env"].get("experiment_case", "main"))
    dimensions = {
        "observation_dims": [288, 288],
        "state_dim": CENTRALIZED_STATE_DIM,
        "padded_action_dims": [PADDED_ACTION_DIMS[agent] for agent in AGENTS],
        "effective_action_dims": [EFFECTIVE_ACTION_DIMS[agent] for agent in AGENTS],
    }
    if metadata:
        checks = {
            # Part-11 MAPPO metadata predates these explicit fields; the resolved
            # config remains authoritative for that documented legacy schema.
            "algorithm_name": (metadata.get("algorithm_name", method.algorithm_name), method.algorithm_name),
            "critic_type": (metadata.get("critic_type", method.critic_type), method.critic_type),
            "method_id": (metadata.get("method_id", method.method_id), method.method_id),
            "agent_order": (metadata.get("agent_order"), list(AGENTS)),
            "observation_dims": (metadata.get("observation_dims"), dimensions["observation_dims"]),
            "state_dim": (metadata.get("state_dim"), dimensions["state_dim"]),
            "action_dims": (metadata.get("action_dims"), dimensions["padded_action_dims"]),
            "effective_action_dims": (metadata.get("agent_effective_action_dims"), dimensions["effective_action_dims"]),
            "action_padding_strategy": (metadata.get("action_padding_strategy"), ACTION_PADDING_STRATEGY),
            "bounded_box_actions": (metadata.get("use_bounded_box_actions"), True),
        }
        bad = [f"{name}={actual!r}" for name, (actual, expected) in checks.items() if actual != expected]
        if bad:
            raise ModelCompatibilityError("Training metadata is incompatible: " + "; ".join(bad))
    result = {
        "schema": MODEL_EVALUATION_SCHEMA_VERSION,
        **method.as_dict(),
        "agent_order": list(AGENTS),
        **dimensions,
        "action_padding_strategy": ACTION_PADDING_STRATEGY,
        "bounded_box_actions": True,
        "actor_network": {
            "hidden_sizes": list(config["model"]["hidden_sizes"]),
            "activation_func": config["model"]["activation_func"],
            "use_feature_normalization": bool(config["model"]["use_feature_normalization"]),
            "use_naive_recurrent_policy": bool(config["model"]["use_naive_recurrent_policy"]),
            "use_recurrent_policy": bool(config["model"]["use_recurrent_policy"]),
            "recurrent_n": int(config["model"]["recurrent_n"]),
        },
        "experiment_case": experiment_case,
        **environment_fingerprints(experiment_case),
    }
    rng = config.get("rng")
    if isinstance(rng, Mapping):
        rng_fields = (
            "rng_isolation_version",
            "critic_init_seed_rule",
            "critic_init_seed",
            "actor_sampling_rng_isolated_from_critic_init",
        )
        missing_rng = [key for key in rng_fields if key not in rng]
        if missing_rng:
            raise ModelCompatibilityError(
                "Resolved RNG isolation config is incomplete: " + ", ".join(missing_rng)
            )
        for key in rng_fields:
            result[key] = copy.deepcopy(rng[key])
        if metadata:
            bad_rng = [
                f"{key}={metadata.get(key)!r}"
                for key in rng_fields
                if metadata.get(key) != rng[key]
            ]
            if bad_rng:
                raise ModelCompatibilityError(
                    "Training RNG isolation metadata is incompatible: " + "; ".join(bad_rng)
                )
    graph = graph_metadata or (metadata or {}).get("graph_metadata")
    if graph:
        result["graph_metadata"] = copy.deepcopy(dict(graph))
    return result


def validate_model_suite_compatibility(model: LoadedModel, suite: Mapping[str, Any]) -> None:
    algorithm = model.compatibility.get(
        "algorithm", model.compatibility.get("algorithm_name")
    )
    if algorithm not in {
        "mappo",
        "happo",
    }:
        raise ModelCompatibilityError(
            f"Unsupported evaluation algorithm metadata: {algorithm!r}."
        )
    expected = {
        "agent_order": list(AGENTS),
        "observation_dims": [288, 288],
        "state_dim": CENTRALIZED_STATE_DIM,
        "padded_action_dims": [22, 22],
        "effective_action_dims": [22, 1],
        "action_padding_strategy": ACTION_PADDING_STRATEGY,
        "bounded_box_actions": True,
        "environment_config_fingerprint": suite["environment_config_fingerprint"],
        "reward_config_fingerprint": suite["reward_config_fingerprint"],
        "data_config_fingerprint": suite["data_config_fingerprint"],
    }
    differences = [
        f"{key}: model={model.compatibility.get(key)!r}, suite={value!r}"
        for key, value in expected.items()
        if model.compatibility.get(key) != value
    ]
    if differences:
        raise ModelCompatibilityError("Model/suite compatibility failure: " + "; ".join(differences))


def _build_actors(config: Mapping[str, Any], state_dicts: list[Mapping[str, Any]], device: torch.device) -> list[Any]:
    args = {**config["model"], **config["algo"]}
    obs_space = gym.spaces.Box(-np.inf, np.inf, shape=(288,), dtype=np.float32)
    action_space = gym.spaces.Box(0.0, 1.0, shape=(22,), dtype=np.float32)
    actors = []
    algorithm = str(config["main"]["algorithm_name"])
    actor_class = EFFECTIVE_ACTION_ALGO_REGISTRY[algorithm]
    for agent, state in zip(AGENTS, state_dicts, strict=True):
        actor = actor_class(
            args,
            obs_space,
            action_space,
            device=device,
            effective_action_dim=EFFECTIVE_ACTION_DIMS[agent],
        )
        actor.actor.load_state_dict(dict(state), strict=True)
        actor.actor.eval()
        actor.actor_optimizer = None
        for parameter in actor.actor.parameters():
            parameter.grad = None
        actors.append(actor)
    return actors


def load_model(
    *,
    run_dir: str | Path | None = None,
    checkpoint: str | Path | None = None,
    model_dir: str | Path | None = None,
    model_config: str | Path | None = None,
    device: str = "cpu",
    model_id: str | None = None,
) -> LoadedModel:
    supplied = [run_dir is not None, checkpoint is not None, model_dir is not None]
    if sum(supplied) != 1:
        raise ValueError("Exactly one of run_dir, checkpoint, or model_dir is required.")
    torch_device = torch.device(device)
    if torch_device.type != "cpu":
        raise ValueError("Fixed evaluation currently requires device=cpu.")

    metadata: dict[str, Any] | None = None
    payload: dict[str, Any] | None = None
    inferred_run: Path | None = None
    training_update: int | None = None
    if run_dir is not None:
        source = Path(run_dir).resolve()
        inferred_run = source
        metadata = _load_json(source / "run_metadata.json", "run metadata")
        config_path = _resolve_config_path(source, model_config, inferred_run)
        state_paths = [source / "models" / "actor_agent0.pt", source / "models" / "actor_agent1.pt"]
        state_dicts = [torch.load(path, map_location="cpu", weights_only=False) for path in state_paths]
        source_kind = "run_dir"
        source_hashes = [file_sha256(path) for path in state_paths]
        model_hash = canonical_sha256(source_hashes)
        training_update = int(metadata.get("completed_updates", metadata.get("updates", 0)))
    elif checkpoint is not None:
        source = Path(checkpoint).resolve()
        payload = torch.load(source, map_location="cpu", weights_only=False)
        if payload.get("checkpoint_type") != "training_resume":
            raise ModelCompatibilityError("Checkpoint is not a formal training_resume payload.")
        provenance = payload.get("provenance", {})
        run_value = provenance.get("run_dir")
        inferred_run = Path(run_value).resolve() if run_value else None
        if inferred_run and (inferred_run / "run_metadata.json").is_file():
            metadata = _load_json(inferred_run / "run_metadata.json", "run metadata")
        config_path = _resolve_config_path(source.parent, model_config, inferred_run)
        model_state = payload.get("model_state", {})
        if model_state.get("agent_order") != list(AGENTS):
            raise ModelCompatibilityError("Checkpoint agent order is incompatible.")
        state_dicts = [model_state["actor_agent0_state_dict"], model_state["actor_agent1_state_dict"]]
        source_kind = "checkpoint"
        model_hash = file_sha256(source)
        training_update = int(payload["saved_at_update"])
    else:
        source = Path(model_dir).resolve()
        inferred_run = source.parent if (source.parent / "resolved_config.json").is_file() else None
        if inferred_run and (inferred_run / "run_metadata.json").is_file():
            metadata = _load_json(inferred_run / "run_metadata.json", "run metadata")
        config_path = _resolve_config_path(source, model_config, inferred_run)
        state_paths = [source / "actor_agent0.pt", source / "actor_agent1.pt"]
        state_dicts = [torch.load(path, map_location="cpu", weights_only=False) for path in state_paths]
        source_kind = "model_dir"
        source_hashes = [file_sha256(path) for path in state_paths]
        model_hash = canonical_sha256(source_hashes)
        training_update = None if metadata is None else int(metadata.get("completed_updates", 0))

    config = _validate_resolved_config(_load_json(config_path, "resolved model config"))
    method = derive_method(config["main"]["algorithm_name"], config["critic"]["type"])
    compatibility = _compatibility(
        config,
        metadata,
        None if payload is None else payload.get("graph_metadata"),
    )
    if payload is not None:
        checkpoint_compat = payload.get("compatibility", {})
        checks = {
            "algorithm": method.algorithm_name,
            "critic_type": method.critic_type,
            "method_id": method.method_id,
            "agent_order": list(AGENTS),
            "bounded_box_actions": True,
        }
        for key, expected in checks.items():
            if key in {"critic_type", "method_id"} and key not in checkpoint_compat:
                continue
            if checkpoint_compat.get(key) != expected:
                raise ModelCompatibilityError(f"Checkpoint compatibility field {key} differs.")
        if isinstance(config.get("rng"), Mapping):
            for key in (
                "rng_isolation_version",
                "critic_init_seed_rule",
                "critic_init_seed",
                "actor_sampling_rng_isolated_from_critic_init",
            ):
                if checkpoint_compat.get(key) != config["rng"][key]:
                    raise ModelCompatibilityError(
                        f"Checkpoint RNG isolation field {key} differs."
                    )
        dims = checkpoint_compat.get("dims", {})
        for key, expected in {
            "observation": [288, 288], "state": CENTRALIZED_STATE_DIM,
            "padded_action": [22, 22], "effective_action": [22, 1],
        }.items():
            if dims.get(key) != expected:
                raise ModelCompatibilityError(f"Checkpoint dimension {key} differs: {dims.get(key)!r}.")

    actors = _build_actors(config, state_dicts, torch_device)
    training_run_id = None if metadata is None else str(metadata.get("run_id", inferred_run.name if inferred_run else ""))
    training_seed = int(config["seed"]["seed"]) if config["seed"].get("seed") is not None else None
    return LoadedModel(
        model_id=model_id or f"{source_kind}-{source.stem}",
        source_kind=source_kind,
        source_path=source,
        source_description=str(source),
        model_sha256=model_hash,
        actors=actors,
        resolved_config=config,
        training_run_id=training_run_id,
        training_seed=training_seed,
        training_update=training_update,
        compatibility=compatibility,
    )
