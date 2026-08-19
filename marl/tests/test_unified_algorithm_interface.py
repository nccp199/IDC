"""Part-13 unit coverage for explicit method and critic selection."""

from __future__ import annotations

import copy

import pytest
import numpy as np
import torch
import gymnasium as gym

from marl.algorithms import EffectiveActionMAPPO
from marl.evaluation.model_loader import _build_actors
from marl.checkpointing.training_checkpoint import (
    CheckpointCompatibilityError,
    _validate_method_metadata,
)
from marl.methods import CRITIC_INTERFACE_VERSION, derive_method
from train.train_harl_mappo_short import resolve_config


def _base_config() -> dict:
    return {
        "main": {"algorithm_name": "mappo", "env_name": "gym", "experiment_name": "idc_bess_mappo_short"},
        "critic": {"type": "mlp"},
        "project": {"harl_upstream_commit": "x", "expected_harl_head": "y"},
        "seed": {"seed_specify": True, "seed": 7110},
        "device": {"cuda": False, "cuda_deterministic": True, "torch_threads": 1},
        "parallel": {"worker_seed_stride": 1000, "multiprocessing_start_method": "spawn", "blas_threads": 1},
        "train": {"updates": 1, "num_env_steps": None, "n_rollout_threads": 2, "episode_length": 24, "log_interval": 1, "eval_interval": 1000, "use_valuenorm": False, "use_linear_lr_decay": False, "use_proper_time_limits": True, "model_dir": None},
        "eval": {"use_eval": False, "n_eval_rollout_threads": 1, "eval_episodes": 1, "seed": 2026},
        "render": {"use_render": False, "render_episodes": 1},
        "model": {"hidden_sizes": [32, 32], "activation_func": "relu", "use_feature_normalization": True, "initialization_method": "orthogonal_", "gain": 0.01, "use_naive_recurrent_policy": False, "use_recurrent_policy": False, "recurrent_n": 1, "data_chunk_length": 24, "lr": 0.0005, "critic_lr": 0.0005, "opti_eps": 1e-5, "weight_decay": 0, "std_x_coef": 1, "std_y_coef": 0.5, "use_bounded_box_actions": True},
        "algo": {"ppo_epoch": 1, "critic_epoch": 1, "use_clipped_value_loss": True, "clip_param": 0.2, "actor_num_mini_batch": 1, "critic_num_mini_batch": 1, "entropy_coef": 0.01, "value_loss_coef": 1.0, "use_max_grad_norm": True, "max_grad_norm": 10.0, "use_gae": True, "gamma": 0.99, "gae_lambda": 0.95, "use_huber_loss": True, "use_policy_active_masks": True, "huber_delta": 10.0, "action_aggregation": "prod", "share_param": False, "fixed_order": True},
        "logger": {"log_dir": "runs/test", "step_logging_enabled": True, "episode_logging_enabled": True, "tensorboard_enabled": True},
        "checkpoint": {"enabled": True, "interval_updates": 5, "save_final": True, "keep_last": 5},
        "env": {"scenario": "idc_bess_padding", "state_type": "EP", "experiment_case": "main"},
    }


def test_method_ids_are_derived_from_explicit_pairs() -> None:
    assert derive_method("mappo", "mlp").method_id == "MAPPO_MLP"
    assert derive_method("happo", "mlp").method_id == "HAPPO_MLP"


def test_happo_hgta_is_registered_and_implemented() -> None:
    registered = derive_method("happo", "hgta")
    assert registered.method_id == "HAPPO_HGTA"
    assert registered.implemented is True


@pytest.mark.parametrize(
    ("algorithm", "critic"),
    [("mappo", "hgta"), ("unknown", "mlp"), ("happo", "unknown")],
)
def test_invalid_method_combinations_fail_fast(algorithm: str, critic: str) -> None:
    with pytest.raises(ValueError, match="Unsupported algorithm/critic combination"):
        derive_method(algorithm, critic)


def test_resolved_config_records_all_method_compatibility_fields() -> None:
    config = resolve_config(
        _base_config(), algorithm="happo", critic_type="mlp"
    )
    assert config["main"]["algorithm_name"] == "happo"
    assert config["critic"]["type"] == "mlp"
    assert config["main"]["method_id"] == "HAPPO_MLP"
    assert config["main"]["algorithm_implementation_version"]
    assert config["main"]["critic_interface_version"] == CRITIC_INTERFACE_VERSION
    assert "HAPPO_MLP" in config["main"]["experiment_name"]
    assert config["rng"] == {
        "rng_isolation_version": "critic-init-isolation-v1",
        "critic_init_seed_rule": "base_seed + critic_init_seed_offset",
        "critic_init_seed_offset": 200000,
        "critic_init_seed": 207110,
        "actor_sampling_rng_isolated_from_critic_init": True,
    }


def test_legacy_mappo_resolution_keeps_training_values() -> None:
    base = _base_config()
    before = copy.deepcopy(base)
    resolved = resolve_config(base)
    for section in ("seed", "device", "parallel", "train", "eval", "render", "model", "algo", "env"):
        expected = before[section]
        if section == "train":
            expected = {**expected, "num_env_steps": 48}
        assert resolved[section] == expected
    assert resolved["main"]["algorithm_name"] == "mappo"
    assert resolved["main"]["method_id"] == "MAPPO_MLP"


def test_identical_actor_state_has_identical_deterministic_eval_interface() -> None:
    mappo = resolve_config(_base_config())
    happo = resolve_config(_base_config(), algorithm="happo", critic_type="mlp")
    args = {**mappo["model"], **mappo["algo"]}
    obs_space = gym.spaces.Box(-np.inf, np.inf, shape=(288,), dtype=np.float32)
    act_space = gym.spaces.Box(0.0, 1.0, shape=(22,), dtype=np.float32)
    torch.manual_seed(7110)
    source = EffectiveActionMAPPO(args, obs_space, act_space, effective_action_dim=22)
    states = [copy.deepcopy(source.actor.state_dict()) for _ in range(2)]
    mappo_actors = _build_actors(mappo, states, torch.device("cpu"))
    happo_actors = _build_actors(happo, states, torch.device("cpu"))
    observation = np.zeros((1, 288), np.float32)
    rnn = np.zeros((1, 1, 32), np.float32)
    masks = np.ones((1, 1), np.float32)
    with torch.no_grad():
        for left, right in zip(mappo_actors, happo_actors, strict=True):
            left_action = left.act(observation, rnn, masks, None, deterministic=True)[0]
            right_action = right.act(observation, rnn, masks, None, deterministic=True)[0]
            assert torch.equal(left_action, right_action)


def test_checkpoint_rejects_cross_algorithm_and_cross_critic_metadata() -> None:
    runner = type("Runner", (), {"method": derive_method("happo", "mlp"), "fixed_order": True})()
    valid = {
        "algorithm_name": "happo",
        "critic_type": "mlp",
        "method_id": "HAPPO_MLP",
        "agent_update_order_policy": "fixed",
    }
    _validate_method_metadata(valid, runner)
    for changed in (
        {**valid, "algorithm_name": "mappo", "method_id": "MAPPO_MLP"},
        {**valid, "critic_type": "hgta", "method_id": "HAPPO_HGTA"},
        {**valid, "agent_update_order_policy": "torch_randperm"},
    ):
        with pytest.raises(CheckpointCompatibilityError, match="compatibility differs"):
            _validate_method_metadata(changed, runner)
