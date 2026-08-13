from __future__ import annotations

import copy

import gymnasium as gym
import numpy as np
import pytest
import torch

from marl.critics import build_critic
from marl.methods import derive_method


def _args(hgta_config):
    return {
        "hidden_sizes": [32, 32],
        "activation_func": "relu",
        "use_feature_normalization": True,
        "initialization_method": "orthogonal_",
        "gain": 0.01,
        "use_naive_recurrent_policy": False,
        "use_recurrent_policy": False,
        "recurrent_n": 1,
        "data_chunk_length": 24,
        "critic_lr": 5e-4,
        "opti_eps": 1e-5,
        "weight_decay": 0,
        "clip_param": 0.2,
        "critic_epoch": 1,
        "critic_num_mini_batch": 1,
        "value_loss_coef": 1.0,
        "max_grad_norm": 10.0,
        "huber_delta": 10.0,
        "use_max_grad_norm": True,
        "use_clipped_value_loss": True,
        "use_huber_loss": True,
        "use_policy_active_masks": True,
        "hgta": copy.deepcopy(hgta_config),
    }


def _critic(hgta_config):
    return build_critic(
        "hgta",
        _args(hgta_config),
        gym.spaces.Box(-np.inf, np.inf, shape=(364,), dtype=np.float32),
        device=torch.device("cpu"),
    )


class _OneBatch:
    def __init__(self, state):
        self.state = state

    def feed_forward_generator_critic(self, _):
        batch = self.state.shape[0]
        yield (
            self.state,
            np.zeros((batch, 1, 32), np.float32),
            np.zeros((batch, 1), np.float32),
            np.ones((batch, 1), np.float32),
            np.ones((batch, 1), np.float32),
        )


def test_happo_hgta_is_formally_registered():
    method = derive_method("happo", "hgta")
    assert method.method_id == "HAPPO_HGTA"
    assert method.implemented is True
    with pytest.raises(ValueError, match="Unsupported"):
        derive_method("mappo", "hgta")


def test_external_interface_forward_train_state_and_modes(hgta_config, synthetic_state):
    torch.manual_seed(7110)
    critic = _critic(hgta_config)
    state = synthetic_state.repeat(2, 1).numpy()
    rnn = np.zeros((2, 1, 32), np.float32)
    masks = np.ones((2, 1), np.float32)
    values, next_rnn = critic.get_values(state, rnn, masks)
    assert values.shape == (2, 1)
    assert next_rnn.shape == (2, 1, 32)
    assert torch.isfinite(values).all()
    assert critic.graph_metadata["hgta_node_count"] == 39
    assert critic.graph_metadata["hgta_edge_count"] == 90

    train_info = critic.train(_OneBatch(state))
    assert torch.isfinite(torch.tensor(train_info["value_loss"]))
    assert train_info["value_prediction_nonfinite_count"] == 0
    saved = copy.deepcopy(critic.state_dict())
    optimizer_state = critic.optimizer_state()
    assert optimizer_state["state"]
    clone = _critic(hgta_config)
    clone.load_state_dict(saved, strict=True)
    critic.to("cpu")
    critic.train_mode()
    assert critic.network.training
    critic.eval_mode()
    assert not critic.network.training


def test_bad_shape_and_recurrent_config_fail_fast(hgta_config):
    critic = _critic(hgta_config)
    with pytest.raises(ValueError, match=r"\[batch, 364\]"):
        critic.get_values(np.zeros((1, 363), np.float32), None, None)
    args = _args(hgta_config)
    args["use_recurrent_policy"] = True
    with pytest.raises(NotImplementedError, match="feed-forward"):
        build_critic(
            "hgta",
            args,
            gym.spaces.Box(-np.inf, np.inf, shape=(364,), dtype=np.float32),
            device=torch.device("cpu"),
        )
