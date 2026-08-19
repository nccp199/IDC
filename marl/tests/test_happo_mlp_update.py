"""Focused HAPPO ordering/factor tests without entering the physical environment."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import torch

from marl.algorithms.update_strategy import HAPPOUpdateStrategy


class _Buffer:
    def __init__(self, episode_length: int, threads: int, action_dim: int) -> None:
        self.obs = np.zeros((episode_length + 1, threads, 1), np.float32)
        self.rnn_states = np.zeros((episode_length + 1, threads, 1, 1), np.float32)
        self.actions = np.zeros((episode_length, threads, action_dim), np.float32)
        self.masks = np.ones((episode_length + 1, threads, 1), np.float32)
        self.active_masks = np.ones((episode_length + 1, threads, 1), np.float32)
        self.available_actions = None
        self.factor = None

    def update_factor(self, factor: np.ndarray) -> None:
        self.factor = factor.copy()


class _Actor:
    def __init__(self, effective_dim: int, action_dim: int, delta: torch.Tensor) -> None:
        self.effective_action_mask = torch.zeros(1, action_dim)
        self.effective_action_mask[:, :effective_dim] = 1.0
        self.log_probs = torch.zeros(4, action_dim)
        self.delta = delta
        self.train_calls = 0

    def evaluate_actions(self, *args):
        return self.log_probs.clone(), torch.tensor(0.0), None

    def effective_ratio(self, new: torch.Tensor, old: torch.Tensor) -> torch.Tensor:
        return torch.exp(((new - old) * self.effective_action_mask).sum(-1, keepdim=True))

    def train(self, buffer: _Buffer, advantages: np.ndarray, state_type: str):
        assert buffer.factor is not None
        self.train_calls += 1
        self.log_probs = self.log_probs + self.delta
        return {
            "policy_loss": 0.1,
            "dist_entropy": 0.2,
            "actor_grad_norm": 0.3,
            "ratio": 1.0,
        }


class _Critic:
    def __init__(self) -> None:
        self.train_calls = 0

    def train(self, buffer, normalizer):
        self.train_calls += 1
        return {"value_loss": 0.4, "critic_grad_norm": 0.5}


def _runner(*, fixed_order: bool = True):
    episode_length, threads, action_dim = 2, 2, 22
    idc_delta = torch.full((4, action_dim), 0.001)
    bess_delta = torch.cat((torch.full((4, 1), 0.02), torch.full((4, 21), 2.0)), dim=1)
    buffers = [_Buffer(episode_length, threads, action_dim) for _ in range(2)]
    critic_buffer = SimpleNamespace(
        returns=np.ones((episode_length + 1, threads, 1), np.float32),
        value_preds=np.zeros((episode_length + 1, threads, 1), np.float32),
    )
    return SimpleNamespace(
        algo_args={"train": {"episode_length": episode_length, "n_rollout_threads": threads}},
        value_normalizer=None,
        critic_buffer=critic_buffer,
        state_type="EP",
        actor_buffer=buffers,
        actor=[_Actor(22, 22, idc_delta), _Actor(1, 22, bess_delta)],
        fixed_order=fixed_order,
        num_agents=2,
        agent_names=("idc", "bess"),
        critic=_Critic(),
        last_algorithm_update={},
    )


def test_happo_updates_each_actor_and_critic_exactly_once() -> None:
    runner = _runner()
    infos, critic_info = HAPPOUpdateStrategy().train(runner)
    assert len(infos) == 2
    assert critic_info["value_loss"] == 0.4
    assert [actor.train_calls for actor in runner.actor] == [1, 1]
    assert runner.critic.train_calls == 1
    assert runner.last_algorithm_update["agent_update_order"] == ["idc", "bess"]
    assert runner.last_algorithm_update["happo_factor_shape"] == [2, 2, 1]
    assert runner.last_algorithm_update["actor_update_count_idc"] == 1
    assert runner.last_algorithm_update["actor_update_count_bess"] == 1
    assert runner.last_algorithm_update["critic_update_count"] == 1


def test_second_agent_receives_first_agent_factor_and_bess_virtual_dims_do_not_contribute() -> None:
    runner = _runner()
    HAPPOUpdateStrategy().train(runner)
    expected_after_idc = np.exp(22 * 0.001)
    assert np.max(np.abs(runner.actor_buffer[1].factor - expected_after_idc)) < 1e-6
    diagnostics = runner.last_algorithm_update
    assert diagnostics["happo_factor_initial_mean"] == 1.0
    assert abs(diagnostics["happo_factor_after_first_agent_mean"] - expected_after_idc) < 1e-6
    assert diagnostics["bess_happo_factor_effective_physical_max_diff"] == 0.0
    assert diagnostics["bess_happo_virtual_factor_contribution"] == 0.0
    assert diagnostics["happo_factor_nonfinite_count"] == 0
    assert diagnostics["happo_factor_nonpositive_count"] == 0
    assert diagnostics["happo_factor_initial_exactly_one_fraction"] == 1.0
    assert diagnostics["happo_factor_after_idc_reconstruction_max_diff"] == 0.0
    assert diagnostics["happo_factor_final_reconstruction_max_diff"] == 0.0
    assert diagnostics["idc_action_buffer_mutation_max_diff"] == 0.0
    assert diagnostics["bess_action_buffer_mutation_max_diff"] == 0.0
    assert diagnostics["critic_return_buffer_mutation_max_diff"] == 0.0
    expected_final = expected_after_idc * np.exp(0.02)
    assert abs(diagnostics["happo_factor_final_mean"] - expected_final) < 1e-6


def test_random_agent_order_is_reproducible_from_torch_rng() -> None:
    orders = []
    for _ in range(2):
        torch.manual_seed(7110)
        runner = _runner(fixed_order=False)
        HAPPOUpdateStrategy().train(runner)
        orders.append(runner.last_algorithm_update["agent_update_order"])
    assert orders[0] == orders[1]
