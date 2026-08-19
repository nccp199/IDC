from __future__ import annotations

import copy
import unittest

import numpy as np
import torch
from gymnasium import spaces

from marl.algorithms import (
    EffectiveActionHAPPO,
    EffectiveActionMAPPO,
    effective_entropy_from_log_probs,
    effective_ratio_from_log_probs,
)
from marl.diagnostics import compute_bess_policy_diagnostics
from marl.envs.harl_env_factory import make_harl_single_env
from marl.specs import (
    BESS_AGENT,
    EFFECTIVE_ACTION_DIMS,
    IDC_AGENT,
    PADDED_ACTION_DIMS,
    effective_action_mask,
)


def _algo_args() -> dict[str, object]:
    return {
        "hidden_sizes": [8, 8],
        "activation_func": "relu",
        "use_feature_normalization": True,
        "initialization_method": "orthogonal_",
        "gain": 0.01,
        "use_naive_recurrent_policy": False,
        "use_recurrent_policy": False,
        "recurrent_n": 1,
        "data_chunk_length": 4,
        "lr": 5e-4,
        "opti_eps": 1e-5,
        "weight_decay": 0.0,
        "std_x_coef": 1.0,
        "std_y_coef": 0.5,
        "use_bounded_box_actions": True,
        "use_policy_active_masks": True,
        "action_aggregation": "prod",
        "clip_param": 0.2,
        "ppo_epoch": 1,
        "actor_num_mini_batch": 1,
        "entropy_coef": 0.01,
        "use_max_grad_norm": True,
        "max_grad_norm": 10.0,
    }


def _spaces():
    return (
        spaces.Box(-np.inf, np.inf, shape=(12,), dtype=np.float32),
        spaces.Box(0.0, 1.0, shape=(22,), dtype=np.float32),
    )


def _actor_batch(algo, *, batch_size: int = 8):
    rng = np.random.default_rng(7301)
    obs = rng.normal(size=(batch_size, 12)).astype(np.float32)
    rnn = np.zeros((batch_size, 1, 8), dtype=np.float32)
    masks = np.ones((batch_size, 1), dtype=np.float32)
    active = np.ones((batch_size, 1), dtype=np.float32)
    with torch.no_grad():
        actions, old_log_probs, _ = algo.get_actions(obs, rnn, masks)
    advantages = np.linspace(-1.0, 1.0, batch_size, dtype=np.float32).reshape(-1, 1)
    return (
        obs,
        rnn,
        actions.cpu().numpy(),
        masks,
        active,
        old_log_probs.cpu().numpy(),
        advantages,
        None,
    )


class EffectiveActionMaskUnitTest(unittest.TestCase):
    def test_agent_masks_match_contract(self) -> None:
        idc = np.asarray(effective_action_mask(IDC_AGENT), dtype=np.float32)
        bess = np.asarray(effective_action_mask(BESS_AGENT), dtype=np.float32)
        self.assertEqual(idc.shape, (22,))
        self.assertEqual(float(idc.sum()), 22.0)
        self.assertTrue(np.all(idc == 1.0))
        self.assertEqual(bess.shape, (22,))
        self.assertEqual(float(bess[0]), 1.0)
        self.assertTrue(np.all(bess[1:] == 0.0))
        self.assertEqual(float(bess.sum()), 1.0)
        self.assertEqual(PADDED_ACTION_DIMS, {IDC_AGENT: 22, BESS_AGENT: 22})
        self.assertEqual(EFFECTIVE_ACTION_DIMS, {IDC_AGENT: 22, BESS_AGENT: 1})

    def test_bess_ratio_ignores_virtual_dimensions(self) -> None:
        old = torch.zeros(3, 22)
        new = old.clone()
        new[:, 1:] = torch.linspace(-0.7, 0.9, 21)
        mask = torch.tensor(effective_action_mask(BESS_AGENT))
        ratio = effective_ratio_from_log_probs(new, old, mask)
        torch.testing.assert_close(ratio, torch.ones(3, 1))

        new[:, 0] = 0.25
        ratio = effective_ratio_from_log_probs(new, old, mask)
        torch.testing.assert_close(ratio, torch.full((3, 1), np.exp(0.25)))

    def test_idc_ratio_uses_every_dimension(self) -> None:
        old = torch.zeros(1, 22)
        mask = torch.tensor(effective_action_mask(IDC_AGENT))
        baseline = effective_ratio_from_log_probs(old, old, mask)
        for index in (0, 7, 21):
            new = old.clone()
            new[:, index] = 0.1
            changed = effective_ratio_from_log_probs(new, old, mask)
            self.assertGreater(float(changed), float(baseline))
            torch.testing.assert_close(changed, torch.full((1, 1), np.exp(0.1)))

    def test_bess_entropy_uses_only_physical_dimension(self) -> None:
        mask = torch.tensor(effective_action_mask(BESS_AGENT))
        log_probs = torch.full((4, 22), -0.5)
        baseline = effective_entropy_from_log_probs(log_probs, mask)
        virtual_changed = log_probs.clone()
        virtual_changed[:, 1:] = 100.0
        torch.testing.assert_close(
            effective_entropy_from_log_probs(virtual_changed, mask), baseline
        )
        physical_changed = log_probs.clone()
        physical_changed[:, 0] = -1.5
        self.assertNotEqual(
            float(effective_entropy_from_log_probs(physical_changed, mask)),
            float(baseline),
        )

    def test_unspecified_effective_dim_defaults_to_all_dimensions(self) -> None:
        obs_space, act_space = _spaces()
        algo = EffectiveActionMAPPO(_algo_args(), obs_space, act_space)
        self.assertEqual(algo.effective_action_dim, 22)
        torch.testing.assert_close(
            algo.effective_action_mask, torch.ones(1, 22)
        )

    def test_mappo_virtual_output_gradients_are_zero(self) -> None:
        obs_space, act_space = _spaces()
        algo = EffectiveActionMAPPO(
            _algo_args(), obs_space, act_space, effective_action_dim=1
        )
        sample = _actor_batch(algo)
        algo.update(sample)
        mean = algo.actor.act.action_out.fc_mean
        log_std = algo.actor.act.action_out.log_std
        self.assertGreater(float(torch.linalg.vector_norm(mean.weight.grad[0])), 0.0)
        self.assertGreater(float(torch.abs(mean.bias.grad[0])), 0.0)
        self.assertGreater(float(torch.abs(log_std.grad[0])), 0.0)
        self.assertLessEqual(float(torch.max(torch.abs(mean.weight.grad[1:]))), 1e-10)
        self.assertLessEqual(float(torch.max(torch.abs(mean.bias.grad[1:]))), 1e-10)
        self.assertLessEqual(float(torch.max(torch.abs(log_std.grad[1:]))), 1e-10)
        trunk_norm = torch.linalg.vector_norm(
            torch.cat(
                [
                    parameter.grad.reshape(-1)
                    for parameter in algo.actor.base.parameters()
                    if parameter.grad is not None
                ]
            )
        )
        self.assertGreater(float(trunk_norm), 0.0)

    def test_masked_and_physical_only_trunk_gradients_match(self) -> None:
        obs_space, act_space = _spaces()
        algo = EffectiveActionMAPPO(
            _algo_args(), obs_space, act_space, effective_action_dim=1
        )
        sample = _actor_batch(algo)
        obs, rnn, actions, masks, active, old_log_probs, advantages, _ = sample
        old = torch.as_tensor(old_log_probs)
        advantage = torch.as_tensor(advantages)

        new_masked, _, _ = algo.evaluate_actions(
            obs, rnn, actions, masks, None, active
        )
        ratio_masked = algo.effective_ratio(new_masked, old)
        entropy_masked = algo.effective_entropy(
            new_masked, torch.as_tensor(active)
        )
        masked_loss = -torch.minimum(
            ratio_masked * advantage,
            torch.clamp(ratio_masked, 0.8, 1.2) * advantage,
        ).mean() - 0.01 * entropy_masked
        trunk = list(algo.actor.base.parameters())
        masked_grads = torch.autograd.grad(masked_loss, trunk)

        new_physical, _, _ = algo.evaluate_actions(
            obs, rnn, actions, masks, None, active
        )
        ratio_physical = torch.exp(new_physical[:, :1] - old[:, :1])
        entropy_physical = -new_physical[:, 0].mean()
        physical_loss = -torch.minimum(
            ratio_physical * advantage,
            torch.clamp(ratio_physical, 0.8, 1.2) * advantage,
        ).mean() - 0.01 * entropy_physical
        physical_grads = torch.autograd.grad(physical_loss, trunk)

        for masked, physical in zip(masked_grads, physical_grads, strict=True):
            torch.testing.assert_close(masked, physical, rtol=1e-6, atol=1e-8)

    def test_mappo_ratio_clip_and_kl_match_physical_dimension(self) -> None:
        mask = torch.tensor(effective_action_mask(BESS_AGENT))
        old = torch.zeros(5, 22)
        new = torch.zeros(5, 22)
        new[:, 0] = torch.tensor([-0.3, -0.1, 0.0, 0.1, 0.3])
        new[:, 1:] = torch.linspace(-1.0, 1.0, 21)
        effective = effective_ratio_from_log_probs(new, old, mask).squeeze(-1)
        physical = torch.exp(new[:, 0] - old[:, 0])
        torch.testing.assert_close(effective, physical)
        effective_clip = (effective < 0.8) | (effective > 1.2)
        physical_clip = (physical < 0.8) | (physical > 1.2)
        self.assertTrue(torch.equal(effective_clip, physical_clip))
        effective_log_ratio = torch.log(effective)
        physical_log_ratio = new[:, 0] - old[:, 0]
        effective_kl = ((effective - 1.0) - effective_log_ratio).mean()
        physical_kl = ((physical - 1.0) - physical_log_ratio).mean()
        torch.testing.assert_close(effective_kl, physical_kl)

    def test_happo_update_uses_the_same_masked_ratio(self) -> None:
        obs_space, act_space = _spaces()
        algo = EffectiveActionHAPPO(
            _algo_args(), obs_space, act_space, effective_action_dim=1
        )
        old = torch.zeros(4, 22)
        new = torch.zeros(4, 22)
        new[:, 1:] = torch.linspace(-0.8, 0.8, 21)
        torch.testing.assert_close(algo.effective_ratio(new, old), torch.ones(4, 1))

        sample = _actor_batch(algo) + (np.ones((8, 1), dtype=np.float32),)
        algo.update(sample)
        mean = algo.actor.act.action_out.fc_mean
        log_std = algo.actor.act.action_out.log_std
        self.assertLessEqual(float(torch.max(torch.abs(mean.weight.grad[1:]))), 1e-10)
        self.assertLessEqual(float(torch.max(torch.abs(mean.bias.grad[1:]))), 1e-10)
        self.assertLessEqual(float(torch.max(torch.abs(log_std.grad[1:]))), 1e-10)

    def test_bess_diagnostics_use_the_training_mask(self) -> None:
        old = np.zeros((3, 22), dtype=np.float64)
        new = old.copy()
        new[:, 1:] = 0.5
        metrics = compute_bess_policy_diagnostics(
            actions=np.full((3, 22), 0.5, dtype=np.float64),
            old_log_prob=old,
            new_log_prob=new,
            advantages=np.asarray([-1.0, 0.0, 1.0]),
            clip_param=0.2,
            action_mask=effective_action_mask(BESS_AGENT),
        )
        self.assertEqual(metrics["effective_ratio"], 1.0)
        self.assertEqual(metrics["effective_ratio_clip_fraction"], 0.0)
        self.assertEqual(metrics["effective_approx_kl"], 0.0)
        self.assertGreater(metrics["full_ratio"], 1.0)

    def test_existing_output_22_actor_state_loads_strictly(self) -> None:
        from harl.algorithms.actors.mappo import MAPPO

        obs_space, act_space = _spaces()
        upstream = MAPPO(_algo_args(), obs_space, act_space)
        masked = EffectiveActionMAPPO(
            _algo_args(), obs_space, act_space, effective_action_dim=1
        )
        incompatible = masked.actor.load_state_dict(
            copy.deepcopy(upstream.actor.state_dict()), strict=True
        )
        self.assertEqual(incompatible.missing_keys, [])
        self.assertEqual(incompatible.unexpected_keys, [])


class EffectiveActionMaskPhysicalRegressionTest(unittest.TestCase):
    def test_four_virtual_action_patterns_leave_full_episode_unchanged(self) -> None:
        seed = 7110
        envs = [make_harl_single_env(seed=seed) for _ in range(4)]
        for env in envs:
            env.seed(seed)
        fixed_random = np.random.default_rng(7401).uniform(0.0, 1.0, 21).astype(np.float32)
        changing_rng = np.random.default_rng(7402)
        idc = np.full(22, 0.5, dtype=np.float32)
        try:
            resets = [env.reset() for env in envs]
            for other in resets[1:]:
                self._assert_nested_equal(resets[0], other)

            for _ in range(24):
                virtuals = (
                    np.zeros(21, dtype=np.float32),
                    np.ones(21, dtype=np.float32),
                    fixed_random,
                    changing_rng.uniform(0.0, 1.0, 21).astype(np.float32),
                )
                results = []
                for env, virtual in zip(envs, virtuals, strict=True):
                    bess = np.concatenate(
                        (np.asarray([0.5], dtype=np.float32), virtual)
                    )
                    results.append(env.step(np.stack((idc, bess))))
                for other in results[1:]:
                    self._assert_nested_equal(results[0], other)
        finally:
            for env in envs:
                env.close()

    def _assert_nested_equal(self, left, right) -> None:
        if isinstance(left, dict):
            self.assertEqual(set(left), set(right))
            for key in left:
                self._assert_nested_equal(left[key], right[key])
            return
        if isinstance(left, (tuple, list)):
            self.assertEqual(len(left), len(right))
            for first, second in zip(left, right, strict=True):
                self._assert_nested_equal(first, second)
            return
        if isinstance(left, np.ndarray) or isinstance(right, np.ndarray):
            np.testing.assert_array_equal(np.asarray(left), np.asarray(right))
            return
        if isinstance(left, (float, int, bool, np.number)):
            self.assertEqual(float(left), float(right))
            return
        self.assertEqual(left, right)


if __name__ == "__main__":
    unittest.main()
