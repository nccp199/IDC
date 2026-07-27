from __future__ import annotations

import unittest

import numpy as np
from gymnasium import spaces

from marl import IDCGridMultiAgentEnv
from marl.bridges import HARL_UPSTREAM_COMMIT, HarlIDCGridBridge
from marl.tests.helpers import make_grid_env


class _TruncatingGenericEnv:
    agents = ("idc", "bess")

    def __init__(self) -> None:
        self.observation_spaces = {
            "idc": spaces.Box(-np.inf, np.inf, shape=(288,), dtype=np.float32),
            "bess": spaces.Box(-np.inf, np.inf, shape=(164,), dtype=np.float32),
        }
        self.action_spaces = {
            "idc": spaces.Box(0.0, 1.0, shape=(22,), dtype=np.float32),
            "bess": spaces.Box(0.0, 1.0, shape=(1,), dtype=np.float32),
        }
        self.state_space = spaces.Box(-np.inf, np.inf, shape=(294,), dtype=np.float32)

    def reset(self, seed=None, options=None):
        return (
            {
                "idc": np.zeros(288, dtype=np.float32),
                "bess": np.zeros(164, dtype=np.float32),
            },
            np.zeros(294, dtype=np.float32),
            {},
        )

    def step(self, action_dict):
        return (
            {
                "idc": np.zeros(288, dtype=np.float32),
                "bess": np.zeros(164, dtype=np.float32),
            },
            np.zeros(294, dtype=np.float32),
            {"idc": 1.0, "bess": 1.0},
            {"idc": False, "bess": False, "__all__": False},
            {"idc": True, "bess": True, "__all__": True},
            {"timeout_marker": True},
        )


class HarlBridgeShapeTest(unittest.TestCase):
    def test_spaces_reset_and_step_protocol(self) -> None:
        grid_env = make_grid_env(seed=4404)
        bridge = HarlIDCGridBridge(IDCGridMultiAgentEnv(grid_env))
        try:
            self.assertEqual(bridge.n_agents, 2)
            self.assertEqual(bridge.agent_order, ("idc", "bess"))
            self.assertEqual(bridge.upstream_commit, HARL_UPSTREAM_COMMIT)
            self.assertEqual(tuple(space.shape for space in bridge.observation_space), ((288,), (164,)))
            self.assertEqual(tuple(space.shape for space in bridge.action_space), ((22,), (1,)))
            self.assertEqual(
                tuple(space.shape for space in bridge.share_observation_space),
                ((294,), (294,)),
            )

            self.assertEqual(bridge.seed(4404), [4404])
            obs, share_obs, available_actions = bridge.reset()
            self.assertIsInstance(obs, tuple)
            self.assertEqual(tuple(value.shape for value in obs), ((288,), (164,)))
            self.assertEqual(share_obs.shape, (2, 294))
            self.assertIsNone(available_actions)
            self.assertTrue(all(value.dtype == np.float32 for value in obs))
            self.assertEqual(share_obs.dtype, np.float32)
            self.assertTrue(all(np.isfinite(value).all() for value in obs))
            self.assertTrue(np.isfinite(share_obs).all())
            np.testing.assert_array_equal(share_obs[0], share_obs[1])

            result = bridge.step(
                [
                    np.full(22, 0.5, dtype=np.float32),
                    np.asarray([0.5], dtype=np.float32),
                ]
            )
            next_obs, next_share_obs, rewards, dones, infos, next_available = result
            self.assertEqual(tuple(value.shape for value in next_obs), ((288,), (164,)))
            self.assertEqual(next_share_obs.shape, (2, 294))
            self.assertEqual(rewards.shape, (2, 1))
            self.assertEqual(rewards.dtype, np.float32)
            self.assertEqual(dones.shape, (2,))
            self.assertEqual(dones.dtype, np.bool_)
            self.assertEqual(len(infos), 2)
            self.assertEqual([info["agent_id"] for info in infos], ["idc", "bess"])
            self.assertTrue(all(info["bad_transition"] is False for info in infos))
            self.assertIsNone(next_available)
            self.assertTrue(np.isfinite(rewards).all())
            bridge.close()
        finally:
            grid_env.close()

    def test_rejects_unordered_or_wrong_length_actions(self) -> None:
        grid_env = make_grid_env(seed=4405)
        bridge = HarlIDCGridBridge(IDCGridMultiAgentEnv(grid_env))
        try:
            bridge.seed(4405)
            bridge.reset()
            with self.assertRaises(TypeError):
                bridge.step({"idc": np.zeros(22), "bess": np.zeros(1)})
            with self.assertRaises(ValueError):
                bridge.step([np.zeros(22)])
        finally:
            grid_env.close()

    def test_truncation_sets_done_and_bad_transition(self) -> None:
        bridge = HarlIDCGridBridge(_TruncatingGenericEnv())
        bridge.seed(4406)
        bridge.reset()
        _, _, _, dones, infos, _ = bridge.step(
            [np.zeros(22, dtype=np.float32), np.zeros(1, dtype=np.float32)]
        )
        np.testing.assert_array_equal(dones, np.asarray([True, True]))
        self.assertTrue(all(info["bad_transition"] for info in infos))
        self.assertTrue(all(info["truncated"] for info in infos))
        self.assertTrue(all(not info["terminated"] for info in infos))


if __name__ == "__main__":
    unittest.main()
