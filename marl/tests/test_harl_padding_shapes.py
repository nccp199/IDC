from __future__ import annotations

import unittest

import numpy as np

from marl import IDCGridMultiAgentEnv
from marl.bridges import HarlIDCGridBridge, HarlPaddedBridge
from marl.tests.helpers import make_grid_env


class HarlPaddingShapeTest(unittest.TestCase):
    def test_spaces_reset_and_step_are_stock_stackable(self) -> None:
        grid_env = make_grid_env(seed=7101)
        bridge = HarlPaddedBridge(HarlIDCGridBridge(IDCGridMultiAgentEnv(grid_env)))
        try:
            bridge.seed(7101)
            obs, share_obs, available = bridge.reset()
            self.assertEqual(obs.shape, (2, 288))
            self.assertEqual(share_obs.shape, (2, 364))
            self.assertEqual(obs.dtype, np.float32)
            self.assertEqual(share_obs.dtype, np.float32)
            self.assertEqual(tuple(space.shape for space in bridge.action_space), ((22,), (22,)))
            self.assertEqual(tuple(space.dtype for space in bridge.action_space), (np.float32, np.float32))
            np.testing.assert_array_equal(bridge.action_space[0].low, np.zeros(22, np.float32))
            np.testing.assert_array_equal(bridge.action_space[0].high, np.ones(22, np.float32))
            self.assertIsNone(available)

            actions = np.stack([space.sample() for space in bridge.action_space]).astype(np.float32)
            obs, share_obs, rewards, dones, infos, available = bridge.step(actions)
            self.assertEqual(obs.shape, (2, 288))
            self.assertEqual(share_obs.shape, (2, 364))
            self.assertEqual(rewards.shape, (2, 1))
            self.assertEqual(dones.shape, (2,))
            self.assertEqual(rewards.dtype, np.float32)
            self.assertEqual(dones.dtype, np.bool_)
            self.assertEqual(len(infos), 2)
            self.assertIsNone(available)
        finally:
            bridge.close()


if __name__ == "__main__":
    unittest.main()
