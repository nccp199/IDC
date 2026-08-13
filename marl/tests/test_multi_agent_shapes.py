from __future__ import annotations

import unittest

import numpy as np

from marl import IDCGridMultiAgentEnv
from marl.tests.helpers import make_grid_env


class MultiAgentShapeTest(unittest.TestCase):
    def test_reset_and_step_contract(self) -> None:
        grid_env = make_grid_env(seed=1101)
        env = IDCGridMultiAgentEnv(grid_env)
        try:
            obs, state, info = env.reset(seed=1101)
            self.assertEqual(tuple(obs), env.agents)
            self.assertEqual(env.action_spaces["idc"].shape, (22,))
            self.assertEqual(env.action_spaces["bess"].shape, (1,))
            self.assertEqual(obs["idc"].shape, (288,))
            self.assertEqual(obs["bess"].shape, (164,))
            self.assertEqual(state.shape, (364,))
            self.assertEqual(obs["idc"].dtype, np.float32)
            self.assertEqual(obs["bess"].dtype, np.float32)
            self.assertEqual(state.dtype, np.float32)
            self.assertTrue(np.isfinite(obs["idc"]).all())
            self.assertTrue(np.isfinite(obs["bess"]).all())
            self.assertTrue(np.isfinite(state).all())
            self.assertIsInstance(info, dict)
            np.testing.assert_array_equal(state[290:294], np.zeros(4, dtype=np.float32))

            action = {
                "idc": np.full(22, 0.5, dtype=np.float32),
                "bess": np.asarray([0.5], dtype=np.float32),
            }
            next_obs, next_state, reward, terminated, truncated, next_info = env.step(action)
            self.assertEqual(set(next_obs), set(env.agents))
            self.assertEqual(set(reward), set(env.agents))
            self.assertEqual(set(terminated), {"idc", "bess", "__all__"})
            self.assertEqual(set(truncated), {"idc", "bess", "__all__"})
            self.assertEqual(next_obs["idc"].shape, (288,))
            self.assertEqual(next_obs["bess"].shape, (164,))
            self.assertEqual(next_state.shape, (364,))
            self.assertTrue(np.isfinite(next_obs["idc"]).all())
            self.assertTrue(np.isfinite(next_obs["bess"]).all())
            self.assertTrue(np.isfinite(next_state).all())
            self.assertTrue(all(np.isfinite(value) for value in reward.values()))
            self.assertIsInstance(next_info, dict)
        finally:
            grid_env.close()


if __name__ == "__main__":
    unittest.main()
