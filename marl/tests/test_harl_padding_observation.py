from __future__ import annotations

import unittest

import numpy as np

from marl import IDCGridMultiAgentEnv
from marl.bridges import HarlIDCGridBridge, HarlPaddedBridge
from marl.tests.helpers import make_grid_env


class HarlPaddingObservationTest(unittest.TestCase):
    def test_bess_prefix_and_zero_tail_hold_for_complete_episode(self) -> None:
        seed = 7102
        grid_env = make_grid_env(seed=seed)
        raw_bridge = HarlIDCGridBridge(IDCGridMultiAgentEnv(grid_env))
        bridge = HarlPaddedBridge(raw_bridge)
        try:
            bridge.seed(seed)
            padded_obs, share_obs, _ = bridge.reset()
            true_obs = bridge.last_true_observations
            self.assertIsNotNone(true_obs)
            self.assertEqual(padded_obs.shape, (2, 288))
            self.assertTrue(np.isfinite(share_obs).all())
            np.testing.assert_array_equal(padded_obs[0], true_obs[0])
            np.testing.assert_array_equal(padded_obs[1, :164], true_obs[1])
            np.testing.assert_array_equal(padded_obs[1, 164:], np.zeros(124, np.float32))

            for step in range(1, 25):
                actions = np.stack([space.sample() for space in bridge.action_space]).astype(np.float32)
                padded_obs, share_obs, rewards, dones, infos, _ = bridge.step(actions)
                true_obs = bridge.last_true_observations
                np.testing.assert_array_equal(padded_obs[0], true_obs[0])
                np.testing.assert_array_equal(padded_obs[1, :164], true_obs[1])
                np.testing.assert_array_equal(padded_obs[1, 164:], np.zeros(124, np.float32))
                self.assertTrue(np.isfinite(padded_obs).all())
                self.assertTrue(np.isfinite(share_obs).all())
                self.assertTrue(np.isfinite(rewards).all())
                self.assertEqual(bool(dones[0]), step == 24)
                self.assertTrue(bool(infos[0]["grid_opf_success"]))
                self.assertTrue(bool(infos[0]["grid_mef_success"]))
                print(
                    f"harl_padding_step={step:02d} obs={padded_obs.shape} "
                    f"state={share_obs.shape} reward={rewards.shape} done={bool(dones[0])}"
                )
            print("harl_padding_rollout_summary steps=24 opf=24/24 mef=24/24 finite=True")
        finally:
            bridge.close()


if __name__ == "__main__":
    unittest.main()
