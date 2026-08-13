from __future__ import annotations

import unittest

import numpy as np

from marl import IDCGridMultiAgentEnv
from marl.bridges import HarlIDCGridBridge
from marl.tests.helpers import make_grid_env


class HarlBridgeRolloutTest(unittest.TestCase):
    def test_complete_24_step_harl_protocol_rollout(self) -> None:
        seed = 6606
        grid_env = make_grid_env(seed=seed)
        bridge = HarlIDCGridBridge(IDCGridMultiAgentEnv(grid_env))
        opf_successes = 0
        mef_successes = 0
        try:
            bridge.seed(seed)
            obs, share_obs, available = bridge.reset()
            self.assertEqual(tuple(value.shape for value in obs), ((288,), (164,)))
            self.assertEqual(share_obs.shape, (2, 364))
            self.assertIsNone(available)

            for step in range(1, 25):
                ordered_actions = [space.sample() for space in bridge.action_space]
                obs, share_obs, rewards, dones, infos, available = bridge.step(ordered_actions)
                self.assertEqual(tuple(value.shape for value in obs), ((288,), (164,)))
                self.assertEqual(share_obs.shape, (2, 364))
                self.assertEqual(rewards.shape, (2, 1))
                self.assertTrue(all(np.isfinite(value).all() for value in obs))
                self.assertTrue(np.isfinite(share_obs).all())
                self.assertTrue(np.isfinite(rewards).all())
                self.assertIsNone(available)
                self.assertEqual([info["agent_id"] for info in infos], ["idc", "bess"])
                self.assertEqual(bool(dones[0]), step == 24)
                self.assertEqual(bool(dones[1]), step == 24)
                self.assertFalse(infos[0]["truncated"])
                self.assertFalse(infos[0]["bad_transition"])
                self.assertGreaterEqual(infos[0]["bess_soc"], 0.1 - 1e-9)
                self.assertLessEqual(infos[0]["bess_soc"], 0.9 + 1e-9)
                opf_successes += int(bool(infos[0]["grid_opf_success"]))
                mef_successes += int(bool(infos[0]["grid_mef_success"]))
                print(
                    f"harl_step={step:02d} obs=((288,),(164,)) state={share_obs.shape} "
                    f"reward={rewards.shape} done={bool(dones[0])} "
                    f"opf={bool(infos[0]['grid_opf_success'])} soc={infos[0]['bess_soc']:.6f}"
                )

            self.assertEqual(opf_successes, 24)
            self.assertEqual(mef_successes, 24)
            print("harl_rollout_summary steps=24 opf=24/24 mef=24/24 finite=True")
        finally:
            grid_env.close()


if __name__ == "__main__":
    unittest.main()
