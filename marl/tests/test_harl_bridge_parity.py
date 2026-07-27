from __future__ import annotations

import unittest

import numpy as np

from marl import IDCGridMultiAgentEnv
from marl.bridges import HarlIDCGridBridge
from marl.tests.helpers import make_grid_env


class HarlBridgeParityTest(unittest.TestCase):
    def test_bridge_matches_generic_environment(self) -> None:
        seed = 5505
        direct_grid_env = make_grid_env(seed=seed)
        bridge_grid_env = make_grid_env(seed=seed)
        direct_env = IDCGridMultiAgentEnv(direct_grid_env)
        bridge = HarlIDCGridBridge(IDCGridMultiAgentEnv(bridge_grid_env))
        rng = np.random.default_rng(seed)
        actions = [
            (
                rng.uniform(0.0, 1.0, size=22).astype(np.float32),
                rng.uniform(0.0, 1.0, size=1).astype(np.float32),
            )
            for _ in range(3)
        ]
        info_keys = (
            "P_IDC_kW",
            "P_grid_kW",
            "bess_soc",
            "grid_lmp",
            "grid_mef_plus",
            "grid_mef_minus",
        )
        try:
            direct_obs, direct_state, _ = direct_env.reset(seed=seed)
            bridge.seed(seed)
            bridge_obs, bridge_state, available = bridge.reset()
            self.assertIsNone(available)
            np.testing.assert_allclose(bridge_obs[0], direct_obs["idc"], rtol=0.0, atol=0.0)
            np.testing.assert_allclose(bridge_obs[1], direct_obs["bess"], rtol=0.0, atol=0.0)
            np.testing.assert_allclose(bridge_state[0], direct_state, rtol=0.0, atol=0.0)
            np.testing.assert_allclose(bridge_state[1], direct_state, rtol=0.0, atol=0.0)

            for idc_action, bess_action in actions:
                direct_result = direct_env.step({"idc": idc_action, "bess": bess_action})
                bridge_result = bridge.step([idc_action, bess_action])
                direct_obs, direct_state, direct_rewards, direct_term, direct_trunc, direct_info = direct_result
                bridge_obs, bridge_state, rewards, dones, infos, available = bridge_result

                np.testing.assert_allclose(bridge_obs[0], direct_obs["idc"], rtol=1e-7, atol=1e-7)
                np.testing.assert_allclose(bridge_obs[1], direct_obs["bess"], rtol=1e-7, atol=1e-7)
                np.testing.assert_allclose(bridge_state[0], direct_state, rtol=1e-7, atol=1e-7)
                np.testing.assert_allclose(bridge_state[1], direct_state, rtol=1e-7, atol=1e-7)
                np.testing.assert_allclose(
                    rewards[:, 0],
                    [direct_rewards["idc"], direct_rewards["bess"]],
                    rtol=1e-7,
                    atol=1e-7,
                )
                expected_done = direct_term["__all__"] or direct_trunc["__all__"]
                np.testing.assert_array_equal(dones, np.asarray([expected_done, expected_done]))
                self.assertEqual(infos[0]["terminated"], direct_term["__all__"])
                self.assertEqual(infos[0]["truncated"], direct_trunc["__all__"])
                self.assertIsNone(available)
                for key in info_keys:
                    np.testing.assert_allclose(infos[0][key], direct_info[key], rtol=1e-7, atol=1e-7)
                    np.testing.assert_allclose(infos[1][key], direct_info[key], rtol=1e-7, atol=1e-7)
        finally:
            direct_grid_env.close()
            bridge_grid_env.close()


if __name__ == "__main__":
    unittest.main()
