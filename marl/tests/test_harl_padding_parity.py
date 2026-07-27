from __future__ import annotations

import unittest

import numpy as np

from marl import IDCGridMultiAgentEnv
from marl.bridges import HarlIDCGridBridge, HarlPaddedBridge
from marl.tests.helpers import make_grid_env


class HarlPaddingParityTest(unittest.TestCase):
    def test_padded_bridge_matches_true_bridge_after_removing_padding(self) -> None:
        seed = 7104
        raw_grid = make_grid_env(seed=seed)
        padded_grid = make_grid_env(seed=seed)
        raw_bridge = HarlIDCGridBridge(IDCGridMultiAgentEnv(raw_grid))
        padded_bridge = HarlPaddedBridge(HarlIDCGridBridge(IDCGridMultiAgentEnv(padded_grid)))
        try:
            raw_bridge.seed(seed)
            padded_bridge.seed(seed)
            raw_obs, raw_state, _ = raw_bridge.reset()
            padded_obs, padded_state, _ = padded_bridge.reset()
            np.testing.assert_array_equal(padded_obs[0], raw_obs[0])
            np.testing.assert_array_equal(padded_obs[1, :164], raw_obs[1])
            np.testing.assert_array_equal(padded_state, raw_state)

            rng = np.random.default_rng(seed)
            for _ in range(3):
                idc = rng.uniform(0.0, 1.0, 22).astype(np.float32)
                bess = rng.uniform(0.0, 1.0, 22).astype(np.float32)
                raw_result = raw_bridge.step((idc, bess[:1]))
                padded_result = padded_bridge.step(np.stack((idc, bess)))
                np.testing.assert_array_equal(padded_result[0][0], raw_result[0][0])
                np.testing.assert_array_equal(padded_result[0][1, :164], raw_result[0][1])
                np.testing.assert_array_equal(padded_result[0][1, 164:], np.zeros(124, np.float32))
                np.testing.assert_array_equal(padded_result[1], raw_result[1])
                np.testing.assert_array_equal(padded_result[2], raw_result[2])
                np.testing.assert_array_equal(padded_result[3], raw_result[3])
                for key in (
                    "bess_soc",
                    "bess_charge_power_kW",
                    "bess_discharge_power_kW",
                    "grid_opf_success",
                    "grid_mef_success",
                    "grid_lmp",
                    "grid_mef_plus",
                    "grid_mef_minus",
                ):
                    self.assertEqual(padded_result[4][0][key], raw_result[4][0][key], key)
        finally:
            raw_bridge.close()
            padded_bridge.close()


if __name__ == "__main__":
    unittest.main()
