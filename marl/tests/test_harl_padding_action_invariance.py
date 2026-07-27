from __future__ import annotations

import unittest

import numpy as np

from marl import IDCGridMultiAgentEnv
from marl.bridges import HarlIDCGridBridge, HarlPaddedBridge
from marl.tests.helpers import make_grid_env


class HarlPaddingActionInvarianceTest(unittest.TestCase):
    def test_virtual_bess_dimensions_cannot_change_transition(self) -> None:
        seed = 7103
        grid_a = make_grid_env(seed=seed)
        grid_b = make_grid_env(seed=seed)
        generic_a = IDCGridMultiAgentEnv(grid_a)
        generic_b = IDCGridMultiAgentEnv(grid_b)
        bridge_a = HarlPaddedBridge(HarlIDCGridBridge(generic_a))
        bridge_b = HarlPaddedBridge(HarlIDCGridBridge(generic_b))
        try:
            bridge_a.seed(seed)
            bridge_b.seed(seed)
            reset_a = bridge_a.reset()
            reset_b = bridge_b.reset()
            np.testing.assert_array_equal(reset_a[0], reset_b[0])
            np.testing.assert_array_equal(reset_a[1], reset_b[1])

            idc_action = np.linspace(0.1, 0.9, 22, dtype=np.float32)
            bess_a = np.zeros(22, dtype=np.float32)
            bess_b = np.random.default_rng(seed).uniform(0.0, 1.0, 22).astype(np.float32)
            bess_a[0] = bess_b[0] = np.float32(0.73)
            result_a = bridge_a.step(np.stack((idc_action, bess_a)))
            result_b = bridge_b.step(np.stack((idc_action, bess_b)))

            np.testing.assert_array_equal(generic_a.last_flat_action, generic_b.last_flat_action)
            np.testing.assert_array_equal(result_a[0], result_b[0])
            np.testing.assert_array_equal(result_a[1], result_b[1])
            np.testing.assert_array_equal(result_a[2], result_b[2])
            np.testing.assert_array_equal(result_a[3], result_b[3])
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
                self.assertEqual(result_a[4][0][key], result_b[4][0][key], key)
        finally:
            bridge_a.close()
            bridge_b.close()


if __name__ == "__main__":
    unittest.main()
