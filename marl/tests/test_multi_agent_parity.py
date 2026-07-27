from __future__ import annotations

import unittest

import numpy as np

from marl import IDCGridMultiAgentEnv
from marl.tests.helpers import make_grid_env


class MultiAgentParityTest(unittest.TestCase):
    def test_wrapping_preserves_grid_coupled_transitions(self) -> None:
        seed = 2202
        raw_env = make_grid_env(seed=seed)
        wrapped_grid_env = make_grid_env(seed=seed)
        multi_env = IDCGridMultiAgentEnv(wrapped_grid_env)
        rng = np.random.default_rng(seed)
        flat_actions = [rng.uniform(0.0, 1.0, size=23).astype(np.float32) for _ in range(3)]
        numeric_info_keys = (
            "P_IDC_kW",
            "P_grid_kW",
            "bess_soc",
            "bess_energy_kWh",
            "bess_charge_power_kW",
            "bess_discharge_power_kW",
            "grid_lmp",
            "grid_mef_plus",
            "grid_mef_minus",
            "grid_min_voltage_pu",
            "grid_max_line_loading_percent",
        )
        try:
            raw_obs, raw_reset_info = raw_env.reset(seed=seed)
            _, _, multi_reset_info = multi_env.reset(seed=seed)
            np.testing.assert_allclose(raw_obs, multi_env.last_raw_observation, rtol=0.0, atol=0.0)
            self.assertEqual(
                bool(raw_reset_info["grid_opf_success"]),
                bool(multi_reset_info["grid_opf_success"]),
            )

            for flat in flat_actions:
                idc_action, bess_action = multi_env.split_action(flat)
                raw_result = raw_env.step(flat)
                multi_result = multi_env.step({"idc": idc_action, "bess": bess_action})
                raw_obs, raw_reward, raw_terminated, raw_truncated, raw_info = raw_result
                _, _, reward, terminated, truncated, multi_info = multi_result

                np.testing.assert_allclose(raw_obs, multi_env.last_raw_observation, rtol=1e-7, atol=1e-7)
                np.testing.assert_allclose(multi_env.last_flat_action, flat, rtol=0.0, atol=0.0)
                self.assertAlmostEqual(raw_reward, reward["idc"], places=7)
                self.assertAlmostEqual(raw_reward, reward["bess"], places=7)
                self.assertEqual(raw_terminated, terminated["__all__"])
                self.assertEqual(raw_truncated, truncated["__all__"])
                self.assertEqual(raw_info["grid_opf_success"], multi_info["grid_opf_success"])
                self.assertEqual(raw_info["grid_mef_success"], multi_info["grid_mef_success"])
                for key in numeric_info_keys:
                    np.testing.assert_allclose(raw_info[key], multi_info[key], rtol=1e-7, atol=1e-7)
        finally:
            raw_env.close()
            wrapped_grid_env.close()


if __name__ == "__main__":
    unittest.main()
