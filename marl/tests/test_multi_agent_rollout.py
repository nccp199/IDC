from __future__ import annotations

import unittest

import numpy as np

from marl import IDCGridMultiAgentEnv
from marl.tests.helpers import make_grid_env


class MultiAgentRolloutTest(unittest.TestCase):
    def test_complete_24_step_rollout(self) -> None:
        seed = 3303
        grid_env = make_grid_env(seed=seed)
        env = IDCGridMultiAgentEnv(grid_env)
        required_info = {
            "P_IDC_kW",
            "P_grid_kW",
            "bess_soc",
            "bess_energy_kWh",
            "bess_charge_power_kW",
            "bess_discharge_power_kW",
            "grid_lmp",
            "grid_mef_plus",
            "grid_mef_minus",
            "grid_opf_success",
            "grid_mef_success",
        }
        opf_successes = 0
        try:
            obs, state, _ = env.reset(seed=seed)
            self.assertTrue(np.isfinite(obs["idc"]).all())
            self.assertTrue(np.isfinite(obs["bess"]).all())
            self.assertTrue(np.isfinite(state).all())

            for step in range(1, 25):
                action = {
                    "idc": env.action_spaces["idc"].sample(),
                    "bess": env.action_spaces["bess"].sample(),
                }
                obs, state, rewards, terminated, truncated, info = env.step(action)
                self.assertEqual(env.last_flat_action.shape, (23,))
                np.testing.assert_array_equal(env.last_flat_action[:22], action["idc"])
                np.testing.assert_array_equal(env.last_flat_action[22:], action["bess"])
                self.assertTrue(np.isfinite(obs["idc"]).all())
                self.assertTrue(np.isfinite(obs["bess"]).all())
                self.assertTrue(np.isfinite(state).all())
                self.assertTrue(all(np.isfinite(value) for value in rewards.values()))
                self.assertTrue(required_info.issubset(info))
                self.assertTrue(bool(info["grid_opf_success"]))
                self.assertTrue(bool(info["grid_mef_success"]))
                self.assertGreaterEqual(info["bess_soc"], grid_env.env.bess_soc_min - 1e-9)
                self.assertLessEqual(info["bess_soc"], grid_env.env.bess_soc_max + 1e-9)
                self.assertFalse(truncated["__all__"])
                self.assertEqual(terminated["__all__"], step == 24)
                opf_successes += int(bool(info["grid_opf_success"]))
                print(
                    f"step={step:02d} idc_obs={obs['idc'].shape} "
                    f"bess_obs={obs['bess'].shape} state={state.shape} "
                    f"opf={bool(info['grid_opf_success'])} soc={info['bess_soc']:.6f}"
                )

            self.assertEqual(opf_successes, 24)
            print("rollout_summary steps=24 opf_success=24/24 finite=True terminated=True truncated=False")
        finally:
            grid_env.close()


if __name__ == "__main__":
    unittest.main()
