from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

import numpy as np

from marl.diagnostics import BESSVirtualActionMonitor, compute_bess_policy_diagnostics


class _ScalarWriter:
    def __init__(self) -> None:
        self.scalars = []

    def add_scalar(self, name, value, step) -> None:
        self.scalars.append((name, value, step))


class BESSVirtualActionMonitorTest(unittest.TestCase):
    def test_policy_diagnostics_use_bounded_actions_and_transformed_log_probs(self) -> None:
        actions = np.linspace(0.05, 0.95, 44, dtype=np.float64).reshape(2, 22)
        old_log_prob = np.full((2, 22), -0.75, dtype=np.float64)
        new_log_prob = np.full((2, 22), -0.5, dtype=np.float64)
        metrics = compute_bess_policy_diagnostics(
            actions=actions,
            old_log_prob=old_log_prob,
            new_log_prob=new_log_prob,
            advantages=np.asarray([1.0, -1.0]),
            clip_param=0.2,
        )
        self.assertAlmostEqual(metrics["effective_log_prob"], -0.5)
        self.assertAlmostEqual(metrics["virtual_log_prob_sum"], -10.5)
        self.assertAlmostEqual(metrics["full_log_prob"], -11.0)
        self.assertAlmostEqual(metrics["effective_entropy"], 0.5)
        self.assertAlmostEqual(metrics["virtual_entropy_sum"], 10.5)
        self.assertAlmostEqual(metrics["full_entropy"], 11.0)

    def test_csv_full_vector_tensorboard_and_non_overwrite(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            writer = _ScalarWriter()
            action = np.linspace(0.0, 1.0, 22, dtype=np.float32)
            monitor = BESSVirtualActionMonitor(
                temp_dir,
                save_full_virtual_action_vector=True,
                tensorboard_writer=writer,
            )
            monitor.record_step(
                global_step=1,
                episode=0,
                episode_step=1,
                bess_action=action,
                reward=-2.5,
                info={
                    "bess_soc": 0.55,
                    "bess_charge_power_kW": 2.0,
                    "bess_discharge_power_kW": 0.0,
                    "price": 0.65,
                    "carbon_factor": 0.4,
                },
                terminated=False,
                truncated=False,
            )
            path = monitor.path
            monitor.close()
            self.assertEqual(monitor.rows_written, 1)
            self.assertIsNone(monitor.last_error)
            with path.open(encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 1)
            self.assertEqual(float(rows[0]["bess_effective_action"]), float(action[0]))
            self.assertEqual(float(rows[0]["virtual_action_21"]), float(action[21]))
            self.assertTrue(writer.scalars)

            second = BESSVirtualActionMonitor(temp_dir)
            self.assertNotEqual(second.path, path)
            second.close()
            self.assertTrue(Path(path).exists())

    def test_disabled_monitor_is_a_noop(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            monitor = BESSVirtualActionMonitor(temp_dir, enabled=False)
            monitor.record_step(
                global_step=1,
                episode=0,
                episode_step=1,
                bess_action=np.zeros(22),
                reward=0.0,
                info={},
                terminated=False,
                truncated=False,
            )
            self.assertIsNone(monitor.path)
            self.assertFalse(list(Path(temp_dir).iterdir()))


if __name__ == "__main__":
    unittest.main()
