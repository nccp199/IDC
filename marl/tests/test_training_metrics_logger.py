from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from marl.logging import (
    EPISODE_COLUMNS,
    STEP_COLUMNS,
    UPDATE_COLUMNS,
    CompositeTrainingLogger,
    TrainingMetricsLogger,
)


class _Writer:
    def __init__(self) -> None:
        self.values: list[tuple[str, float, int]] = []

    def add_scalar(self, tag, value, step) -> None:
        self.values.append((str(tag), float(value), int(step)))


class _Upstream:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def init(self, episodes):
        self.calls.append("init")

    def episode_init(self, episode):
        self.calls.append("episode_init")

    def per_step(self, data):
        self.calls.append("per_step")

    def episode_log(self, *args, **kwargs):
        self.calls.append("episode_log")

    def close(self):
        self.calls.append("close")


def _synthetic_info(hour: int, *, terminal: bool = False, reward: float = 1.0) -> dict:
    components = {
        "r_done": reward,
        "r_finished_task": 0.0,
        "r_priority_finish": 0.0,
        "r_cost": 0.0,
        "r_carbon": 0.0,
        "r_queue": 0.0,
        "r_queue_overflow": 0.0,
        "r_urgent_backlog": 0.0,
        "r_waiting": 0.0,
        "r_deadline": 0.0,
        "r_sla": 0.0,
        "r_unused": 0.0,
        "r_grid_peak": 0.0,
        "r_peak_load": 0.0,
        "r_pause": 0.0,
        "r_resume": 0.0,
        "r_non_interruptible": 0.0,
        "r_load_smooth": 0.0,
        "r_action_smooth": 0.0,
        "r_bess_degradation": 0.0,
        "r_bess_invalid_action": 0.0,
        "r_final_queue": 0.0,
        "r_soc_final": 0.0,
    }
    return {
        "hour": hour,
        "terminated": terminal,
        "truncated": False,
        "reward_total": reward,
        "base_reward": reward,
        "grid_adjusted_reward": reward,
        "grid_reward_penalty": 0.0,
        **components,
        "completed_work": 2.0,
        "total_completed_work": 2.0 * (hour + 1),
        "newly_finished_count": 1,
        "finished_task_count": hour + 1,
        "completion_rate": (hour + 1) / 24.0,
        "task_completion_rate": (hour + 1) / 24.0,
        "backlog_work": 24.0 - hour,
        "Q": 24.0 - hour,
        "overflow_work": 0.0,
        "urgent_backlog_work": 1.0,
        "task_forecast_mode": "noisy",
        "forecast_error_level": 0.20,
        "task_forecast_mae": 2.0,
        "task_forecast_rmse": 3.0,
        "task_forecast_mape_nonzero_percent": 10.0,
        **(
            {
                "true_task_arrival_profile": np.arange(24, dtype=np.float64),
                "task_arrival_forecast": np.arange(24, dtype=np.float64) + 1.0,
            }
            if terminal
            else {}
        ),
        "unfinished_task_count": 23 - hour,
        "avg_waiting_time": 0.5,
        "new_deadline_miss_count": 0,
        "deadline_miss_count": 0,
        "sla_violation_count": 0,
        "sla_penalty": 0.0,
        "pause_count_this_step": 0,
        "resume_count_this_step": 0,
        "non_interruptible_interruption_this_step": 0,
        "P_IDC_kW": 1000.0,
        "P_IT": 800.0,
        "P_cooling": 200.0,
        "PUE": 1.25,
        "actual_task_load_mean": 0.5,
        "grid_energy_kWh": 900.0,
        "price": 0.2,
        "carbon_factor": 0.4,
        "cost": 180.0,
        "total_cost": 180.0 * (hour + 1),
        "carbon_emission": 360.0,
        "total_carbon_emission": 360.0 * (hour + 1),
        "grid_power_kW": 900.0,
        "P_bus_net_kW": 900.0,
        "grid_peak_power_kW": 900.0,
        "grid_peak_excess_kW": 0.0,
        "pv_available_kW": 100.0,
        "pv_used_kW": 100.0,
        "pv_available_kWh": 100.0,
        "pv_used_kWh": 100.0,
        "grid_reference_usep": 120.0,
        "bess_raw_action": 0.5,
        "bess_charge_power_kW": 0.0,
        "bess_discharge_power_kW": 0.0,
        "bess_soc": 0.5,
        "bess_energy_kWh": 500.0,
        "bess_charge_kWh": 0.0,
        "bess_discharge_kWh": 0.0,
        "bess_cycle_throughput_kWh": 0.0,
        "bess_degradation_cost": 0.0,
        "invalid_bess_action": 0.0,
        "grid_load_scale": 1.0,
        "grid_lmp": 50.0,
        "grid_mef_plus": 400.0,
        "grid_mef_minus": 390.0,
        "grid_min_voltage_pu": 0.98,
        "grid_max_voltage_pu": 1.02,
        "grid_max_line_loading_percent": 40.0,
        "grid_max_transformer_loading_percent": 25.0,
        "grid_network_loss_mw": 0.01,
        "grid_opf_success": True,
        "grid_mef_success": True,
        "grid_voltage_violation_count": 0,
        "grid_line_overload_count": 0,
        "grid_transformer_overload_count": 0,
        "grid_bus_dynamic_clip_count": 0,
        "grid_bus_dynamic_clip_fraction": 0.0,
        "grid_bus_dynamic_missing_value_count": 0,
        "grid_bus_dynamic_fallback_used": False,
        "safe_violation_opf": 0.0,
        "safe_violation_cost": 0.0,
        "safe_cost_total": 0.0,
        "grid_security_penalty": 0.0,
        "grid_opf_cache_hit": hour > 0,
        "grid_mef_cache_hit": hour > 0,
        "grid_cache_opf_size": hour + 1,
        "grid_cache_mef_size": hour + 1,
        "grid_opf_cache_hit_count": hour,
        "grid_opf_cache_miss_count": 1,
        "grid_mef_cache_hit_count": hour,
        "grid_mef_cache_miss_count": 1,
        "grid_cache_load_bin_mw": 0.01,
        "grid_cache_mef_load_bin_mw": 0.01,
        "grid_reward_enabled": False,
    }


def _data(infos: list[dict], *, virtual: float = 0.25):
    workers = len(infos)
    rewards = np.asarray(
        [[[info["grid_adjusted_reward"]], [info["grid_adjusted_reward"]]] for info in infos],
        dtype=np.float32,
    )
    dones = np.asarray(
        [[bool(info["terminated"] or info["truncated"])] * 2 for info in infos],
        dtype=np.bool_,
    )
    info_array = np.empty((workers, 2), dtype=object)
    actions = np.full((workers, 2, 22), 0.5, dtype=np.float32)
    actions[:, 1, 1:] = virtual
    for worker, info in enumerate(infos):
        info_array[worker, 0] = {**info, "agent_id": "idc"}
        info_array[worker, 1] = {**info, "agent_id": "bess"}
    return (None, None, rewards, dones, info_array, None, None, actions, None, None, None)


def _diagnostics(effective_dim: int) -> dict:
    return {
        "effective_action_dim": effective_dim,
        "padded_action_dim": 22,
        "effective_ratio_mean": 1.0,
        "effective_ratio_std": 0.01,
        "effective_ratio_min": 0.98,
        "effective_ratio_max": 1.02,
        "effective_clip_fraction": 0.0,
        "effective_approx_kl": 0.0001,
        "effective_physical_ratio_max_diff": 0.0,
        "virtual_mean_rows_grad_norm": 0.0,
        "virtual_log_std_grad_norm": 0.0,
        "virtual_entropy_optimization_contribution": 0.0,
    }


class TrainingMetricsLoggerTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        from marl.envs.harl_env_factory import make_harl_single_env

        env = make_harl_single_env(seed=7110)
        env.seed(7110)
        try:
            env.reset()
            actions = np.full((2, 22), 0.5, dtype=np.float32)
            cls.real_transition = env.step(actions)
            cls.real_actions = actions
        finally:
            env.close()

    def _logger(self, root: Path, **kwargs) -> TrainingMetricsLogger:
        root.mkdir()
        return TrainingMetricsLogger(run_dir=root, seed=7110, writer=_Writer(), **kwargs)

    def _complete_episode(self, logger: TrainingMetricsLogger, *, workers: int = 1) -> None:
        logger.begin_update(1)
        for hour in range(24):
            logger.record_step(
                _data(
                    [
                        _synthetic_info(hour, terminal=hour == 23, reward=float(worker + 1))
                        for worker in range(workers)
                    ]
                )
            )

    def _record_fake_update(self, logger: TrainingMetricsLogger, *, update: int = 1) -> dict:
        return logger.record_update(
            update=update,
            actor_infos=[
                {"policy_loss": 0.1, "dist_entropy": 1.1, "actor_grad_norm": 2.1},
                {"policy_loss": 0.2, "dist_entropy": 1.2, "actor_grad_norm": 2.2},
            ],
            critic_info={"value_loss": 0.3, "critic_grad_norm": 3.0},
            actor_buffers=[
                SimpleNamespace(actions=np.full((24, 1, 22), 0.4)),
                SimpleNamespace(actions=np.full((24, 1, 22), 0.5)),
            ],
            critic_buffer=SimpleNamespace(
                value_preds=np.zeros((25, 1, 1)),
                returns=np.linspace(0.0, 1.0, 25).reshape(25, 1, 1),
            ),
            actors=[
                SimpleNamespace(last_update_diagnostics=_diagnostics(22)),
                SimpleNamespace(last_update_diagnostics=_diagnostics(1)),
            ],
            update_time_seconds=0.01,
            rollout_time_seconds=0.02,
        )

    def test_real_environment_info_extracts_required_step_fields(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            logger = self._logger(Path(tmp) / "real")
            logger.begin_update(1)
            obs, share, rewards, dones, infos, avail = self.real_transition
            data = (
                obs[None], share[None], rewards[None], dones[None],
                np.asarray([infos], dtype=object), avail, None,
                self.real_actions[None], None, None, None,
            )
            try:
                logger.record_step(data)
            finally:
                logger.close()
            with logger.step_path.open(encoding="utf-8", newline="") as stream:
                row = next(csv.DictReader(stream))
            self.assertEqual(tuple(row), STEP_COLUMNS)
            self.assertEqual(int(row["hour"]), 0)
            self.assertEqual(row["ambient_temperature_C"], "")
            self.assertEqual(row["server_load_max"], "")

    def test_reward_reconstruction_and_grid_identity(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            logger = self._logger(Path(tmp) / "identity")
            logger.begin_update(1)
            logger.record_step(_data([_synthetic_info(0)]))
            logger.close()
            with logger.step_path.open(encoding="utf-8", newline="") as stream:
                row = next(csv.DictReader(stream))
            self.assertEqual(float(row["reward_reconstruction_error"]), 0.0)
            self.assertEqual(float(row["reward_returned"]), float(row["reward_total"]))
            self.assertEqual(float(row["base_reward"]), float(row["grid_adjusted_reward"]))
            self.assertEqual(float(row["grid_penalty"]), 0.0)

    def test_episode_aggregation_uses_sum_mean_max_final_and_rate(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            logger = self._logger(Path(tmp) / "aggregate")
            self._complete_episode(logger)
            logger.close()
            with logger.episode_path.open(encoding="utf-8", newline="") as stream:
                row = next(csv.DictReader(stream))
            self.assertEqual(int(row["episode_steps"]), 24)
            self.assertEqual(float(row["episode_reward"]), 24.0)
            self.assertEqual(float(row["completed_work_sum"]), 48.0)
            self.assertEqual(float(row["mean_queue"]), 12.5)
            self.assertEqual(float(row["max_queue"]), 24.0)
            self.assertEqual(float(row["final_backlog"]), 1.0)
            self.assertEqual(float(row["opf_success_rate"]), 1.0)

    def test_terminal_info_is_written_once_to_the_current_episode(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            logger = self._logger(Path(tmp) / "terminal")
            self._complete_episode(logger)
            logger.close()
            with logger.episode_path.open(encoding="utf-8", newline="") as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(len(rows), 1)
            self.assertEqual(float(rows[0]["terminal_reward"]), 1.0)
            self.assertEqual(float(rows[0]["r_final_queue_sum"]), 0.0)
            self.assertEqual(float(rows[0]["r_final_soc_sum"]), 0.0)

    def test_vec_auto_reset_metadata_does_not_enter_next_episode(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            logger = self._logger(Path(tmp) / "autoreset")
            self._complete_episode(logger)
            self._record_fake_update(logger)
            logger.begin_update(2)
            reset_step = _synthetic_info(0)
            reset_step["original_obs"] = np.full((2, 288), 99.0)
            logger.record_step(_data([reset_step]))
            logger.close()
            with logger.step_path.open(encoding="utf-8", newline="") as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(len(rows), 25)
            self.assertEqual(int(rows[-1]["episode_id"]), 1)
            self.assertEqual(int(rows[-1]["episode_step"]), 0)
            self.assertEqual(int(rows[-1]["hour"]), 0)

    def test_incomplete_episode_is_not_promoted_to_episode_csv(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            logger = self._logger(Path(tmp) / "incomplete")
            logger.begin_update(1)
            logger.record_step(_data([_synthetic_info(0)]))
            summary = logger.finalize(status="failed", termination_reason="test")
            logger.close()
            with logger.episode_path.open(encoding="utf-8", newline="") as stream:
                self.assertEqual(list(csv.DictReader(stream)), [])
            self.assertEqual(summary["incomplete_episode_steps_by_worker"], {"0": 1})
            self.assertFalse(summary["incomplete_episodes_written_to_episode_csv"])

    def test_mask_diagnostics_are_persisted_in_update_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            logger = self._logger(Path(tmp) / "update")
            self._complete_episode(logger)
            row = self._record_fake_update(logger)
            logger.close()
            self.assertEqual(row["bess_effective_action_dim"], 1)
            self.assertEqual(row["bess_virtual_action_dim"], 21)
            self.assertEqual(row["bess_effective_physical_ratio_max_diff"], 0.0)
            self.assertEqual(row["bess_virtual_mean_grad_norm"], 0.0)
            self.assertEqual(row["bess_virtual_log_std_grad_norm"], 0.0)
            with logger.update_path.open(encoding="utf-8", newline="") as stream:
                written = next(csv.DictReader(stream))
            self.assertEqual(tuple(written), UPDATE_COLUMNS)

    def test_shared_team_reward_is_not_double_counted(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            logger = self._logger(Path(tmp) / "shared")
            self._complete_episode(logger)
            logger.close()
            with logger.episode_path.open(encoding="utf-8", newline="") as stream:
                row = next(csv.DictReader(stream))
            self.assertEqual(float(row["episode_reward"]), 24.0)
            self.assertEqual(float(row["shared_reward_max_diff"]), 0.0)

    def test_csv_schema_order_is_stable(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            logger = self._logger(Path(tmp) / "schema")
            self._complete_episode(logger)
            logger.close()
            with logger.step_path.open(encoding="utf-8", newline="") as stream:
                rows = list(csv.reader(stream))
            self.assertEqual(tuple(rows[0]), STEP_COLUMNS)
            self.assertTrue(all(len(row) == len(STEP_COLUMNS) for row in rows[1:]))
            with logger.episode_path.open(encoding="utf-8", newline="") as stream:
                self.assertEqual(tuple(next(csv.reader(stream))), EPISODE_COLUMNS)
            schema = json.loads(logger.schema_path.read_text(encoding="utf-8"))
            self.assertEqual(schema["reward_aliases"], {"r_grid_peak": "r_peak_load"})

    def test_non_finite_critical_metric_fails_instead_of_becoming_zero(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            logger = self._logger(Path(tmp) / "finite")
            logger.begin_update(1)
            info = _synthetic_info(0)
            info["reward_total"] = np.nan
            try:
                with self.assertRaisesRegex(FloatingPointError, "reward_total"):
                    logger.record_step(_data([info]))
            finally:
                logger.close()

    def test_two_worker_accumulators_do_not_mix_episodes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            logger = self._logger(Path(tmp) / "workers")
            self._complete_episode(logger, workers=2)
            logger.close()
            with logger.episode_path.open(encoding="utf-8", newline="") as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(len(rows), 2)
            by_worker = {int(row["worker_id"]): float(row["episode_reward"]) for row in rows}
            self.assertEqual(by_worker, {0: 24.0, 1: 48.0})
            self.assertEqual({int(row["episode_id"]) for row in rows}, {0})

    def test_existing_metrics_directory_is_never_silently_appended_or_overwritten(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp) / "collision"
            logger = self._logger(run)
            with self.assertRaises(FileExistsError):
                TrainingMetricsLogger(run_dir=run, seed=7110, writer=None)
            logger.close()

    def test_composite_logger_delegates_and_closes_both_layers(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            metrics = self._logger(Path(tmp) / "composite")
            upstream = _Upstream()
            logger = CompositeTrainingLogger(upstream, metrics)
            logger.init(1)
            logger.episode_init(1)
            logger.per_step(_data([_synthetic_info(0)]))
            logger.episode_log(None)
            logger.close()
            self.assertEqual(
                upstream.calls,
                ["init", "episode_init", "per_step", "episode_log", "close"],
            )

    def test_composite_logger_routes_upstream_add_scalars_to_safe_single_scalars(self) -> None:
        class Writer:
            def __init__(self):
                self.values = []

            def add_scalars(self, *args, **kwargs):
                raise AssertionError("unsafe add_scalars should be intercepted")

            def add_scalar(self, tag, value, step, walltime=None):
                self.values.append((tag, value, step, walltime))

        writer = Writer()

        class Upstream:
            writter = writer

            def episode_log(self):
                self.writter.add_scalars(
                    "critic/value_loss", {"critic/value_loss": 2.5}, 144
                )

        with tempfile.TemporaryDirectory() as tmp:
            metrics = self._logger(Path(tmp) / "safe-scalars")
            CompositeTrainingLogger(Upstream(), metrics).episode_log()
            metrics.close()
        self.assertEqual(writer.values, [("critic/value_loss", 2.5, 144, None)])


if __name__ == "__main__":
    unittest.main()
