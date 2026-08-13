"""Structured step, episode, update, and TensorBoard metrics for IDC/BESS MAPPO."""

from __future__ import annotations

import csv
import json
import math
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


LOGGER_VERSION = "idc-on-policy-metrics-v3"
METRIC_SCHEMA_VERSION = "2.1.0"
REWARD_RECONSTRUCTION_TOLERANCE = 1e-6

# Canonical log name -> physical-environment info name. r_grid_peak is deliberately
# absent: it aliases r_peak_load and must never be included in reconstruction.
REWARD_COMPONENT_SOURCES = (
    ("r_done", "r_done"),
    ("r_finished_tasks", "r_finished_task"),
    ("r_priority", "r_priority_finish"),
    ("r_cost", "r_cost"),
    ("r_carbon", "r_carbon"),
    ("r_queue", "r_queue"),
    ("r_overflow", "r_queue_overflow"),
    ("r_urgent", "r_urgent_backlog"),
    ("r_waiting", "r_waiting"),
    ("r_deadline", "r_deadline"),
    ("r_sla", "r_sla"),
    ("r_unused", "r_unused"),
    ("r_peak_load", "r_peak_load"),
    ("r_pause", "r_pause"),
    ("r_resume", "r_resume"),
    ("r_noninterruptible", "r_non_interruptible"),
    ("r_load_smooth", "r_load_smooth"),
    ("r_action_smooth", "r_action_smooth"),
    ("r_degradation", "r_bess_degradation"),
    ("r_invalid_action", "r_bess_invalid_action"),
    ("r_final_queue", "r_final_queue"),
    ("r_final_soc", "r_soc_final"),
)
REWARD_COMPONENTS = tuple(name for name, _ in REWARD_COMPONENT_SOURCES)

STEP_COLUMNS = (
    "run_id", "seed", "update", "global_step", "worker_id", "episode_id",
    "episode_step", "hour", "terminated", "truncated", "is_terminal",
    "reward_returned", "reward_total", "base_reward", "grid_adjusted_reward",
    "grid_penalty", *REWARD_COMPONENTS, "r_grid_peak",
    "reward_reconstruction_error", "idc_reward", "bess_reward",
    "shared_reward_abs_diff", "completed_work_step", "completed_work_total",
    "finished_tasks_step", "finished_tasks_total", "completion_rate",
    "task_completion_rate", "backlog_work", "queue_work", "overflow_work",
    "urgent_backlog", "active_task_count", "waiting_time_mean",
    "task_forecast_mode", "task_forecast_error_level", "task_forecast_mae",
    "task_forecast_rmse", "task_forecast_mape_nonzero_percent",
    "true_task_arrival_profile", "task_arrival_forecast",
    "deadline_miss_step", "deadline_miss_total", "sla_violation_count",
    "sla_value", "pause_count", "resume_count",
    "noninterruptible_violation_count", "idc_power_kW", "it_power_kW",
    "cooling_power_kW", "pue", "ambient_temperature_C",
    "ambient_temperature_C_available", "server_load_mean", "server_load_max",
    "server_load_max_available", "grid_energy_kWh", "price", "carbon_factor",
    "cost_step", "cost_total", "carbon_step_kg", "carbon_total_kg",
    "grid_power_kW", "p_bus_net_kW", "peak_power_kW", "peak_excess_kW",
    "pv_available_kW", "pv_used_kW", "pv_available_kWh", "pv_used_kWh",
    "grid_reference_usep_sgd_per_mwh", "bess_action_physical",
    "bess_action_padded_dim0", "bess_virtual_action_mean",
    "bess_virtual_action_std", "bess_virtual_action_min",
    "bess_virtual_action_max", "bess_charge_power_kW",
    "bess_discharge_power_kW", "bess_soc", "bess_energy_kWh",
    "bess_charge_kWh", "bess_discharge_kWh", "bess_throughput_kWh",
    "bess_degradation_cost", "bess_invalid_request",
    "bess_infeasible_request_power_kW", "grid_load_scale", "lmp",
    "lmp_available", "mef_plus", "mef_minus", "mef_available",
    "voltage_min_pu", "voltage_max_pu", "line_loading_max_pct",
    "grid_loss_mw", "opf_success", "mef_success", "voltage_violation",
    "line_violation", "opf_violation", "safe_violation_total", "safe_cost",
    "grid_security_penalty", "opf_cache_hit", "mef_cache_hit",
    "opf_cache_size", "mef_cache_size", "opf_cache_hits_total",
    "opf_cache_misses_total", "mef_cache_hits_total", "mef_cache_misses_total",
    "cache_opf_load_bin_mw", "cache_mef_load_bin_mw", "grid_reward_enabled",
    "safe_rl_enabled",
)

EPISODE_COLUMNS = (
    "run_id", "seed", "update", "worker_id", "episode_id", "complete",
    "episode_steps", "global_step_end", "episode_reward", "average_step_reward",
    "terminal_reward", *(f"{name}_sum" for name in REWARD_COMPONENTS),
    "reward_reconstruction_error_max_abs", "completed_work_sum",
    "finished_tasks_sum", "final_completion_rate", "final_task_completion_rate",
    "mean_queue", "max_queue", "final_backlog", "waiting_time_mean",
    "task_forecast_mode", "task_forecast_error_level", "task_forecast_mae",
    "task_forecast_rmse", "task_forecast_mape_nonzero_percent",
    "true_task_arrival_profile", "task_arrival_forecast",
    "deadline_miss_sum", "final_deadline_miss_total", "final_sla_violation_count",
    "pause_count_sum", "resume_count_sum", "noninterruptible_violation_sum",
    "grid_energy_kWh_sum", "cost_sum", "carbon_kg_sum", "pv_used_kWh_sum",
    "mean_pue", "mean_grid_power_kW", "max_grid_power_kW",
    "max_peak_excess_kW", "mean_bess_soc", "final_bess_soc",
    "bess_charge_kWh_sum", "bess_discharge_kWh_sum", "bess_throughput_kWh_sum",
    "bess_degradation_cost_sum", "max_bess_invalid_request_kW",
    "bess_physical_action_mean", "bess_physical_action_std",
    "charge_hour_fraction", "discharge_hour_fraction", "idle_hour_fraction",
    "opf_success_rate", "mef_success_rate", "opf_cache_hit_rate",
    "mef_cache_hit_rate", "voltage_violation_rate", "line_violation_rate",
    "min_voltage_pu", "max_voltage_pu", "mean_voltage_pu",
    "max_line_loading_pct", "mean_lmp", "mean_mef_plus",
    "safe_violation_sum", "max_safe_violation", "grid_security_penalty_sum",
    "final_cost_total", "final_carbon_total_kg", "final_opf_cache_size",
    "final_mef_cache_size", "shared_reward_max_diff",
)

UPDATE_COLUMNS = (
    "run_id", "seed", "update", "global_step", "episodes_completed",
    "method_id", "algorithm_name", "critic_type", "agent_update_order",
    "graph_schema_hash", "topology_hash", "hgta_hidden_dim", "hgta_heads",
    "hgta_layers", "hgta_node_count", "hgta_edge_count", "hgta_parameter_count",
    "agent_update_order_policy", "happo_available",
    "wall_time_seconds", "rollout_time_seconds", "update_time_seconds",
    "steps_per_second", "rollout_episode_reward_mean", "rollout_step_reward_mean",
    "value_loss", "critic_grad_norm", "critic_grad_norm_postclip",
    "critic_parameter_delta_norm", "critic_optimizer_step_delta",
    "critic_optimizer_step_count",
    "critic_update_count", "value_prediction_mean", "value_prediction_std",
    "graph_input_nonfinite_count", "node_embedding_nonfinite_count",
    "attention_nonfinite_count", "graph_embedding_nonfinite_count",
    "forecast_embedding_nonfinite_count", "value_prediction_nonfinite_count",
    "return_mean", "return_std",
    "advantage_mean", "advantage_std", "explained_variance",
    "idc_policy_loss", "idc_entropy", "idc_actor_grad_norm", "idc_ratio_mean",
    "idc_ratio_std", "idc_ratio_min", "idc_ratio_max", "idc_clip_fraction",
    "idc_approx_kl", "idc_action_mean", "idc_action_std",
    "idc_actor_grad_norm_preclip", "idc_actor_grad_norm_postclip",
    "idc_actor_parameter_delta_norm", "idc_actor_optimizer_step_delta",
    "idc_actor_optimizer_step_count", "actor_update_count_idc",
    "idc_rollout_log_prob_max_diff",
    "idc_action_buffer_mutation_max_diff", "idc_log_prob_buffer_mutation_max_diff",
    "bess_policy_loss", "bess_effective_entropy", "bess_actor_grad_norm",
    "bess_effective_ratio_mean", "bess_effective_ratio_std",
    "bess_effective_ratio_min", "bess_effective_ratio_max",
    "bess_effective_clip_fraction", "bess_effective_approx_kl",
    "bess_physical_action_mean", "bess_physical_action_std",
    "bess_physical_action_min", "bess_physical_action_max",
    "bess_charge_hour_fraction", "bess_discharge_hour_fraction",
    "bess_idle_hour_fraction", "bess_actor_grad_norm_preclip",
    "bess_actor_grad_norm_postclip", "bess_actor_parameter_delta_norm",
    "bess_actor_optimizer_step_delta", "bess_actor_optimizer_step_count",
    "actor_update_count_bess",
    "bess_rollout_log_prob_max_diff", "bess_action_buffer_mutation_max_diff",
    "bess_log_prob_buffer_mutation_max_diff",
    "bess_effective_action_dim", "bess_padded_action_dim",
    "bess_virtual_action_dim", "bess_effective_physical_ratio_max_diff",
    "bess_virtual_mean_grad_norm", "bess_virtual_log_std_grad_norm",
    "bess_virtual_entropy_optimization_contribution", "bess_mask_enabled",
    "happo_factor_initial_mean", "happo_factor_initial_min",
    "happo_factor_initial_max", "happo_factor_initial_p01",
    "happo_factor_initial_p50", "happo_factor_initial_p99",
    "happo_factor_initial_exactly_one_fraction",
    "happo_factor_after_first_agent_mean",
    "happo_factor_after_first_agent_min", "happo_factor_after_first_agent_max",
    "happo_factor_after_first_agent_p01", "happo_factor_after_first_agent_p50",
    "happo_factor_after_first_agent_p99",
    "happo_factor_after_first_agent_exactly_one_fraction",
    "happo_factor_after_idc_mean", "happo_factor_after_idc_min",
    "happo_factor_after_idc_max", "happo_factor_after_idc_p01",
    "happo_factor_after_idc_p50", "happo_factor_after_idc_p99",
    "happo_factor_after_idc_exactly_one_fraction",
    "happo_factor_final_mean", "happo_factor_final_min",
    "happo_factor_final_max", "happo_factor_final_p01",
    "happo_factor_final_p50", "happo_factor_final_p99",
    "happo_factor_final_exactly_one_fraction", "happo_factor_nonfinite_count",
    "happo_factor_nonpositive_count",
    "happo_factor_after_idc_reconstruction_max_diff",
    "happo_factor_final_reconstruction_max_diff",
    "critic_return_buffer_mutation_max_diff", "advantage_mutation_max_diff",
    "first_updated_agent", "second_updated_agent",
    "bess_happo_factor_effective_physical_max_diff",
    "bess_happo_virtual_factor_contribution",
    "idc_reward_mean", "bess_reward_mean", "shared_reward_max_diff",
    "opf_success_rate", "mef_success_rate", "safe_violation_sum",
)


class TrainingMetricsLogger:
    """Write stable project metrics without changing environment or optimizer semantics."""

    def __init__(
        self,
        *,
        run_dir: str | Path,
        seed: int,
        writer: Any | None,
        episode_length: int = 24,
        step_logging_enabled: bool = True,
        episode_logging_enabled: bool = True,
        tensorboard_enabled: bool = True,
        reward_tolerance: float = REWARD_RECONSTRUCTION_TOLERANCE,
        method_metadata: Mapping[str, Any] | None = None,
    ) -> None:
        self.run_dir = Path(run_dir).resolve()
        self.run_id = self.run_dir.name
        self.seed = int(seed)
        self.writer = writer
        self.episode_length = int(episode_length)
        self.step_logging_enabled = bool(step_logging_enabled)
        self.episode_logging_enabled = bool(episode_logging_enabled)
        self.tensorboard_enabled = bool(tensorboard_enabled)
        self.reward_tolerance = float(reward_tolerance)
        self.method_metadata = dict(method_metadata or {
            "method_id": "MAPPO_MLP", "algorithm_name": "mappo", "critic_type": "mlp"
        })
        self.metrics_dir = self.run_dir / "metrics"
        self.metrics_dir.mkdir(parents=False, exist_ok=False)
        self.step_path = self.metrics_dir / "step_metrics.csv"
        self.episode_path = self.metrics_dir / "episode_metrics.csv"
        self.update_path = self.metrics_dir / "update_metrics.csv"
        self.schema_path = self.metrics_dir / "metric_schema.json"
        self.summary_path = self.metrics_dir / "run_summary.json"
        self._step_stream, self._step_writer = self._open_csv(
            self.step_path, STEP_COLUMNS, enabled=self.step_logging_enabled
        )
        self._episode_stream, self._episode_writer = self._open_csv(
            self.episode_path, EPISODE_COLUMNS, enabled=self.episode_logging_enabled
        )
        self._update_stream, self._update_writer = self._open_csv(
            self.update_path, UPDATE_COLUMNS, enabled=True
        )
        self._write_schema()

        self._start_time = time.perf_counter()
        self._update_start_time = self._start_time
        self._update_start_global_step = 0
        self._current_update = 0
        self._global_step = 0
        self._episodes_completed = 0
        self._episode_ids: dict[int, int] = {}
        self._episode_rows: dict[int, list[dict[str, Any]]] = {}
        self._completed_this_update: list[dict[str, Any]] = []
        self._agent_rewards_this_update: list[tuple[float, float]] = []
        self._last_episode: dict[str, Any] | None = None
        self._best_episode: dict[str, Any] | None = None
        self._last_update: dict[str, Any] | None = None
        self._opf_successes = 0
        self._mef_successes = 0
        self._transition_count = 0
        self._finalized = False
        self._resumed = False
        self._resume_parent: str | None = None
        self._segment_start_update = 0
        self._segment_start_global_step = 0
        self._segment_start_episodes = 0

    @staticmethod
    def _open_csv(path: Path, columns: Sequence[str], *, enabled: bool):
        if not enabled:
            return None, None
        stream = path.open("x", encoding="utf-8", newline="")
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="raise")
        writer.writeheader()
        stream.flush()
        return stream, writer

    @property
    def global_step(self) -> int:
        return self._global_step

    @property
    def episodes_completed(self) -> int:
        return self._episodes_completed

    @property
    def finalized(self) -> bool:
        return self._finalized

    @property
    def last_update_metrics(self) -> dict[str, Any] | None:
        return None if self._last_update is None else dict(self._last_update)

    @property
    def last_episode_metrics(self) -> dict[str, Any] | None:
        return None if self._last_episode is None else dict(self._last_episode)

    def state_dict(self) -> dict[str, Any]:
        """Return counters needed to continue numbering at a post-update boundary."""
        if any(self._episode_rows.values()):
            raise RuntimeError("Logger state can only be checkpointed at an episode boundary.")
        if self._completed_this_update or self._agent_rewards_this_update:
            raise RuntimeError("Logger state can only be checkpointed after record_update().")
        self._flush_all()
        return {
            "version": "training-metrics-state-v1",
            "current_update": int(self._current_update),
            "global_step": int(self._global_step),
            "episodes_completed": int(self._episodes_completed),
            "episode_ids": {int(key): int(value) for key, value in self._episode_ids.items()},
            "last_episode": None if self._last_episode is None else dict(self._last_episode),
            "best_episode": None if self._best_episode is None else dict(self._best_episode),
            "last_update": None if self._last_update is None else dict(self._last_update),
            "opf_successes": int(self._opf_successes),
            "mef_successes": int(self._mef_successes),
            "transition_count": int(self._transition_count),
        }

    def load_state_dict(
        self, state: Mapping[str, Any], *, resume_parent: str | None = None
    ) -> None:
        """Continue counters in this new run segment without appending old CSV files."""
        if state.get("version") != "training-metrics-state-v1":
            raise ValueError(f"Unsupported logger checkpoint state {state.get('version')!r}.")
        self._current_update = int(state["current_update"])
        self._global_step = int(state["global_step"])
        self._episodes_completed = int(state["episodes_completed"])
        self._episode_ids = {
            int(key): int(value) for key, value in state["episode_ids"].items()
        }
        self._episode_rows = {key: [] for key in self._episode_ids}
        self._last_episode = None if state["last_episode"] is None else dict(state["last_episode"])
        self._best_episode = None if state["best_episode"] is None else dict(state["best_episode"])
        self._last_update = None if state["last_update"] is None else dict(state["last_update"])
        self._opf_successes = int(state["opf_successes"])
        self._mef_successes = int(state["mef_successes"])
        self._transition_count = int(state["transition_count"])
        self._completed_this_update = []
        self._agent_rewards_this_update = []
        self._resumed = True
        self._resume_parent = resume_parent
        self._segment_start_update = self._current_update
        self._segment_start_global_step = self._global_step
        self._segment_start_episodes = self._episodes_completed

    def begin_update(self, update: int) -> None:
        if self._current_update and self._completed_this_update:
            raise RuntimeError("Previous update metrics were not finalized before the next rollout.")
        self._current_update = int(update)
        self._update_start_time = time.perf_counter()
        self._update_start_global_step = self._global_step
        self._completed_this_update = []
        self._agent_rewards_this_update = []

    def begin_policy_update(self) -> float:
        """Return rollout/return-computation time before optimizer work starts."""
        return float(time.perf_counter() - self._update_start_time)

    def record_step(self, data: Sequence[Any]) -> None:
        rewards, dones, infos, actions = data[2], data[3], data[4], data[7]
        rewards_array = np.asarray(rewards)
        dones_array = np.asarray(dones)
        actions_array = np.asarray(actions)
        n_workers = int(rewards_array.shape[0])
        if rewards_array.shape[:2] != (n_workers, 2):
            raise ValueError(f"Expected two-agent rewards, got shape {rewards_array.shape!r}.")
        if actions_array.shape[:2] != (n_workers, 2) or actions_array.shape[-1] != 22:
            raise ValueError(f"Expected padded actions (workers,2,22), got {actions_array.shape!r}.")

        for worker_id in range(n_workers):
            worker_infos = infos[worker_id]
            if len(worker_infos) != 2:
                raise ValueError("Each worker must expose exactly two agent info dictionaries.")
            info = worker_infos[0]
            if not isinstance(info, Mapping):
                raise TypeError("Agent 0 info must be a mapping.")
            other_info = worker_infos[1]
            if not isinstance(other_info, Mapping):
                raise TypeError("Agent 1 info must be a mapping.")
            for key in ("reward_total", "hour", "terminated", "truncated"):
                if not self._shared_equal(info.get(key), other_info.get(key)):
                    raise ValueError(f"Two-agent info mismatch for shared field {key!r}.")

            episode_id = self._episode_ids.setdefault(worker_id, 0)
            accumulator = self._episode_rows.setdefault(worker_id, [])
            episode_step = len(accumulator)
            if episode_step >= self.episode_length:
                raise RuntimeError("Episode accumulator exceeded the configured episode length.")
            self._global_step += 1
            idc_reward = self._required_float(rewards_array[worker_id, 0, 0], "idc_reward")
            bess_reward = self._required_float(rewards_array[worker_id, 1, 0], "bess_reward")
            shared_diff = abs(idc_reward - bess_reward)
            self._agent_rewards_this_update.append((idc_reward, bess_reward))

            terminated = bool(info["terminated"])
            truncated = bool(info["truncated"])
            is_terminal = terminated or truncated or bool(np.all(dones_array[worker_id]))
            row = self._extract_step_row(
                info=info,
                actions=actions_array[worker_id],
                worker_id=worker_id,
                episode_id=episode_id,
                episode_step=episode_step,
                idc_reward=idc_reward,
                bess_reward=bess_reward,
                shared_diff=shared_diff,
                terminated=terminated,
                truncated=truncated,
                is_terminal=is_terminal,
            )
            if self._step_writer is not None:
                self._step_writer.writerow(row)
            accumulator.append(row)
            self._transition_count += 1
            self._opf_successes += int(row["opf_success"])
            self._mef_successes += int(row["mef_success"])

            if is_terminal:
                if len(accumulator) != self.episode_length:
                    raise RuntimeError(
                        f"Worker {worker_id} terminated after {len(accumulator)} steps; "
                        f"expected {self.episode_length}."
                    )
                episode = self._aggregate_episode(accumulator)
                if self._episode_writer is not None:
                    self._episode_writer.writerow(episode)
                self._flush_episode_files()
                self._completed_this_update.append(episode)
                self._last_episode = dict(episode)
                if (
                    self._best_episode is None
                    or float(episode["episode_reward"])
                    > float(self._best_episode["episode_reward"])
                ):
                    self._best_episode = dict(episode)
                self._episodes_completed += 1
                self._episode_ids[worker_id] = episode_id + 1
                self._episode_rows[worker_id] = []
                self._write_episode_tensorboard(episode)

    def _extract_step_row(
        self,
        *,
        info: Mapping[str, Any],
        actions: np.ndarray,
        worker_id: int,
        episode_id: int,
        episode_step: int,
        idc_reward: float,
        bess_reward: float,
        shared_diff: float,
        terminated: bool,
        truncated: bool,
        is_terminal: bool,
    ) -> dict[str, Any]:
        required_info = {
            source for _, source in REWARD_COMPONENT_SOURCES
        } | {
            "hour", "reward_total", "base_reward", "grid_adjusted_reward",
            "grid_reward_penalty", "completed_work", "total_completed_work",
            "newly_finished_count", "finished_task_count", "completion_rate",
            "task_completion_rate", "backlog_work", "Q", "overflow_work",
            "urgent_backlog_work", "unfinished_task_count", "avg_waiting_time",
            "new_deadline_miss_count", "deadline_miss_count", "sla_violation_count",
            "sla_penalty", "pause_count_this_step", "resume_count_this_step",
            "non_interruptible_interruption_this_step", "P_IDC_kW", "P_IT",
            "P_cooling", "PUE", "actual_task_load_mean", "grid_energy_kWh",
            "price", "carbon_factor", "cost", "total_cost", "carbon_emission",
            "total_carbon_emission", "grid_power_kW", "P_bus_net_kW",
            "grid_peak_power_kW", "grid_peak_excess_kW", "pv_available_kW",
            "pv_used_kW", "pv_available_kWh", "pv_used_kWh", "grid_reference_usep",
            "bess_raw_action", "bess_charge_power_kW", "bess_discharge_power_kW",
            "bess_soc", "bess_energy_kWh", "bess_cycle_throughput_kWh",
            "bess_charge_kWh", "bess_discharge_kWh",
            "bess_degradation_cost", "invalid_bess_action", "grid_load_scale",
            "grid_opf_success", "grid_mef_success", "grid_voltage_violation_count",
            "grid_line_overload_count", "safe_violation_opf", "safe_violation_cost",
            "safe_cost_total", "grid_security_penalty", "grid_opf_cache_hit",
            "grid_mef_cache_hit", "grid_cache_opf_size", "grid_cache_mef_size",
            "grid_opf_cache_hit_count", "grid_opf_cache_miss_count",
            "grid_mef_cache_hit_count", "grid_mef_cache_miss_count",
            "grid_cache_load_bin_mw", "grid_cache_mef_load_bin_mw",
            "grid_reward_enabled", "task_forecast_mode", "forecast_error_level",
            "task_forecast_mae", "task_forecast_rmse",
            "task_forecast_mape_nonzero_percent",
        }
        if is_terminal:
            required_info.update(
                {"true_task_arrival_profile", "task_arrival_forecast"}
            )
        missing = sorted(required_info.difference(info))
        if missing:
            raise KeyError("Training step info is missing required metrics: " + ", ".join(missing))

        reward_components = {
            canonical: self._required_float(info[source], source)
            for canonical, source in REWARD_COMPONENT_SOURCES
        }
        reward_total = self._required_float(info["reward_total"], "reward_total")
        reconstruction = float(sum(reward_components.values()))
        reconstruction_error = reconstruction - reward_total
        if abs(reconstruction_error) > self.reward_tolerance:
            raise ValueError(
                "Reward reconstruction failed: "
                f"components={reconstruction}, reward_total={reward_total}, "
                f"error={reconstruction_error}."
            )
        reward_returned = idc_reward
        adjusted = self._required_float(info["grid_adjusted_reward"], "grid_adjusted_reward")
        if abs(reward_returned - adjusted) > self.reward_tolerance:
            raise ValueError("Returned team reward does not equal grid_adjusted_reward.")

        hour = self._required_int(info["hour"], "hour")
        if hour != episode_step or not 0 <= hour < self.episode_length:
            raise ValueError(
                f"hour={hour} and episode_step={episode_step} are not aligned."
            )
        padded_bess = np.asarray(actions[1], dtype=np.float64)
        if padded_bess.shape != (22,) or not np.isfinite(padded_bess).all():
            raise ValueError("BESS padded action must be finite with shape (22,).")
        physical = self._required_float(info["bess_raw_action"], "bess_raw_action")
        virtual = padded_bess[1:]

        lmp = self._optional_float(info.get("grid_lmp"))
        mef_plus = self._optional_float(info.get("grid_mef_plus"))
        mef_minus = self._optional_float(info.get("grid_mef_minus"))
        row: dict[str, Any] = {
            "run_id": self.run_id,
            "seed": self.seed,
            "update": self._current_update,
            "global_step": self._global_step,
            "worker_id": worker_id,
            "episode_id": episode_id,
            "episode_step": episode_step,
            "hour": hour,
            "terminated": terminated,
            "truncated": truncated,
            "is_terminal": is_terminal,
            "reward_returned": reward_returned,
            "reward_total": reward_total,
            "base_reward": self._required_float(info["base_reward"], "base_reward"),
            "grid_adjusted_reward": adjusted,
            "grid_penalty": self._required_float(info["grid_reward_penalty"], "grid_reward_penalty"),
            **reward_components,
            "r_grid_peak": self._required_float(info["r_grid_peak"], "r_grid_peak"),
            "reward_reconstruction_error": reconstruction_error,
            "idc_reward": idc_reward,
            "bess_reward": bess_reward,
            "shared_reward_abs_diff": shared_diff,
            "completed_work_step": self._required_float(info["completed_work"], "completed_work"),
            "completed_work_total": self._required_float(info["total_completed_work"], "total_completed_work"),
            "finished_tasks_step": self._required_int(info["newly_finished_count"], "newly_finished_count"),
            "finished_tasks_total": self._required_int(info["finished_task_count"], "finished_task_count"),
            "completion_rate": self._required_float(info["completion_rate"], "completion_rate"),
            "task_completion_rate": self._required_float(info["task_completion_rate"], "task_completion_rate"),
            "backlog_work": self._required_float(info["backlog_work"], "backlog_work"),
            "queue_work": self._required_float(info["Q"], "Q"),
            "overflow_work": self._required_float(info["overflow_work"], "overflow_work"),
            "urgent_backlog": self._required_float(info["urgent_backlog_work"], "urgent_backlog_work"),
            "active_task_count": self._required_int(info["unfinished_task_count"], "unfinished_task_count"),
            "waiting_time_mean": self._required_float(info["avg_waiting_time"], "avg_waiting_time"),
            "task_forecast_mode": str(info["task_forecast_mode"]),
            "task_forecast_error_level": self._required_float(
                info["forecast_error_level"], "forecast_error_level"
            ),
            "task_forecast_mae": self._required_float(
                info["task_forecast_mae"], "task_forecast_mae"
            ),
            "task_forecast_rmse": self._required_float(
                info["task_forecast_rmse"], "task_forecast_rmse"
            ),
            "task_forecast_mape_nonzero_percent": self._required_float(
                info["task_forecast_mape_nonzero_percent"],
                "task_forecast_mape_nonzero_percent",
            ),
            "true_task_arrival_profile": (
                json.dumps(
                    np.asarray(info["true_task_arrival_profile"], dtype=np.float64).tolist(),
                    separators=(",", ":"),
                )
                if is_terminal
                else ""
            ),
            "task_arrival_forecast": (
                json.dumps(
                    np.asarray(info["task_arrival_forecast"], dtype=np.float64).tolist(),
                    separators=(",", ":"),
                )
                if is_terminal
                else ""
            ),
            "deadline_miss_step": self._required_int(info["new_deadline_miss_count"], "new_deadline_miss_count"),
            "deadline_miss_total": self._required_int(info["deadline_miss_count"], "deadline_miss_count"),
            "sla_violation_count": self._required_int(info["sla_violation_count"], "sla_violation_count"),
            "sla_value": self._required_float(info["sla_penalty"], "sla_penalty"),
            "pause_count": self._required_int(info["pause_count_this_step"], "pause_count_this_step"),
            "resume_count": self._required_int(info["resume_count_this_step"], "resume_count_this_step"),
            "noninterruptible_violation_count": self._required_int(
                info["non_interruptible_interruption_this_step"],
                "non_interruptible_interruption_this_step",
            ),
            "idc_power_kW": self._required_float(info["P_IDC_kW"], "P_IDC_kW"),
            "it_power_kW": self._required_float(info["P_IT"], "P_IT"),
            "cooling_power_kW": self._required_float(info["P_cooling"], "P_cooling"),
            "pue": self._required_float(info["PUE"], "PUE"),
            "ambient_temperature_C": "",
            "ambient_temperature_C_available": False,
            "server_load_mean": self._required_float(info["actual_task_load_mean"], "actual_task_load_mean"),
            "server_load_max": "",
            "server_load_max_available": False,
            "grid_energy_kWh": self._required_float(info["grid_energy_kWh"], "grid_energy_kWh"),
            "price": self._required_float(info["price"], "price"),
            "carbon_factor": self._required_float(info["carbon_factor"], "carbon_factor"),
            "cost_step": self._required_float(info["cost"], "cost"),
            "cost_total": self._required_float(info["total_cost"], "total_cost"),
            "carbon_step_kg": self._required_float(info["carbon_emission"], "carbon_emission"),
            "carbon_total_kg": self._required_float(info["total_carbon_emission"], "total_carbon_emission"),
            "grid_power_kW": self._required_float(info["grid_power_kW"], "grid_power_kW"),
            "p_bus_net_kW": self._required_float(info["P_bus_net_kW"], "P_bus_net_kW"),
            "peak_power_kW": self._required_float(info["grid_peak_power_kW"], "grid_peak_power_kW"),
            "peak_excess_kW": self._required_float(info["grid_peak_excess_kW"], "grid_peak_excess_kW"),
            "pv_available_kW": self._required_float(info["pv_available_kW"], "pv_available_kW"),
            "pv_used_kW": self._required_float(info["pv_used_kW"], "pv_used_kW"),
            "pv_available_kWh": self._required_float(info["pv_available_kWh"], "pv_available_kWh"),
            "pv_used_kWh": self._required_float(info["pv_used_kWh"], "pv_used_kWh"),
            "grid_reference_usep_sgd_per_mwh": self._required_float(info["grid_reference_usep"], "grid_reference_usep"),
            "bess_action_physical": physical,
            "bess_action_padded_dim0": float(padded_bess[0]),
            "bess_virtual_action_mean": float(np.mean(virtual)),
            "bess_virtual_action_std": float(np.std(virtual)),
            "bess_virtual_action_min": float(np.min(virtual)),
            "bess_virtual_action_max": float(np.max(virtual)),
            "bess_charge_power_kW": self._required_float(info["bess_charge_power_kW"], "bess_charge_power_kW"),
            "bess_discharge_power_kW": self._required_float(info["bess_discharge_power_kW"], "bess_discharge_power_kW"),
            "bess_soc": self._required_float(info["bess_soc"], "bess_soc"),
            "bess_energy_kWh": self._required_float(info["bess_energy_kWh"], "bess_energy_kWh"),
            "bess_charge_kWh": self._required_float(info["bess_charge_kWh"], "bess_charge_kWh"),
            "bess_discharge_kWh": self._required_float(info["bess_discharge_kWh"], "bess_discharge_kWh"),
            "bess_throughput_kWh": self._required_float(info["bess_cycle_throughput_kWh"], "bess_cycle_throughput_kWh"),
            "bess_degradation_cost": self._required_float(info["bess_degradation_cost"], "bess_degradation_cost"),
            "bess_invalid_request": abs(self._required_float(info["invalid_bess_action"], "invalid_bess_action")) > 1e-12,
            "bess_infeasible_request_power_kW": self._required_float(info["invalid_bess_action"], "invalid_bess_action"),
            "grid_load_scale": self._required_float(info["grid_load_scale"], "grid_load_scale"),
            "lmp": lmp,
            "lmp_available": lmp != "",
            "mef_plus": mef_plus,
            "mef_minus": mef_minus,
            "mef_available": mef_plus != "" and mef_minus != "",
            "voltage_min_pu": self._optional_float(info.get("grid_min_voltage_pu")),
            "voltage_max_pu": self._optional_float(info.get("grid_max_voltage_pu")),
            "line_loading_max_pct": self._optional_float(info.get("grid_max_line_loading_percent")),
            "grid_loss_mw": self._optional_float(info.get("grid_network_loss_mw")),
            "opf_success": bool(info["grid_opf_success"]),
            "mef_success": bool(info["grid_mef_success"]),
            "voltage_violation": self._required_int(info["grid_voltage_violation_count"], "grid_voltage_violation_count") > 0,
            "line_violation": self._required_int(info["grid_line_overload_count"], "grid_line_overload_count") > 0,
            "opf_violation": self._required_float(info["safe_violation_opf"], "safe_violation_opf") > 0.0,
            "safe_violation_total": self._required_float(info["safe_violation_cost"], "safe_violation_cost"),
            "safe_cost": self._required_float(info["safe_cost_total"], "safe_cost_total"),
            "grid_security_penalty": self._required_float(info["grid_security_penalty"], "grid_security_penalty"),
            "opf_cache_hit": bool(info["grid_opf_cache_hit"]),
            "mef_cache_hit": bool(info["grid_mef_cache_hit"]),
            "opf_cache_size": self._required_int(info["grid_cache_opf_size"], "grid_cache_opf_size"),
            "mef_cache_size": self._required_int(info["grid_cache_mef_size"], "grid_cache_mef_size"),
            "opf_cache_hits_total": self._required_int(info["grid_opf_cache_hit_count"], "grid_opf_cache_hit_count"),
            "opf_cache_misses_total": self._required_int(info["grid_opf_cache_miss_count"], "grid_opf_cache_miss_count"),
            "mef_cache_hits_total": self._required_int(info["grid_mef_cache_hit_count"], "grid_mef_cache_hit_count"),
            "mef_cache_misses_total": self._required_int(info["grid_mef_cache_miss_count"], "grid_mef_cache_miss_count"),
            "cache_opf_load_bin_mw": self._required_float(info["grid_cache_load_bin_mw"], "grid_cache_load_bin_mw"),
            "cache_mef_load_bin_mw": self._required_float(info["grid_cache_mef_load_bin_mw"], "grid_cache_mef_load_bin_mw"),
            "grid_reward_enabled": bool(info["grid_reward_enabled"]),
            "safe_rl_enabled": False,
        }
        if set(row) != set(STEP_COLUMNS):
            raise AssertionError(f"Step schema mismatch: {set(STEP_COLUMNS).symmetric_difference(row)}")
        return row

    def _aggregate_episode(self, rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        final = rows[-1]
        values = lambda field: np.asarray([float(row[field]) for row in rows], dtype=np.float64)
        optional_values = lambda field: np.asarray(
            [float(row[field]) for row in rows if row[field] != ""], dtype=np.float64
        )
        charge = values("bess_charge_power_kW")
        discharge = values("bess_discharge_power_kW")
        idle = (charge <= 1e-12) & (discharge <= 1e-12)
        voltage = np.concatenate(
            (optional_values("voltage_min_pu"), optional_values("voltage_max_pu"))
        )
        lmp = optional_values("lmp")
        mef = optional_values("mef_plus")
        episode: dict[str, Any] = {
            "run_id": self.run_id,
            "seed": self.seed,
            "update": final["update"],
            "worker_id": final["worker_id"],
            "episode_id": final["episode_id"],
            "complete": True,
            "episode_steps": len(rows),
            "global_step_end": final["global_step"],
            "episode_reward": float(values("reward_returned").sum()),
            "average_step_reward": float(values("reward_returned").mean()),
            "terminal_reward": float(final["reward_returned"]),
            **{f"{name}_sum": float(values(name).sum()) for name in REWARD_COMPONENTS},
            "reward_reconstruction_error_max_abs": float(np.max(np.abs(values("reward_reconstruction_error")))),
            "completed_work_sum": float(values("completed_work_step").sum()),
            "finished_tasks_sum": int(values("finished_tasks_step").sum()),
            "final_completion_rate": float(final["completion_rate"]),
            "final_task_completion_rate": float(final["task_completion_rate"]),
            "mean_queue": float(values("queue_work").mean()),
            "max_queue": float(values("queue_work").max()),
            "final_backlog": float(final["backlog_work"]),
            "waiting_time_mean": float(values("waiting_time_mean").mean()),
            "task_forecast_mode": str(final["task_forecast_mode"]),
            "task_forecast_error_level": float(final["task_forecast_error_level"]),
            "task_forecast_mae": float(final["task_forecast_mae"]),
            "task_forecast_rmse": float(final["task_forecast_rmse"]),
            "task_forecast_mape_nonzero_percent": float(
                final["task_forecast_mape_nonzero_percent"]
            ),
            "true_task_arrival_profile": str(final["true_task_arrival_profile"]),
            "task_arrival_forecast": str(final["task_arrival_forecast"]),
            "deadline_miss_sum": int(values("deadline_miss_step").sum()),
            "final_deadline_miss_total": int(final["deadline_miss_total"]),
            "final_sla_violation_count": int(final["sla_violation_count"]),
            "pause_count_sum": int(values("pause_count").sum()),
            "resume_count_sum": int(values("resume_count").sum()),
            "noninterruptible_violation_sum": int(values("noninterruptible_violation_count").sum()),
            "grid_energy_kWh_sum": float(values("grid_energy_kWh").sum()),
            "cost_sum": float(values("cost_step").sum()),
            "carbon_kg_sum": float(values("carbon_step_kg").sum()),
            "pv_used_kWh_sum": float(values("pv_used_kWh").sum()),
            "mean_pue": float(values("pue").mean()),
            "mean_grid_power_kW": float(values("grid_power_kW").mean()),
            "max_grid_power_kW": float(values("grid_power_kW").max()),
            "max_peak_excess_kW": float(values("peak_excess_kW").max()),
            "mean_bess_soc": float(values("bess_soc").mean()),
            "final_bess_soc": float(final["bess_soc"]),
            "bess_charge_kWh_sum": float(values("bess_charge_kWh").sum()),
            "bess_discharge_kWh_sum": float(values("bess_discharge_kWh").sum()),
            "bess_throughput_kWh_sum": float(values("bess_throughput_kWh").sum()),
            "bess_degradation_cost_sum": float(values("bess_degradation_cost").sum()),
            "max_bess_invalid_request_kW": float(np.max(np.abs(values("bess_infeasible_request_power_kW")))),
            "bess_physical_action_mean": float(values("bess_action_physical").mean()),
            "bess_physical_action_std": float(values("bess_action_physical").std()),
            "charge_hour_fraction": float(np.mean(charge > 1e-12)),
            "discharge_hour_fraction": float(np.mean(discharge > 1e-12)),
            "idle_hour_fraction": float(np.mean(idle)),
            "opf_success_rate": float(values("opf_success").mean()),
            "mef_success_rate": float(values("mef_success").mean()),
            "opf_cache_hit_rate": float(values("opf_cache_hit").mean()),
            "mef_cache_hit_rate": float(values("mef_cache_hit").mean()),
            "voltage_violation_rate": float(values("voltage_violation").mean()),
            "line_violation_rate": float(values("line_violation").mean()),
            "min_voltage_pu": float(voltage.min()) if voltage.size else "",
            "max_voltage_pu": float(voltage.max()) if voltage.size else "",
            "mean_voltage_pu": float(voltage.mean()) if voltage.size else "",
            "max_line_loading_pct": self._optional_max(rows, "line_loading_max_pct"),
            "mean_lmp": float(lmp.mean()) if lmp.size else "",
            "mean_mef_plus": float(mef.mean()) if mef.size else "",
            "safe_violation_sum": float(values("safe_violation_total").sum()),
            "max_safe_violation": float(values("safe_violation_total").max()),
            "grid_security_penalty_sum": float(values("grid_security_penalty").sum()),
            "final_cost_total": float(final["cost_total"]),
            "final_carbon_total_kg": float(final["carbon_total_kg"]),
            "final_opf_cache_size": int(final["opf_cache_size"]),
            "final_mef_cache_size": int(final["mef_cache_size"]),
            "shared_reward_max_diff": float(values("shared_reward_abs_diff").max()),
        }
        if set(episode) != set(EPISODE_COLUMNS):
            raise AssertionError(f"Episode schema mismatch: {set(EPISODE_COLUMNS).symmetric_difference(episode)}")
        return episode

    def record_update(
        self,
        *,
        update: int,
        actor_infos: Sequence[Mapping[str, Any]],
        critic_info: Mapping[str, Any],
        actor_buffers: Sequence[Any],
        critic_buffer: Any,
        actors: Sequence[Any],
        update_time_seconds: float,
        rollout_time_seconds: float,
        algorithm_update: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        if int(update) != self._current_update:
            raise ValueError("Update number does not match the active rollout.")
        if len(actor_infos) != 2 or len(actor_buffers) != 2 or len(actors) != 2:
            raise ValueError("Structured metrics require IDC and BESS actor data.")
        if not self._completed_this_update:
            raise RuntimeError("An update completed without a complete logged episode.")
        idc_diag = self._require_diagnostics(actors[0], "IDC")
        bess_diag = self._require_diagnostics(actors[1], "BESS")
        rewards = np.asarray(self._agent_rewards_this_update, dtype=np.float64)
        idc_actions = np.asarray(actor_buffers[0].actions, dtype=np.float64)
        bess_actions = np.asarray(actor_buffers[1].actions, dtype=np.float64)[..., 0]
        value_preds = np.asarray(critic_buffer.value_preds[:-1], dtype=np.float64)
        returns = np.asarray(critic_buffer.returns[:-1], dtype=np.float64)
        advantages = returns - value_preds
        variance = float(np.var(returns))
        explained_variance = (
            1.0 - float(np.var(returns - value_preds)) / variance
            if variance > 1e-12
            else 0.0
        )
        wall = float(time.perf_counter() - self._start_time)
        update_elapsed = float(rollout_time_seconds + update_time_seconds)
        steps = self._global_step - self._update_start_global_step
        episode_rewards = np.asarray(
            [float(row["episode_reward"]) for row in self._completed_this_update]
        )
        opf_rates = np.asarray(
            [float(row["opf_success_rate"]) for row in self._completed_this_update]
        )
        mef_rates = np.asarray(
            [float(row["mef_success_rate"]) for row in self._completed_this_update]
        )
        safe_sums = np.asarray(
            [float(row["safe_violation_sum"]) for row in self._completed_this_update]
        )
        charge_fractions = np.asarray(
            [float(row["charge_hour_fraction"]) for row in self._completed_this_update]
        )
        discharge_fractions = np.asarray(
            [float(row["discharge_hour_fraction"]) for row in self._completed_this_update]
        )
        idle_fractions = np.asarray(
            [float(row["idle_hour_fraction"]) for row in self._completed_this_update]
        )
        algorithm_update = dict(algorithm_update or {})
        happo_available = bool(algorithm_update.get("happo_available", False))
        happo = lambda key: algorithm_update.get(key, "") if happo_available else ""
        row = {
            "run_id": self.run_id,
            "seed": self.seed,
            "update": int(update),
            "global_step": self._global_step,
            "episodes_completed": self._episodes_completed,
            "method_id": self.method_metadata["method_id"],
            "algorithm_name": self.method_metadata["algorithm_name"],
            "critic_type": self.method_metadata["critic_type"],
            "graph_schema_hash": self.method_metadata.get("graph_schema_hash", ""),
            "topology_hash": self.method_metadata.get("topology_hash", ""),
            "hgta_hidden_dim": int(self.method_metadata.get("hgta_hidden_dim", 0)),
            "hgta_heads": int(self.method_metadata.get("hgta_heads", 0)),
            "hgta_layers": int(self.method_metadata.get("hgta_layers", 0)),
            "hgta_node_count": int(self.method_metadata.get("hgta_node_count", 0)),
            "hgta_edge_count": int(self.method_metadata.get("hgta_edge_count", 0)),
            "hgta_parameter_count": int(self.method_metadata.get("hgta_parameter_count", 0)),
            "agent_update_order": ",".join(algorithm_update.get("agent_update_order", ())),
            "agent_update_order_policy": algorithm_update.get("agent_update_order_policy", ""),
            "happo_available": happo_available,
            "wall_time_seconds": wall,
            "rollout_time_seconds": float(rollout_time_seconds),
            "update_time_seconds": float(update_time_seconds),
            "steps_per_second": float(steps / update_elapsed) if update_elapsed > 0 else 0.0,
            "rollout_episode_reward_mean": float(episode_rewards.mean()),
            "rollout_step_reward_mean": float(rewards[:, 0].mean()),
            "value_loss": self._metric_float(critic_info["value_loss"], "value_loss"),
            "critic_grad_norm": self._metric_float(critic_info["critic_grad_norm"], "critic_grad_norm"),
            "critic_grad_norm_postclip": happo("critic_grad_norm_postclip"),
            "critic_parameter_delta_norm": happo("critic_parameter_delta_norm"),
            "critic_optimizer_step_delta": happo("critic_optimizer_step_delta"),
            "critic_optimizer_step_count": happo("critic_optimizer_step_count"),
            "critic_update_count": happo("critic_update_count"),
            "value_prediction_mean": float(value_preds.mean()),
            "value_prediction_std": float(value_preds.std()),
            "graph_input_nonfinite_count": int(critic_info.get("graph_input_nonfinite_count", 0)),
            "node_embedding_nonfinite_count": int(critic_info.get("node_embedding_nonfinite_count", 0)),
            "attention_nonfinite_count": int(critic_info.get("attention_nonfinite_count", 0)),
            "graph_embedding_nonfinite_count": int(critic_info.get("graph_embedding_nonfinite_count", 0)),
            "forecast_embedding_nonfinite_count": int(critic_info.get("forecast_embedding_nonfinite_count", 0)),
            "value_prediction_nonfinite_count": int(critic_info.get("value_prediction_nonfinite_count", 0)),
            "return_mean": float(returns.mean()),
            "return_std": float(returns.std()),
            "advantage_mean": float(advantages.mean()),
            "advantage_std": float(advantages.std()),
            "explained_variance": explained_variance,
            "idc_policy_loss": self._metric_float(actor_infos[0]["policy_loss"], "idc_policy_loss"),
            "idc_entropy": self._metric_float(actor_infos[0]["dist_entropy"], "idc_entropy"),
            "idc_actor_grad_norm": self._metric_float(actor_infos[0]["actor_grad_norm"], "idc_actor_grad_norm"),
            "idc_ratio_mean": self._diag_float(idc_diag, "effective_ratio_mean"),
            "idc_ratio_std": self._diag_float(idc_diag, "effective_ratio_std"),
            "idc_ratio_min": self._diag_float(idc_diag, "effective_ratio_min"),
            "idc_ratio_max": self._diag_float(idc_diag, "effective_ratio_max"),
            "idc_clip_fraction": self._diag_float(idc_diag, "effective_clip_fraction"),
            "idc_approx_kl": self._diag_float(idc_diag, "effective_approx_kl"),
            "idc_action_mean": float(idc_actions.mean()),
            "idc_action_std": float(idc_actions.std()),
            "idc_actor_grad_norm_preclip": self._metric_float(
                idc_diag.get("total_actor_grad_norm_preclip", actor_infos[0]["actor_grad_norm"]),
                "idc_actor_grad_norm_preclip",
            ),
            "idc_actor_grad_norm_postclip": self._metric_float(
                idc_diag.get("total_actor_grad_norm_postclip", actor_infos[0]["actor_grad_norm"]),
                "idc_actor_grad_norm_postclip",
            ),
            "idc_actor_parameter_delta_norm": happo("idc_actor_parameter_delta_norm"),
            "idc_actor_optimizer_step_delta": happo("idc_actor_optimizer_step_delta"),
            "idc_actor_optimizer_step_count": happo("idc_actor_optimizer_step_count"),
            "actor_update_count_idc": happo("actor_update_count_idc"),
            "idc_rollout_log_prob_max_diff": happo("idc_rollout_log_prob_max_diff"),
            "idc_action_buffer_mutation_max_diff": happo("idc_action_buffer_mutation_max_diff"),
            "idc_log_prob_buffer_mutation_max_diff": happo("idc_log_prob_buffer_mutation_max_diff"),
            "bess_policy_loss": self._metric_float(actor_infos[1]["policy_loss"], "bess_policy_loss"),
            "bess_effective_entropy": self._metric_float(actor_infos[1]["dist_entropy"], "bess_effective_entropy"),
            "bess_actor_grad_norm": self._metric_float(actor_infos[1]["actor_grad_norm"], "bess_actor_grad_norm"),
            "bess_effective_ratio_mean": self._diag_float(bess_diag, "effective_ratio_mean"),
            "bess_effective_ratio_std": self._diag_float(bess_diag, "effective_ratio_std"),
            "bess_effective_ratio_min": self._diag_float(bess_diag, "effective_ratio_min"),
            "bess_effective_ratio_max": self._diag_float(bess_diag, "effective_ratio_max"),
            "bess_effective_clip_fraction": self._diag_float(bess_diag, "effective_clip_fraction"),
            "bess_effective_approx_kl": self._diag_float(bess_diag, "effective_approx_kl"),
            "bess_physical_action_mean": float(bess_actions.mean()),
            "bess_physical_action_std": float(bess_actions.std()),
            "bess_physical_action_min": float(bess_actions.min()),
            "bess_physical_action_max": float(bess_actions.max()),
            "bess_charge_hour_fraction": float(charge_fractions.mean()),
            "bess_discharge_hour_fraction": float(discharge_fractions.mean()),
            "bess_idle_hour_fraction": float(idle_fractions.mean()),
            "bess_actor_grad_norm_preclip": self._metric_float(
                bess_diag.get("total_actor_grad_norm_preclip", actor_infos[1]["actor_grad_norm"]),
                "bess_actor_grad_norm_preclip",
            ),
            "bess_actor_grad_norm_postclip": self._metric_float(
                bess_diag.get("total_actor_grad_norm_postclip", actor_infos[1]["actor_grad_norm"]),
                "bess_actor_grad_norm_postclip",
            ),
            "bess_actor_parameter_delta_norm": happo("bess_actor_parameter_delta_norm"),
            "bess_actor_optimizer_step_delta": happo("bess_actor_optimizer_step_delta"),
            "bess_actor_optimizer_step_count": happo("bess_actor_optimizer_step_count"),
            "actor_update_count_bess": happo("actor_update_count_bess"),
            "bess_rollout_log_prob_max_diff": happo("bess_rollout_log_prob_max_diff"),
            "bess_action_buffer_mutation_max_diff": happo("bess_action_buffer_mutation_max_diff"),
            "bess_log_prob_buffer_mutation_max_diff": happo("bess_log_prob_buffer_mutation_max_diff"),
            "bess_effective_action_dim": int(bess_diag["effective_action_dim"]),
            "bess_padded_action_dim": int(bess_diag["padded_action_dim"]),
            "bess_virtual_action_dim": int(bess_diag["padded_action_dim"] - bess_diag["effective_action_dim"]),
            "bess_effective_physical_ratio_max_diff": self._diag_float(bess_diag, "effective_physical_ratio_max_diff"),
            "bess_virtual_mean_grad_norm": self._diag_float(bess_diag, "virtual_mean_rows_grad_norm"),
            "bess_virtual_log_std_grad_norm": self._diag_float(bess_diag, "virtual_log_std_grad_norm"),
            "bess_virtual_entropy_optimization_contribution": self._diag_float(
                bess_diag, "virtual_entropy_optimization_contribution"
            ),
            "bess_mask_enabled": int(bess_diag["effective_action_dim"]) < int(bess_diag["padded_action_dim"]),
            "happo_factor_initial_mean": happo("happo_factor_initial_mean"),
            "happo_factor_initial_min": happo("happo_factor_initial_min"),
            "happo_factor_initial_max": happo("happo_factor_initial_max"),
            "happo_factor_initial_p01": happo("happo_factor_initial_p01"),
            "happo_factor_initial_p50": happo("happo_factor_initial_p50"),
            "happo_factor_initial_p99": happo("happo_factor_initial_p99"),
            "happo_factor_initial_exactly_one_fraction": happo("happo_factor_initial_exactly_one_fraction"),
            "happo_factor_after_first_agent_mean": happo("happo_factor_after_first_agent_mean"),
            "happo_factor_after_first_agent_min": happo("happo_factor_after_first_agent_min"),
            "happo_factor_after_first_agent_max": happo("happo_factor_after_first_agent_max"),
            "happo_factor_after_first_agent_p01": happo("happo_factor_after_first_agent_p01"),
            "happo_factor_after_first_agent_p50": happo("happo_factor_after_first_agent_p50"),
            "happo_factor_after_first_agent_p99": happo("happo_factor_after_first_agent_p99"),
            "happo_factor_after_first_agent_exactly_one_fraction": happo("happo_factor_after_first_agent_exactly_one_fraction"),
            "happo_factor_after_idc_mean": happo("happo_factor_after_idc_mean"),
            "happo_factor_after_idc_min": happo("happo_factor_after_idc_min"),
            "happo_factor_after_idc_max": happo("happo_factor_after_idc_max"),
            "happo_factor_after_idc_p01": happo("happo_factor_after_idc_p01"),
            "happo_factor_after_idc_p50": happo("happo_factor_after_idc_p50"),
            "happo_factor_after_idc_p99": happo("happo_factor_after_idc_p99"),
            "happo_factor_after_idc_exactly_one_fraction": happo(
                "happo_factor_after_idc_exactly_one_fraction"
            ),
            "happo_factor_final_mean": happo("happo_factor_final_mean"),
            "happo_factor_final_min": happo("happo_factor_final_min"),
            "happo_factor_final_max": happo("happo_factor_final_max"),
            "happo_factor_final_p01": happo("happo_factor_final_p01"),
            "happo_factor_final_p50": happo("happo_factor_final_p50"),
            "happo_factor_final_p99": happo("happo_factor_final_p99"),
            "happo_factor_final_exactly_one_fraction": happo("happo_factor_final_exactly_one_fraction"),
            "happo_factor_nonfinite_count": happo("happo_factor_nonfinite_count"),
            "happo_factor_nonpositive_count": happo("happo_factor_nonpositive_count"),
            "happo_factor_after_idc_reconstruction_max_diff": happo(
                "happo_factor_after_idc_reconstruction_max_diff"
            ),
            "happo_factor_final_reconstruction_max_diff": happo(
                "happo_factor_final_reconstruction_max_diff"
            ),
            "critic_return_buffer_mutation_max_diff": happo(
                "critic_return_buffer_mutation_max_diff"
            ),
            "advantage_mutation_max_diff": happo("advantage_mutation_max_diff"),
            "first_updated_agent": happo("first_updated_agent"),
            "second_updated_agent": happo("second_updated_agent"),
            "bess_happo_factor_effective_physical_max_diff": happo(
                "bess_happo_factor_effective_physical_max_diff"
            ),
            "bess_happo_virtual_factor_contribution": happo(
                "bess_happo_virtual_factor_contribution"
            ),
            "idc_reward_mean": float(rewards[:, 0].mean()),
            "bess_reward_mean": float(rewards[:, 1].mean()),
            "shared_reward_max_diff": float(np.max(np.abs(rewards[:, 0] - rewards[:, 1]))),
            "opf_success_rate": float(opf_rates.mean()),
            "mef_success_rate": float(mef_rates.mean()),
            "safe_violation_sum": float(safe_sums.sum()),
        }
        self._assert_finite_row(row, exceptions=())
        self._update_writer.writerow(row)
        self._update_stream.flush()
        self._write_update_tensorboard(row)
        self._last_update = dict(row)
        self._completed_this_update = []
        self._agent_rewards_this_update = []
        return row

    def finalize(
        self,
        *,
        status: str,
        termination_reason: str,
        model_paths: Sequence[str | Path] = (),
        nan_or_inf_detected: bool = False,
    ) -> dict[str, Any]:
        if self._finalized:
            raise RuntimeError("Training metrics summary was already finalized.")
        incomplete = {
            str(worker): len(rows)
            for worker, rows in self._episode_rows.items()
            if rows
        }
        summary = {
            "logger_version": LOGGER_VERSION,
            "metric_schema_version": METRIC_SCHEMA_VERSION,
            "status": str(status),
            "termination_reason": str(termination_reason),
            "updates_completed": self._current_update if status == "completed" else max(self._current_update - 1, 0),
            "total_updates_completed": self._current_update if status == "completed" else max(self._current_update - 1, 0),
            "updates_completed_this_segment": max(
                (self._current_update if status == "completed" else max(self._current_update - 1, 0))
                - self._segment_start_update,
                0,
            ),
            "total_environment_steps": self._global_step,
            "environment_steps_this_segment": self._global_step - self._segment_start_global_step,
            "episodes_completed": self._episodes_completed,
            "episodes_completed_this_segment": self._episodes_completed - self._segment_start_episodes,
            "resumed": self._resumed,
            "resume_parent": self._resume_parent,
            "incomplete_episode_steps_by_worker": incomplete,
            "incomplete_episodes_written_to_episode_csv": False,
            "last_episode_metrics": self._last_episode,
            "best_episode_reward": None if self._best_episode is None else self._best_episode["episode_reward"],
            "best_episode_update": None if self._best_episode is None else self._best_episode["update"],
            "final_model_paths": [str(Path(path).resolve()) for path in model_paths],
            "nan_or_inf_detected": bool(nan_or_inf_detected),
            "opf_success_rate": self._opf_successes / self._transition_count if self._transition_count else None,
            "mef_success_rate": self._mef_successes / self._transition_count if self._transition_count else None,
            "log_paths": {
                "step_metrics": str(self.step_path) if self.step_logging_enabled else None,
                "episode_metrics": str(self.episode_path) if self.episode_logging_enabled else None,
                "update_metrics": str(self.update_path),
                "metric_schema": str(self.schema_path),
                "run_summary": str(self.summary_path),
                "tensorboard": str(self.run_dir / "logs") if self.tensorboard_enabled else None,
            },
        }
        with self.summary_path.open("x", encoding="utf-8") as stream:
            json.dump(summary, stream, ensure_ascii=False, indent=2, sort_keys=True)
            stream.write("\n")
        self._finalized = True
        self._flush_all()
        return summary

    def close(self) -> None:
        if not self._finalized:
            self.finalize(
                status="interrupted",
                termination_reason="logger_closed_before_formal_finalize",
            )
        for stream in (self._step_stream, self._episode_stream, self._update_stream):
            if stream is not None and not stream.closed:
                stream.flush()
                stream.close()

    def _write_schema(self) -> None:
        reward_sources = dict(REWARD_COMPONENT_SOURCES)
        schema: dict[str, Any] = {
            "logger_version": LOGGER_VERSION,
            "metric_schema_version": METRIC_SCHEMA_VERSION,
            "reward_reconstruction_tolerance": self.reward_tolerance,
            "reward_aliases": {"r_grid_peak": "r_peak_load"},
            "semantic_notes": {
                "price": "Artificial IDC price used by the physical cost and reward.",
                "grid_reference_usep_sgd_per_mwh": "NEMS USEP reference only; it does not enter the current reward.",
                "grid_safe_and_security_fields": "Monitoring only; Safe RL and grid reward are disabled in this stage.",
                "bess_action_physical": "Signed physical request after the existing action adapter mapping.",
                "bess_action_padded_dim0": "Bounded HARL actor output before the existing BESS action mapping.",
            },
            "levels": {},
        }
        for level, columns in (
            ("step", STEP_COLUMNS),
            ("episode", EPISODE_COLUMNS),
            ("update", UPDATE_COLUMNS),
        ):
            schema["levels"][level] = [
                {
                    "field": field,
                    "unit": self._unit(field),
                    "aggregation": self._aggregation(field, level),
                    "cumulative": self._is_cumulative(field),
                    "enters_reward": field in REWARD_COMPONENTS,
                    "alias_of": "r_peak_load" if field == "r_grid_peak" else None,
                    "source": reward_sources.get(field),
                    "missing_value_semantics": self._missing_semantics(field),
                }
                for field in columns
            ]
        with self.schema_path.open("x", encoding="utf-8") as stream:
            json.dump(schema, stream, ensure_ascii=False, indent=2, sort_keys=True)
            stream.write("\n")

    def _write_episode_tensorboard(self, row: Mapping[str, Any]) -> None:
        if not self.tensorboard_enabled or self.writer is None:
            return
        step = int(row["global_step_end"])
        tags = {
            "reward/episode_total": row["episode_reward"],
            "reward/step_mean": row["average_step_reward"],
            **{f"reward/{name}": row[f"{name}_sum"] for name in REWARD_COMPONENTS},
            "task/completion_rate": row["final_completion_rate"],
            "task/completed_work": row["completed_work_sum"],
            "task/final_backlog": row["final_backlog"],
            "task/deadline_miss": row["final_deadline_miss_total"],
            "task/sla_violation": row["final_sla_violation_count"],
            "energy/cost": row["cost_sum"],
            "energy/carbon_kg": row["carbon_kg_sum"],
            "energy/grid_kWh": row["grid_energy_kWh_sum"],
            "energy/peak_power_kW": row["max_grid_power_kW"],
            "energy/pv_used_kWh": row["pv_used_kWh_sum"],
            "bess/soc_final": row["final_bess_soc"],
            "bess/soc_mean": row["mean_bess_soc"],
            "bess/charge_kWh": row["bess_charge_kWh_sum"],
            "bess/discharge_kWh": row["bess_discharge_kWh_sum"],
            "bess/throughput_kWh": row["bess_throughput_kWh_sum"],
            "bess/invalid_request": row["max_bess_invalid_request_kW"],
            "bess/physical_action_mean": row["bess_physical_action_mean"],
            "grid/opf_success_rate": row["opf_success_rate"],
            "grid/mef_success_rate": row["mef_success_rate"],
            "grid/line_loading_max_pct": row["max_line_loading_pct"],
            "grid/lmp_mean": row["mean_lmp"],
            "grid/mef_mean": row["mean_mef_plus"],
            "grid/safe_violation_total": row["safe_violation_sum"],
        }
        if row["min_voltage_pu"] != "":
            tags["grid/voltage_min_pu"] = row["min_voltage_pu"]
        self._add_scalars(tags, step)

    def _write_update_tensorboard(self, row: Mapping[str, Any]) -> None:
        if not self.tensorboard_enabled or self.writer is None:
            return
        tags = {
            "train/value_loss": row["value_loss"],
            "train/critic_grad_norm": row["critic_grad_norm"],
            "train/idc_policy_loss": row["idc_policy_loss"],
            "train/bess_policy_loss": row["bess_policy_loss"],
            "train/idc_entropy": row["idc_entropy"],
            "train/bess_effective_entropy": row["bess_effective_entropy"],
            "train/idc_clip_fraction": row["idc_clip_fraction"],
            "train/bess_clip_fraction": row["bess_effective_clip_fraction"],
            "mask/bess_effective_ratio_diff": row["bess_effective_physical_ratio_max_diff"],
            "mask/bess_virtual_mean_grad": row["bess_virtual_mean_grad_norm"],
            "mask/bess_virtual_log_std_grad": row["bess_virtual_log_std_grad_norm"],
            "mask/bess_virtual_entropy_contribution": row["bess_virtual_entropy_optimization_contribution"],
        }
        self._add_scalars(tags, int(row["global_step"]))

    def _add_scalars(self, tags: Mapping[str, Any], step: int) -> None:
        for tag, value in tags.items():
            if value == "":
                continue
            number = self._metric_float(value, tag)
            self.writer.add_scalar(tag, number, step)

    @staticmethod
    def _require_diagnostics(actor: Any, name: str) -> Mapping[str, Any]:
        diagnostics = getattr(actor, "last_update_diagnostics", None)
        if not isinstance(diagnostics, Mapping):
            raise RuntimeError(f"{name} effective-action diagnostics are missing.")
        return diagnostics

    @classmethod
    def _diag_float(cls, diagnostics: Mapping[str, Any], key: str) -> float:
        if key not in diagnostics:
            raise KeyError(f"Effective-action diagnostics missing {key!r}.")
        return cls._metric_float(diagnostics[key], key)

    @staticmethod
    def _metric_float(value: Any, field: str) -> float:
        if hasattr(value, "detach"):
            value = value.detach().cpu().item()
        return TrainingMetricsLogger._required_float(value, field)

    @staticmethod
    def _required_float(value: Any, field: str) -> float:
        try:
            result = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Metric {field!r} must be numeric.") from exc
        if not math.isfinite(result):
            raise FloatingPointError(f"Metric {field!r} is NaN or inf.")
        return result

    @staticmethod
    def _required_int(value: Any, field: str) -> int:
        number = TrainingMetricsLogger._required_float(value, field)
        result = int(number)
        if number != result:
            raise ValueError(f"Metric {field!r} must be integral, got {number}.")
        return result

    @staticmethod
    def _optional_float(value: Any) -> float | str:
        try:
            result = float(value)
        except (TypeError, ValueError):
            return ""
        return result if math.isfinite(result) else ""

    @staticmethod
    def _shared_equal(left: Any, right: Any) -> bool:
        try:
            if math.isnan(float(left)) and math.isnan(float(right)):
                return True
        except (TypeError, ValueError):
            pass
        return bool(left == right)

    @staticmethod
    def _optional_max(rows: Sequence[Mapping[str, Any]], field: str) -> float | str:
        values = [float(row[field]) for row in rows if row[field] != ""]
        return max(values) if values else ""

    @staticmethod
    def _assert_finite_row(row: Mapping[str, Any], exceptions: Sequence[str]) -> None:
        excluded = set(exceptions)
        for key, value in row.items():
            if key in excluded or isinstance(value, (str, bool)):
                continue
            if not math.isfinite(float(value)):
                raise FloatingPointError(f"Metric {key!r} is NaN or inf.")

    def _flush_episode_files(self) -> None:
        for stream in (self._step_stream, self._episode_stream):
            if stream is not None:
                stream.flush()

    def _flush_all(self) -> None:
        for stream in (self._step_stream, self._episode_stream, self._update_stream):
            if stream is not None:
                stream.flush()

    @staticmethod
    def _unit(field: str) -> str:
        lower = field.lower()
        if lower.endswith("_kw") or "power_kw" in lower or "request_power_kw" in lower:
            return "kW"
        if lower.endswith("_kwh") or "energy_kwh" in lower or "throughput_kwh" in lower:
            return "kWh"
        if lower.endswith("_mw"):
            return "MW"
        if "carbon" in lower and ("kg" in lower or "emission" in lower):
            return "kgCO2e"
        if lower.endswith("_pu"):
            return "p.u."
        if lower.endswith("_pct"):
            return "%"
        if "time_seconds" in lower:
            return "s"
        if "rate" in lower or "fraction" in lower or "completion" in lower or lower == "pue":
            return "ratio"
        return "dimensionless"

    @staticmethod
    def _aggregation(field: str, level: str) -> str:
        if level == "step":
            return "raw"
        if field.startswith("final_") or field == "terminal_reward":
            return "final"
        if field.startswith("max_"):
            return "max"
        if field.startswith("min_"):
            return "min"
        if field.startswith("mean_") or field.endswith("_mean") or field.endswith("_rate") or field.endswith("_fraction"):
            return "mean/rate"
        if field.endswith("_sum") or field == "episode_reward":
            return "sum"
        return "update/raw"

    @staticmethod
    def _is_cumulative(field: str) -> bool:
        return field.endswith("_total") or field in {
            "completed_work_total", "cost_total", "carbon_total_kg",
            "opf_cache_hits_total", "opf_cache_misses_total",
            "mef_cache_hits_total", "mef_cache_misses_total",
        }

    @staticmethod
    def _missing_semantics(field: str) -> str:
        if field in {"ambient_temperature_C", "server_load_max"}:
            return "blank: current environment info does not expose this field"
        if field in {
            "lmp", "mef_plus", "mef_minus", "voltage_min_pu", "voltage_max_pu",
            "line_loading_max_pct", "grid_loss_mw", "min_voltage_pu",
            "max_voltage_pu", "mean_voltage_pu", "max_line_loading_pct",
            "mean_lmp", "mean_mef_plus",
        }:
            return "blank: solver value unavailable or non-finite; success flag remains authoritative"
        return "not allowed"


class CanonicalMetricBuilder:
    """Build canonical step/episode rows without opening training log files.

    Fixed evaluation uses this adapter so reward reconstruction and physical
    metric semantics stay byte-for-byte aligned with the Part-8 training
    logger instead of maintaining a second field mapping.
    """

    _extract_step_row = TrainingMetricsLogger._extract_step_row
    _aggregate_episode = TrainingMetricsLogger._aggregate_episode
    _required_float = staticmethod(TrainingMetricsLogger._required_float)
    _required_int = staticmethod(TrainingMetricsLogger._required_int)
    _optional_float = staticmethod(TrainingMetricsLogger._optional_float)
    _optional_max = staticmethod(TrainingMetricsLogger._optional_max)

    def __init__(
        self,
        *,
        run_id: str,
        seed: int,
        episode_length: int = 24,
        reward_tolerance: float = REWARD_RECONSTRUCTION_TOLERANCE,
    ) -> None:
        self.run_id = str(run_id)
        self.seed = int(seed)
        self.episode_length = int(episode_length)
        self.reward_tolerance = float(reward_tolerance)
        self._current_update = 0
        self._global_step = 0

    def build_step(
        self,
        *,
        info: Mapping[str, Any],
        actions: np.ndarray,
        episode_step: int,
        idc_reward: float,
        bess_reward: float,
        terminated: bool,
        truncated: bool,
    ) -> dict[str, Any]:
        self._global_step = int(episode_step) + 1
        shared_diff = abs(float(idc_reward) - float(bess_reward))
        return self._extract_step_row(
            info=info,
            actions=actions,
            worker_id=0,
            episode_id=0,
            episode_step=int(episode_step),
            idc_reward=float(idc_reward),
            bess_reward=float(bess_reward),
            shared_diff=shared_diff,
            terminated=bool(terminated),
            truncated=bool(truncated),
            is_terminal=bool(terminated or truncated),
        )

    def aggregate_episode(self, rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        return self._aggregate_episode(rows)


class CompositeTrainingLogger:
    """Delegate HARL logging while persisting project structured metrics."""

    def __init__(self, upstream: Any, metrics: TrainingMetricsLogger) -> None:
        self.upstream = upstream
        self.metrics = metrics

    def init(self, episodes: int) -> None:
        self.upstream.init(episodes)

    def episode_init(self, episode: int) -> None:
        self.metrics.begin_update(episode)
        self.upstream.episode_init(episode)

    def per_step(self, data: Sequence[Any]) -> None:
        self.metrics.record_step(data)
        self.upstream.per_step(data)

    def episode_log(self, *args, **kwargs) -> None:
        # HARL uses tensorboardX.add_scalars with slash-containing tags.  On
        # Windows that API creates nested writer directories; long resume-run
        # paths can exceed the filesystem limit before checkpointing.  Keep
        # the same scalar values and steps, but route this logging-only call
        # through the already-safe single-file add_scalar API.
        writer = getattr(self.upstream, "writter", None)
        original_add_scalars = getattr(writer, "add_scalars", None)
        if original_add_scalars is None:
            self.upstream.episode_log(*args, **kwargs)
            return

        def safe_add_scalars(main_tag, tag_scalar_dict, global_step=None, walltime=None):
            for scalar_tag, value in tag_scalar_dict.items():
                tag = main_tag if scalar_tag == main_tag else f"{main_tag}/{scalar_tag}"
                writer.add_scalar(tag, value, global_step, walltime=walltime)

        writer.add_scalars = safe_add_scalars
        try:
            self.upstream.episode_log(*args, **kwargs)
        finally:
            writer.add_scalars = original_add_scalars

    def __getattr__(self, name: str) -> Any:
        return getattr(self.upstream, name)

    def close(self) -> None:
        primary_error: BaseException | None = None
        try:
            self.upstream.close()
        except BaseException as exc:
            primary_error = exc
        try:
            self.metrics.close()
        except BaseException:
            if primary_error is None:
                raise
        if primary_error is not None:
            raise primary_error
