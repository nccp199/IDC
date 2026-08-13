"""Stable fixed-evaluation CSV/JSON schemas built from Part-8 metrics."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from marl.logging import EPISODE_COLUMNS, STEP_COLUMNS


EVALUATION_SCHEMA_VERSION = "idc-bess-fixed-evaluation-v2"

STEP_ID_COLUMNS = (
    "evaluation_id", "suite_id", "suite_hash", "model_id", "model_source",
    "model_sha256", "training_seed", "training_update", "scenario_id",
    "scenario_seed", "idc_action_mean", "idc_action_std", "idc_action_min",
    "idc_action_max",
)
_DROP_STEP = {"run_id", "seed", "update", "global_step", "worker_id", "episode_id"}
EVALUATION_STEP_COLUMNS = STEP_ID_COLUMNS + tuple(
    column for column in STEP_COLUMNS if column not in _DROP_STEP
)

EPISODE_ID_COLUMNS = (
    "evaluation_id", "suite_id", "suite_hash", "model_id", "model_source",
    "model_sha256", "training_seed", "training_update", "scenario_id",
    "scenario_seed", "status", "failure_type", "failure_message",
    "scenario_load_seconds", "runtime_seconds",
)
_DROP_EPISODE = {"run_id", "seed", "update", "worker_id", "episode_id"}
EVALUATION_EPISODE_COLUMNS = EPISODE_ID_COLUMNS + tuple(
    column for column in EPISODE_COLUMNS if column not in _DROP_EPISODE
)

AGGREGATE_COLUMNS = (
    "evaluation_id", "suite_id", "suite_hash", "suite_type", "model_id",
    "model_source", "model_sha256", "metric", "mean", "sample_std",
    "median", "min", "max", "scenario_count", "successful_scenarios",
    "failed_scenarios", "interpretation",
)

AGGREGATE_METRICS = (
    "episode_reward", "final_completion_rate", "final_task_completion_rate",
    "completed_work_sum", "finished_tasks_sum", "final_backlog",
    "final_deadline_miss_total", "final_sla_violation_count", "cost_sum",
    "carbon_kg_sum", "grid_energy_kWh_sum", "max_grid_power_kW", "mean_pue",
    "final_bess_soc", "bess_charge_kWh_sum", "bess_discharge_kWh_sum",
    "bess_throughput_kWh_sum", "max_bess_invalid_request_kW",
    "opf_success_rate", "mef_success_rate", "min_voltage_pu",
    "max_line_loading_pct", "safe_violation_sum", "runtime_seconds",
)


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


class EvaluationMetricsWriter:
    def __init__(self, output_dir: str | Path) -> None:
        self.output_dir = Path(output_dir).resolve()
        self.output_dir.mkdir(parents=True, exist_ok=False)
        self.step_path = self.output_dir / "step_metrics.csv"
        self.episode_path = self.output_dir / "episode_metrics.csv"
        self.aggregate_csv_path = self.output_dir / "aggregate_metrics.csv"
        self.aggregate_json_path = self.output_dir / "aggregate_metrics.json"
        self.step_stream = self.step_path.open("x", encoding="utf-8", newline="")
        self.episode_stream = self.episode_path.open("x", encoding="utf-8", newline="")
        self.step_writer = csv.DictWriter(self.step_stream, fieldnames=EVALUATION_STEP_COLUMNS, extrasaction="raise")
        self.episode_writer = csv.DictWriter(self.episode_stream, fieldnames=EVALUATION_EPISODE_COLUMNS, extrasaction="raise")
        self.step_writer.writeheader()
        self.episode_writer.writeheader()
        self.episode_rows: list[dict[str, Any]] = []

    @staticmethod
    def _identity(
        *, evaluation_id: str, suite: Mapping[str, Any], model: Any, scenario: Mapping[str, Any]
    ) -> dict[str, Any]:
        return {
            "evaluation_id": evaluation_id,
            "suite_id": suite["suite_id"],
            "suite_hash": suite["suite_sha256"],
            "model_id": model.model_id,
            "model_source": model.source_description,
            "model_sha256": model.model_sha256,
            "training_seed": "" if model.training_seed is None else model.training_seed,
            "training_update": "" if model.training_update is None else model.training_update,
            "scenario_id": scenario["scenario_id"],
            "scenario_seed": scenario["scenario_seed"],
        }

    def write_success(
        self,
        *,
        evaluation_id: str,
        suite: Mapping[str, Any],
        model: Any,
        scenario: Mapping[str, Any],
        steps: Sequence[Mapping[str, Any]],
        episode: Mapping[str, Any],
        scenario_load_seconds: float,
        runtime_seconds: float,
    ) -> dict[str, Any]:
        identity = self._identity(evaluation_id=evaluation_id, suite=suite, model=model, scenario=scenario)
        for canonical in steps:
            idc_action = np.asarray(canonical.pop("_idc_action"), dtype=np.float64)
            row = {
                **identity,
                "idc_action_mean": float(idc_action.mean()),
                "idc_action_std": float(idc_action.std()),
                "idc_action_min": float(idc_action.min()),
                "idc_action_max": float(idc_action.max()),
                **{key: value for key, value in canonical.items() if key not in _DROP_STEP},
            }
            if tuple(row) != EVALUATION_STEP_COLUMNS:
                raise AssertionError("Evaluation step schema order changed.")
            self.step_writer.writerow(row)
        episode_row = {
            **identity,
            "status": "completed",
            "failure_type": "",
            "failure_message": "",
            "scenario_load_seconds": float(scenario_load_seconds),
            "runtime_seconds": float(runtime_seconds),
            **{key: value for key, value in episode.items() if key not in _DROP_EPISODE},
        }
        if tuple(episode_row) != EVALUATION_EPISODE_COLUMNS:
            raise AssertionError("Evaluation episode schema order changed.")
        self.episode_writer.writerow(episode_row)
        self.step_stream.flush()
        self.episode_stream.flush()
        self.episode_rows.append(episode_row)
        return episode_row

    def write_failure(
        self,
        *,
        evaluation_id: str,
        suite: Mapping[str, Any],
        model: Any,
        scenario: Mapping[str, Any],
        error: BaseException,
        scenario_load_seconds: float,
        runtime_seconds: float,
    ) -> dict[str, Any]:
        row = {
            **self._identity(evaluation_id=evaluation_id, suite=suite, model=model, scenario=scenario),
            "status": "failed",
            "failure_type": type(error).__name__,
            "failure_message": str(error),
            "scenario_load_seconds": float(scenario_load_seconds),
            "runtime_seconds": float(runtime_seconds),
            **{column: "" for column in EVALUATION_EPISODE_COLUMNS if column not in EPISODE_ID_COLUMNS},
        }
        if tuple(row) != EVALUATION_EPISODE_COLUMNS:
            raise AssertionError("Evaluation failure schema order changed.")
        self.episode_writer.writerow(row)
        self.episode_stream.flush()
        self.episode_rows.append(row)
        return row

    def finalize(
        self,
        *,
        evaluation_id: str,
        suite: Mapping[str, Any],
        models: Sequence[Any],
    ) -> list[dict[str, Any]]:
        rows = []
        nested: dict[str, Any] = {
            "evaluation_schema_version": EVALUATION_SCHEMA_VERSION,
            "evaluation_id": evaluation_id,
            "suite_id": suite["suite_id"],
            "suite_hash": suite["suite_sha256"],
            "suite_type": suite["suite_type"],
            "interpretation": "engineering smoke only; not statistically sufficient" if suite["suite_type"] == "smoke" else suite["suite_type"],
            "models": {},
        }
        for model in models:
            model_episodes = [row for row in self.episode_rows if row["model_id"] == model.model_id]
            successful = [row for row in model_episodes if row["status"] == "completed"]
            failed = len(model_episodes) - len(successful)
            model_json = {"successful_scenarios": len(successful), "failed_scenarios": failed, "metrics": {}}
            for metric in AGGREGATE_METRICS:
                values = np.asarray([float(row[metric]) for row in successful if row[metric] != ""], dtype=np.float64)
                if values.size and not np.isfinite(values).all():
                    raise FloatingPointError(f"Aggregate metric {metric} contains NaN/inf.")
                stats = {
                    "mean": float(values.mean()) if values.size else "",
                    "sample_std": float(values.std(ddof=1)) if values.size > 1 else 0.0 if values.size else "",
                    "median": float(np.median(values)) if values.size else "",
                    "min": float(values.min()) if values.size else "",
                    "max": float(values.max()) if values.size else "",
                }
                aggregate = {
                    "evaluation_id": evaluation_id,
                    "suite_id": suite["suite_id"],
                    "suite_hash": suite["suite_sha256"],
                    "suite_type": suite["suite_type"],
                    "model_id": model.model_id,
                    "model_source": model.source_description,
                    "model_sha256": model.model_sha256,
                    "metric": metric,
                    **stats,
                    "scenario_count": len(model_episodes),
                    "successful_scenarios": len(successful),
                    "failed_scenarios": failed,
                    "interpretation": nested["interpretation"],
                }
                rows.append(aggregate)
                model_json["metrics"][metric] = stats
            nested["models"][model.model_id] = model_json
        with self.aggregate_csv_path.open("x", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=AGGREGATE_COLUMNS, extrasaction="raise")
            writer.writeheader()
            writer.writerows(rows)
        _write_json(self.aggregate_json_path, nested)
        self.close()
        return rows

    def close(self) -> None:
        if not self.step_stream.closed:
            self.step_stream.close()
        if not self.episode_stream.closed:
            self.episode_stream.close()
