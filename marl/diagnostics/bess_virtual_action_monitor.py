"""Failure-isolated CSV/TensorBoard monitoring for padded BESS actions."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import numpy as np


class BESSVirtualActionMonitor:
    """Record effective and virtual BESS action diagnostics without gradients.

    The monitor accepts only NumPy-compatible values and immediately converts
    them to plain floats. Logging failures are retained in ``last_error`` and do
    not propagate into environment execution.
    """

    BASE_FIELDS = (
        "global_step",
        "episode",
        "episode_step",
        "bess_effective_action",
        "bess_virtual_action_mean",
        "bess_virtual_action_std",
        "bess_virtual_action_abs_mean",
        "bess_virtual_action_abs_max",
        "bess_virtual_action_l2_norm",
        "bess_full_action_l2_norm",
        "bess_soc",
        "bess_charge_power",
        "bess_discharge_power",
        "reward",
        "price",
        "carbon_factor",
        "terminated",
        "truncated",
    )

    def __init__(
        self,
        output_dir: str | Path,
        *,
        enabled: bool = True,
        save_full_virtual_action_vector: bool = False,
        tensorboard_writer: Any | None = None,
        filename: str = "bess_virtual_action_diagnostics.csv",
    ) -> None:
        self.enabled = bool(enabled)
        self.save_full_virtual_action_vector = bool(save_full_virtual_action_vector)
        self.tensorboard_writer = tensorboard_writer
        self.last_error: str | None = None
        self.rows_written = 0
        self.path: Path | None = None
        self._file = None
        self._writer = None

        if self.enabled:
            try:
                output_path = Path(output_dir)
                output_path.mkdir(parents=True, exist_ok=True)
                self.path = self._non_overwriting_path(output_path / filename)
                self._file = self.path.open("x", encoding="utf-8", newline="")
                fields = list(self.BASE_FIELDS)
                if self.save_full_virtual_action_vector:
                    fields.extend(f"virtual_action_{index}" for index in range(1, 22))
                self._writer = csv.DictWriter(self._file, fieldnames=fields)
                self._writer.writeheader()
                self._file.flush()
            except Exception as exc:  # diagnostics must never break execution
                self.last_error = f"{type(exc).__name__}: {exc}"
                self.enabled = False
                self._safe_close_file()

    @staticmethod
    def _non_overwriting_path(path: Path) -> Path:
        if not path.exists():
            return path
        for index in range(1, 10_000):
            candidate = path.with_name(f"{path.stem}_{index}{path.suffix}")
            if not candidate.exists():
                return candidate
        raise FileExistsError(f"Could not allocate a unique diagnostics path below {path.parent}.")

    def record_step(
        self,
        *,
        global_step: int,
        episode: int,
        episode_step: int,
        bess_action: Any,
        reward: float,
        info: dict[str, Any],
        terminated: bool,
        truncated: bool,
    ) -> None:
        if not self.enabled or self._writer is None:
            return
        try:
            full = np.asarray(bess_action, dtype=np.float64)
            if full.shape != (22,) or not np.isfinite(full).all():
                raise ValueError("bess_action must be finite with shape (22,).")
            virtual = full[1:]
            row = {
                "global_step": int(global_step),
                "episode": int(episode),
                "episode_step": int(episode_step),
                "bess_effective_action": float(full[0]),
                "bess_virtual_action_mean": float(np.mean(virtual)),
                "bess_virtual_action_std": float(np.std(virtual)),
                "bess_virtual_action_abs_mean": float(np.mean(np.abs(virtual))),
                "bess_virtual_action_abs_max": float(np.max(np.abs(virtual))),
                "bess_virtual_action_l2_norm": float(np.linalg.norm(virtual)),
                "bess_full_action_l2_norm": float(np.linalg.norm(full)),
                "bess_soc": self._optional_float(info.get("bess_soc")),
                "bess_charge_power": self._optional_float(info.get("bess_charge_power_kW")),
                "bess_discharge_power": self._optional_float(info.get("bess_discharge_power_kW")),
                "reward": float(reward),
                "price": self._optional_float(info.get("price")),
                "carbon_factor": self._optional_float(info.get("carbon_factor")),
                "terminated": bool(terminated),
                "truncated": bool(truncated),
            }
            if self.save_full_virtual_action_vector:
                row.update(
                    {f"virtual_action_{index}": float(value) for index, value in enumerate(virtual, 1)}
                )
            self._writer.writerow(row)
            self._file.flush()
            self.rows_written += 1
            self._write_tensorboard(row, int(global_step))
        except Exception as exc:  # diagnostics must never break execution
            self.last_error = f"{type(exc).__name__}: {exc}"

    @staticmethod
    def _optional_float(value: Any) -> float | str:
        if value is None:
            return ""
        result = float(value)
        return result if np.isfinite(result) else ""

    def _write_tensorboard(self, row: dict[str, Any], step: int) -> None:
        if self.tensorboard_writer is None:
            return
        scalars = {
            "diagnostics/bess_effective_action": row["bess_effective_action"],
            "diagnostics/bess_virtual_abs_mean": row["bess_virtual_action_abs_mean"],
            "diagnostics/bess_virtual_abs_max": row["bess_virtual_action_abs_max"],
            "diagnostics/bess_virtual_l2": row["bess_virtual_action_l2_norm"],
            "diagnostics/bess_full_action_l2": row["bess_full_action_l2_norm"],
        }
        for name, value in scalars.items():
            try:
                self.tensorboard_writer.add_scalar(name, float(value), step)
            except Exception as exc:
                self.last_error = f"{type(exc).__name__}: {exc}"

    def record_policy_diagnostics(self, metrics: dict[str, Any], global_step: int) -> None:
        """Write detached per-update metrics without changing policy computation."""
        if not self.enabled or self.tensorboard_writer is None:
            return
        for name, value in metrics.items():
            try:
                scalar = float(value)
                if np.isfinite(scalar):
                    self.tensorboard_writer.add_scalar(f"diagnostics/bess_{name}", scalar, global_step)
            except Exception as exc:
                self.last_error = f"{type(exc).__name__}: {exc}"

    def close(self) -> None:
        self._safe_close_file()

    def _safe_close_file(self) -> None:
        if self._file is None:
            return
        try:
            self._file.flush()
            self._file.close()
        except Exception as exc:
            self.last_error = f"{type(exc).__name__}: {exc}"
        finally:
            self._file = None
            self._writer = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        self.close()
