"""Read-only exactness audit for a continuous update-5 run and update-3 resume."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch


ALLOWED_FIELDS = {
    "run_id",
    "rollout_time_seconds",
    "update_time_seconds",
    "wall_time_seconds",
    "steps_per_second",
}


def _load(path: Path) -> dict[str, Any]:
    return torch.load(path, map_location="cpu", weights_only=False)


def _rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def _differences(left: Any, right: Any, path: str = "") -> list[str]:
    if torch.is_tensor(left) and torch.is_tensor(right):
        return [] if left.dtype == right.dtype and left.shape == right.shape and torch.equal(left, right) else [path]
    if isinstance(left, np.ndarray) and isinstance(right, np.ndarray):
        return [] if left.dtype == right.dtype and left.shape == right.shape and np.array_equal(left, right) else [path]
    if isinstance(left, Mapping) and isinstance(right, Mapping):
        result = []
        if set(left) != set(right):
            result.append(f"{path}.keys")
        for key in sorted(set(left).intersection(right)):
            result.extend(_differences(left[key], right[key], f"{path}.{key}" if path else str(key)))
        return result
    if isinstance(left, (list, tuple)) and isinstance(right, (list, tuple)):
        result = [] if len(left) == len(right) else [f"{path}.length"]
        for index, (a, b) in enumerate(zip(left, right)):
            result.extend(_differences(a, b, f"{path}[{index}]"))
        return result
    if isinstance(left, float) and isinstance(right, float) and math.isnan(left) and math.isnan(right):
        return []
    return [] if left == right else [path]


def _csv_projection(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    return [
        {key: value for key, value in row.items() if key not in ALLOWED_FIELDS}
        for row in rows
    ]


def _numeric_max(left: list[dict[str, str]], right: list[dict[str, str]]) -> float:
    maximum = 0.0
    for left_row, right_row in zip(left, right, strict=True):
        for key in left_row:
            if key in ALLOWED_FIELDS:
                continue
            try:
                a = float(left_row[key])
                b = float(right_row[key])
            except (TypeError, ValueError):
                continue
            if math.isnan(a) and math.isnan(b):
                continue
            maximum = max(maximum, abs(a - b))
    return maximum


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("continuous", type=Path)
    parser.add_argument("resumed", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    continuous = args.continuous.resolve()
    resumed = args.resumed.resolve()
    left = _load(continuous / "checkpoints" / "final.pt")
    right = _load(resumed / "checkpoints" / "final.pt")
    exact_sections = (
        "model_state",
        "optimizer_state",
        "normalizer_state",
        "rng_state",
        "environment_state",
        "runner_state",
        "graph_metadata",
        "compatibility",
    )
    checkpoint_sections = {
        section: _differences(left[section], right[section], section)
        for section in exact_sections
    }
    verification_differences = _differences(
        left["verification_state"], right["verification_state"], "verification_state"
    )
    unexpected_verification = [
        path for path in verification_differences if path.rsplit(".", 1)[-1] not in ALLOWED_FIELDS
    ]

    csv_specs = {
        "step_last96": ("step_metrics.csv", 96),
        "episode_last4": ("episode_metrics.csv", 4),
        "update_4_5": ("update_metrics.csv", 2),
    }
    csv_audit = {}
    for name, (filename, count) in csv_specs.items():
        left_rows = _rows(continuous / "metrics" / filename)[-count:]
        right_rows = _rows(resumed / "metrics" / filename)
        left_projection = _csv_projection(left_rows)
        right_projection = _csv_projection(right_rows)
        differences = _differences(left_projection, right_projection, name)
        csv_audit[name] = {
            "expected_rows": count,
            "continuous_rows": len(left_rows),
            "resumed_rows": len(right_rows),
            "differences": differences,
            "difference_count": len(differences),
            "numeric_max_abs": (
                _numeric_max(left_rows, right_rows)
                if len(left_rows) == len(right_rows)
                else float("inf")
            ),
        }

    result = {
        "schema_version": "part18-rng-fix-resume-exactness-v1",
        "continuous_run": str(continuous),
        "resumed_run": str(resumed),
        "checkpoint_sections": checkpoint_sections,
        "verification_differences": verification_differences,
        "unexpected_verification_differences": unexpected_verification,
        **csv_audit,
    }
    result["exact_resume_passed"] = bool(
        all(not value for value in checkpoint_sections.values())
        and not unexpected_verification
        and all(
            not csv_audit[name]["differences"]
            and csv_audit[name]["numeric_max_abs"] == 0.0
            for name in csv_specs
        )
    )
    encoded = json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(encoded, encoding="utf-8")
    print(encoded, end="")
    if not result["exact_resume_passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
