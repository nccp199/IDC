"""Read-only exactness audit for two fresh same-method, same-seed runs."""

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


def _rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def _project_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    return [{key: value for key, value in row.items() if key not in ALLOWED_FIELDS} for row in rows]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("left", type=Path)
    parser.add_argument("right", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    left_dir = args.left.resolve()
    right_dir = args.right.resolve()
    left = torch.load(left_dir / "checkpoints" / "final.pt", map_location="cpu", weights_only=False)
    right = torch.load(right_dir / "checkpoints" / "final.pt", map_location="cpu", weights_only=False)
    if left["method_id"] != right["method_id"]:
        raise ValueError("Repeated-run audit requires the same method_id.")

    sections = (
        "model_state", "optimizer_state", "normalizer_state", "rng_state",
        "environment_state", "runner_state", "graph_metadata", "compatibility",
    )
    section_differences = {
        section: _differences(left[section], right[section], section) for section in sections
    }
    verification = _differences(
        left["verification_state"], right["verification_state"], "verification_state"
    )
    unexpected_verification = [
        path for path in verification if path.rsplit(".", 1)[-1] not in ALLOWED_FIELDS
    ]
    csv_differences = {}
    for filename in ("step_metrics.csv", "episode_metrics.csv", "update_metrics.csv"):
        left_rows = _rows(left_dir / "metrics" / filename)
        right_rows = _rows(right_dir / "metrics" / filename)
        csv_differences[filename] = {
            "left_rows": len(left_rows),
            "right_rows": len(right_rows),
            "differences": _differences(
                _project_rows(left_rows), _project_rows(right_rows), filename
            ),
        }
    result = {
        "schema_version": "part18-rng-fix-repeated-run-exactness-v1",
        "method_id": left["method_id"],
        "left_run": str(left_dir),
        "right_run": str(right_dir),
        "checkpoint_section_differences": section_differences,
        "verification_differences": verification,
        "unexpected_verification_differences": unexpected_verification,
        "csv": csv_differences,
    }
    result["repeated_run_exactness_passed"] = bool(
        all(not value for value in section_differences.values())
        and not unexpected_verification
        and all(not value["differences"] for value in csv_differences.values())
    )
    encoded = json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(encoded, encoding="utf-8")
    print(encoded, end="")
    if not result["repeated_run_exactness_passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
