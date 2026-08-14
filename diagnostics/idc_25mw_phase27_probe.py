"""Phase 2.7 formal 25 MW environment and normalized-feature probe."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from configs.config_ultimate import IDC_SCALE_CONFIG  # noqa: E402
from marl import IDCGridMultiAgentEnv  # noqa: E402
from marl.specs import SUPPLEMENTAL_FIELDS  # noqa: E402
from marl.tests.helpers import make_grid_env  # noqa: E402


SEED = 2026


def _stats(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "min": float(array.min()),
        "mean": float(array.mean()),
        "max": float(array.max()),
    }


def _run_case(name: str, server_action: float, bess_schedule: list[float]) -> dict[str, Any]:
    grid_env = make_grid_env(SEED)
    env = IDCGridMultiAgentEnv(grid_env)
    rows: list[dict[str, Any]] = []
    try:
        obs, state, reset_info = env.reset(seed=SEED)
        for hour in range(24):
            idc_action = np.full(22, 0.5, dtype=np.float32)
            idc_action[:20] = server_action
            action = {
                "idc": idc_action,
                "bess": np.asarray([bess_schedule[hour]], dtype=np.float32),
            }
            obs, state, _, terminated, truncated, info = env.step(action)
            normalized = np.asarray(
                info["supplemental_features_normalized"], dtype=np.float64
            )
            rows.append(
                {
                    "hour": hour,
                    "normalized": {
                        field: float(value)
                        for field, value in zip(
                            SUPPLEMENTAL_FIELDS, normalized, strict=True
                        )
                    },
                    "p_idc_mw": float(info["P_IDC_kW"]) / 1000.0,
                    "p_grid_mw": float(info["P_grid_kW"]) / 1000.0,
                    "opf_success": bool(info["grid_opf_success"]),
                    "state_dim": int(state.shape[0]),
                    "idc_obs_dim": int(obs["idc"].shape[0]),
                    "bess_obs_dim": int(obs["bess"].shape[0]),
                }
            )
            if terminated["__all__"] or truncated["__all__"]:
                break
        if len(rows) != 24:
            raise RuntimeError(f"{name}: expected 24 steps, got {len(rows)}")
        feature_stats = {
            field: _stats([row["normalized"][field] for row in rows])
            for field in SUPPLEMENTAL_FIELDS
        }
        return {
            "case": name,
            "server_action": server_action,
            "references": dict(env.supplemental_feature_references),
            "input_semantics_version": env.input_semantics_version,
            "supplemental_normalization_version": env.supplemental_normalization_version,
            "feature_stats": feature_stats,
            "p_idc_mw": _stats([row["p_idc_mw"] for row in rows]),
            "p_grid_mw": _stats([row["p_grid_mw"] for row in rows]),
            "opf_success_count": sum(row["opf_success"] for row in rows),
            "failure_hours": [row["hour"] for row in rows if not row["opf_success"]],
            "dimensions": {
                "idc_actor_observation": rows[0]["idc_obs_dim"],
                "bess_local_observation": rows[0]["bess_obs_dim"],
                "centralized_state": rows[0]["state_dim"],
                "flat_action": 23,
            },
            "hours": rows,
        }
    finally:
        grid_env.close()


def run(output: Path) -> dict[str, Any]:
    normal_schedule = [0.0] * 3 + [1.0] * 3 + [0.5] * 18
    idle_schedule = [0.5] * 24
    payload = {
        "seed": SEED,
        "formal_scale": dict(IDC_SCALE_CONFIG),
        "normal_mixed_bess": _run_case("normal_mixed_bess", 0.5, normal_schedule),
        "high_idle_bess": _run_case("high_idle_bess", 1.0, idle_schedule),
    }
    def check(value: Any) -> None:
        if isinstance(value, dict):
            for item in value.values():
                check(item)
        elif isinstance(value, list):
            for item in value:
                check(item)
        elif isinstance(value, float) and not math.isfinite(value):
            raise ValueError("Probe payload contains NaN or infinity.")
    check(payload)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/idc_25mw_phase27/probe.json"),
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    result = run(args.output)
    print(
        json.dumps(
            {
                name: {
                    "opf_success": result[name]["opf_success_count"],
                    "p_idc_mw": result[name]["p_idc_mw"],
                }
                for name in ("normal_mixed_bess", "high_idle_bess")
            },
            indent=2,
        )
    )
