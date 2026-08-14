from __future__ import annotations

import csv
import json
import os
from pathlib import Path

import pytest
import torch


def _run_dir() -> Path:
    value = os.environ.get("PART16_HGTA_RUN_DIR")
    if not value:
        pytest.skip("Set PART16_HGTA_RUN_DIR after the unique formal one-update run.")
    path = Path(value).resolve()
    if not path.is_dir():
        raise AssertionError(f"PART16_HGTA_RUN_DIR does not exist: {path}")
    return path


def _rows(path: Path):
    with path.open("r", encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def test_formal_hgta_one_update_artifacts():
    run = _run_dir()
    summary = json.loads((run / "metrics" / "run_summary.json").read_text(encoding="utf-8"))
    metadata = json.loads((run / "run_metadata.json").read_text(encoding="utf-8"))
    steps = _rows(run / "metrics" / "step_metrics.csv")
    episodes = _rows(run / "metrics" / "episode_metrics.csv")
    updates = _rows(run / "metrics" / "update_metrics.csv")
    assert summary["status"] == "completed"
    assert summary["total_environment_steps"] == 48
    assert summary["updates_completed"] == 1
    assert len(steps) == 48
    assert len(episodes) == 2
    assert len(updates) == 1
    assert metadata["critic_type"] == "hgta"
    assert metadata["method_id"] == "HAPPO_HGTA"
    row = updates[0]
    assert row["hgta_node_count"] == "39"
    assert row["hgta_edge_count"] == "90"
    assert float(row["critic_update_count"]) == 1.0
    assert float(row["actor_update_count_idc"]) == 1.0
    assert float(row["actor_update_count_bess"]) == 1.0
    assert float(row["happo_factor_nonfinite_count"]) == 0.0
    assert float(row["happo_factor_nonpositive_count"]) == 0.0
    assert float(row["bess_happo_virtual_factor_contribution"]) == 0.0
    assert float(row["bess_effective_physical_ratio_max_diff"]) == 0.0
    assert float(row["shared_reward_max_diff"]) == 0.0
    assert float(row["opf_success_rate"]) == 1.0
    assert float(row["mef_success_rate"]) == 1.0
    for field in (
        "graph_input_nonfinite_count",
        "node_embedding_nonfinite_count",
        "attention_nonfinite_count",
        "graph_embedding_nonfinite_count",
        "forecast_embedding_nonfinite_count",
        "value_prediction_nonfinite_count",
    ):
        assert float(row[field]) == 0.0

    checkpoint = torch.load(
        run / "checkpoints" / "final.pt", map_location="cpu", weights_only=False
    )
    graph = checkpoint["graph_metadata"]
    assert checkpoint["critic_type"] == "hgta"
    assert checkpoint["method_id"] == "HAPPO_HGTA"
    assert graph["graph_schema_version"] == "hgta_graph_v3_normalized_supplemental"
    assert graph["node_count"] == 39
    assert graph["edge_count"] == 90
    assert len(graph["feature_schema_hash"]) == 64
    assert len(graph["topology_hash"]) == 64
    assert len(graph["graph_schema_hash"]) == 64
