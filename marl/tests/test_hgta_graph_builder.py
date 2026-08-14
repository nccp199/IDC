from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from marl.graphs import HGTAGraphBuilder, NODE_TYPE_ORDER, build_graph_schema


def test_state_slices_and_server_feature_major_reorder(hgta_builder, synthetic_state):
    graph = hgta_builder.build(synthetic_state)
    assert graph.batch_size == 1
    assert torch.equal(graph.node_features["task_pool"][0, 0], synthetic_state[6:16])
    expected_server = torch.stack(
        [synthetic_state[start : start + 20] for start in (16, 36, 56, 76, 96, 116)],
        dim=-1,
    )
    assert torch.equal(graph.node_features["server_group"][0], expected_server)
    assert torch.allclose(
        graph.node_features["idc"][0, 0], torch.tensor([0.06, 1200.0 / 27000.0])
    )
    assert torch.allclose(
        graph.node_features["bess"][0, 0], torch.tensor([0.5, 0.5, 0.25, 0.125])
    )
    assert torch.equal(graph.forecast_features[0], synthetic_state[136:280])
    assert torch.equal(graph.node_features["bus"][0, :, 7:], synthetic_state[294:364].reshape(14, 5))


def test_current_pv_hour_recovery_is_exact_for_all_24_hours(hgta_builder, synthetic_state):
    states = synthetic_state.repeat(24, 1)
    hours = torch.arange(24, dtype=torch.float32)
    states[:, 4] = torch.sin(2 * torch.pi * hours / 24)
    states[:, 5] = torch.cos(2 * torch.pi * hours / 24)
    for sample in range(24):
        states[sample, 208:232] = torch.arange(24, dtype=torch.float32) + 100 * sample
    graph = hgta_builder.build(states)
    assert torch.equal(graph.current_hour.cpu(), torch.arange(24))
    assert torch.equal(
        graph.node_features["pv"][:, 0, 0].cpu(),
        torch.arange(24, dtype=torch.float32) * 101,
    )


@pytest.mark.parametrize("batch_size", (1, 2, 48))
def test_fixed_graph_batch_is_isolated(hgta_builder, synthetic_state, batch_size):
    states = synthetic_state.repeat(batch_size, 1)
    graph = hgta_builder.build(states)
    assert graph.node_count_per_graph == 39
    assert graph.edge_count_per_graph == 90
    assert graph.total_node_count == 39 * batch_size
    assert graph.total_edge_count == 90 * batch_size
    for relation in graph.schema.relations:
        key = relation.key
        src_count = graph.schema.node_counts[relation.source_type]
        dst_count = graph.schema.node_counts[relation.target_type]
        edges = graph.batched_edge_index[key]
        edge_batch = graph.edge_batch_index[key]
        assert torch.equal(edges[0] // src_count, edge_batch)
        assert torch.equal(edges[1] // dst_count, edge_batch)

    if batch_size > 1:
        changed = states.clone()
        changed[0, 16] += 10.0
        changed_graph = hgta_builder.build(changed)
        assert not torch.equal(
            graph.node_features["server_group"][0],
            changed_graph.node_features["server_group"][0],
        )
        assert torch.equal(
            graph.node_features["server_group"][1:],
            changed_graph.node_features["server_group"][1:],
        )


def test_frozen_topology_relations_and_hashes(hgta_schema, hgta_config):
    assert hgta_schema.node_type_order == NODE_TYPE_ORDER
    assert hgta_schema.node_count == 39
    assert hgta_schema.edge_count == 90
    assert hgta_schema.node_counts["server_group"] == 20
    assert hgta_schema.node_counts["bus"] == 14
    lengths = {relation.key: len(relation.edges) for relation in hgta_schema.relations}
    assert lengths["bus__line_connected__bus"] == 30
    assert lengths["bus__transformer_connected__bus"] == 10
    assert lengths["server_group__belongs_to__idc"] == 20
    assert lengths["idc__contains__server_group"] == 20
    for key in (
        "idc__attached_to__bus",
        "bess__attached_to__bus",
        "pv__attached_to__bus",
    ):
        relation = next(item for item in hgta_schema.relations if item.key == key)
        assert relation.edges == ((0, 8),)
    assert all(len(value) == 64 for value in (
        hgta_schema.feature_schema_hash,
        hgta_schema.topology_hash,
        hgta_schema.graph_schema_hash,
    ))
    rebuilt = build_graph_schema(hgta_config)
    assert rebuilt.metadata() == hgta_schema.metadata()


def test_shape_and_nonfinite_fail_fast(hgta_builder, synthetic_state):
    with pytest.raises(ValueError, match=r"\[B,364\]"):
        hgta_builder.build(torch.zeros(363))
    for value in (float("nan"), float("inf"), float("-inf")):
        bad = synthetic_state.clone()
        bad[50] = value
        with pytest.raises(FloatingPointError, match="NaN/inf"):
            hgta_builder.build(bad)


def test_information_sensitivity_reaches_declared_features(hgta_builder, hgta_schema, synthetic_state):
    base = hgta_builder.build(synthetic_state)
    changes = {
        "server_group": (16, "server_group"),
        "bess": (288, "bess"),
        "current_price": (1, "global"),
        "future_price": (136, "forecast"),
    }
    for _, (index, target) in changes.items():
        changed = synthetic_state.clone()
        changed[index] += 0.25
        other = hgta_builder.build(changed)
        if target == "forecast":
            assert not torch.equal(base.forecast_features, other.forecast_features)
        else:
            assert not torch.equal(base.node_features[target], other.node_features[target])

    modified_bus = [list(row) for row in hgta_schema.bus_features]
    modified_bus[3][1] += 0.125
    alternate_schema = replace(
        hgta_schema, bus_features=tuple(tuple(row) for row in modified_bus)
    )
    alternate = HGTAGraphBuilder(alternate_schema).build(synthetic_state)
    assert not torch.equal(base.node_features["bus"], alternate.node_features["bus"])


def test_dynamic_bus_values_map_only_to_the_corresponding_bus(
    hgta_builder, synthetic_state
):
    state = synthetic_state.clone()
    dynamic = torch.arange(70, dtype=torch.float32).reshape(14, 5)
    state[294:364] = dynamic.reshape(-1)
    graph = hgta_builder.build(state)
    assert graph.node_features["bus"].shape == (1, 14, 12)
    assert torch.equal(graph.node_features["bus"][0, :, 7:], dynamic)
