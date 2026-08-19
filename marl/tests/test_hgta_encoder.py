from __future__ import annotations

import copy

import pytest
import torch

from marl.critics.hgta_encoder import HGTAEncoder


@pytest.mark.parametrize("batch_size", (1, 2, 48))
def test_forward_shapes_finite_and_all_type_summaries_nonzero(
    hgta_schema, hgta_builder, synthetic_state, batch_size
):
    network = HGTAEncoder(hgta_schema)
    graph = hgta_builder.build(synthetic_state.repeat(batch_size, 1))
    output = network(graph)
    assert output.shape == (batch_size, 1)
    assert torch.isfinite(output).all()
    assert network.last_diagnostics == {
        "graph_input_nonfinite_count": 0,
        "node_embedding_nonfinite_count": 0,
        "attention_nonfinite_count": 0,
        "graph_embedding_nonfinite_count": 0,
        "forecast_embedding_nonfinite_count": 0,
        "value_prediction_nonfinite_count": 0,
    }
    assert set(network.last_type_summary_norms) == set(hgta_schema.node_type_order)
    assert all(value > 0.0 for value in network.last_type_summary_norms.values())
    assert set(network.last_attention_means) == set(hgta_schema.relation_type_order)
    assert all(torch.isfinite(torch.tensor(value)) for value in network.last_attention_means.values())


def test_train_eval_and_seeded_output_are_reproducible(hgta_schema, hgta_builder, synthetic_state):
    graph = hgta_builder.build(synthetic_state.repeat(2, 1))
    torch.manual_seed(7110)
    first = HGTAEncoder(hgta_schema)
    first.train()
    train_value = first(graph)
    first.eval()
    eval_value = first(graph)
    assert torch.equal(train_value, eval_value)

    torch.manual_seed(7110)
    second = HGTAEncoder(hgta_schema)
    second.eval()
    assert torch.equal(eval_value, second(graph))


def test_backward_reaches_all_major_modules_and_changes_relation_parameters(
    hgta_schema, hgta_builder, synthetic_state
):
    torch.manual_seed(7110)
    network = HGTAEncoder(hgta_schema)
    graph = hgta_builder.build(synthetic_state.repeat(2, 1))
    before = {
        name: parameter.detach().clone()
        for name, parameter in network.named_parameters()
        if "relation_" in name
    }
    optimizer = torch.optim.Adam(network.parameters(), lr=5e-4)
    target = torch.tensor([[0.5], [-0.25]])
    loss = (network(graph) - target).square().mean()
    optimizer.zero_grad()
    loss.backward()

    for name, module in network.type_projection.items():
        assert module.weight.grad is not None, name
        assert torch.isfinite(module.weight.grad).all(), name
    for layer_index, layer in enumerate(network.relation_layers):
        gradients = [parameter.grad for parameter in layer.parameters() if parameter.grad is not None]
        assert gradients, layer_index
        assert all(torch.isfinite(gradient).all() for gradient in gradients)
        assert sum(float(gradient.abs().sum()) for gradient in gradients) > 0.0
    for module_name, module in (
        ("forecast", network.forecast_encoder),
        ("value_head", network.value_head),
    ):
        gradients = [parameter.grad for parameter in module.parameters() if parameter.grad is not None]
        assert gradients, module_name
        assert all(torch.isfinite(gradient).all() for gradient in gradients)
        assert sum(float(gradient.abs().sum()) for gradient in gradients) > 0.0

    optimizer.step()
    changed = [
        not torch.equal(before[name], parameter.detach())
        for name, parameter in network.named_parameters()
        if name in before
    ]
    assert changed and any(changed)


def test_sample_isolation_survives_message_passing(hgta_schema, hgta_builder, synthetic_state):
    torch.manual_seed(7110)
    network = HGTAEncoder(hgta_schema).eval()
    states = synthetic_state.repeat(2, 1)
    baseline = network(hgta_builder.build(states)).detach()
    changed = states.clone()
    changed[0, 16] += 5.0
    result = network(hgta_builder.build(changed)).detach()
    assert not torch.equal(baseline[0], result[0])
    assert torch.equal(baseline[1], result[1])
