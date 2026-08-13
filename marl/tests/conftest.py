from __future__ import annotations

import pytest
import torch

from marl.graphs import HGTAGraphBuilder, build_graph_schema


@pytest.fixture(scope="session")
def hgta_config() -> dict:
    return {
        "graph_schema_version": "hgta_graph_v2_dynamic_bus",
        "graph_builder_version": "hgta_builder_v2_dynamic_bus",
        "architecture_version": "hgta_critic_v1",
        "hidden_dim": 32,
        "attention_heads": 4,
        "attention_layers": 2,
        "dropout": 0.0,
        "residual": True,
        "layer_norm": True,
        "activation": "relu",
        "forecast_hidden_dim": 64,
        "forecast_embedding_dim": 32,
        "value_hidden_dim": 64,
        "idc_power_ref_kw": 2000.0,
        "grid_power_ref_kw": 4000.0,
        "bess_capacity_kwh": 10000.0,
        "bess_charge_power_ref_kw": 2000.0,
        "bess_discharge_power_ref_kw": 2000.0,
    }


@pytest.fixture(scope="session")
def hgta_schema(hgta_config):
    return build_graph_schema(hgta_config)


@pytest.fixture(scope="session")
def hgta_builder(hgta_schema):
    return HGTAGraphBuilder(hgta_schema)


@pytest.fixture
def synthetic_state():
    state = torch.zeros(364, dtype=torch.float32)
    state[0:6] = torch.tensor([0.6, 0.4, 0.2, 0.5, 0.0, 1.0])
    state[6:16] = torch.linspace(0.1, 1.0, 10)
    for offset, start in enumerate((16, 36, 56, 76, 96, 116), start=1):
        state[start : start + 20] = torch.arange(20, dtype=torch.float32) / 100 + offset
    state[136:280] = torch.linspace(-1.0, 1.0, 144)
    state[208:232] = torch.linspace(0.0, 1.0, 24)
    state[280:288] = torch.linspace(0.1, 0.8, 8)
    state[288:294] = torch.tensor([0.5, 5000.0, 1500.0, 1200.0, 500.0, 250.0])
    state[294:364] = torch.arange(70, dtype=torch.float32) / 100.0
    return state
