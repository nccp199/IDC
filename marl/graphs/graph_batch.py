"""Typed tensor batch returned by the fixed HGTA graph builder."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import torch

from .graph_schema import GraphSchema


@dataclass(frozen=True)
class HeteroGraphBatch:
    schema: GraphSchema
    node_features: Mapping[str, torch.Tensor]
    forecast_features: torch.Tensor
    edge_index: Mapping[str, torch.Tensor]
    batched_edge_index: Mapping[str, torch.Tensor]
    edge_batch_index: Mapping[str, torch.Tensor]
    batch_index: Mapping[str, torch.Tensor]
    current_hour: torch.Tensor

    @property
    def batch_size(self) -> int:
        return int(self.forecast_features.shape[0])

    @property
    def node_count_per_graph(self) -> int:
        return self.schema.node_count

    @property
    def edge_count_per_graph(self) -> int:
        return self.schema.edge_count

    @property
    def total_node_count(self) -> int:
        return self.batch_size * self.node_count_per_graph

    @property
    def total_edge_count(self) -> int:
        return self.batch_size * self.edge_count_per_graph
