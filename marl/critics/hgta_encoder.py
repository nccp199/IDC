"""Pure-PyTorch heterogeneous relation attention and typed graph readout."""

from __future__ import annotations

from collections import defaultdict
import math
from typing import Callable

import torch
from torch import nn

from marl.graphs import HeteroGraphBatch, NODE_TYPE_ORDER, GraphSchema


def _activation(name: str) -> Callable[[torch.Tensor], torch.Tensor]:
    if name == "relu":
        return torch.relu
    if name == "gelu":
        return torch.nn.functional.gelu
    if name == "tanh":
        return torch.tanh
    raise ValueError(f"Unsupported HGTA activation {name!r}.")


class HeterogeneousRelationAttentionLayer(nn.Module):
    """One non-shared typed Q/K/V layer with relation-specific K/V transforms."""

    def __init__(
        self,
        schema: GraphSchema,
        *,
        hidden_dim: int,
        heads: int,
        dropout: float,
        activation: str,
    ) -> None:
        super().__init__()
        if hidden_dim % heads != 0:
            raise ValueError("hidden_dim must be divisible by heads.")
        self.schema = schema
        self.hidden_dim = int(hidden_dim)
        self.heads = int(heads)
        self.head_dim = self.hidden_dim // self.heads
        self.activation = _activation(activation)
        self.dropout = nn.Dropout(float(dropout))

        self.query = nn.ModuleDict(
            {name: nn.Linear(hidden_dim, hidden_dim, bias=False) for name in NODE_TYPE_ORDER}
        )
        self.key = nn.ModuleDict(
            {name: nn.Linear(hidden_dim, hidden_dim, bias=False) for name in NODE_TYPE_ORDER}
        )
        self.value = nn.ModuleDict(
            {name: nn.Linear(hidden_dim, hidden_dim, bias=False) for name in NODE_TYPE_ORDER}
        )
        self.output = nn.ModuleDict(
            {name: nn.Linear(hidden_dim, hidden_dim) for name in NODE_TYPE_ORDER}
        )
        self.layer_norm = nn.ModuleDict(
            {name: nn.LayerNorm(hidden_dim) for name in NODE_TYPE_ORDER}
        )
        self.relation_key = nn.ParameterDict()
        self.relation_value = nn.ParameterDict()
        self.relation_bias = nn.ParameterDict()
        for relation in schema.relations:
            key_matrix = torch.empty(self.heads, self.head_dim, self.head_dim)
            value_matrix = torch.empty(self.heads, self.head_dim, self.head_dim)
            nn.init.xavier_uniform_(key_matrix)
            nn.init.xavier_uniform_(value_matrix)
            self.relation_key[relation.key] = nn.Parameter(key_matrix)
            self.relation_value[relation.key] = nn.Parameter(value_matrix)
            self.relation_bias[relation.key] = nn.Parameter(torch.zeros(self.heads))

        self.last_attention_nonfinite_count = 0
        self.last_attention_means: dict[str, float] = {}

    def forward(self, hidden: dict[str, torch.Tensor], graph: HeteroGraphBatch) -> dict[str, torch.Tensor]:
        batch_size = graph.batch_size
        projected_query = {
            name: self.query[name](hidden[name]).reshape(
                batch_size, hidden[name].shape[1], self.heads, self.head_dim
            )
            for name in NODE_TYPE_ORDER
        }
        projected_key = {
            name: self.key[name](hidden[name]).reshape(
                batch_size, hidden[name].shape[1], self.heads, self.head_dim
            )
            for name in NODE_TYPE_ORDER
        }
        projected_value = {
            name: self.value[name](hidden[name]).reshape(
                batch_size, hidden[name].shape[1], self.heads, self.head_dim
            )
            for name in NODE_TYPE_ORDER
        }

        incoming: dict[str, list[tuple[str, torch.Tensor, torch.Tensor, torch.Tensor]]] = defaultdict(list)
        for relation in self.schema.relations:
            edges = graph.edge_index[relation.key]
            source_index, target_index = edges[0], edges[1]
            source_key = projected_key[relation.source_type][:, source_index]
            source_value = projected_value[relation.source_type][:, source_index]
            target_query = projected_query[relation.target_type][:, target_index]
            relation_key = torch.einsum(
                "behd,hdf->behf", source_key, self.relation_key[relation.key]
            )
            relation_value = torch.einsum(
                "behd,hdf->behf", source_value, self.relation_value[relation.key]
            )
            scores = (
                (target_query * relation_key).sum(dim=-1) / math.sqrt(self.head_dim)
                + self.relation_bias[relation.key][None, None, :]
            )
            if not torch.isfinite(scores).all() or not torch.isfinite(relation_value).all():
                self.last_attention_nonfinite_count = int(
                    (~torch.isfinite(scores)).sum().item()
                    + (~torch.isfinite(relation_value)).sum().item()
                )
                raise FloatingPointError(
                    f"HGTA relation {relation.key!r} produced NaN/inf attention inputs."
                )
            incoming[relation.target_type].append(
                (relation.key, target_index, scores, relation_value)
            )

        updated: dict[str, torch.Tensor] = {}
        attention_means: dict[str, list[torch.Tensor]] = defaultdict(list)
        self.last_attention_nonfinite_count = 0
        for target_type in NODE_TYPE_ORDER:
            target_count = int(self.schema.node_counts[target_type])
            groups = incoming.get(target_type, [])
            if groups:
                all_target = torch.cat([item[1] for item in groups], dim=0)
                all_scores = torch.cat([item[2] for item in groups], dim=1)
                all_values = torch.cat([item[3] for item in groups], dim=1)
                relation_ranges: list[tuple[str, int, int]] = []
                cursor = 0
                for key, _, scores, _ in groups:
                    relation_ranges.append((key, cursor, cursor + scores.shape[1]))
                    cursor += scores.shape[1]
                all_weights = torch.zeros_like(all_scores)
                node_messages = []
                for target_index in range(target_count):
                    mask = all_target == target_index
                    if bool(mask.any()):
                        weights = torch.softmax(all_scores[:, mask, :], dim=1)
                        if not torch.isfinite(weights).all():
                            self.last_attention_nonfinite_count += int(
                                (~torch.isfinite(weights)).sum().item()
                            )
                            raise FloatingPointError(
                                f"HGTA target {target_type}[{target_index}] attention is NaN/inf."
                            )
                        all_weights[:, mask, :] = weights
                        message = (weights.unsqueeze(-1) * all_values[:, mask]).sum(dim=1)
                    else:
                        message = torch.zeros(
                            batch_size,
                            self.heads,
                            self.head_dim,
                            dtype=hidden[target_type].dtype,
                            device=hidden[target_type].device,
                        )
                    node_messages.append(message)
                aggregate = torch.stack(node_messages, dim=1).reshape(
                    batch_size, target_count, self.hidden_dim
                )
                for key, start, stop in relation_ranges:
                    attention_means[key].append(all_weights[:, start:stop].mean().detach())
            else:
                aggregate = torch.zeros_like(hidden[target_type])

            message_output = self.dropout(self.activation(self.output[target_type](aggregate)))
            result = self.layer_norm[target_type](hidden[target_type] + message_output)
            if not torch.isfinite(result).all():
                raise FloatingPointError(
                    f"HGTA target type {target_type!r} embedding contains NaN/inf."
                )
            updated[target_type] = result

        self.last_attention_means = {
            key: float(torch.stack(values).mean().cpu().item())
            for key, values in attention_means.items()
        }
        return updated


class HGTAEncoder(nn.Module):
    """Two-layer typed relation encoder, type-wise readout, and forecast fusion."""

    def __init__(self, schema: GraphSchema) -> None:
        super().__init__()
        config = schema.architecture_config
        self.schema = schema
        self.hidden_dim = int(config["hidden_dim"])
        self.activation = _activation(str(config["activation"]))
        self.type_projection = nn.ModuleDict(
            {
                name: nn.Linear(len(schema.feature_names[name]), self.hidden_dim)
                for name in NODE_TYPE_ORDER
            }
        )
        self.relation_layers = nn.ModuleList(
            [
                HeterogeneousRelationAttentionLayer(
                    schema,
                    hidden_dim=self.hidden_dim,
                    heads=int(config["attention_heads"]),
                    dropout=float(config["dropout"]),
                    activation=str(config["activation"]),
                )
                for _ in range(int(config["attention_layers"]))
            ]
        )
        self.forecast_encoder = nn.Sequential(
            nn.Linear(144, int(config["forecast_hidden_dim"])),
            self._activation_module(str(config["activation"])),
            nn.Linear(
                int(config["forecast_hidden_dim"]),
                int(config["forecast_embedding_dim"]),
            ),
            self._activation_module(str(config["activation"])),
        )
        graph_summary_dim = len(NODE_TYPE_ORDER) * self.hidden_dim
        fused_dim = graph_summary_dim + int(config["forecast_embedding_dim"])
        if graph_summary_dim != 224 or fused_dim != 256:
            raise ValueError(
                f"HGTA graph v1 requires 224 graph + 32 forecast = 256, got "
                f"{graph_summary_dim} + {int(config['forecast_embedding_dim'])}."
            )
        self.value_head = nn.Sequential(
            nn.Linear(fused_dim, int(config["value_hidden_dim"])),
            self._activation_module(str(config["activation"])),
            nn.Linear(int(config["value_hidden_dim"]), 1),
        )
        self.last_diagnostics: dict[str, int] = {}
        self.last_type_summary_norms: dict[str, float] = {}
        self.last_attention_means: dict[str, float] = {}

    @staticmethod
    def _activation_module(name: str) -> nn.Module:
        return {"relu": nn.ReLU, "gelu": nn.GELU, "tanh": nn.Tanh}[name]()

    @staticmethod
    def _nonfinite_count(tensors: list[torch.Tensor]) -> int:
        return sum(int((~torch.isfinite(value)).sum().item()) for value in tensors)

    def forward(self, graph: HeteroGraphBatch) -> torch.Tensor:
        hidden = {
            name: self.activation(self.type_projection[name](graph.node_features[name]))
            for name in NODE_TYPE_ORDER
        }
        node_nonfinite = self._nonfinite_count(list(hidden.values()))
        if node_nonfinite:
            raise FloatingPointError(
                f"HGTA input projection produced {node_nonfinite} NaN/inf values."
            )
        attention_nonfinite = 0
        attention_means: dict[str, list[float]] = defaultdict(list)
        for layer in self.relation_layers:
            hidden = layer(hidden, graph)
            attention_nonfinite += int(layer.last_attention_nonfinite_count)
            for key, value in layer.last_attention_means.items():
                attention_means[key].append(value)

        summaries = []
        summary_norms = {}
        for name in NODE_TYPE_ORDER:
            if int(self.schema.node_counts[name]) == 1:
                summary = hidden[name][:, 0]
            else:
                summary = hidden[name].mean(dim=1)
            summaries.append(summary)
            summary_norms[name] = float(summary.detach().norm(dim=-1).mean().cpu().item())
        graph_embedding = torch.cat(summaries, dim=-1)
        forecast_embedding = self.forecast_encoder(graph.forecast_features)
        value = self.value_head(torch.cat((graph_embedding, forecast_embedding), dim=-1))
        graph_nonfinite = int((~torch.isfinite(graph_embedding)).sum().item())
        forecast_nonfinite = int((~torch.isfinite(forecast_embedding)).sum().item())
        value_nonfinite = int((~torch.isfinite(value)).sum().item())
        if graph_nonfinite or forecast_nonfinite or value_nonfinite:
            raise FloatingPointError(
                "HGTA readout produced NaN/inf: "
                f"graph={graph_nonfinite}, forecast={forecast_nonfinite}, value={value_nonfinite}."
            )
        self.last_diagnostics = {
            "graph_input_nonfinite_count": 0,
            "node_embedding_nonfinite_count": node_nonfinite,
            "attention_nonfinite_count": attention_nonfinite,
            "graph_embedding_nonfinite_count": graph_nonfinite,
            "forecast_embedding_nonfinite_count": forecast_nonfinite,
            "value_prediction_nonfinite_count": value_nonfinite,
        }
        self.last_type_summary_norms = summary_norms
        self.last_attention_means = {
            key: float(sum(values) / len(values)) for key, values in attention_means.items()
        }
        return value
