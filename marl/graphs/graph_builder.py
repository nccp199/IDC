"""Rebuild the frozen candidate-B heterogeneous graph from state[364]."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from .graph_batch import HeteroGraphBatch
from .graph_schema import GraphSchema
from marl.specs.state_specs import (
    CENTRALIZED_STATE_DIM,
    GRID_BUS_COUNT,
    GRID_BUS_DYNAMIC_FEATURE_DIM,
    GRID_BUS_DYNAMIC_STATE_SLICE,
)


class HGTAGraphBuilder:
    """Deterministic, stateless state slicing and fixed-graph batching."""

    input_dim = CENTRALIZED_STATE_DIM

    def __init__(self, schema: GraphSchema) -> None:
        if schema.node_count != 39 or schema.edge_count != 90:
            raise ValueError("HGTAGraphBuilder requires the frozen 39-node/90-edge schema.")
        self.schema = schema

    @staticmethod
    def _as_state_tensor(state: Any) -> torch.Tensor:
        if torch.is_tensor(state):
            tensor = state.to(dtype=torch.float32)
        else:
            tensor = torch.as_tensor(np.asarray(state), dtype=torch.float32)
        if tensor.ndim == 1:
            tensor = tensor.unsqueeze(0)
        if tensor.ndim != 2 or tensor.shape[1] != HGTAGraphBuilder.input_dim:
            raise ValueError(
                "HGTA graph builder expects state "
                f"[B,{HGTAGraphBuilder.input_dim}], got {tuple(tensor.shape)}."
            )
        if not torch.isfinite(tensor).all():
            count = int((~torch.isfinite(tensor)).sum().item())
            raise FloatingPointError(f"HGTA state contains {count} NaN/inf values.")
        return tensor

    @staticmethod
    def _current_hour(state: torch.Tensor) -> torch.Tensor:
        hours = torch.arange(24, dtype=state.dtype, device=state.device)
        angles = 2.0 * torch.pi * hours / 24.0
        standard = torch.stack((torch.sin(angles), torch.cos(angles)), dim=-1)
        current = state[:, (4, 5)]
        squared_distance = ((current[:, None, :] - standard[None, :, :]) ** 2).sum(-1)
        return torch.argmin(squared_distance, dim=1)

    def build(self, state: Any) -> HeteroGraphBatch:
        values = self._as_state_tensor(state)
        batch_size = int(values.shape[0])
        device = values.device
        refs = self.schema.normalization_references

        server_group = torch.stack(
            tuple(values[:, start : start + 20] for start in (16, 36, 56, 76, 96, 116)),
            dim=-1,
        )
        task_pool = values[:, 6:16].unsqueeze(1)
        idc = torch.stack(
            (
                values[:, 290] / float(refs["idc_power_ref_kw"]),
                values[:, 291] / float(refs["grid_power_ref_kw"]),
            ),
            dim=-1,
        ).unsqueeze(1)
        bess = torch.stack(
            (
                values[:, 288],
                values[:, 289] / float(refs["bess_capacity_kwh"]),
                values[:, 292] / float(refs["bess_charge_power_ref_kw"]),
                values[:, 293] / float(refs["bess_discharge_power_ref_kw"]),
            ),
            dim=-1,
        ).unsqueeze(1)
        pv_forecast = values[:, 208:232]
        current_hour = self._current_hour(values)
        current_pv = pv_forecast.gather(1, current_hour[:, None]).squeeze(1)
        pv = torch.stack(
            (current_pv, pv_forecast.mean(dim=1), pv_forecast.max(dim=1).values),
            dim=-1,
        ).unsqueeze(1)
        global_features = torch.cat(
            (values[:, (0, 1, 2, 4, 5)], values[:, 280:288]), dim=-1
        ).unsqueeze(1)
        bus_template = torch.tensor(
            self.schema.bus_features, dtype=values.dtype, device=device
        )
        dynamic_bus = values[:, GRID_BUS_DYNAMIC_STATE_SLICE].reshape(
            batch_size, GRID_BUS_COUNT, GRID_BUS_DYNAMIC_FEATURE_DIM
        )
        bus = torch.cat(
            (
                bus_template.unsqueeze(0).expand(batch_size, -1, -1),
                dynamic_bus,
            ),
            dim=-1,
        )
        forecast = values[:, 136:280]

        extracted = {
            "global": global_features,
            "idc": idc,
            "task_pool": task_pool,
            "server_group": server_group,
            "bess": bess,
            "pv": pv,
            "bus": bus,
        }
        node_features = {name: extracted[name] for name in self.schema.node_type_order}
        for name in self.schema.node_type_order:
            tensor = node_features[name]
            expected = (
                batch_size,
                int(self.schema.node_counts[name]),
                len(self.schema.feature_names[name]),
            )
            if tuple(tensor.shape) != expected:
                raise RuntimeError(
                    f"HGTA node feature {name!r} has shape {tuple(tensor.shape)}, expected {expected}."
                )
            if not torch.isfinite(tensor).all():
                raise FloatingPointError(f"HGTA node feature {name!r} contains NaN/inf.")

        edge_index: dict[str, torch.Tensor] = {}
        batched_edge_index: dict[str, torch.Tensor] = {}
        edge_batch_index: dict[str, torch.Tensor] = {}
        for relation in self.schema.relations:
            base = torch.tensor(relation.edges, dtype=torch.long, device=device).t().contiguous()
            edge_index[relation.key] = base
            source_count = int(self.schema.node_counts[relation.source_type])
            target_count = int(self.schema.node_counts[relation.target_type])
            copies = []
            for sample in range(batch_size):
                offset = torch.tensor(
                    [[sample * source_count], [sample * target_count]],
                    dtype=torch.long,
                    device=device,
                )
                copies.append(base + offset)
            batched_edge_index[relation.key] = torch.cat(copies, dim=1)
            edge_batch_index[relation.key] = torch.arange(
                batch_size, dtype=torch.long, device=device
            ).repeat_interleave(base.shape[1])

        batch_index = {
            name: torch.arange(batch_size, dtype=torch.long, device=device).repeat_interleave(
                int(self.schema.node_counts[name])
            )
            for name in self.schema.node_type_order
        }
        return HeteroGraphBatch(
            schema=self.schema,
            node_features=node_features,
            forecast_features=forecast,
            edge_index=edge_index,
            batched_edge_index=batched_edge_index,
            edge_batch_index=edge_batch_index,
            batch_index=batch_index,
            current_hour=current_hour,
        )
