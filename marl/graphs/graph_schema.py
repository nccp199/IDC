"""Versioned candidate-B graph schema derived from the fixed IEEE-14 case."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Any, Mapping

import numpy as np


GRAPH_SCHEMA_VERSION = "hgta_graph_v3_normalized_supplemental"
GRAPH_BUILDER_VERSION = "hgta_builder_v3_normalized_supplemental"
HGTA_ARCHITECTURE_VERSION = "hgta_critic_v1"

NODE_TYPE_ORDER = (
    "global",
    "idc",
    "task_pool",
    "server_group",
    "bess",
    "pv",
    "bus",
)

NODE_COUNTS = {
    "global": 1,
    "idc": 1,
    "task_pool": 1,
    "server_group": 20,
    "bess": 1,
    "pv": 1,
    "bus": 14,
}

NODE_FEATURE_NAMES = {
    "global": (
        "current_temperature",
        "current_price",
        "current_task_arrival",
        "current_time_sin",
        "current_time_cos",
        "grid_lmp",
        "grid_mef_plus",
        "grid_mef_minus",
        "grid_min_voltage",
        "grid_max_line_loading",
        "grid_network_loss",
        "grid_security_penalty",
        "grid_opf_success",
    ),
    "idc": ("p_idc", "p_grid"),
    "task_pool": (
        "waiting_count",
        "running_count",
        "finished_count",
        "unfinished_count",
        "urgent_work",
        "overdue_work",
        "average_deadline_left",
        "average_priority",
        "parallelizable_work",
        "interruptible_work",
    ),
    "server_group": (
        "load",
        "capacity",
        "efficiency",
        "unit_cost",
        "available",
        "estimated_temperature",
    ),
    "bess": (
        "soc",
        "energy_fraction",
        "charge_power_fraction",
        "discharge_power_fraction",
    ),
    "pv": ("current_forecast", "forecast_mean", "forecast_max"),
    "bus": (
        "nominal_voltage",
        "base_active_load",
        "base_reactive_load",
        "has_generator",
        "has_external_grid",
        "has_idc",
        "bus_index",
        "current_voltage_deviation",
        "current_net_active_power",
        "current_net_reactive_power",
        "current_nodal_lmp",
        "current_incident_branch_max_loading",
    ),
}


def _canonical_hash(value: Any) -> str:
    encoded = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class RelationSpec:
    """One ordered heterogeneous relation with type-local directed edges."""

    source_type: str
    name: str
    target_type: str
    edges: tuple[tuple[int, int], ...]

    @property
    def key(self) -> str:
        return f"{self.source_type}__{self.name}__{self.target_type}"

    def as_dict(self) -> dict[str, Any]:
        return {
            "source_type": self.source_type,
            "name": self.name,
            "target_type": self.target_type,
            "edges": [list(edge) for edge in self.edges],
        }


@dataclass(frozen=True)
class GraphSchema:
    """Canonical graph structure, features, normalization, and fingerprints."""

    node_type_order: tuple[str, ...]
    node_counts: Mapping[str, int]
    feature_names: Mapping[str, tuple[str, ...]]
    relations: tuple[RelationSpec, ...]
    bus_features: tuple[tuple[float, ...], ...]
    line_edges: tuple[tuple[int, int], ...]
    transformer_edges: tuple[tuple[int, int], ...]
    normalization_references: Mapping[str, float]
    architecture_config: Mapping[str, Any]
    feature_schema_hash: str
    topology_hash: str
    graph_schema_hash: str
    graph_schema_version: str = GRAPH_SCHEMA_VERSION
    graph_builder_version: str = GRAPH_BUILDER_VERSION
    hgta_architecture_version: str = HGTA_ARCHITECTURE_VERSION

    @property
    def relation_type_order(self) -> tuple[str, ...]:
        return tuple(relation.key for relation in self.relations)

    @property
    def node_count(self) -> int:
        return sum(int(self.node_counts[name]) for name in self.node_type_order)

    @property
    def edge_count(self) -> int:
        return sum(len(relation.edges) for relation in self.relations)

    @property
    def feature_dims(self) -> dict[str, int]:
        return {name: len(self.feature_names[name]) for name in self.node_type_order}

    def metadata(self) -> dict[str, Any]:
        return {
            "graph_schema_version": self.graph_schema_version,
            "graph_builder_version": self.graph_builder_version,
            "hgta_architecture_version": self.hgta_architecture_version,
            "node_type_order": list(self.node_type_order),
            "relation_type_order": list(self.relation_type_order),
            "feature_names": {
                name: list(self.feature_names[name]) for name in self.node_type_order
            },
            "feature_dims": self.feature_dims,
            "node_counts": {
                name: int(self.node_counts[name]) for name in self.node_type_order
            },
            "node_count": self.node_count,
            "edge_count": self.edge_count,
            "line_edges": [list(edge) for edge in self.line_edges],
            "transformer_edges": [list(edge) for edge in self.transformer_edges],
            "normalization_references": dict(self.normalization_references),
            "architecture_config": dict(self.architecture_config),
            "feature_schema_hash": self.feature_schema_hash,
            "topology_hash": self.topology_hash,
            "graph_schema_hash": self.graph_schema_hash,
            "pooling": {
                "global": "identity",
                "idc": "identity",
                "task_pool": "identity",
                "server_group": "mean",
                "bess": "identity",
                "pv": "identity",
                "bus": "mean",
            },
            "forecast_encoder": [144, 64, 32],
            "value_head": [256, 64, 1],
        }


def _required_positive(config: Mapping[str, Any], key: str) -> float:
    try:
        value = float(config[key])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"critic.hgta.{key} must be a positive number.") from exc
    if not np.isfinite(value) or value <= 0.0:
        raise ValueError(f"critic.hgta.{key} must be a positive finite number.")
    return value


def _architecture_config(config: Mapping[str, Any]) -> dict[str, Any]:
    required_ints = (
        "hidden_dim",
        "attention_heads",
        "attention_layers",
        "forecast_hidden_dim",
        "forecast_embedding_dim",
        "value_hidden_dim",
    )
    result: dict[str, Any] = {}
    for key in required_ints:
        value = config.get(key)
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise ValueError(f"critic.hgta.{key} must be a positive integer.")
        result[key] = int(value)
    if result["hidden_dim"] % result["attention_heads"] != 0:
        raise ValueError("critic.hgta.hidden_dim must be divisible by attention_heads.")
    if result["attention_layers"] != 2:
        raise ValueError("HGTA graph v1 requires exactly two attention layers.")
    dropout = float(config.get("dropout", 0.0))
    if not np.isfinite(dropout) or not 0.0 <= dropout < 1.0:
        raise ValueError("critic.hgta.dropout must be in [0, 1).")
    result.update(
        {
            "dropout": dropout,
            "residual": bool(config.get("residual", True)),
            "layer_norm": bool(config.get("layer_norm", True)),
            "activation": str(config.get("activation", "relu")).lower(),
        }
    )
    if not result["residual"] or not result["layer_norm"]:
        raise ValueError("HGTA graph v1 requires residual and layer_norm enabled.")
    if result["activation"] not in {"relu", "gelu", "tanh"}:
        raise ValueError("critic.hgta.activation must be relu, gelu, or tanh.")
    return result


def build_graph_schema(config: Mapping[str, Any]) -> GraphSchema:
    """Build and fingerprint the frozen 39-node/90-edge candidate-B graph."""

    from grid_model.ieee14_loader import load_ieee14_case

    architecture = _architecture_config(config)
    normalization = {
        "idc_power_ref_kw": _required_positive(config, "idc_power_ref_kw"),
        "grid_power_ref_kw": _required_positive(config, "grid_power_ref_kw"),
        "bess_capacity_kwh": _required_positive(config, "bess_capacity_kwh"),
        "bess_charge_power_ref_kw": _required_positive(
            config, "bess_charge_power_ref_kw"
        ),
        "bess_discharge_power_ref_kw": _required_positive(
            config, "bess_discharge_power_ref_kw"
        ),
        "bus_voltage_center_pu": 1.0,
        "bus_voltage_deviation_ref_pu": 0.10,
        "bus_active_power_ref_mw": 100.0,
        "bus_reactive_power_ref_mvar": 100.0,
        "bus_lmp_ref": 100.0,
        "bus_branch_loading_ref_percent": 100.0,
    }

    grid_case = load_ieee14_case()
    net = grid_case.raw_network
    bus_ids = tuple(int(index) for index in net.bus.index.tolist())
    if bus_ids != tuple(range(14)):
        raise ValueError(f"HGTA graph v1 requires canonical IEEE-14 bus ids 0..13, got {bus_ids}.")
    line_edges = tuple(
        (int(row.from_bus), int(row.to_bus)) for _, row in net.line.iterrows()
    )
    transformer_edges = tuple(
        (int(row.hv_bus), int(row.lv_bus)) for _, row in net.trafo.iterrows()
    )
    if len(line_edges) != 15 or len(transformer_edges) != 5:
        raise ValueError("HGTA graph v1 requires 15 IEEE-14 lines and 5 transformers.")

    active_load = np.zeros(14, dtype=np.float64)
    reactive_load = np.zeros(14, dtype=np.float64)
    for _, row in net.load.iterrows():
        bus = int(row.bus)
        active_load[bus] += float(row.p_mw)
        reactive_load[bus] += float(row.q_mvar)
    nominal_voltage = np.asarray(net.bus.vn_kv, dtype=np.float64)
    voltage_ref = max(float(np.max(np.abs(nominal_voltage))), 1e-9)
    active_ref = max(float(np.max(np.abs(active_load))), 1e-9)
    reactive_ref = max(float(np.max(np.abs(reactive_load))), 1e-9)
    normalization.update(
        {
            "bus_nominal_voltage_ref_kv": voltage_ref,
            "bus_active_load_ref_mw": active_ref,
            "bus_reactive_load_ref_mvar": reactive_ref,
            "bus_index_ref": 13.0,
        }
    )
    generator_buses = {int(bus) for bus in net.gen.bus.tolist()}
    external_grid_buses = {int(bus) for bus in net.ext_grid.bus.tolist()}
    bus_features = tuple(
        (
            float(nominal_voltage[index] / voltage_ref),
            float(active_load[index] / active_ref),
            float(reactive_load[index] / reactive_ref),
            1.0 if index in generator_buses else 0.0,
            1.0 if index in external_grid_buses else 0.0,
            1.0 if index == 8 else 0.0,
            float(index / 13.0),
        )
        for index in range(14)
    )

    line_directed = tuple(
        directed
        for left, right in line_edges
        for directed in ((left, right), (right, left))
    )
    transformer_directed = tuple(
        directed
        for left, right in transformer_edges
        for directed in ((left, right), (right, left))
    )
    relations = (
        RelationSpec("server_group", "belongs_to", "idc", tuple((i, 0) for i in range(20))),
        RelationSpec("idc", "contains", "server_group", tuple((0, i) for i in range(20))),
        RelationSpec("task_pool", "queued_at", "idc", ((0, 0),)),
        RelationSpec("idc", "manages", "task_pool", ((0, 0),)),
        RelationSpec("idc", "attached_to", "bus", ((0, 8),)),
        RelationSpec("bus", "hosts_idc", "idc", ((8, 0),)),
        RelationSpec("bess", "attached_to", "bus", ((0, 8),)),
        RelationSpec("bus", "hosts_bess", "bess", ((8, 0),)),
        RelationSpec("pv", "attached_to", "bus", ((0, 8),)),
        RelationSpec("bus", "hosts_pv", "pv", ((8, 0),)),
        RelationSpec("global", "provides_context", "idc", ((0, 0),)),
        RelationSpec("idc", "reports_state", "global", ((0, 0),)),
        RelationSpec("bus", "line_connected", "bus", line_directed),
        RelationSpec("bus", "transformer_connected", "bus", transformer_directed),
    )

    feature_payload = {
        "node_type_order": list(NODE_TYPE_ORDER),
        "node_counts": NODE_COUNTS,
        "feature_names": {name: list(NODE_FEATURE_NAMES[name]) for name in NODE_TYPE_ORDER},
    }
    topology_payload = {
        "bus_ids": list(bus_ids),
        "line_edges": [list(edge) for edge in line_edges],
        "transformer_edges": [list(edge) for edge in transformer_edges],
        "bus_features": [list(values) for values in bus_features],
        "normalization_references": normalization,
    }
    relation_payload = [relation.as_dict() for relation in relations]
    feature_hash = _canonical_hash(feature_payload)
    topology_hash = _canonical_hash(topology_payload)
    graph_hash = _canonical_hash(
        {
            "versions": [
                GRAPH_SCHEMA_VERSION,
                GRAPH_BUILDER_VERSION,
                HGTA_ARCHITECTURE_VERSION,
            ],
            "features": feature_payload,
            "topology": topology_payload,
            "relations": relation_payload,
            "architecture": architecture,
        }
    )
    schema = GraphSchema(
        node_type_order=NODE_TYPE_ORDER,
        node_counts=dict(NODE_COUNTS),
        feature_names=dict(NODE_FEATURE_NAMES),
        relations=relations,
        bus_features=bus_features,
        line_edges=line_edges,
        transformer_edges=transformer_edges,
        normalization_references=normalization,
        architecture_config=architecture,
        feature_schema_hash=feature_hash,
        topology_hash=topology_hash,
        graph_schema_hash=graph_hash,
    )
    if schema.node_count != 39 or schema.edge_count != 90:
        raise RuntimeError(
            f"Frozen HGTA graph must have 39 nodes and 90 edges, got "
            f"{schema.node_count} and {schema.edge_count}."
        )
    return schema
