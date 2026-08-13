"""HARL-compatible centralized value network backed by the fixed HGTA graph."""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np
import torch
from torch import nn

from marl.graphs import HGTAGraphBuilder, build_graph_schema
from marl.specs.state_specs import CENTRALIZED_STATE_DIM

from .hgta_encoder import HGTAEncoder


class HGTAValueNetwork(nn.Module):
    """Drop-in replacement for HARL VNet under the verified feed-forward config."""

    def __init__(self, args: Mapping[str, Any], hgta_config: Mapping[str, Any]) -> None:
        super().__init__()
        if bool(args.get("use_recurrent_policy", False)) or bool(
            args.get("use_naive_recurrent_policy", False)
        ):
            raise NotImplementedError(
                "HGTA critic v1 supports feed-forward value estimation only; "
                "recurrent masks/states cannot be silently ignored."
            )
        self.schema = build_graph_schema(hgta_config)
        self.graph_builder = HGTAGraphBuilder(self.schema)
        self.encoder = HGTAEncoder(self.schema)

    @property
    def diagnostics(self) -> dict[str, int]:
        return dict(self.encoder.last_diagnostics)

    @property
    def graph_metadata(self) -> dict[str, Any]:
        return self.schema.metadata()

    def forward(self, cent_obs: Any, rnn_states: Any, masks: Any):
        parameter = next(self.parameters())
        if torch.is_tensor(cent_obs):
            state = cent_obs.to(device=parameter.device, dtype=torch.float32)
        else:
            state = torch.as_tensor(
                np.asarray(cent_obs), dtype=torch.float32, device=parameter.device
            )
        graph = self.graph_builder.build(state)
        values = self.encoder(graph)
        if rnn_states is None:
            next_rnn_states = None
        elif torch.is_tensor(rnn_states):
            next_rnn_states = rnn_states.to(device=parameter.device, dtype=torch.float32)
        else:
            next_rnn_states = torch.as_tensor(
                np.asarray(rnn_states), dtype=torch.float32, device=parameter.device
            )
        if next_rnn_states is not None and not torch.isfinite(next_rnn_states).all():
            raise FloatingPointError("HGTA critic received non-finite recurrent state.")
        if masks is not None:
            mask_tensor = (
                masks.to(device=parameter.device, dtype=torch.float32)
                if torch.is_tensor(masks)
                else torch.as_tensor(np.asarray(masks), dtype=torch.float32, device=parameter.device)
            )
            if not torch.isfinite(mask_tensor).all():
                raise FloatingPointError("HGTA critic received non-finite masks.")
        return values, next_rnn_states


class HGTACentralizedCritic:
    """Stable project adapter around HARL's unchanged VCritic update machinery."""

    critic_type = "hgta"
    input_kind = "centralized_state_tensor_to_fixed_heterograph"
    input_dim = CENTRALIZED_STATE_DIM

    def __init__(self, implementation: Any, network: HGTAValueNetwork, interface_version: str) -> None:
        self.implementation = implementation
        self.network = network
        self.interface_version = interface_version

    def __getattr__(self, name: str) -> Any:
        return getattr(self.implementation, name)

    @property
    def graph_metadata(self) -> dict[str, Any]:
        metadata = self.network.graph_metadata
        metadata.update(
            {
                "hgta_hidden_dim": int(self.network.schema.architecture_config["hidden_dim"]),
                "hgta_heads": int(self.network.schema.architecture_config["attention_heads"]),
                "hgta_layers": int(self.network.schema.architecture_config["attention_layers"]),
                "hgta_node_count": int(self.network.schema.node_count),
                "hgta_edge_count": int(self.network.schema.edge_count),
                "hgta_parameter_count": sum(
                    parameter.numel() for parameter in self.network.parameters()
                ),
            }
        )
        return metadata

    def forward(self, share_obs: Any, rnn_states_critic: Any, masks: Any) -> Any:
        return self.get_values(share_obs, rnn_states_critic, masks)

    def get_values(self, share_obs: Any, rnn_states_critic: Any, masks: Any) -> Any:
        if getattr(share_obs, "shape", (None,))[-1] != self.input_dim:
            raise ValueError(
                "HGTA critic expects centralized state shape "
                f"[batch, {self.input_dim}]."
            )
        return self.implementation.get_values(share_obs, rnn_states_critic, masks)

    def evaluate(self, share_obs: Any, rnn_states_critic: Any, masks: Any) -> Any:
        return self.get_values(share_obs, rnn_states_critic, masks)

    def train(self, critic_buffer: Any, value_normalizer: Any = None) -> dict[str, Any]:
        result = dict(self.implementation.train(critic_buffer, value_normalizer))
        result.update(self.network.diagnostics)
        return result

    update = train

    def state_dict(self):
        return self.network.state_dict()

    def load_state_dict(self, state: Mapping[str, Any], *, strict: bool = True):
        return self.network.load_state_dict(dict(state), strict=strict)

    def optimizer_state(self):
        return self.implementation.critic_optimizer.state_dict()

    def to(self, device: torch.device | str) -> "HGTACentralizedCritic":
        self.network.to(device)
        return self

    def train_mode(self) -> None:
        self.implementation.prep_training()

    def eval_mode(self) -> None:
        self.implementation.prep_rollout()
