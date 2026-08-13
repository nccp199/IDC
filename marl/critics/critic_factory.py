"""Stable centralized-critic boundary for MLP now and heterogeneous graphs later."""

from __future__ import annotations

from typing import Any, Mapping

import torch

from marl.methods import CRITIC_INTERFACE_VERSION
from marl.specs.state_specs import CENTRALIZED_STATE_DIM


class MLPCentralizedCritic:
    """Transparent adapter around HARL VCritic with one shared state contract."""

    critic_type = "mlp"
    interface_version = CRITIC_INTERFACE_VERSION
    input_kind = "centralized_state_tensor"
    input_dim = CENTRALIZED_STATE_DIM

    def __init__(self, implementation: Any) -> None:
        self.implementation = implementation

    def __getattr__(self, name: str) -> Any:
        return getattr(self.implementation, name)

    def forward(self, share_obs: Any, rnn_states_critic: Any, masks: Any) -> Any:
        return self.get_values(share_obs, rnn_states_critic, masks)

    def get_values(self, share_obs: Any, rnn_states_critic: Any, masks: Any) -> Any:
        if getattr(share_obs, "shape", (None,))[-1] != self.input_dim:
            raise ValueError(
                f"MLP critic expects centralized state shape [batch, {self.input_dim}]."
            )
        return self.implementation.get_values(share_obs, rnn_states_critic, masks)

    def evaluate(self, share_obs: Any, rnn_states_critic: Any, masks: Any) -> Any:
        return self.get_values(share_obs, rnn_states_critic, masks)

    def train(self, critic_buffer: Any, value_normalizer: Any = None) -> Any:
        return self.implementation.train(critic_buffer, value_normalizer)

    update = train

    def state_dict(self) -> Mapping[str, Any]:
        return self.implementation.critic.state_dict()

    def load_state_dict(self, state: Mapping[str, Any], *, strict: bool = True) -> Any:
        return self.implementation.critic.load_state_dict(dict(state), strict=strict)

    def optimizer_state(self) -> Mapping[str, Any]:
        return self.implementation.critic_optimizer.state_dict()

    def to(self, device: torch.device | str) -> "MLPCentralizedCritic":
        self.implementation.critic.to(device)
        return self

    def train_mode(self) -> None:
        self.implementation.prep_training()

    def eval_mode(self) -> None:
        self.implementation.prep_rollout()


def build_critic(
    critic_type: str,
    args: Mapping[str, Any],
    input_space: Any,
    *,
    device: torch.device,
) -> Any:
    """Construct the selected centralized critic behind one stable interface."""
    kind = str(critic_type).strip().lower()
    if kind == "hgta":
        if tuple(getattr(input_space, "shape", ())) != (MLPCentralizedCritic.input_dim,):
            raise ValueError(
                f"HGTA critic input space must be ({CENTRALIZED_STATE_DIM},)."
            )
        hgta_config = args.get("hgta")
        if not isinstance(hgta_config, Mapping):
            raise ValueError("HGTA critic requires a complete critic.hgta configuration.")
        from harl.algorithms.critics.v_critic import VCritic
        from marl.critics.hgta_critic import HGTACentralizedCritic, HGTAValueNetwork

        implementation = VCritic(dict(args), input_space, device=device)
        network = HGTAValueNetwork(args, hgta_config).to(device)
        implementation.critic = network
        implementation.critic_optimizer = torch.optim.Adam(
            network.parameters(),
            lr=float(args["critic_lr"]),
            eps=float(args["opti_eps"]),
            weight_decay=float(args["weight_decay"]),
        )
        return HGTACentralizedCritic(
            implementation, network, interface_version=CRITIC_INTERFACE_VERSION
        )
    if kind != "mlp":
        raise ValueError(f"Unknown critic type {critic_type!r}.")
    if tuple(getattr(input_space, "shape", ())) != (MLPCentralizedCritic.input_dim,):
        raise ValueError(
            f"MLP critic input space must be ({MLPCentralizedCritic.input_dim},), "
            f"got {getattr(input_space, 'shape', None)!r}."
        )
    from harl.algorithms.critics.v_critic import VCritic

    return MLPCentralizedCritic(VCritic(dict(args), input_space, device=device))
