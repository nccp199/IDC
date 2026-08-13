"""MAPPO/HAPPO adapters that optimize only each agent's effective Box actions."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn as nn

from harl.algorithms.actors.happo import HAPPO
from harl.algorithms.actors.mappo import MAPPO
from harl.utils.envs_tools import check
from harl.utils.models_tools import get_grad_norm


def _normalized_mask(action_mask: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    mask = action_mask.to(dtype=reference.dtype, device=reference.device)
    if mask.ndim == 1:
        mask = mask.unsqueeze(0)
    if mask.ndim != 2 or mask.shape[-1] != reference.shape[-1]:
        raise ValueError(
            "action_mask must be broadcastable over the per-dimension log probabilities."
        )
    return mask


def effective_ratio_from_log_probs(
    new_log_probs: torch.Tensor,
    old_log_probs: torch.Tensor,
    action_mask: torch.Tensor,
) -> torch.Tensor:
    """Compute exp(sum(mask * delta-log-prob)) without zeroing a product."""
    if new_log_probs.shape != old_log_probs.shape:
        raise ValueError("new and old per-dimension log probabilities must have equal shapes.")
    mask = _normalized_mask(action_mask, new_log_probs)
    effective_log_ratio = ((new_log_probs - old_log_probs) * mask).sum(
        dim=-1, keepdim=True
    )
    return torch.exp(effective_log_ratio)


def effective_entropy_from_log_probs(
    action_log_probs: torch.Tensor,
    action_mask: torch.Tensor,
    active_masks: torch.Tensor | None = None,
) -> torch.Tensor:
    """Return the masked affine-tanh sample entropy used by this integration."""
    mask = _normalized_mask(action_mask, action_log_probs)
    entropy_per_sample = -(action_log_probs * mask).sum(dim=-1)
    if active_masks is None:
        return entropy_per_sample.mean()
    active = active_masks.to(
        dtype=action_log_probs.dtype, device=action_log_probs.device
    ).squeeze(-1)
    return (entropy_per_sample * active).sum() / active.sum()


class _EffectiveActionMaskMixin:
    """Shared masked ratio/entropy update for the project MAPPO and HAPPO actors."""

    effective_action_mask: torch.Tensor

    def _initialize_effective_action_mask(
        self,
        act_space: Any,
        effective_action_dim: int | None,
        device: torch.device,
    ) -> None:
        if act_space.__class__.__name__ != "Box":
            raise TypeError("Effective-action masking currently supports continuous Box actions only.")
        if self.action_aggregation != "prod":
            raise ValueError("Effective-action masking requires action_aggregation='prod'.")
        padded_dim = int(act_space.shape[0])
        effective_dim = padded_dim if effective_action_dim is None else int(effective_action_dim)
        if effective_dim <= 0 or effective_dim > padded_dim:
            raise ValueError(
                f"effective_action_dim must be in [1, {padded_dim}], got {effective_dim}."
            )
        mask = torch.zeros(padded_dim, dtype=torch.float32, device=device)
        mask[:effective_dim] = 1.0
        self.effective_action_dim = effective_dim
        self.padded_action_dim = padded_dim
        self.effective_action_mask = mask.unsqueeze(0)
        self.last_update_diagnostics: dict[str, Any] | None = None

    def effective_ratio(
        self, new_log_probs: torch.Tensor, old_log_probs: torch.Tensor
    ) -> torch.Tensor:
        return effective_ratio_from_log_probs(
            new_log_probs, old_log_probs, self.effective_action_mask
        )

    def effective_entropy(
        self,
        action_log_probs: torch.Tensor,
        active_masks: torch.Tensor | None,
    ) -> torch.Tensor:
        return effective_entropy_from_log_probs(
            action_log_probs,
            self.effective_action_mask,
            active_masks if self.use_policy_active_masks else None,
        )

    def _masked_update(
        self,
        *,
        obs_batch,
        rnn_states_batch,
        actions_batch,
        masks_batch,
        active_masks_batch,
        old_action_log_probs_batch,
        adv_targ,
        available_actions_batch,
        factor_batch=None,
    ):
        old_action_log_probs_batch = check(old_action_log_probs_batch).to(**self.tpdv)
        adv_targ = check(adv_targ).to(**self.tpdv)
        active_masks_batch = check(active_masks_batch).to(**self.tpdv)
        if factor_batch is not None:
            factor_batch = check(factor_batch).to(**self.tpdv)

        action_log_probs, _, _ = self.evaluate_actions(
            obs_batch,
            rnn_states_batch,
            actions_batch,
            masks_batch,
            available_actions_batch,
            active_masks_batch,
        )
        imp_weights = self.effective_ratio(
            action_log_probs, old_action_log_probs_batch
        )
        dist_entropy = self.effective_entropy(action_log_probs, active_masks_batch)

        surr1 = imp_weights * adv_targ
        surr2 = (
            torch.clamp(
                imp_weights, 1.0 - self.clip_param, 1.0 + self.clip_param
            )
            * adv_targ
        )
        surrogate = torch.min(surr1, surr2)
        if factor_batch is not None:
            surrogate = factor_batch * surrogate

        if self.use_policy_active_masks:
            policy_action_loss = (
                -torch.sum(surrogate, dim=-1, keepdim=True) * active_masks_batch
            ).sum() / active_masks_batch.sum()
        else:
            policy_action_loss = -torch.sum(
                surrogate, dim=-1, keepdim=True
            ).mean()

        self.actor_optimizer.zero_grad()
        (policy_action_loss - dist_entropy * self.entropy_coef).backward()

        mean_layer = self.actor.act.action_out.fc_mean
        log_std = self.actor.act.action_out.log_std
        mean_row_norms = torch.sqrt(
            torch.sum(mean_layer.weight.grad.square(), dim=1)
            + mean_layer.bias.grad.square()
        )
        virtual_mean_rows = mean_row_norms[self.effective_action_dim :]
        virtual_log_std = log_std.grad[self.effective_action_dim :]
        trunk_gradients = [
            parameter.grad.reshape(-1)
            for parameter in self.actor.base.parameters()
            if parameter.grad is not None
        ]
        trunk_grad_norm = torch.linalg.vector_norm(torch.cat(trunk_gradients))
        preclip_grad_norm = get_grad_norm(self.actor.parameters())

        if self.use_max_grad_norm:
            actor_grad_norm = nn.utils.clip_grad_norm_(
                self.actor.parameters(), self.max_grad_norm
            )
        else:
            actor_grad_norm = get_grad_norm(self.actor.parameters())
        postclip_grad_norm = get_grad_norm(self.actor.parameters())
        self.actor_optimizer.step()

        with torch.no_grad():
            post_log_probs, _, _ = self.evaluate_actions(
                obs_batch,
                rnn_states_batch,
                actions_batch,
                masks_batch,
                available_actions_batch,
                active_masks_batch,
            )
            effective_ratio = self.effective_ratio(
                post_log_probs, old_action_log_probs_batch
            )
            if self.effective_action_dim == 1:
                physical_ratio = torch.exp(
                    post_log_probs[:, :1] - old_action_log_probs_batch[:, :1]
                )
            else:
                physical_ratio = effective_ratio
            effective_log_ratio = torch.log(effective_ratio)
            physical_log_ratio = torch.log(physical_ratio)
            effective_clipped = (effective_ratio < 1.0 - self.clip_param) | (
                effective_ratio > 1.0 + self.clip_param
            )
            physical_clipped = (physical_ratio < 1.0 - self.clip_param) | (
                physical_ratio > 1.0 + self.clip_param
            )
            post_entropy = self.effective_entropy(
                post_log_probs, active_masks_batch
            )
            self.last_update_diagnostics = {
                "effective_action_dim": self.effective_action_dim,
                "padded_action_dim": self.padded_action_dim,
                "effective_action_mask": self.effective_action_mask.squeeze(0)
                .detach()
                .cpu()
                .tolist(),
                "effective_ratio_mean": float(effective_ratio.mean().cpu()),
                "effective_ratio_std": float(effective_ratio.std(unbiased=False).cpu()),
                "effective_ratio_min": float(effective_ratio.min().cpu()),
                "effective_ratio_max": float(effective_ratio.max().cpu()),
                "physical_ratio_mean": float(physical_ratio.mean().cpu()),
                "physical_ratio_std": float(physical_ratio.std(unbiased=False).cpu()),
                "effective_physical_ratio_max_diff": float(
                    torch.max(torch.abs(effective_ratio - physical_ratio)).cpu()
                ),
                "effective_clip_fraction": float(
                    effective_clipped.float().mean().cpu()
                ),
                "physical_clip_fraction": float(
                    physical_clipped.float().mean().cpu()
                ),
                "clip_disagreement_fraction": float(
                    (effective_clipped != physical_clipped).float().mean().cpu()
                ),
                "effective_approx_kl": float(
                    torch.mean(
                        (effective_ratio - 1.0) - effective_log_ratio
                    ).cpu()
                ),
                "physical_approx_kl": float(
                    torch.mean((physical_ratio - 1.0) - physical_log_ratio).cpu()
                ),
                "effective_entropy": float(post_entropy.cpu()),
                "virtual_entropy_optimization_contribution": 0.0,
                "physical_mean_row_grad_norm": float(mean_row_norms[0].cpu()),
                "virtual_mean_rows_grad_norm": float(
                    torch.linalg.vector_norm(virtual_mean_rows).cpu()
                    if virtual_mean_rows.numel()
                    else 0.0
                ),
                "virtual_mean_rows_grad_max_abs": float(
                    torch.max(
                        torch.abs(
                            mean_layer.weight.grad[self.effective_action_dim :]
                        )
                    ).cpu()
                    if virtual_mean_rows.numel()
                    else 0.0
                ),
                "physical_log_std_grad_abs": float(torch.abs(log_std.grad[0]).cpu()),
                "virtual_log_std_grad_norm": float(
                    torch.linalg.vector_norm(virtual_log_std).cpu()
                    if virtual_log_std.numel()
                    else 0.0
                ),
                "virtual_log_std_grad_max_abs": float(
                    torch.max(torch.abs(virtual_log_std)).cpu()
                    if virtual_log_std.numel()
                    else 0.0
                ),
                "shared_trunk_grad_norm": float(trunk_grad_norm.cpu()),
                "total_actor_grad_norm_preclip": float(preclip_grad_norm),
                "total_actor_grad_norm_postclip": float(postclip_grad_norm),
                "all_finite": bool(
                    np.isfinite(
                        [
                            float(effective_ratio.mean().cpu()),
                            float(post_entropy.cpu()),
                            float(trunk_grad_norm.cpu()),
                            float(preclip_grad_norm),
                        ]
                    ).all()
                ),
            }
        return policy_action_loss, dist_entropy, actor_grad_norm, imp_weights


class EffectiveActionMAPPO(_EffectiveActionMaskMixin, MAPPO):
    """MAPPO with a persistent per-agent effective-action mask."""

    def __init__(
        self,
        args,
        obs_space,
        act_space,
        device=torch.device("cpu"),
        *,
        effective_action_dim: int | None = None,
    ) -> None:
        super().__init__(args, obs_space, act_space, device)
        self._initialize_effective_action_mask(
            act_space, effective_action_dim, device
        )

    def update(self, sample):
        (
            obs_batch,
            rnn_states_batch,
            actions_batch,
            masks_batch,
            active_masks_batch,
            old_action_log_probs_batch,
            adv_targ,
            available_actions_batch,
        ) = sample
        return self._masked_update(
            obs_batch=obs_batch,
            rnn_states_batch=rnn_states_batch,
            actions_batch=actions_batch,
            masks_batch=masks_batch,
            active_masks_batch=active_masks_batch,
            old_action_log_probs_batch=old_action_log_probs_batch,
            adv_targ=adv_targ,
            available_actions_batch=available_actions_batch,
        )


class EffectiveActionHAPPO(_EffectiveActionMaskMixin, HAPPO):
    """HAPPO with the same effective-action probability semantics as MAPPO."""

    def __init__(
        self,
        args,
        obs_space,
        act_space,
        device=torch.device("cpu"),
        *,
        effective_action_dim: int | None = None,
    ) -> None:
        super().__init__(args, obs_space, act_space, device)
        self._initialize_effective_action_mask(
            act_space, effective_action_dim, device
        )

    def update(self, sample):
        (
            obs_batch,
            rnn_states_batch,
            actions_batch,
            masks_batch,
            active_masks_batch,
            old_action_log_probs_batch,
            adv_targ,
            available_actions_batch,
            factor_batch,
        ) = sample
        return self._masked_update(
            obs_batch=obs_batch,
            rnn_states_batch=rnn_states_batch,
            actions_batch=actions_batch,
            masks_batch=masks_batch,
            active_masks_batch=active_masks_batch,
            old_action_log_probs_batch=old_action_log_probs_batch,
            adv_targ=adv_targ,
            available_actions_batch=available_actions_batch,
            factor_batch=factor_batch,
        )


EFFECTIVE_ACTION_ALGO_REGISTRY = {
    "mappo": EffectiveActionMAPPO,
    "happo": EffectiveActionHAPPO,
}
