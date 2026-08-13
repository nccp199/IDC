"""Algorithm-specific actor update strategies for the shared IDC runner."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np
import torch


def _factor_stats(factor: np.ndarray) -> dict[str, float]:
    return {
        "mean": float(np.mean(factor)),
        "min": float(np.min(factor)),
        "max": float(np.max(factor)),
        "p01": float(np.percentile(factor, 1)),
        "p50": float(np.percentile(factor, 50)),
        "p99": float(np.percentile(factor, 99)),
        "exactly_one_fraction": float(np.mean(factor == 1.0)),
    }


def _module_parameters(owner: Any, attribute: str) -> list[torch.Tensor]:
    module = getattr(owner, attribute, None)
    if module is None or not hasattr(module, "parameters"):
        return []
    return list(module.parameters())


def _parameter_snapshot(parameters: list[torch.Tensor]) -> list[torch.Tensor]:
    return [parameter.detach().clone() for parameter in parameters]


def _parameter_delta_norm(
    before: list[torch.Tensor], parameters: list[torch.Tensor]
) -> float:
    if not before:
        return 0.0
    squared = sum(
        float(torch.sum((parameter.detach() - previous).double().square()).cpu())
        for previous, parameter in zip(before, parameters, strict=True)
    )
    return float(np.sqrt(squared))


def _optimizer_step(optimizer: Any) -> int:
    if optimizer is None:
        return 0
    steps: list[int] = []
    for state in optimizer.state.values():
        step = state.get("step", 0)
        steps.append(int(step.item()) if torch.is_tensor(step) else int(step))
    return max(steps, default=0)


def _max_abs_diff(left: np.ndarray, right: np.ndarray) -> float:
    if left.shape != right.shape:
        return float("inf")
    if left.size == 0:
        return 0.0
    return float(np.max(np.abs(left.astype(np.float64) - right.astype(np.float64))))


class AlgorithmUpdateStrategy(ABC):
    algorithm_name: str

    @abstractmethod
    def train(self, runner: Any) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        raise NotImplementedError


class MAPPOUpdateStrategy(AlgorithmUpdateStrategy):
    algorithm_name = "mappo"

    def train(self, runner: Any) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        from harl.runners.on_policy_ma_runner import OnPolicyMARunner

        actor_infos, critic_info = OnPolicyMARunner.train(runner)
        runner.last_algorithm_update = {
            "agent_update_order": list(runner.agent_names),
            "agent_update_order_policy": "parallel_independent",
            "happo_available": False,
        }
        return actor_infos, critic_info


class HAPPOUpdateStrategy(AlgorithmUpdateStrategy):
    """HARL OnPolicyHARunner semantics with effective-action factor ratios."""

    algorithm_name = "happo"

    @staticmethod
    def _evaluate(runner: Any, agent_id: int) -> torch.Tensor:
        buffer = runner.actor_buffer[agent_id]
        available_actions = (
            None
            if buffer.available_actions is None
            else buffer.available_actions[:-1].reshape(
                -1, *buffer.available_actions.shape[2:]
            )
        )
        log_probs, _, _ = runner.actor[agent_id].evaluate_actions(
            buffer.obs[:-1].reshape(-1, *buffer.obs.shape[2:]),
            buffer.rnn_states[0:1].reshape(-1, *buffer.rnn_states.shape[2:]),
            buffer.actions.reshape(-1, *buffer.actions.shape[2:]),
            buffer.masks[:-1].reshape(-1, *buffer.masks.shape[2:]),
            available_actions,
            buffer.active_masks[:-1].reshape(-1, *buffer.active_masks.shape[2:]),
        )
        return log_probs

    def train(self, runner: Any) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        episode_length = int(runner.algo_args["train"]["episode_length"])
        n_threads = int(runner.algo_args["train"]["n_rollout_threads"])
        factor = np.ones((episode_length, n_threads, 1), dtype=np.float32)
        initial = _factor_stats(factor)

        if runner.value_normalizer is not None:
            advantages = runner.critic_buffer.returns[:-1] - runner.value_normalizer.denormalize(
                runner.critic_buffer.value_preds[:-1]
            )
        else:
            advantages = runner.critic_buffer.returns[:-1] - runner.critic_buffer.value_preds[:-1]

        if runner.state_type == "FP":
            active = np.stack(
                [buffer.active_masks for buffer in runner.actor_buffer], axis=2
            )
            copy_advantages = advantages.copy()
            copy_advantages[active[:-1] == 0.0] = np.nan
            advantages = (
                advantages - np.nanmean(copy_advantages)
            ) / (np.nanstd(copy_advantages) + 1e-5)

        order = (
            list(range(runner.num_agents))
            if runner.fixed_order
            else list(torch.randperm(runner.num_agents).cpu().numpy())
        )
        actor_infos: list[dict[str, Any] | None] = [None] * runner.num_agents
        after_first: dict[str, float] | None = None
        bess_factor_diff = 0.0
        bess_virtual_contribution = 0.0
        initial_factor = factor.copy()
        factor_after_idc: np.ndarray | None = None
        idc_effective_ratio: np.ndarray | None = None
        bess_physical_ratio: np.ndarray | None = None
        actor_parameter_deltas = [0.0] * runner.num_agents
        actor_optimizer_step_deltas = [0] * runner.num_agents
        rollout_log_prob_diffs = [0.0] * runner.num_agents
        action_snapshots = [buffer.actions.copy() for buffer in runner.actor_buffer]
        log_prob_snapshots = [
            getattr(buffer, "action_log_probs", np.empty(0)).copy()
            for buffer in runner.actor_buffer
        ]
        return_snapshot = runner.critic_buffer.returns.copy()
        advantage_snapshot = advantages.copy()

        for position, agent_id in enumerate(order):
            buffer = runner.actor_buffer[agent_id]
            buffer.update_factor(factor)
            old_log_probs = self._evaluate(runner, agent_id).detach()
            stored_log_probs = getattr(buffer, "action_log_probs", None)
            if stored_log_probs is not None:
                rollout_log_prob_diffs[agent_id] = float(
                    torch.max(
                        torch.abs(
                            old_log_probs
                            - torch.as_tensor(
                                stored_log_probs.reshape(old_log_probs.shape),
                                device=old_log_probs.device,
                                dtype=old_log_probs.dtype,
                            )
                        )
                    ).cpu()
                )
            actor_parameters = _module_parameters(runner.actor[agent_id], "actor")
            actor_before = _parameter_snapshot(actor_parameters)
            actor_optimizer = getattr(runner.actor[agent_id], "actor_optimizer", None)
            optimizer_step_before = _optimizer_step(actor_optimizer)
            if runner.state_type == "EP":
                info = runner.actor[agent_id].train(buffer, advantages.copy(), "EP")
            else:
                info = runner.actor[agent_id].train(
                    buffer, advantages[:, :, agent_id].copy(), "FP"
                )
            actor_parameter_deltas[agent_id] = _parameter_delta_norm(
                actor_before, actor_parameters
            )
            actor_optimizer_step_deltas[agent_id] = (
                _optimizer_step(actor_optimizer) - optimizer_step_before
            )
            new_log_probs = self._evaluate(runner, agent_id).detach()
            ratio = runner.actor[agent_id].effective_ratio(new_log_probs, old_log_probs)
            ratio_array = ratio.cpu().numpy().reshape(episode_length, n_threads, 1)
            factor = factor * ratio_array
            if not np.isfinite(factor).all():
                raise FloatingPointError("HAPPO importance factor became non-finite.")
            actor_infos[agent_id] = info
            if position == 0:
                after_first = _factor_stats(factor)
            if runner.agent_names[agent_id] == "bess":
                physical_ratio = torch.exp(new_log_probs[:, :1] - old_log_probs[:, :1])
                bess_physical_ratio = physical_ratio.cpu().numpy().reshape(
                    episode_length, n_threads, 1
                )
                bess_factor_diff = float(torch.max(torch.abs(ratio - physical_ratio)).cpu())
                virtual_mask = 1.0 - runner.actor[agent_id].effective_action_mask
                actual_virtual_log_ratio = (
                    (new_log_probs - old_log_probs) * virtual_mask * 0.0
                ).sum(dim=-1, keepdim=True)
                bess_virtual_contribution = float(
                    torch.max(torch.abs(actual_virtual_log_ratio)).cpu()
                )
            elif runner.agent_names[agent_id] == "idc":
                idc_effective_ratio = ratio_array.copy()
                factor_after_idc = factor.copy()

        critic_parameters = _module_parameters(runner.critic, "critic")
        critic_before = _parameter_snapshot(critic_parameters)
        critic_optimizer = getattr(runner.critic, "critic_optimizer", None)
        critic_optimizer_step_before = _optimizer_step(critic_optimizer)
        critic_info = runner.critic.train(runner.critic_buffer, runner.value_normalizer)
        critic_parameter_delta = _parameter_delta_norm(critic_before, critic_parameters)
        critic_optimizer_step_delta = (
            _optimizer_step(critic_optimizer) - critic_optimizer_step_before
        )
        critic_postclip_grad_norm = float(
            np.sqrt(
                sum(
                    float(torch.sum(parameter.grad.detach().double().square()).cpu())
                    for parameter in critic_parameters
                    if parameter.grad is not None
                )
            )
        )
        final = _factor_stats(factor)
        assert after_first is not None
        if idc_effective_ratio is None or factor_after_idc is None or bess_physical_ratio is None:
            raise RuntimeError("HAPPO audit requires the fixed IDC/BESS agent set.")
        reconstructed_after_idc = initial_factor * idc_effective_ratio
        reconstructed_final = reconstructed_after_idc * bess_physical_ratio
        runner.last_algorithm_update = {
            "agent_update_order": [runner.agent_names[index] for index in order],
            "agent_update_order_policy": "fixed" if runner.fixed_order else "torch_randperm",
            "happo_available": True,
            "happo_factor_shape": list(factor.shape),
            "happo_factor_initial_mean": initial["mean"],
            "happo_factor_initial_min": initial["min"],
            "happo_factor_initial_max": initial["max"],
            "happo_factor_initial_p01": initial["p01"],
            "happo_factor_initial_p50": initial["p50"],
            "happo_factor_initial_p99": initial["p99"],
            "happo_factor_initial_exactly_one_fraction": initial["exactly_one_fraction"],
            "happo_factor_after_first_agent_mean": after_first["mean"],
            "happo_factor_after_first_agent_min": after_first["min"],
            "happo_factor_after_first_agent_max": after_first["max"],
            "happo_factor_after_first_agent_p01": after_first["p01"],
            "happo_factor_after_first_agent_p50": after_first["p50"],
            "happo_factor_after_first_agent_p99": after_first["p99"],
            "happo_factor_after_first_agent_exactly_one_fraction": after_first["exactly_one_fraction"],
            "happo_factor_after_idc_mean": _factor_stats(factor_after_idc)["mean"],
            "happo_factor_after_idc_min": _factor_stats(factor_after_idc)["min"],
            "happo_factor_after_idc_max": _factor_stats(factor_after_idc)["max"],
            "happo_factor_after_idc_p01": _factor_stats(factor_after_idc)["p01"],
            "happo_factor_after_idc_p50": _factor_stats(factor_after_idc)["p50"],
            "happo_factor_after_idc_p99": _factor_stats(factor_after_idc)["p99"],
            "happo_factor_after_idc_exactly_one_fraction": _factor_stats(factor_after_idc)["exactly_one_fraction"],
            "happo_factor_final_mean": final["mean"],
            "happo_factor_final_min": final["min"],
            "happo_factor_final_max": final["max"],
            "happo_factor_final_p01": final["p01"],
            "happo_factor_final_p50": final["p50"],
            "happo_factor_final_p99": final["p99"],
            "happo_factor_final_exactly_one_fraction": final["exactly_one_fraction"],
            "happo_factor_nonfinite_count": int((~np.isfinite(factor)).sum()),
            "happo_factor_nonpositive_count": int((factor <= 0.0).sum()),
            "happo_factor_after_idc_reconstruction_max_diff": _max_abs_diff(
                reconstructed_after_idc, factor_after_idc
            ),
            "happo_factor_final_reconstruction_max_diff": _max_abs_diff(
                reconstructed_final, factor
            ),
            "first_updated_agent": runner.agent_names[order[0]],
            "second_updated_agent": runner.agent_names[order[1]],
            "bess_happo_factor_effective_physical_max_diff": bess_factor_diff,
            "bess_happo_virtual_factor_contribution": bess_virtual_contribution,
            "actor_update_count_idc": 1,
            "actor_update_count_bess": 1,
            "critic_update_count": 1,
            "idc_actor_parameter_delta_norm": actor_parameter_deltas[0],
            "bess_actor_parameter_delta_norm": actor_parameter_deltas[1],
            "critic_parameter_delta_norm": critic_parameter_delta,
            "idc_actor_optimizer_step_delta": actor_optimizer_step_deltas[0],
            "bess_actor_optimizer_step_delta": actor_optimizer_step_deltas[1],
            "critic_optimizer_step_delta": critic_optimizer_step_delta,
            "idc_actor_optimizer_step_count": _optimizer_step(
                getattr(runner.actor[0], "actor_optimizer", None)
            ),
            "bess_actor_optimizer_step_count": _optimizer_step(
                getattr(runner.actor[1], "actor_optimizer", None)
            ),
            "critic_optimizer_step_count": _optimizer_step(critic_optimizer),
            "critic_grad_norm_postclip": critic_postclip_grad_norm,
            "idc_rollout_log_prob_max_diff": rollout_log_prob_diffs[0],
            "bess_rollout_log_prob_max_diff": rollout_log_prob_diffs[1],
            "idc_action_buffer_mutation_max_diff": _max_abs_diff(
                action_snapshots[0], runner.actor_buffer[0].actions
            ),
            "bess_action_buffer_mutation_max_diff": _max_abs_diff(
                action_snapshots[1], runner.actor_buffer[1].actions
            ),
            "idc_log_prob_buffer_mutation_max_diff": _max_abs_diff(
                log_prob_snapshots[0],
                getattr(runner.actor_buffer[0], "action_log_probs", np.empty(0)),
            ),
            "bess_log_prob_buffer_mutation_max_diff": _max_abs_diff(
                log_prob_snapshots[1],
                getattr(runner.actor_buffer[1], "action_log_probs", np.empty(0)),
            ),
            "critic_return_buffer_mutation_max_diff": _max_abs_diff(
                return_snapshot, runner.critic_buffer.returns
            ),
            "advantage_mutation_max_diff": _max_abs_diff(
                advantage_snapshot, advantages
            ),
            "factor_audit": {
                "initial": initial_factor,
                "idc_effective_ratio": idc_effective_ratio,
                "after_idc": factor_after_idc,
                "bess_physical_ratio": bess_physical_ratio,
                "final": factor.copy(),
            },
        }
        return [info for info in actor_infos if info is not None], critic_info


def build_update_strategy(algorithm_name: str) -> AlgorithmUpdateStrategy:
    algorithm = str(algorithm_name).strip().lower()
    if algorithm == "mappo":
        return MAPPOUpdateStrategy()
    if algorithm == "happo":
        return HAPPOUpdateStrategy()
    raise ValueError(f"No update strategy for algorithm {algorithm_name!r}.")
