"""Shared IDC/BESS on-policy runner with pluggable MAPPO/HAPPO updates."""

from __future__ import annotations

import time
from typing import Any

import numpy as np
import setproctitle
import torch

from harl.common.buffers.on_policy_actor_buffer import OnPolicyActorBuffer
from harl.common.buffers.on_policy_critic_buffer_ep import OnPolicyCriticBufferEP
from harl.common.buffers.on_policy_critic_buffer_fp import OnPolicyCriticBufferFP
from harl.common.valuenorm import ValueNorm
from harl.envs import LOGGER_REGISTRY
from harl.runners.on_policy_ma_runner import OnPolicyMARunner
from harl.utils.configs_tools import init_dir, save_config
from harl.utils.envs_tools import set_seed
from harl.utils.models_tools import init_device

from marl.algorithms import EFFECTIVE_ACTION_ALGO_REGISTRY, build_update_strategy
from marl.critics import build_critic
from marl.logging import CompositeTrainingLogger, TrainingMetricsLogger
from marl.methods import derive_method
from marl.specs import AGENTS, EFFECTIVE_ACTION_DIMS, PADDED_ACTION_DIMS, effective_action_mask
from marl.utils.rng_isolation import build_with_isolated_rng


class IDCOnPolicyMARunner(OnPolicyMARunner):
    """Use one HARL rollout/GAE/buffer loop for MAPPO and HAPPO.

    HARL's pinned base runner calls module-global environment factory functions
    from its constructor and has no external-VecEnv injection point. This class
    only replaces that constructor. Algorithm differences are isolated in the
    update strategy; sampling, bridge, buffers, GAE, logging, and checkpoints
    remain shared.
    """

    def __init__(
        self,
        args: dict[str, Any],
        algo_args: dict[str, Any],
        env_args: dict[str, Any],
        *,
        train_envs: Any,
        eval_envs: Any | None = None,
    ) -> None:
        critic_type = args.get("critic_type")
        self.method = derive_method(args.get("algo", ""), critic_type)
        if algo_args["render"]["use_render"]:
            raise ValueError("The formal short-training runner does not support render mode.")
        if int(getattr(train_envs, "n_agents", -1)) != 2:
            raise ValueError("The formal training VecEnv must expose n_agents=2.")
        if algo_args["eval"]["use_eval"] and eval_envs is None:
            raise ValueError("use_eval=True requires an independently constructed eval VecEnv.")

        self.args = args
        self.algo_args = algo_args
        self.env_args = env_args
        self.hidden_sizes = algo_args["model"]["hidden_sizes"]
        self.rnn_hidden_size = self.hidden_sizes[-1]
        self.recurrent_n = algo_args["model"]["recurrent_n"]
        self.action_aggregation = algo_args["algo"]["action_aggregation"]
        self.state_type = env_args.get("state_type", "EP")
        self.share_param = algo_args["algo"]["share_param"]
        self.fixed_order = algo_args["algo"]["fixed_order"]
        self.agent_names = AGENTS
        self.update_strategy = build_update_strategy(self.method.algorithm_name)
        self.last_algorithm_update: dict[str, Any] = {}
        self.stability_counters = {
            "action_overflow_count": 0,
            "nonfinite_observation_count": 0,
            "nonfinite_state_count": 0,
            "nonfinite_action_count": 0,
            "nonfinite_reward_count": 0,
        }
        self.completed_updates = 0
        self.resume_start_update = 0
        self.resumed = False
        self.total_updates_target = (
            int(algo_args["train"]["num_env_steps"])
            // int(algo_args["train"]["episode_length"])
            // int(algo_args["train"]["n_rollout_threads"])
        )
        self.checkpoint_manager: Any | None = None
        if self.share_param:
            raise ValueError("IDC/BESS effective-action masks require share_param=false.")
        if tuple(getattr(train_envs, "agent_order", ())) != AGENTS:
            raise ValueError(f"Formal runner requires agent order {AGENTS!r}.")
        if args["algo"] not in EFFECTIVE_ACTION_ALGO_REGISTRY:
            raise ValueError(
                f"No project effective-action adapter for algorithm {args['algo']!r}."
            )
        self.effective_action_dims = tuple(
            EFFECTIVE_ACTION_DIMS[agent] for agent in AGENTS
        )
        self.padded_action_dims = tuple(PADDED_ACTION_DIMS[agent] for agent in AGENTS)
        self.effective_action_masks = tuple(
            effective_action_mask(agent) for agent in AGENTS
        )

        set_seed(algo_args["seed"])
        self.device = init_device(algo_args["device"])
        self.run_dir, self.log_dir, self.save_dir, self.writter = init_dir(
            args["env"],
            env_args,
            args["algo"],
            args["exp_name"],
            algo_args["seed"]["seed"],
            logger_path=algo_args["logger"]["log_dir"],
        )
        save_config(args, algo_args, env_args, self.run_dir)
        setproctitle.setproctitle(
            f"{args['algo']}-{args['env']}-{args['exp_name']}"
        )

        self.envs = train_envs
        self.eval_envs = eval_envs
        self.num_agents = int(train_envs.n_agents)
        self.actor = []
        actor_class = EFFECTIVE_ACTION_ALGO_REGISTRY[args["algo"]]
        for agent_id in range(self.num_agents):
            actual_padded_dim = int(self.envs.action_space[agent_id].shape[0])
            if actual_padded_dim != self.padded_action_dims[agent_id]:
                raise ValueError(
                    f"Agent {agent_id} padded action dim must be "
                    f"{self.padded_action_dims[agent_id]}, got {actual_padded_dim}."
                )
            self.actor.append(
                actor_class(
                    {**algo_args["model"], **algo_args["algo"]},
                    self.envs.observation_space[agent_id],
                    self.envs.action_space[agent_id],
                    device=self.device,
                    effective_action_dim=self.effective_action_dims[agent_id],
                )
            )

        self.actor_buffer = [
            OnPolicyActorBuffer(
                {**algo_args["train"], **algo_args["model"]},
                self.envs.observation_space[agent_id],
                self.envs.action_space[agent_id],
            )
            for agent_id in range(self.num_agents)
        ]

        share_observation_space = self.envs.share_observation_space[0]
        critic_args = {**algo_args["model"], **algo_args["algo"]}
        if self.method.critic_type == "hgta":
            critic_args["hgta"] = dict(algo_args["critic"]["hgta"])
        self.rng_isolation_metadata = dict(algo_args["rng"])
        self.critic = build_with_isolated_rng(
            lambda: build_critic(
                self.method.critic_type,
                critic_args,
                share_observation_space,
                device=self.device,
            ),
            seed=int(self.rng_isolation_metadata["critic_init_seed"]),
        )
        if self.state_type == "EP":
            self.critic_buffer = OnPolicyCriticBufferEP(
                {**algo_args["train"], **algo_args["model"], **algo_args["algo"]},
                share_observation_space,
            )
        elif self.state_type == "FP":
            self.critic_buffer = OnPolicyCriticBufferFP(
                {**algo_args["train"], **algo_args["model"], **algo_args["algo"]},
                share_observation_space,
                self.num_agents,
            )
        else:
            raise ValueError(f"Unsupported centralized state type {self.state_type!r}.")

        self.value_normalizer = (
            ValueNorm(1, device=self.device)
            if algo_args["train"]["use_valuenorm"]
            else None
        )
        upstream_logger = LOGGER_REGISTRY[args["env"]](
            args, algo_args, env_args, self.num_agents, self.writter, self.run_dir
        )
        logger_config = algo_args["logger"]
        method_metadata = self.method.as_dict()
        method_metadata.update(getattr(self.critic, "graph_metadata", {}))
        self.metrics_logger = TrainingMetricsLogger(
            run_dir=self.run_dir,
            seed=algo_args["seed"]["seed"],
            writer=self.writter,
            episode_length=algo_args["train"]["episode_length"],
            step_logging_enabled=logger_config.get("step_logging_enabled", True),
            episode_logging_enabled=logger_config.get(
                "episode_logging_enabled", True
            ),
            tensorboard_enabled=logger_config.get("tensorboard_enabled", True),
            method_metadata=method_metadata,
        )
        self.logger = CompositeTrainingLogger(upstream_logger, self.metrics_logger)
        if algo_args["train"]["model_dir"] is not None:
            self.restore()

    def set_checkpoint_manager(self, manager: Any) -> None:
        """Attach the project checkpoint owner without changing HARL optimizers."""
        self.checkpoint_manager = manager

    def run(self):
        """Run HARL's stock loop with post-update checkpoint/resume boundaries.

        A resumed runner already owns the next episode's reset observation and
        environment state, so it intentionally skips ``warmup()``.
        """
        if self.algo_args["render"]["use_render"]:
            self.render()
            return
        print("start running")
        if not self.resumed:
            self.warmup()
        self._check_rollout_state_finite()
        if self.total_updates_target <= self.completed_updates:
            raise ValueError(
                "The total target updates must exceed the checkpoint's saved update: "
                f"target={self.total_updates_target}, saved={self.completed_updates}."
            )

        self.logger.init(self.total_updates_target)
        for update in range(self.completed_updates + 1, self.total_updates_target + 1):
            if self.algo_args["train"]["use_linear_lr_decay"]:
                if self.share_param:
                    self.actor[0].lr_decay(update, self.total_updates_target)
                else:
                    for actor in self.actor:
                        actor.lr_decay(update, self.total_updates_target)
                self.critic.lr_decay(update, self.total_updates_target)

            self.logger.episode_init(update)
            self.prep_rollout()
            for step in range(self.algo_args["train"]["episode_length"]):
                values, actions, action_log_probs, rnn_states, rnn_states_critic = self.collect(step)
                obs, share_obs, rewards, dones, infos, available_actions = self.envs.step(actions)
                self._check_environment_output(obs, share_obs, rewards)
                data = (
                    obs, share_obs, rewards, dones, infos, available_actions,
                    values, actions, action_log_probs, rnn_states, rnn_states_critic,
                )
                self.logger.per_step(data)
                self.insert(data)

            self.compute()
            self.prep_training()
            actor_train_infos, critic_train_info = self.train()
            if update % self.algo_args["train"]["log_interval"] == 0:
                self.logger.episode_log(
                    actor_train_infos,
                    critic_train_info,
                    self.actor_buffer,
                    self.critic_buffer,
                )
            if update % self.algo_args["train"]["eval_interval"] == 0:
                if self.algo_args["eval"]["use_eval"]:
                    self.prep_rollout()
                    self.eval()
                self.save()

            # This makes buffer index 0 and the auto-reset environment describe
            # precisely the next collect.  Checkpoints are never mid-rollout.
            self.after_update()
            if self.checkpoint_manager is not None:
                self.checkpoint_manager.maybe_save(update)

    @torch.no_grad()
    def collect(self, step):
        """Collect with strict finite and Box-bound checks; never clip actions."""
        result = super().collect(step)
        actions = result[1]
        if not np.isfinite(actions).all():
            self.stability_counters["nonfinite_action_count"] += int(
                (~np.isfinite(actions)).sum()
            )
            raise FloatingPointError(f"Non-finite action at rollout step {step}.")
        for agent_id, action_space in enumerate(self.envs.action_space):
            agent_actions = actions[:, agent_id]
            if np.any(agent_actions < action_space.low) or np.any(
                agent_actions > action_space.high
            ):
                self.stability_counters["action_overflow_count"] += int(
                    np.sum(agent_actions < action_space.low)
                    + np.sum(agent_actions > action_space.high)
                )
                raise ValueError(
                    f"Agent {agent_id} produced an out-of-bounds action at step {step}; "
                    "actions are not clipped."
                )
        return result

    def _check_rollout_state_finite(self) -> None:
        observations = np.stack([buffer.obs[0] for buffer in self.actor_buffer], axis=1)
        states = self.critic_buffer.share_obs[0]
        self._check_environment_output(observations, states, None)

    def _check_environment_output(self, observations, states, rewards) -> None:
        checks = (
            ("nonfinite_observation_count", observations, "observation"),
            ("nonfinite_state_count", states, "shared state"),
            ("nonfinite_reward_count", rewards, "reward"),
        )
        for counter, value, label in checks:
            if value is None:
                continue
            nonfinite = int((~np.isfinite(np.asarray(value))).sum())
            if nonfinite:
                self.stability_counters[counter] += nonfinite
                raise FloatingPointError(
                    f"Environment produced {nonfinite} non-finite {label} values."
                )

    def train(self):
        """Run the selected update strategy and reject non-finite diagnostics."""
        rollout_time_seconds = self.metrics_logger.begin_policy_update()
        update_started = time.perf_counter()
        actor_infos, critic_info = self.update_strategy.train(self)
        update_time_seconds = time.perf_counter() - update_started
        diagnostics = [
            value
            for actor_info in actor_infos
            for value in actor_info.values()
        ] + list(critic_info.values())
        for value in diagnostics:
            if torch.is_tensor(value):
                value = value.detach().cpu().numpy()
            if not np.isfinite(np.asarray(value)).all():
                raise FloatingPointError(
                    f"{self.method.algorithm_name.upper()} update produced a non-finite diagnostic."
                )
        update = self.completed_updates + 1
        self.metrics_logger.record_update(
            update=update,
            actor_infos=actor_infos,
            critic_info=critic_info,
            actor_buffers=self.actor_buffer,
            critic_buffer=self.critic_buffer,
            actors=self.actor,
            update_time_seconds=update_time_seconds,
            rollout_time_seconds=rollout_time_seconds,
            algorithm_update=self.last_algorithm_update,
        )
        self.completed_updates = update
        return actor_infos, critic_info
