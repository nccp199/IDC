from __future__ import annotations

import copy
import json
import os
import sys
import types
import unittest
from pathlib import Path

import numpy as np

from marl import IDCGridMultiAgentEnv
from marl.bridges import HarlIDCGridBridge, HarlPaddedBridge
from marl.diagnostics import BESSVirtualActionMonitor, compute_bess_policy_diagnostics
from marl.tests.helpers import make_grid_env


HARL_COMMIT = "b1af98b0dbab72a2eee9d160751cd09aedbb8ce2"


def _smoke_algo_args(seed: int, output_dir: Path) -> dict[str, object]:
    """Return the pinned HARL MAPPO defaults with smoke-only overrides."""
    return {
        "seed": {"seed_specify": True, "seed": seed},
        "device": {"cuda": False, "cuda_deterministic": True, "torch_threads": 1},
        "train": {
            "n_rollout_threads": 1,
            "num_env_steps": 24,
            "episode_length": 24,
            "log_interval": 1,
            "eval_interval": 1000,
            "use_valuenorm": False,
            "use_linear_lr_decay": False,
            "use_proper_time_limits": True,
            "model_dir": None,
        },
        "eval": {"use_eval": False, "n_eval_rollout_threads": 1, "eval_episodes": 1},
        "render": {"use_render": False, "render_episodes": 1},
        "model": {
            "hidden_sizes": [32, 32],
            "activation_func": "relu",
            "use_feature_normalization": True,
            "initialization_method": "orthogonal_",
            "gain": 0.01,
            "use_naive_recurrent_policy": False,
            "use_recurrent_policy": False,
            "recurrent_n": 1,
            "data_chunk_length": 24,
            "lr": 0.0005,
            "critic_lr": 0.0005,
            "opti_eps": 0.00001,
            "weight_decay": 0,
            "std_x_coef": 1,
            "std_y_coef": 0.5,
            "use_bounded_box_actions": True,
        },
        "algo": {
            "ppo_epoch": 1,
            "critic_epoch": 1,
            "use_clipped_value_loss": True,
            "clip_param": 0.2,
            "actor_num_mini_batch": 1,
            "critic_num_mini_batch": 1,
            "entropy_coef": 0.01,
            "value_loss_coef": 1,
            "use_max_grad_norm": True,
            "max_grad_norm": 10.0,
            "use_gae": True,
            "gamma": 0.99,
            "gae_lambda": 0.95,
            "use_huber_loss": True,
            "use_policy_active_masks": True,
            "huber_delta": 10.0,
            "action_aggregation": "prod",
            "share_param": False,
            "fixed_order": True,
        },
        "logger": {"log_dir": str(output_dir / "harl_results")},
    }


def run_harl_mappo_padding_smoke(
    *,
    harl_source: str | Path,
    runtime_dependencies: str | Path | None,
    output_dir: str | Path,
    seed: int = 7110,
) -> dict[str, object]:
    """Run one 24-step rollout and one stock HARL MAPPO update."""
    harl_source = Path(harl_source).resolve()
    if runtime_dependencies is not None:
        sys.path.insert(0, str(Path(runtime_dependencies).resolve()))
    sys.path.insert(0, str(harl_source))

    import torch
    from torch.utils.tensorboard import SummaryWriter as TorchSummaryWriter

    class CompatibleSummaryWriter(TorchSummaryWriter):
        def export_scalars_to_json(self, path) -> None:
            Path(path).write_text(json.dumps({}), encoding="utf-8")

    tensorboard_module = types.ModuleType("tensorboardX")
    tensorboard_module.SummaryWriter = CompatibleSummaryWriter
    sys.modules["tensorboardX"] = tensorboard_module
    process_title_module = types.ModuleType("setproctitle")
    process_title_module.setproctitle = lambda unused_title: None
    sys.modules["setproctitle"] = process_title_module
    sys.modules["yaml"] = types.ModuleType("yaml")

    from harl.envs.env_wrappers import ShareDummyVecEnv
    from harl.runners.on_policy_ma_runner import OnPolicyMARunner
    import harl.runners.on_policy_base_runner as runner_module

    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    algo_args = _smoke_algo_args(seed, output_dir)
    args = {"algo": "mappo", "env": "gym", "exp_name": "idc_bess_padding_smoke"}
    env_args = {"scenario": "idc_bess_padding", "state_type": "EP"}

    monitors: list[BESSVirtualActionMonitor] = []

    def env_factory():
        monitor = BESSVirtualActionMonitor(
            output_dir,
            enabled=True,
            save_full_virtual_action_vector=True,
        )
        monitors.append(monitor)
        return HarlPaddedBridge(
            HarlIDCGridBridge(IDCGridMultiAgentEnv(make_grid_env(seed=seed))),
            diagnostics=monitor,
        )

    original_make_train_env = runner_module.make_train_env
    original_get_num_agents = runner_module.get_num_agents
    try:
        runner_module.make_train_env = lambda *unused, **unused_kwargs: ShareDummyVecEnv(
            [env_factory]
        )
        runner_module.get_num_agents = lambda *unused, **unused_kwargs: 2
        runner = OnPolicyMARunner(args, copy.deepcopy(algo_args), env_args)
    finally:
        runner_module.make_train_env = original_make_train_env
        runner_module.get_num_agents = original_get_num_agents

    monitor = monitors[0]
    monitor.tensorboard_writer = runner.writter
    actor_train_infos = None
    critic_train_info = None
    action_mins = np.full(2, np.inf, dtype=np.float64)
    action_maxs = np.full(2, -np.inf, dtype=np.float64)
    try:
        runner.warmup()
        runner.logger.init(1)
        runner.logger.episode_init(1)
        runner.prep_rollout()
        for step in range(24):
            values, actions, action_log_probs, rnn_states, rnn_states_critic = runner.collect(step)
            if actions.shape != (1, 2, 22) or not np.isfinite(actions).all():
                raise AssertionError(f"Unexpected finite padded action batch: {actions!r}.")
            if np.any(actions < 0.0) or np.any(actions > 1.0):
                raise AssertionError(
                    f"Affine-tanh actions escaped [0, 1] at step {step}: {actions!r}."
                )
            action_mins = np.minimum(action_mins, np.min(actions[0], axis=-1))
            action_maxs = np.maximum(action_maxs, np.max(actions[0], axis=-1))
            obs, share_obs, rewards, dones, infos, available_actions = runner.envs.step(actions)
            data = (
                obs,
                share_obs,
                rewards,
                dones,
                infos,
                available_actions,
                values,
                actions,
                action_log_probs,
                rnn_states,
                rnn_states_critic,
            )
            runner.logger.per_step(data)
            runner.insert(data)

        old_bess_log_prob = runner.actor_buffer[1].action_log_probs.copy().reshape(-1, 22)
        bess_obs = runner.actor_buffer[1].obs[:-1].reshape(-1, 288)
        bess_rnn = runner.actor_buffer[1].rnn_states[:-1].reshape(-1, 1, 32)
        bess_actions = runner.actor_buffer[1].actions.reshape(-1, 22)
        bess_masks = runner.actor_buffer[1].masks[:-1].reshape(-1, 1)
        bess_active_masks = runner.actor_buffer[1].active_masks[:-1].reshape(-1, 1)

        runner.compute()
        advantages = runner.critic_buffer.returns[:-1] - runner.critic_buffer.value_preds[:-1]
        diagnostic_advantages = advantages.copy()
        diagnostic_advantages[runner.actor_buffer[1].active_masks[:-1] == 0.0] = np.nan
        diagnostic_advantages = (
            advantages - np.nanmean(diagnostic_advantages)
        ) / (np.nanstd(diagnostic_advantages) + 1e-5)
        diagnostic_advantages = diagnostic_advantages.reshape(-1)

        runner.prep_training()
        actor_train_infos, critic_train_info = runner.train()
        with torch.no_grad():
            new_log_prob, _, distribution = runner.actor[1].evaluate_actions(
                bess_obs,
                bess_rnn,
                bess_actions,
                bess_masks,
                None,
                bess_active_masks,
            )
        policy_metrics = compute_bess_policy_diagnostics(
            actions=bess_actions,
            old_log_prob=old_bess_log_prob,
            new_log_prob=new_log_prob.detach().cpu().numpy(),
            advantages=diagnostic_advantages,
            clip_param=algo_args["algo"]["clip_param"],
        )
        monitor.record_policy_diagnostics(policy_metrics, global_step=24)

        runner.save()
        model_files = sorted(Path(runner.save_dir).glob("*.pt"))
        if len(model_files) != 3:
            raise AssertionError(f"Expected 3 saved model files, got {model_files}.")
        saved_actor = copy.deepcopy(runner.actor[1].actor.state_dict())
        saved_critic = copy.deepcopy(runner.critic.critic.state_dict())

        reload_args = copy.deepcopy(algo_args)
        reload_args["train"]["model_dir"] = str(runner.save_dir)
        original_make_train_env = runner_module.make_train_env
        original_get_num_agents = runner_module.get_num_agents
        try:
            runner_module.make_train_env = lambda *unused, **unused_kwargs: ShareDummyVecEnv(
                [env_factory]
            )
            runner_module.get_num_agents = lambda *unused, **unused_kwargs: 2
            reloaded = OnPolicyMARunner(args, reload_args, env_args)
        finally:
            runner_module.make_train_env = original_make_train_env
            runner_module.get_num_agents = original_get_num_agents
        try:
            for key, value in saved_actor.items():
                torch.testing.assert_close(reloaded.actor[1].actor.state_dict()[key], value)
            for key, value in saved_critic.items():
                torch.testing.assert_close(reloaded.critic.critic.state_dict()[key], value)
        finally:
            reloaded.close()

        flat_values = [
            float(value)
            for info in actor_train_infos
            for value in info.values()
        ] + [float(value) for value in critic_train_info.values()]
        if not np.isfinite(flat_values).all() or not np.isfinite(list(policy_metrics.values())).all():
            raise AssertionError("MAPPO smoke produced a non-finite training diagnostic.")
        if monitor.rows_written != 24:
            raise AssertionError(f"Expected 24 BESS diagnostics rows, got {monitor.rows_written}.")
        event_files = list(Path(runner.log_dir).glob("events.out.tfevents.*"))
        if not event_files:
            raise AssertionError("TensorBoard event file was not created.")
        return {
            "harl_commit": HARL_COMMIT,
            "total_steps": 24,
            "updates": 1,
            "action_ranges": {
                "idc": (float(action_mins[0]), float(action_maxs[0])),
                "bess_padded": (float(action_mins[1]), float(action_maxs[1])),
            },
            "actor_update_completed": actor_train_infos is not None,
            "critic_update_completed": critic_train_info is not None,
            "model_saved": True,
            "model_reloaded": True,
            "actor_train_infos": actor_train_infos,
            "critic_train_info": critic_train_info,
            "policy_metrics": policy_metrics,
            "diagnostics_csv": str(monitor.path),
            "tensorboard_dir": str(runner.log_dir),
            "model_dir": str(runner.save_dir),
            "nan_detected": False,
        }
    finally:
        runner.close()


class HarlMAPPOSmokeTest(unittest.TestCase):
    def test_one_stock_mappo_update_save_and_reload(self) -> None:
        source = os.environ.get("HARL_SOURCE_PATH")
        if not source or not Path(source).exists():
            self.skipTest("Set HARL_SOURCE_PATH to the pinned HARL checkout for integration smoke.")
        runtime = os.environ.get("HARL_RUNTIME_PATH")
        output = os.environ.get("HARL_SMOKE_OUTPUT_DIR", "runs/idc_bess_mappo_padding_smoke")
        result = run_harl_mappo_padding_smoke(
            harl_source=source,
            runtime_dependencies=runtime,
            output_dir=output,
        )
        self.assertEqual(result["total_steps"], 24)
        self.assertEqual(result["updates"], 1)
        self.assertFalse(result["nan_detected"])
        for minimum, maximum in result["action_ranges"].values():
            self.assertGreaterEqual(minimum, 0.0)
            self.assertLessEqual(maximum, 1.0)
        self.assertTrue(result["actor_update_completed"])
        self.assertTrue(result["critic_update_completed"])
        self.assertTrue(result["model_saved"])
        self.assertTrue(result["model_reloaded"])
        self.assertTrue(Path(result["diagnostics_csv"]).is_file())
        self.assertTrue(Path(result["tensorboard_dir"]).is_dir())
        self.assertTrue(Path(result["model_dir"]).is_dir())


if __name__ == "__main__":
    unittest.main()
