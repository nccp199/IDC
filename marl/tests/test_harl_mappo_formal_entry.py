from __future__ import annotations

import csv
import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from marl.logging import LOGGER_VERSION, METRIC_SCHEMA_VERSION
from train.train_harl_mappo_short import (
    DEFAULT_CONFIG_PATH,
    PROJECT_ROOT,
    configure_import_paths,
    derive_num_env_steps,
    load_yaml_config,
    main,
    prepare_training_environment,
    resolve_config,
    validate_environment_contract,
    validate_grid_startup,
)


def _unwrap_formal_env(vec_env):
    padded = vec_env.envs[0]
    bridge = padded.env
    multi_agent = bridge.env
    grid = multi_agent.env
    base = grid.env
    return padded, bridge, multi_agent, grid, base


def _task_summary(base_env):
    return tuple(
        (
            task.task_id,
            task.profile_key,
            task.arrival_time,
            task.duration,
            tuple(np.asarray(task.load_profile, dtype=np.float64)),
            task.workload,
            task.deadline,
            task.priority,
            task.interruptible,
            task.parallelizable,
        )
        for task in base_env.tasks
    )


def _fake_grid_vec(*, source="grid.csv", opf_success=True, mef_success=True):
    info = {
        "grid_scenario_enabled": True,
        "grid_scenario_source": source,
        "grid_scenario_message": "Loaded dynamic grid load scale.",
        "grid_opf_success": opf_success,
        "grid_opf_message": "OPF test message",
        "grid_mef_success": mef_success,
        "grid_mef_message": "MEF test message",
    }
    env = SimpleNamespace(
        last_info=info,
        grid_enabled=True,
        grid_scenario_source=source,
        grid_scenario_message=info["grid_scenario_message"],
        opf_mode="ac",
        use_mef=True,
    )
    return SimpleNamespace(envs=[env])


class HarlMAPPOFormalEntryTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.harl_source = Path(
            os.environ.get("HARL_SOURCE_PATH", PROJECT_ROOT.parent / "HARL")
        ).resolve()
        runtime_value = os.environ.get("HARL_RUNTIME_PATH")
        cls.runtime_path = (
            Path(runtime_value).resolve()
            if runtime_value
            else (PROJECT_ROOT / ".tmp_harl_runtime").resolve()
        )
        configure_import_paths(cls.harl_source, cls.runtime_path)

    def test_update_to_num_env_steps_derivation(self) -> None:
        self.assertEqual(derive_num_env_steps(5, 24, 1), 120)
        self.assertEqual(derive_num_env_steps(5, 24, 2), 240)

    def test_resolved_config_keeps_bounded_box_explicit(self) -> None:
        resolved = resolve_config(load_yaml_config(DEFAULT_CONFIG_PATH), updates=5)
        self.assertEqual(resolved["train"]["updates"], 5)
        self.assertEqual(resolved["train"]["n_rollout_threads"], 2)
        self.assertEqual(resolved["train"]["num_env_steps"], 240)
        self.assertIs(resolved["model"]["use_bounded_box_actions"], True)
        self.assertEqual(resolved["algo"]["action_aggregation"], "prod")
        self.assertIs(resolved["algo"]["share_param"], False)

    def test_formal_environment_factory_contract(self) -> None:
        from marl.envs.harl_env_factory import make_harl_eval_env, make_harl_train_env

        config = resolve_config(
            load_yaml_config(DEFAULT_CONFIG_PATH), updates=1, rollout_threads=1
        )
        train_env = make_harl_train_env(
            seed=config["seed"]["seed"],
            n_rollout_threads=1,
            scenario=config["env"]["scenario"],
            experiment_case=config["env"]["experiment_case"],
        )
        eval_env = make_harl_eval_env(
            seed=config["eval"]["seed"],
            n_eval_rollout_threads=1,
            scenario=config["env"]["scenario"],
            experiment_case=config["env"]["experiment_case"],
        )
        try:
            dimensions = validate_environment_contract(train_env, config)
            self.assertEqual(dimensions["n_agents"], 2)
            self.assertEqual(dimensions["agent_order"], ["idc", "bess"])
            self.assertEqual(dimensions["observation_dims"], [288, 288])
            self.assertEqual(dimensions["action_dims"], [22, 22])
            self.assertEqual(dimensions["state_dims"], [364, 364])
            self.assertIsNot(train_env.envs[0], eval_env.envs[0])
        finally:
            train_env.close()
            eval_env.close()

    def test_probe_is_independent_and_preserves_first_task_realization(self) -> None:
        from marl.envs.harl_env_factory import make_harl_train_env

        config = resolve_config(
            load_yaml_config(DEFAULT_CONFIG_PATH), updates=1, rollout_threads=1
        )
        kwargs = {
            "seed": config["seed"]["seed"],
            "n_rollout_threads": 1,
            "scenario": config["env"]["scenario"],
            "experiment_case": config["env"]["experiment_case"],
        }
        baseline = make_harl_train_env(**kwargs)
        created = []

        def recording_factory(**factory_kwargs):
            env = make_harl_train_env(**factory_kwargs)
            created.append(env)
            return env

        train_env = None
        try:
            baseline_obs, baseline_state, _ = baseline.reset()
            baseline_info = baseline.envs[0].last_info
            baseline_base = _unwrap_formal_env(baseline)[-1]
            baseline_tasks = _task_summary(baseline_base)
            baseline_lambda = baseline_base.lambda_t.copy()

            train_env, dimensions, startup = prepare_training_environment(
                config, env_factory=recording_factory
            )
            self.assertEqual(len(created), 2)
            probe_env, fresh_train_env = created
            self.assertIs(train_env, fresh_train_env)
            self.assertIsNone(fresh_train_env.envs[0].last_info)

            probe_chain = _unwrap_formal_env(probe_env)
            train_chain = _unwrap_formal_env(fresh_train_env)
            for probe_object, train_object in zip(probe_chain, train_chain, strict=True):
                self.assertIsNot(probe_object, train_object)
            self.assertIsNot(probe_chain[3].grid_cache, train_chain[3].grid_cache)

            train_obs, train_state, _ = train_env.reset()
            train_info = train_env.envs[0].last_info
            train_base = _unwrap_formal_env(train_env)[-1]
            np.testing.assert_array_equal(train_obs, baseline_obs)
            np.testing.assert_array_equal(train_state, baseline_state)
            np.testing.assert_array_equal(train_base.lambda_t, baseline_lambda)
            self.assertEqual(_task_summary(train_base), baseline_tasks)
            self.assertEqual(
                train_info["total_task_count"], baseline_info["total_task_count"]
            )
            self.assertEqual(
                train_info["initial_backlog_work"],
                baseline_info["initial_backlog_work"],
            )
            self.assertEqual(dimensions["n_agents"], 2)
            self.assertFalse(startup["grid_scenario_source"].lower().startswith("fallback"))
        finally:
            baseline.close()
            if train_env is not None:
                train_env.close()

    def test_main_grid_startup_check_passes(self) -> None:
        from marl.envs.harl_env_factory import make_harl_train_env

        config = resolve_config(load_yaml_config(DEFAULT_CONFIG_PATH), updates=1)
        env = make_harl_train_env(
            seed=config["seed"]["seed"],
            n_rollout_threads=1,
            scenario=config["env"]["scenario"],
            experiment_case=config["env"]["experiment_case"],
        )
        try:
            validate_environment_contract(env, config)
            startup = validate_grid_startup(env)
            self.assertTrue(startup["grid_enabled"])
            self.assertEqual(startup["grid_opf_mode"], "ac")
            self.assertTrue(startup["grid_use_mef"])
            self.assertTrue(startup["initial_opf_success"])
            self.assertTrue(startup["initial_mef_success"])
        finally:
            env.close()

    def test_fallback_blocks_before_fresh_train_environment_is_created(self) -> None:
        from marl.envs.harl_env_factory import make_harl_train_env

        config = resolve_config(load_yaml_config(DEFAULT_CONFIG_PATH), updates=1)
        created = []

        def fallback_factory(**kwargs):
            env = make_harl_train_env(**kwargs)
            created.append(env)
            grid_env = _unwrap_formal_env(env)[3]
            grid_env.grid_scenario_source = "fallback: test-grid.csv"
            grid_env.grid_scenario_message = "Synthetic fallback for startup test."
            return env

        with self.assertRaisesRegex(RuntimeError, "grid_scenario_source_not_fallback"):
            prepare_training_environment(config, env_factory=fallback_factory)
        self.assertEqual(len(created), 1)

    def test_initial_opf_and_mef_failures_block_startup(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "initial_opf_success"):
            validate_grid_startup(_fake_grid_vec(opf_success=False))
        with self.assertRaisesRegex(RuntimeError, "initial_mef_success"):
            validate_grid_startup(_fake_grid_vec(mef_success=False))

    def test_formal_sources_do_not_depend_on_test_modules(self) -> None:
        source_paths = (
            PROJECT_ROOT / "train" / "train_harl_mappo_short.py",
            PROJECT_ROOT / "marl" / "envs" / "harl_env_factory.py",
            PROJECT_ROOT / "marl" / "runners" / "idc_mappo_runner.py",
        )
        forbidden = ("marl.tests", "pytest", "unittest.mock")
        for source_path in source_paths:
            text = source_path.read_text(encoding="utf-8")
            for token in forbidden:
                self.assertNotIn(token, text, f"{source_path} contains {token!r}")

    def test_one_update_formal_cli(self) -> None:
        with tempfile.TemporaryDirectory(prefix="harl-mappo-formal-") as temporary:
            argv = [
                "--config",
                str(DEFAULT_CONFIG_PATH),
                "--harl-source",
                str(self.harl_source),
                "--runtime-path",
                str(self.runtime_path),
                "--seed",
                "7110",
                "--updates",
                "1",
                "--episode-length",
                "24",
                "--rollout-threads",
                "1",
                "--output-dir",
                temporary,
                "--scenario",
                "idc_bess_padding",
                "--device",
                "cpu",
            ]
            result = main(argv)

            self.assertEqual(result["status"], "completed")
            self.assertEqual(result["updates"], 1)
            self.assertEqual(result["num_env_steps"], 24)
            self.assertFalse(result["nan_or_inf_detected"])
            self.assertTrue(result["action_bounds_checks_passed"])
            self.assertTrue(Path(result["resolved_config"]).is_file())
            self.assertTrue(Path(result["run_metadata"]).is_file())
            self.assertEqual(len(result["model_files"]), 3)
            self.assertTrue(all(Path(path).is_file() for path in result["model_files"]))

            metadata = json.loads(Path(result["run_metadata"]).read_text(encoding="utf-8"))
            resolved_snapshot = json.loads(
                Path(result["resolved_config"]).read_text(encoding="utf-8")
            )
            self.assertEqual(metadata["status"], "completed")
            self.assertEqual(metadata["completed_updates"], 1)
            self.assertEqual(
                metadata["harl_expected_git_head"],
                "050ad6a294fe9f7572985dea910d59ea6d4f94b4",
            )
            self.assertEqual(metadata["agent_order"], ["idc", "bess"])
            self.assertEqual(metadata["observation_dims"], [288, 288])
            self.assertEqual(metadata["action_dims"], [22, 22])
            self.assertEqual(
                metadata["action_padding_strategy"],
                "padding-v1-effective-mask",
            )
            self.assertIs(metadata["effective_action_mask_enabled"], True)
            self.assertEqual(metadata["agent_effective_action_dims"], [22, 1])
            self.assertEqual(metadata["agent_padded_action_dims"], [22, 22])
            self.assertEqual(metadata["agent_virtual_action_dims"], [0, 21])
            self.assertEqual(
                metadata["effective_action_masks"],
                [[1.0] * 22, [1.0] + [0.0] * 21],
            )
            idc_update, bess_update = metadata["effective_action_update_diagnostics"]
            self.assertEqual(idc_update["effective_action_dim"], 22)
            self.assertEqual(bess_update["effective_action_dim"], 1)
            self.assertEqual(bess_update["effective_physical_ratio_max_diff"], 0.0)
            self.assertEqual(bess_update["clip_disagreement_fraction"], 0.0)
            self.assertEqual(bess_update["virtual_mean_rows_grad_norm"], 0.0)
            self.assertEqual(bess_update["virtual_log_std_grad_norm"], 0.0)
            self.assertGreater(bess_update["physical_mean_row_grad_norm"], 0.0)
            self.assertGreater(bess_update["physical_log_std_grad_abs"], 0.0)
            self.assertGreater(bess_update["shared_trunk_grad_norm"], 0.0)
            self.assertTrue(idc_update["all_finite"])
            self.assertTrue(bess_update["all_finite"])
            self.assertEqual(metadata["state_dim"], 364)
            self.assertEqual(metadata["state_dims"], [364, 364])
            self.assertIs(metadata["use_bounded_box_actions"], True)
            expected_runtime_metadata = {
                "harl_scenario": "idc_bess_padding",
                "experiment_case": "main",
                "algorithm_seed": 7110,
                "environment_seed": 7110,
                "server_seed": 7110,
                "task_seed": 7110,
                "rng_isolation_version": "critic-init-isolation-v1",
                "critic_init_seed_rule": "base_seed + critic_init_seed_offset",
                "critic_init_seed": 207110,
                "actor_sampling_rng_isolated_from_critic_init": True,
            }
            for key, expected in expected_runtime_metadata.items():
                self.assertEqual(metadata[key], expected)
                self.assertEqual(resolved_snapshot["runtime"][key], expected)
            self.assertFalse(metadata["grid_scenario_source"].lower().startswith("fallback"))
            self.assertEqual(
                metadata["grid_scenario_source"],
                resolved_snapshot["runtime"]["grid_scenario_source"],
            )
            self.assertEqual(
                metadata["grid_scenario_message"],
                resolved_snapshot["runtime"]["grid_scenario_message"],
            )
            self.assertEqual(metadata["logger_version"], LOGGER_VERSION)
            self.assertEqual(metadata["metric_schema_version"], METRIC_SCHEMA_VERSION)
            self.assertTrue(metadata["step_logging_enabled"])
            self.assertTrue(metadata["episode_logging_enabled"])
            self.assertTrue(metadata["tensorboard_enabled"])
            self.assertEqual(metadata["reward_aliases"], {"r_grid_peak": "r_peak_load"})
            self.assertFalse(metadata["grid_reward_enabled"])
            self.assertFalse(metadata["safe_rl_enabled"])
            self.assertEqual(len(metadata["grid_scenario_sha256"]), 64)

            metrics_dir = Path(result["run_dir"]) / "metrics"
            step_path = metrics_dir / "step_metrics.csv"
            episode_path = metrics_dir / "episode_metrics.csv"
            update_path = metrics_dir / "update_metrics.csv"
            schema_path = metrics_dir / "metric_schema.json"
            summary_path = metrics_dir / "run_summary.json"
            for path in (step_path, episode_path, update_path, schema_path, summary_path):
                self.assertTrue(path.is_file(), path)
            with step_path.open(encoding="utf-8", newline="") as stream:
                step_rows = list(csv.DictReader(stream))
            with episode_path.open(encoding="utf-8", newline="") as stream:
                episode_rows = list(csv.DictReader(stream))
            with update_path.open(encoding="utf-8", newline="") as stream:
                update_rows = list(csv.DictReader(stream))
            self.assertEqual(len(step_rows), 24)
            self.assertEqual(len(episode_rows), 1)
            self.assertEqual(len(update_rows), 1)
            episode_reward = float(episode_rows[0]["episode_reward"])
            self.assertAlmostEqual(
                sum(float(row["reward_returned"]) for row in step_rows),
                episode_reward,
                places=6,
            )
            self.assertAlmostEqual(
                float(update_rows[0]["rollout_episode_reward_mean"]),
                episode_reward,
                places=6,
            )
            for component in metadata["reward_component_names"]:
                self.assertAlmostEqual(
                    sum(float(row[component]) for row in step_rows),
                    float(episode_rows[0][f"{component}_sum"]),
                    places=6,
                )
            self.assertTrue(step_rows[-1]["is_terminal"].lower() == "true")
            self.assertAlmostEqual(
                float(step_rows[-1]["r_final_queue"]),
                float(episode_rows[0]["r_final_queue_sum"]),
                places=6,
            )
            self.assertAlmostEqual(
                float(step_rows[-1]["r_final_soc"]),
                float(episode_rows[0]["r_final_soc_sum"]),
                places=6,
            )
            self.assertEqual(float(update_rows[0]["shared_reward_max_diff"]), 0.0)
            self.assertEqual(
                float(update_rows[0]["bess_effective_physical_ratio_max_diff"]), 0.0
            )
            self.assertEqual(float(update_rows[0]["bess_virtual_mean_grad_norm"]), 0.0)
            self.assertEqual(
                float(update_rows[0]["bess_virtual_log_std_grad_norm"]), 0.0
            )
            schema = json.loads(schema_path.read_text(encoding="utf-8"))
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            self.assertEqual(schema["metric_schema_version"], METRIC_SCHEMA_VERSION)
            self.assertEqual(summary["updates_completed"], 1)
            self.assertEqual(summary["total_environment_steps"], 24)
            self.assertEqual(summary["episodes_completed"], 1)
            self.assertFalse(summary["nan_or_inf_detected"])
            self.assertEqual(summary["termination_reason"], "requested_updates_completed")
            self.assertTrue(list((Path(result["run_dir"]) / "logs").glob("events*")))


if __name__ == "__main__":
    unittest.main()
