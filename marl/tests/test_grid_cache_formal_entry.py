from __future__ import annotations

import copy
import functools
import json
import os
import sys
import time
import unittest
import warnings
from pathlib import Path
from unittest.mock import patch

import numpy as np

from configs.config_ultimate import GRID_CACHE_CONFIG
from configs.experiment_cases import get_experiment_case
from grid_model.grid_cache import GridResultCache
from grid_model.grid_case import MEFResult, OPFResult
from train.train_ppo_ultimate import make_unmonitored_env

import env_wrappers.grid_coupled_env as grid_wrapper_module
import grid_model.mef_calculator as mef_module


PROJECT_ROOT = Path(__file__).resolve().parents[2]
HARL_SOURCE = Path(
    os.environ.get("HARL_SOURCE_PATH", PROJECT_ROOT.parent / "HARL")
).resolve()
if str(HARL_SOURCE) not in sys.path:
    sys.path.insert(0, str(HARL_SOURCE))

SEED = 7110
SCENARIO = "idc_bess_padding"
EXPERIMENT_CASE = "main"


def _make_grid_env(*, cache_enabled: bool = True, cache_config=None):
    case = get_experiment_case(EXPERIMENT_CASE)
    config = copy.deepcopy(GRID_CACHE_CONFIG)
    config["enable_grid_cache"] = bool(cache_enabled)
    if cache_config:
        config.update(cache_config)
    return make_unmonitored_env(
        case["env_config"],
        case["reward_config"],
        case["data_config"],
        seed=SEED,
        grid_cache_config=config,
    )


def _unwrap_grid_env(vec_env):
    return vec_env.envs[0].env.env.env


def _fixed_actions() -> list[np.ndarray]:
    actions = []
    for step in range(24):
        action = np.full(23, 0.5, dtype=np.float32)
        action[:20] = np.clip(
            0.42 + 0.28 * np.sin((step + np.arange(20) * 0.31) * 0.47),
            0.08,
            0.92,
        )
        action[20] = 0.25 + 0.5 * ((step % 6) / 5.0)
        action[21] = 0.75 - 0.5 * ((step % 5) / 4.0)
        action[22] = (0.18, 0.82, 0.50)[step % 3]
        actions.append(action)
    return actions


_PHYSICAL_FIELDS = (
    "bess_soc",
    "bess_charge_power_kW",
    "bess_discharge_power_kW",
    "P_IDC_kW",
    "P_grid_kW",
    "cost",
    "carbon_emission",
    "grid_lmp",
    "grid_mef_plus",
    "grid_mef_minus",
    "grid_min_voltage_pu",
    "grid_max_voltage_pu",
    "grid_max_line_loading_percent",
    "grid_network_loss_mw",
    "task_completion_rate",
    "backlog_work",
)


@functools.lru_cache(maxsize=1)
def _episode_regression_result():
    warnings.filterwarnings("ignore")
    cache_off = _make_grid_env(cache_enabled=False)
    cache_on = _make_grid_env(cache_enabled=True)
    actions = _fixed_actions()
    original_task_rng_state = copy.deepcopy(
        cache_on.env.model.task_rng.bit_generator.state
    )

    original_opf = grid_wrapper_module.solve_opf
    original_mef = grid_wrapper_module.calculate_nodal_mef
    original_mef_opf = mef_module.solve_opf
    counters = {"wrapper_opf": 0, "mef": 0, "mef_internal_opf": 0}

    def counted_opf(*args, **kwargs):
        counters["wrapper_opf"] += 1
        return original_opf(*args, **kwargs)

    def counted_mef(*args, **kwargs):
        counters["mef"] += 1
        return original_mef(*args, **kwargs)

    def counted_mef_opf(*args, **kwargs):
        counters["mef_internal_opf"] += 1
        return original_mef_opf(*args, **kwargs)

    def run_episode(env):
        rows = []
        started = time.perf_counter()
        _, reset_info = env.reset(seed=SEED)
        for action in actions:
            _, reward, terminated, truncated, info = env.step(action)
            rows.append(
                {
                    "reward": float(reward),
                    **{field: info[field] for field in _PHYSICAL_FIELDS},
                    "grid_opf_success": bool(info["grid_opf_success"]),
                    "grid_mef_success": bool(info["grid_mef_success"]),
                    "done": bool(terminated or truncated),
                }
            )
        return {
            "elapsed": time.perf_counter() - started,
            "rows": rows,
            "reset_info": reset_info,
            "stats": env.grid_cache.stats(),
        }

    def delta(before, after):
        return {key: after[key] - before[key] for key in before}

    try:
        with (
            patch.object(grid_wrapper_module, "solve_opf", side_effect=counted_opf),
            patch.object(
                grid_wrapper_module,
                "calculate_nodal_mef",
                side_effect=counted_mef,
            ),
            patch.object(mef_module, "solve_opf", side_effect=counted_mef_opf),
        ):
            before = dict(counters)
            off_run = run_episode(cache_off)
            after_off = dict(counters)

            cache_on.env.model.task_rng.bit_generator.state = copy.deepcopy(
                original_task_rng_state
            )
            on_first = run_episode(cache_on)
            after_first = dict(counters)

            cache_on.env.model.task_rng.bit_generator.state = copy.deepcopy(
                original_task_rng_state
            )
            before_repeat_stats = cache_on.grid_cache.stats()
            on_repeat = run_episode(cache_on)
            after_repeat = dict(counters)

        max_differences = {}
        for field in ("reward", *_PHYSICAL_FIELDS):
            max_differences[field] = max(
                abs(float(left[field]) - float(right[field]))
                for left, right in zip(
                    off_run["rows"], on_first["rows"], strict=True
                )
            )
        discrete_equal = {
            field: all(
                left[field] == right[field]
                for left, right in zip(
                    off_run["rows"], on_first["rows"], strict=True
                )
            )
            for field in ("grid_opf_success", "grid_mef_success", "done")
        }
        return {
            "max_differences": max_differences,
            "discrete_equal": discrete_equal,
            "cache_off": {
                "elapsed": off_run["elapsed"],
                "calls": delta(before, after_off),
                "stats": off_run["stats"],
            },
            "cache_on_first": {
                "elapsed": on_first["elapsed"],
                "calls": delta(after_off, after_first),
                "stats": on_first["stats"],
                "mef_info_bin": on_first["reset_info"][
                    "grid_cache_mef_load_bin_mw"
                ],
            },
            "cache_on_repeat": {
                "elapsed": on_repeat["elapsed"],
                "calls": delta(after_first, after_repeat),
                "stats_before": before_repeat_stats,
                "stats_after": on_repeat["stats"],
            },
        }
    finally:
        cache_off.close()
        cache_on.close()


class GridCacheFormalEntryTest(unittest.TestCase):
    def test_config_stats_and_legacy_fallback(self) -> None:
        default_cache = GridResultCache()
        self.assertEqual(default_cache.cache_mef_load_bin_mw, 0.01)

        formal = GridResultCache(GRID_CACHE_CONFIG)
        self.assertEqual(formal.cache_load_bin_mw, 0.1)
        self.assertEqual(formal.cache_mef_load_bin_mw, 0.01)
        self.assertEqual(formal.float_digits, 6)
        self.assertEqual(formal.stats()["cache_mef_load_bin_mw"], 0.01)

        legacy = GridResultCache({"cache_load_bin_mw": 0.2})
        self.assertEqual(legacy.cache_mef_load_bin_mw, 0.2)
        self.assertEqual(legacy.config["cache_mef_load_bin_mw"], 0.2)

    def test_opf_key_rule_is_unchanged(self) -> None:
        cache = GridResultCache(GRID_CACHE_CONFIG)
        zero = cache.make_opf_key("ac", 8, 0, 1.0, 0.0)
        nearby = cache.make_opf_key("ac", 8, 0, 1.0, 0.04)
        self.assertEqual(zero, nearby)
        self.assertEqual(zero, ("opf", "ac", 8, 0, 1.0, 0.0))

    def test_mef_boundary_loads_use_distinct_exact_keys(self) -> None:
        cache = GridResultCache(GRID_CACHE_CONFIG)
        keys = {
            load: cache.make_mef_key("ac", 8, 0, 1.0, load, 0.1)
            for load in (0.0, 0.03, 0.04, 0.1)
        }
        self.assertEqual(len(set(keys.values())), len(keys))
        for load, key in keys.items():
            self.assertEqual(key[5], ("exact", round(load, 6)))
        self.assertNotEqual(keys[0.0], keys[0.04])

    def test_mef_normal_region_uses_independent_bin(self) -> None:
        cache = GridResultCache(GRID_CACHE_CONFIG)
        first = cache.make_mef_key("ac", 8, 2, 1.0, 0.201, 0.1)
        same_bin = cache.make_mef_key("ac", 8, 2, 1.0, 0.204, 0.1)
        next_bin = cache.make_mef_key("ac", 8, 2, 1.0, 0.206, 0.1)
        self.assertEqual(first[5], ("binned", 0.2))
        self.assertEqual(first, same_bin)
        self.assertEqual(next_bin[5], ("binned", 0.21))
        self.assertNotEqual(first, next_bin)

    def test_delta_stays_in_key_and_moves_boundary(self) -> None:
        cache = GridResultCache(GRID_CACHE_CONFIG)
        exact = cache.make_mef_key("ac", 8, 2, 1.0, 0.08, 0.1)
        binned = cache.make_mef_key("ac", 8, 2, 1.0, 0.08, 0.05)
        self.assertEqual(exact[5], ("exact", 0.08))
        self.assertEqual(binned[5], ("binned", 0.08))
        self.assertEqual(exact[-1], 0.1)
        self.assertEqual(binned[-1], 0.05)
        self.assertNotEqual(exact, binned)

    def test_zero_and_point_zero_four_real_mef_do_not_cross_hit(self) -> None:
        env = _make_grid_env(cache_enabled=True)
        try:
            results = {}
            for order in ((0.0, 0.04), (0.04, 0.0)):
                env.grid_cache.clear()
                first, first_hit = env._calculate_mef_with_cache(order[0], 0, 1.0)
                second, second_hit = env._calculate_mef_with_cache(order[1], 0, 1.0)
                exact_second = grid_wrapper_module.calculate_nodal_mef(
                    grid_case=env.grid_case,
                    bus_id=env.idc_bus_idx,
                    mode=env.opf_mode,
                    delta_p_mw=env.delta_p_mw,
                    load_scale=1.0,
                    base_idc_load_mw=order[1],
                    clamp_minus_load=True,
                    gen_emission_factors_kg_per_mwh=env.gen_emission_factors,
                )
                self.assertFalse(first_hit)
                self.assertFalse(second_hit)
                self.assertAlmostEqual(
                    second.mef_minus_kg_per_mwh,
                    exact_second.mef_minus_kg_per_mwh,
                    places=9,
                )
                self.assertEqual(env.grid_cache.stats()["mef_size"], 2)
                results[str(order)] = {
                    "first_minus": first.mef_minus_kg_per_mwh,
                    "second_minus": second.mef_minus_kg_per_mwh,
                }

            keys = {
                str(load): env.grid_cache.make_mef_key(
                    env.opf_mode,
                    env.idc_bus_idx,
                    0,
                    1.0,
                    load,
                    env.delta_p_mw,
                )
                for load in (0.0, 0.03, 0.04, 0.1)
            }
            print(json.dumps({"mef_boundary_results": results, "keys": keys}))
        finally:
            env.close()

    def test_identical_and_normal_binned_mef_states_hit(self) -> None:
        env = _make_grid_env(cache_enabled=True)
        original = grid_wrapper_module.calculate_nodal_mef
        calls = {"count": 0}

        def counted(*args, **kwargs):
            calls["count"] += 1
            return original(*args, **kwargs)

        try:
            env.grid_cache.clear()
            with patch.object(
                grid_wrapper_module, "calculate_nodal_mef", side_effect=counted
            ):
                _, first_hit = env._calculate_mef_with_cache(0.04, 0, 1.0)
                _, second_hit = env._calculate_mef_with_cache(0.04, 0, 1.0)
            self.assertEqual((first_hit, second_hit), (False, True))
            self.assertEqual(calls["count"], 1)

            env.grid_cache.clear()
            calls["count"] = 0
            with patch.object(
                grid_wrapper_module, "calculate_nodal_mef", side_effect=counted
            ):
                _, first_hit = env._calculate_mef_with_cache(0.201, 2, 1.0)
                _, same_bin_hit = env._calculate_mef_with_cache(0.204, 2, 1.0)
            self.assertEqual((first_hit, same_bin_hit), (False, True))
            self.assertEqual(calls["count"], 1)
        finally:
            env.close()

    def test_failed_opf_and_mef_results_are_not_cached(self) -> None:
        env = _make_grid_env(cache_enabled=True)
        opf_calls = {"count": 0}
        mef_calls = {"count": 0}

        def failed_opf(*args, **kwargs):
            opf_calls["count"] += 1
            return OPFResult(False, "ac", "synthetic OPF failure")

        def failed_mef(*args, **kwargs):
            mef_calls["count"] += 1
            return MEFResult(
                False, "ac", env.idc_bus_idx, env.delta_p_mw, message="failure"
            )

        try:
            env.grid_cache.clear()
            with patch.object(grid_wrapper_module, "solve_opf", side_effect=failed_opf):
                env._solve_opf_with_cache(0.5, 3, 1.0)
                env._solve_opf_with_cache(0.5, 3, 1.0)
            with patch.object(
                grid_wrapper_module,
                "calculate_nodal_mef",
                side_effect=failed_mef,
            ):
                env._calculate_mef_with_cache(0.5, 3, 1.0)
                env._calculate_mef_with_cache(0.5, 3, 1.0)
            stats = env.grid_cache.stats()
            self.assertEqual(opf_calls["count"], 2)
            self.assertEqual(mef_calls["count"], 2)
            self.assertEqual(stats["opf_size"], 0)
            self.assertEqual(stats["mef_size"], 0)
        finally:
            env.close()

    def test_probe_train_and_independent_formal_envs_have_isolated_caches(self) -> None:
        from marl.envs.harl_env_factory import make_harl_train_env

        kwargs = {
            "seed": SEED,
            "n_rollout_threads": 1,
            "scenario": SCENARIO,
            "experiment_case": EXPERIMENT_CASE,
        }
        probe = make_harl_train_env(**kwargs)
        train_a = None
        train_b = None
        try:
            probe.reset()
            probe_grid = _unwrap_grid_env(probe)
            train_a = make_harl_train_env(**kwargs)
            train_b = make_harl_train_env(**kwargs)
            train_a_grid = _unwrap_grid_env(train_a)
            train_b_grid = _unwrap_grid_env(train_b)
            self.assertEqual(
                len(
                    {
                        id(probe_grid.grid_cache),
                        id(train_a_grid.grid_cache),
                        id(train_b_grid.grid_cache),
                    }
                ),
                3,
            )
            self.assertEqual(probe_grid.grid_cache.stats()["mef_size"], 1)
            self.assertEqual(train_a_grid.grid_cache.stats()["mef_size"], 0)
            self.assertEqual(train_b_grid.grid_cache.stats()["mef_size"], 0)
            self.assertIsNone(train_a.envs[0].last_info)
            self.assertIsNone(train_b.envs[0].last_info)
        finally:
            probe.close()
            if train_a is not None:
                train_a.close()
            if train_b is not None:
                train_b.close()

    def test_24_step_cache_on_off_physics_match(self) -> None:
        result = _episode_regression_result()
        for field, difference in result["max_differences"].items():
            self.assertLessEqual(difference, 1e-9, field)
        self.assertTrue(all(result["discrete_equal"].values()))
        self.assertEqual(result["cache_on_first"]["mef_info_bin"], 0.01)

    def test_repeated_episode_preserves_cache_performance(self) -> None:
        result = _episode_regression_result()
        off_calls = result["cache_off"]["calls"]
        first_calls = result["cache_on_first"]["calls"]
        repeat_calls = result["cache_on_repeat"]["calls"]
        self.assertEqual(off_calls, first_calls)
        self.assertGreater(off_calls["wrapper_opf"], 0)
        self.assertGreater(off_calls["mef_internal_opf"], 0)
        self.assertEqual(repeat_calls["wrapper_opf"], 0)
        self.assertEqual(repeat_calls["mef"], 0)
        self.assertEqual(repeat_calls["mef_internal_opf"], 0)
        before = result["cache_on_repeat"]["stats_before"]
        after = result["cache_on_repeat"]["stats_after"]
        self.assertEqual(after["opf_hit_count"] - before["opf_hit_count"], 25)
        self.assertEqual(after["mef_hit_count"] - before["mef_hit_count"], 25)
        print(json.dumps({"episode_performance": result}, default=float))


if __name__ == "__main__":
    unittest.main()
