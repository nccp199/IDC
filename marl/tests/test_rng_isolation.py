"""Unit gates for Critic-initialization RNG isolation."""

from __future__ import annotations

import copy
import random
from pathlib import Path
from typing import Any, Mapping

import gymnasium as gym
import numpy as np
import pytest
import torch

from harl.utils.envs_tools import set_seed
from marl.algorithms import EFFECTIVE_ACTION_ALGO_REGISTRY
from marl.critics import build_critic
from marl.graphs import HGTAGraphBuilder, build_graph_schema
from marl.specs import AGENTS, EFFECTIVE_ACTION_DIMS
from marl.utils.rng_isolation import (
    build_with_isolated_rng,
    capture_global_rng_state,
    isolated_global_rng,
)
from train.train_harl_mappo_short import load_yaml_config, resolve_config


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CRITIC_SEED = 207_110


def _spaces():
    return (
        gym.spaces.Box(-np.inf, np.inf, shape=(288,), dtype=np.float32),
        gym.spaces.Box(0.0, 1.0, shape=(22,), dtype=np.float32),
        gym.spaces.Box(-np.inf, np.inf, shape=(364,), dtype=np.float32),
    )


def _resolved(algorithm: str, critic_type: str) -> dict[str, Any]:
    return resolve_config(
        load_yaml_config(PROJECT_ROOT / "configs" / "harl_mappo_short.yaml"),
        algorithm=algorithm,
        critic_type=critic_type,
        seed=7110,
        updates=1,
        episode_length=24,
        rollout_threads=2,
        device="cpu",
        checkpoint_interval=1,
    )


def _build_critic(config: Mapping[str, Any]):
    _, _, state_space = _spaces()
    critic_args = {**config["model"], **config["algo"]}
    if config["critic"]["type"] == "hgta":
        critic_args["hgta"] = dict(config["critic"]["hgta"])
    return build_critic(
        config["critic"]["type"], critic_args, state_space, device=torch.device("cpu")
    )


def _assert_rng_state_exact(left: Mapping[str, Any], right: Mapping[str, Any]) -> None:
    assert left["python"] == right["python"]
    assert left["numpy"][0] == right["numpy"][0]
    assert np.array_equal(left["numpy"][1], right["numpy"][1])
    assert left["numpy"][2:] == right["numpy"][2:]
    assert torch.equal(left["torch_cpu"], right["torch_cpu"])
    assert left["cuda_available"] == right["cuda_available"]
    if left["torch_cuda"] is None:
        assert right["torch_cuda"] is None
    else:
        assert len(left["torch_cuda"]) == len(right["torch_cuda"])
        assert all(
            torch.equal(a, b)
            for a, b in zip(left["torch_cuda"], right["torch_cuda"], strict=True)
        )


def _assert_state_dict_exact(left: Mapping[str, torch.Tensor], right: Mapping[str, torch.Tensor]) -> None:
    assert list(left) == list(right)
    for key in left:
        assert left[key].dtype == right[key].dtype
        assert left[key].shape == right[key].shape
        assert torch.equal(left[key], right[key]), key


@pytest.mark.parametrize("critic_type", ["mlp", "hgta"])
def test_critic_construction_restores_all_global_rng_states(critic_type: str) -> None:
    config = _resolved("happo" if critic_type == "hgta" else "mappo", critic_type)
    set_seed({"seed_specify": True, "seed": 9917})
    before = capture_global_rng_state()
    critic = build_with_isolated_rng(lambda: _build_critic(config), seed=CRITIC_SEED)
    after = capture_global_rng_state()
    assert critic is not None
    _assert_rng_state_exact(before, after)


def test_mlp_and_hgta_leave_identical_subsequent_random_sequence() -> None:
    outputs = []
    for critic_type in ("mlp", "hgta"):
        config = _resolved("happo" if critic_type == "hgta" else "mappo", critic_type)
        set_seed({"seed_specify": True, "seed": 7721})
        build_with_isolated_rng(lambda: _build_critic(config), seed=CRITIC_SEED)
        outputs.append(
            {
                "python": [random.random() for _ in range(8)],
                "numpy": np.random.random(8),
                "torch": torch.rand(8),
            }
        )
    assert outputs[0]["python"] == outputs[1]["python"]
    assert np.array_equal(outputs[0]["numpy"], outputs[1]["numpy"])
    assert torch.equal(outputs[0]["torch"], outputs[1]["torch"])


@pytest.mark.parametrize("critic_type", ["mlp", "hgta"])
def test_same_critic_type_is_reproducible_from_independent_seed(critic_type: str) -> None:
    config = _resolved("happo" if critic_type == "hgta" else "mappo", critic_type)
    left = build_with_isolated_rng(lambda: _build_critic(config), seed=CRITIC_SEED)
    torch.rand(100)
    np.random.random(100)
    random.random()
    right = build_with_isolated_rng(lambda: _build_critic(config), seed=CRITIC_SEED)
    _assert_state_dict_exact(left.state_dict(), right.state_dict())


def test_exception_path_restores_all_global_rng_states() -> None:
    set_seed({"seed_specify": True, "seed": 8621})
    before = capture_global_rng_state()
    with pytest.raises(RuntimeError, match="intentional isolated failure"):
        with isolated_global_rng(CRITIC_SEED):
            random.random()
            np.random.random(16)
            torch.rand(16)
            raise RuntimeError("intentional isolated failure")
    after = capture_global_rng_state()
    _assert_rng_state_exact(before, after)


def test_actor_parameters_and_optimizer_state_do_not_depend_on_method_or_critic() -> None:
    obs_space, action_space, state_space = _spaces()
    records = []
    for algorithm, critic_type in (("mappo", "mlp"), ("happo", "mlp"), ("happo", "hgta")):
        config = _resolved(algorithm, critic_type)
        set_seed(config["seed"])
        actor_class = EFFECTIVE_ACTION_ALGO_REGISTRY[algorithm]
        actors = [
            actor_class(
                {**config["model"], **config["algo"]},
                obs_space,
                action_space,
                device=torch.device("cpu"),
                effective_action_dim=EFFECTIVE_ACTION_DIMS[agent],
            )
            for agent in AGENTS
        ]
        critic_args = {**config["model"], **config["algo"]}
        if critic_type == "hgta":
            critic_args["hgta"] = dict(config["critic"]["hgta"])
        build_with_isolated_rng(
            lambda: build_critic(
                critic_type, critic_args, state_space, device=torch.device("cpu")
            ),
            seed=int(config["rng"]["critic_init_seed"]),
        )
        records.append(
            (
                [copy.deepcopy(actor.actor.state_dict()) for actor in actors],
                [copy.deepcopy(actor.actor_optimizer.state_dict()) for actor in actors],
            )
        )
    for candidate in records[1:]:
        for agent_id in range(2):
            _assert_state_dict_exact(records[0][0][agent_id], candidate[0][agent_id])
            assert records[0][1][agent_id] == candidate[1][agent_id]


def test_schema_builder_and_optimizer_construction_do_not_consume_rng(hgta_config) -> None:
    set_seed({"seed_specify": True, "seed": 9317})
    before_graph = capture_global_rng_state()
    schema = build_graph_schema(hgta_config)
    HGTAGraphBuilder(schema)
    after_graph = capture_global_rng_state()
    _assert_rng_state_exact(before_graph, after_graph)

    parameter = torch.nn.Parameter(torch.zeros(3))
    before_optimizer = capture_global_rng_state()
    torch.optim.Adam([parameter], lr=5e-4)
    after_optimizer = capture_global_rng_state()
    _assert_rng_state_exact(before_optimizer, after_optimizer)
