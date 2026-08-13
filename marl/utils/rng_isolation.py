"""Exception-safe global RNG isolation for deterministic model construction."""

from __future__ import annotations

import copy
import random
from contextlib import contextmanager
from typing import Any, Callable, Iterator, TypeVar

import numpy as np
import torch


RNG_ISOLATION_VERSION = "critic-init-isolation-v1"
CRITIC_INIT_SEED_RULE = "base_seed + critic_init_seed_offset"
DEFAULT_CRITIC_INIT_SEED_OFFSET = 200_000

_T = TypeVar("_T")


def capture_global_rng_state() -> dict[str, Any]:
    """Capture every global RNG family that Critic construction may consume."""
    cuda_available = bool(torch.cuda.is_available())
    return {
        "python": random.getstate(),
        "numpy": copy.deepcopy(np.random.get_state()),
        "torch_cpu": torch.get_rng_state().clone(),
        "torch_cuda": (
            [state.clone() for state in torch.cuda.get_rng_state_all()]
            if cuda_available
            else None
        ),
        "cuda_available": cuda_available,
    }


def restore_global_rng_state(state: dict[str, Any]) -> None:
    """Restore a state returned by :func:`capture_global_rng_state`."""
    random.setstate(state["python"])
    np.random.set_state(copy.deepcopy(state["numpy"]))
    torch.set_rng_state(state["torch_cpu"].cpu())
    cuda_states = state.get("torch_cuda")
    if cuda_states is not None:
        if not torch.cuda.is_available():
            raise RuntimeError(
                "Cannot restore captured CUDA RNG state because CUDA is unavailable."
            )
        torch.cuda.set_rng_state_all([value.cpu() for value in cuda_states])


def _seed_isolated_scope(seed: int) -> None:
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError(f"Isolated RNG seed must be a non-negative integer, got {seed!r}.")
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


@contextmanager
def isolated_global_rng(seed: int) -> Iterator[None]:
    """Run a scope from ``seed`` and always restore the caller's RNG streams."""
    outer_state = capture_global_rng_state()
    try:
        _seed_isolated_scope(seed)
        yield
    finally:
        restore_global_rng_state(outer_state)


def build_with_isolated_rng(builder: Callable[[], _T], *, seed: int) -> _T:
    """Build and return an object without changing any caller-visible RNG state."""
    if not callable(builder):
        raise TypeError("builder must be callable.")
    with isolated_global_rng(seed):
        return builder()
