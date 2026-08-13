"""Unified algorithm/critic method identities."""

from .method_registry import (
    ALGORITHM_IMPLEMENTATION_VERSIONS,
    CRITIC_INTERFACE_VERSION,
    MethodSpec,
    derive_method,
)

__all__ = [
    "ALGORITHM_IMPLEMENTATION_VERSIONS",
    "CRITIC_INTERFACE_VERSION",
    "MethodSpec",
    "derive_method",
]
