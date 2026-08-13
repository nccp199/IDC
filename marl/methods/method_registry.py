"""Explicit method registry; no class-name or filename inference is allowed."""

from __future__ import annotations

from dataclasses import dataclass


CRITIC_INTERFACE_VERSION = "idc-centralized-critic-v1"
ALGORITHM_IMPLEMENTATION_VERSIONS = {
    "mappo": "idc-effective-action-mappo-v1",
    "happo": "idc-effective-action-happo-v1",
}


@dataclass(frozen=True)
class MethodSpec:
    algorithm_name: str
    critic_type: str
    method_id: str
    implemented: bool
    algorithm_implementation_version: str
    critic_interface_version: str = CRITIC_INTERFACE_VERSION

    def require_implemented(self) -> "MethodSpec":
        if not self.implemented:
            raise NotImplementedError(f"Method {self.method_id} is not implemented.")
        return self

    def as_dict(self) -> dict[str, str | bool]:
        return {
            "algorithm_name": self.algorithm_name,
            "critic_type": self.critic_type,
            "method_id": self.method_id,
            "algorithm_implementation_version": self.algorithm_implementation_version,
            "critic_interface_version": self.critic_interface_version,
            "implemented": self.implemented,
        }


_METHODS = {
    ("mappo", "mlp"): MethodSpec(
        "mappo", "mlp", "MAPPO_MLP", True, ALGORITHM_IMPLEMENTATION_VERSIONS["mappo"]
    ),
    ("happo", "mlp"): MethodSpec(
        "happo", "mlp", "HAPPO_MLP", True, ALGORITHM_IMPLEMENTATION_VERSIONS["happo"]
    ),
    ("happo", "hgta"): MethodSpec(
        "happo", "hgta", "HAPPO_HGTA", True, ALGORITHM_IMPLEMENTATION_VERSIONS["happo"]
    ),
}


def derive_method(
    algorithm_name: str, critic_type: str, *, require_implemented: bool = True
) -> MethodSpec:
    """Resolve an explicit algorithm/critic pair and fail fast on unsupported pairs."""
    algorithm = str(algorithm_name).strip().lower()
    critic = str(critic_type).strip().lower()
    try:
        method = _METHODS[(algorithm, critic)]
    except KeyError as exc:
        raise ValueError(
            f"Unsupported algorithm/critic combination: algorithm={algorithm!r}, "
            f"critic_type={critic!r}."
        ) from exc
    return method.require_implemented() if require_implemented else method
