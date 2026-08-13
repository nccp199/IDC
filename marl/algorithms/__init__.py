"""Project-owned algorithm adapters for heterogeneous effective actions."""

from marl.algorithms.effective_action_mask import (
    EFFECTIVE_ACTION_ALGO_REGISTRY,
    EffectiveActionHAPPO,
    EffectiveActionMAPPO,
    effective_entropy_from_log_probs,
    effective_ratio_from_log_probs,
)
from marl.algorithms.update_strategy import (
    AlgorithmUpdateStrategy,
    HAPPOUpdateStrategy,
    MAPPOUpdateStrategy,
    build_update_strategy,
)

__all__ = [
    "EFFECTIVE_ACTION_ALGO_REGISTRY",
    "EffectiveActionHAPPO",
    "EffectiveActionMAPPO",
    "effective_entropy_from_log_probs",
    "effective_ratio_from_log_probs",
    "AlgorithmUpdateStrategy",
    "HAPPOUpdateStrategy",
    "MAPPOUpdateStrategy",
    "build_update_strategy",
]
