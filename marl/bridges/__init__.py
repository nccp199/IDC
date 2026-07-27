"""Framework-specific adapters over the generic multi-agent environment."""

from marl.bridges.harl_bridge import HARL_UPSTREAM_COMMIT, HarlIDCGridBridge
from marl.bridges.harl_padded_bridge import HarlPaddedBridge

__all__ = ["HARL_UPSTREAM_COMMIT", "HarlIDCGridBridge", "HarlPaddedBridge"]
