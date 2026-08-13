"""Centralized critic adapters and construction."""

from .critic_factory import MLPCentralizedCritic, build_critic

__all__ = ["MLPCentralizedCritic", "build_critic"]
