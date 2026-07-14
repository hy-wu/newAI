"""Re-exporting DesignAgentInterface from upper-level Feynman.agent package for backward compatibility."""

from Feynman.agent.design_agent import (
    DesignAgentInterface, MutationAction, MutationType, Observation
)

__all__ = [
    "DesignAgentInterface",
    "MutationAction",
    "MutationType",
    "Observation",
]
