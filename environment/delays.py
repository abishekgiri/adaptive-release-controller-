"""Configurable delayed-reward model; rewards arrive k ~ Geom(p) steps after the action."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from data.schemas import Action, Outcome, Reward


@dataclass(frozen=True)
class DelayConfig:
    """Parameters controlling the reward-delay distribution."""

    geom_p: float = 0.3          # success probability of the geometric delay distribution
    max_delay: int = 50          # cap on delay; rewards beyond this are treated as censored
    censoring_rate: float = 0.0  # fraction of rewards that are permanently censored


class DelayedRewardSampler:
    """Samples reward-arrival delays and tracks pending (action_id, reveal_step) pairs."""

    def __init__(self, config: DelayConfig, rng: np.random.Generator) -> None:
        # TODO: store config and rng
        self._config = config
        self._rng = rng

    def sample_delay(self) -> int:
        """Draw k ~ Geom(p), capped at max_delay. Returns max_delay+1 if censored."""
        # TODO: draw from geometric distribution; apply censoring
        raise NotImplementedError

    def is_censored(self, delay: int) -> bool:
        """True if this delay represents a censored (never-observed) reward."""
        # TODO: check against max_delay and censoring_rate
        raise NotImplementedError

    def schedule(
        self,
        action_id: str,
        action: Action,
        outcome: Outcome,
        cost: float,
        current_step: int,
    ) -> tuple[int, Reward]:
        """Schedule a reward for future delivery; returns (reveal_at_step, reward)."""
        # TODO: sample delay; construct Reward with censored flag; return (current_step+k, reward)
        raise NotImplementedError
