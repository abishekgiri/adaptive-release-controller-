"""Abstract deployment-environment interface for trajectory generation and replay."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

from data.schemas import Action, Context, Reward


class DeploymentEnvironment(ABC):
    """Environment contract shared by synthetic and replay deployment settings.

    Implementations expose only pre-action ``Context`` objects through
    :meth:`observe`. Actions are applied through :meth:`step`, while delayed
    rewards are surfaced by :meth:`advance_time` after their observation step.
    """

    @abstractmethod
    def observe(self) -> Context:
        """Return the current pre-action context visible to the policy.

        Must contain only pre-action observable features — no hidden state,
        no outcome information.
        """
        raise NotImplementedError

    @abstractmethod
    def step(self, action: Action) -> Optional[Reward]:
        """Apply ``action`` at the current decision step.

        Delayed environments should return ``None`` here and release rewards
        through :meth:`advance_time`. Replay-style environments may return an
        already-observed reward when no simulated delay is being modeled.
        """
        raise NotImplementedError

    @abstractmethod
    def advance_time(self) -> list[Reward]:
        """Advance the environment clock by one step and return all rewards that matured.

        Policies must call this at the start of each decision step to flush
        pending rewards from the delayed buffer.
        """
        raise NotImplementedError

    @abstractmethod
    def reset(self) -> Context:
        """Reset the environment to its initial state and return the first context."""
        raise NotImplementedError

    @property
    @abstractmethod
    def current_step(self) -> int:
        """Current discrete time step index."""
        raise NotImplementedError

    @property
    @abstractmethod
    def done(self) -> bool:
        """True when the trajectory is exhausted (replay) or time horizon reached (synthetic)."""
        raise NotImplementedError
