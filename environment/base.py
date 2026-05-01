"""Abstract base class for deployment environments; all environments implement this interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

from data.schemas import Action, Context, Reward


class DeploymentEnvironment(ABC):
    """Abstract deployment environment: observe a context, take an action, receive a delayed reward."""

    @abstractmethod
    def observe(self) -> Context:
        """Return the current pre-action context visible to the policy.

        Must contain only pre-action observable features — no hidden state,
        no outcome information.
        """
        # TODO: implement in subclasses

    @abstractmethod
    def step(self, action: Action) -> Optional[Reward]:
        """Apply action and return any reward that has become observable this step.

        Returns None if no reward has arrived yet (delayed feedback still in flight).
        The reward for this action may arrive in a future call to advance_time().
        """
        # TODO: implement in subclasses

    @abstractmethod
    def advance_time(self) -> list[Reward]:
        """Advance the environment clock by one step and return all rewards that matured.

        Policies must call this at the start of each decision step to flush
        pending rewards from the delayed buffer.
        """
        # TODO: implement in subclasses

    @abstractmethod
    def reset(self) -> Context:
        """Reset the environment to its initial state and return the first context."""
        # TODO: implement in subclasses

    @property
    @abstractmethod
    def current_step(self) -> int:
        """Current discrete time step index."""
        # TODO: implement in subclasses

    @property
    @abstractmethod
    def done(self) -> bool:
        """True when the trajectory is exhausted (replay) or time horizon reached (synthetic)."""
        # TODO: implement in subclasses
