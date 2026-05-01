"""Abstract Policy interface; all policies — baselines and bandits — implement this contract."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from data.schemas import Action, Context, Reward


class Policy(ABC):
    """A stateful mapping from context to action, updated by delayed rewards.

    Invariant: update() must NEVER be called with a reward at the same step the action
    was taken. All reward delivery goes through delayed/buffer.py.
    """

    @abstractmethod
    def select_action(self, context: Context) -> tuple[Action, float]:
        """Select a deployment action for the given context.

        Args:
            context: Pre-action observable features. Must contain no outcome information.

        Returns:
            (action, propensity) — the chosen action and the probability assigned to it
            under the current policy. Propensity is required for IPS-based evaluation.
        """
        # TODO: implement in subclasses

    @abstractmethod
    def update(self, context: Context, action: Action, reward: Reward) -> None:
        """Update internal state given a (context, action, delayed_reward) triple.

        This method is called by the experiment loop only after delayed/buffer.py
        confirms the reward has arrived (k steps after the action).

        Args:
            context: The context observed at action time (not at reward time).
            action: The action taken at that step.
            reward: The delayed reward that just matured.
        """
        # TODO: implement in subclasses

    @abstractmethod
    def reset(self) -> None:
        """Reset all learned state; called by drift/adapt.py on detected drift."""
        # TODO: implement in subclasses

    @property
    @abstractmethod
    def policy_id(self) -> str:
        """Stable identifier used for logging and reproducibility."""
        # TODO: implement in subclasses


class FeatureEncoder:
    """Converts a Context dataclass into a fixed-length numpy feature vector."""

    def encode(self, context: Context) -> np.ndarray:
        """Return a 1-D float64 array of pre-action features.

        The feature order is fixed and documented here so that policy weight
        vectors are interpretable across runs.
        """
        # TODO: implement; must match the field order in data/schemas.py Context
        raise NotImplementedError
