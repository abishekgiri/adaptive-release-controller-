"""Strategies for handling censored or in-flight rewards during policy updates."""

from __future__ import annotations

from abc import ABC, abstractmethod

from data.schemas import Action, Context, Reward


class ImputationStrategy(ABC):
    """Abstract strategy for assigning a cost to a reward that has not yet arrived."""

    @abstractmethod
    def impute(self, context: Context, action: Action, steps_elapsed: int) -> float:
        """Return an imputed cost for a reward that is still in flight.

        Args:
            context: The context observed when the action was taken.
            action: The action whose reward has not arrived.
            steps_elapsed: Number of steps since the action was taken.

        Returns:
            Imputed cost scalar used as a proxy update until the real reward arrives.
        """
        # TODO: implement in subclasses


class ZeroImputation(ImputationStrategy):
    """Imputes zero cost for all in-flight rewards; equivalent to assuming outcomes are good."""

    def impute(self, context: Context, action: Action, steps_elapsed: int) -> float:
        # Conservative baseline: contributes nothing to the update until truth arrives.
        return 0.0


class OptimisticImputation(ImputationStrategy):
    """Imputes the minimum possible cost; encourages exploration of uncertain arms."""

    def impute(self, context: Context, action: Action, steps_elapsed: int) -> float:
        # TODO: return minimum cost for the given action from CostConfig
        raise NotImplementedError


class PessimisticImputation(ImputationStrategy):
    """Imputes the maximum possible cost; encourages caution under uncertainty."""

    def impute(self, context: Context, action: Action, steps_elapsed: int) -> float:
        # TODO: return maximum cost for the given action from CostConfig
        raise NotImplementedError


class MeanImputation(ImputationStrategy):
    """Imputes the historical mean cost for the (action) arm as a running estimate."""

    def __init__(self) -> None:
        # TODO: maintain per-arm running mean
        self._counts: dict[Action, int] = {}
        self._means: dict[Action, float] = {}

    def impute(self, context: Context, action: Action, steps_elapsed: int) -> float:
        # TODO: return current running mean for the arm, or global mean if unseen
        raise NotImplementedError

    def observe(self, action: Action, cost: float) -> None:
        """Update the running mean with a newly observed cost."""
        # TODO: Welford online update
        raise NotImplementedError
