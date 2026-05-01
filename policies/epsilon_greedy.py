"""Epsilon-greedy contextual bandit; sanity-check baseline with minimal assumptions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from data.schemas import Action, Context, Reward
from policies.base import FeatureEncoder, Policy


@dataclass(frozen=True)
class EpsilonGreedyConfig:
    """Hyperparameters for epsilon-greedy exploration."""

    epsilon: float = 0.1          # probability of uniform random exploration
    decay: float = 1.0            # multiplicative decay applied to epsilon each step (1.0 = no decay)
    min_epsilon: float = 0.01     # floor on epsilon after decay


class EpsilonGreedyPolicy(Policy):
    """Epsilon-greedy policy with a per-arm linear value estimator.

    With probability epsilon, selects a uniformly random action (exploration).
    Otherwise, selects the action with the highest estimated value (exploitation).
    Updates are gated by delayed/buffer.py.
    """

    def __init__(
        self,
        config: EpsilonGreedyConfig,
        feature_dim: int,
        rng: np.random.Generator,
        encoder: Optional[FeatureEncoder] = None,
        policy_id: str = "epsilon_greedy",
    ) -> None:
        self._config = config
        self._feature_dim = feature_dim
        self._rng = rng
        self._encoder = encoder or FeatureEncoder()
        self._policy_id = policy_id
        self._current_epsilon = config.epsilon

        # TODO: initialise per-arm linear estimators (OLS or ridge)
        self._weights: dict[Action, np.ndarray] = {}

    def select_action(self, context: Context) -> tuple[Action, float]:
        # TODO: with prob epsilon, sample uniformly; else argmax estimated value
        # propensity: epsilon/|A| + (1-epsilon) * I(a == greedy_action)
        raise NotImplementedError

    def update(self, context: Context, action: Action, reward: Reward) -> None:
        # TODO: update per-arm linear estimator; decay epsilon
        raise NotImplementedError

    def reset(self) -> None:
        # TODO: reset weights and restore initial epsilon
        raise NotImplementedError

    @property
    def policy_id(self) -> str:
        return self._policy_id
