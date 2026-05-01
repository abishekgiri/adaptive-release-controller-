"""Thompson Sampling with a Bayesian linear reward model; bandit baseline 2."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from data.schemas import Action, Context, Reward
from policies.base import FeatureEncoder, Policy


@dataclass(frozen=True)
class ThompsonConfig:
    """Hyperparameters for Thompson Sampling with a linear Gaussian model."""

    prior_variance: float = 1.0    # variance of the Gaussian prior on weights
    noise_variance: float = 0.1    # assumed observation noise variance


class ThompsonSamplingPolicy(Policy):
    """Thompson Sampling: sample weight vector from posterior, select greedy arm.

    Maintains a Bayesian linear model per arm (Gaussian posterior over weight vectors).
    Exploration is implicit: arm selection is stochastic via posterior sampling.
    Updates are gated by delayed/buffer.py.
    """

    def __init__(
        self,
        config: ThompsonConfig,
        feature_dim: int,
        rng: np.random.Generator,
        encoder: Optional[FeatureEncoder] = None,
        policy_id: str = "thompson",
    ) -> None:
        self._config = config
        self._feature_dim = feature_dim
        self._rng = rng
        self._encoder = encoder or FeatureEncoder()
        self._policy_id = policy_id

        # TODO: initialise per-arm posterior: mean μ_a = 0, precision Λ_a = (1/v0) * I
        self._mu: dict[Action, np.ndarray] = {}
        self._precision: dict[Action, np.ndarray] = {}

    def select_action(self, context: Context) -> tuple[Action, float]:
        # TODO: for each arm, sample θ_a ~ N(μ_a, Λ_a^{-1}); pick arm with highest θ_a.T @ x
        # propensity: probability of chosen arm under the current posterior (requires integration)
        raise NotImplementedError

    def update(self, context: Context, action: Action, reward: Reward) -> None:
        # TODO: Bayesian update of (μ_action, Λ_action) given (x, r)
        raise NotImplementedError

    def reset(self) -> None:
        # TODO: reinitialise posteriors to prior
        raise NotImplementedError

    @property
    def policy_id(self) -> str:
        return self._policy_id
