"""LinUCB contextual bandit (Li et al. 2010); first real bandit baseline."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from data.schemas import Action, Context, Reward
from policies.base import FeatureEncoder, Policy


@dataclass(frozen=True)
class LinUCBConfig:
    """Hyperparameters for LinUCB."""

    alpha: float = 1.0   # exploration width; scales the UCB confidence interval
    # TODO: add regularisation lambda if needed


class LinUCBPolicy(Policy):
    """Disjoint LinUCB: one independent ridge-regression model per arm.

    Implements Algorithm 1 from Li et al. 2010. Regret bound: O(√(dT log T)).
    All three actions (deploy, canary, block) are treated as separate arms.
    Updates are gated by delayed/buffer.py; this class never receives immediate rewards.
    """

    def __init__(
        self,
        config: LinUCBConfig,
        feature_dim: int,
        rng: np.random.Generator,
        encoder: Optional[FeatureEncoder] = None,
        policy_id: str = "linucb",
    ) -> None:
        self._config = config
        self._feature_dim = feature_dim
        self._rng = rng
        self._encoder = encoder or FeatureEncoder()
        self._policy_id = policy_id

        # TODO: initialise per-arm A matrices (d×d identity) and b vectors (d×1 zeros)
        # A[a] = I_d, b[a] = 0_d for each a in Action
        self._A: dict[Action, np.ndarray] = {}
        self._b: dict[Action, np.ndarray] = {}

    def select_action(self, context: Context) -> tuple[Action, float]:
        # TODO: encode context; compute UCB score for each arm; return argmax and propensity
        # UCB(a) = θ_a.T @ x + alpha * sqrt(x.T @ A_a^{-1} @ x)
        raise NotImplementedError

    def update(self, context: Context, action: Action, reward: Reward) -> None:
        # TODO: update A[action] and b[action] with the delayed reward
        # A[a] += x @ x.T;  b[a] += r * x
        raise NotImplementedError

    def reset(self) -> None:
        # TODO: reinitialise A and b matrices to identity/zero
        raise NotImplementedError

    @property
    def policy_id(self) -> str:
        return self._policy_id
