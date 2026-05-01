"""Cost-sensitive contextual bandit with delayed-reward buffer and drift adaptation; the contribution."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from data.schemas import Action, Context, Reward
from delayed.buffer import RewardBuffer
from drift.detectors import DriftDetector
from policies.base import FeatureEncoder, Policy
from rewards.cost_model import CostConfig


@dataclass(frozen=True)
class CostSensitiveBanditConfig:
    """Hyperparameters for the cost-sensitive delayed contextual bandit."""

    alpha: float = 1.0              # UCB exploration width (LinUCB-style upper bound)
    cost_config: CostConfig = CostConfig()
    drift_detector: str = "adwin"   # detector name passed to drift/detectors.py
    reset_on_drift: bool = True     # full reset vs. windowed retraining on detected drift
    # TODO: add regularisation, window size, and other hyperparameters


class CostSensitiveBandit(Policy):
    """The research contribution: a contextual bandit that is cost-aware, delay-aware, and drift-aware.

    Differences from vanilla LinUCB:
    1. UCB score is weighted by cost(action, outcome) not raw reward — the policy
       explicitly prefers actions whose failure modes are cheaper.
    2. All updates route through delayed/buffer.py; the policy never sees immediate rewards.
    3. A drift detector monitors the reward stream; on detection, drift/adapt.py triggers
       policy reset or windowed retraining, and the regret measurement resets its baseline.
    """

    def __init__(
        self,
        config: CostSensitiveBanditConfig,
        feature_dim: int,
        rng: np.random.Generator,
        buffer: Optional[RewardBuffer] = None,
        detector: Optional[DriftDetector] = None,
        encoder: Optional[FeatureEncoder] = None,
        policy_id: str = "cost_sensitive_bandit",
    ) -> None:
        self._config = config
        self._feature_dim = feature_dim
        self._rng = rng
        self._buffer = buffer  # TODO: inject or construct default RewardBuffer
        self._detector = detector  # TODO: inject or construct default DriftDetector
        self._encoder = encoder or FeatureEncoder()
        self._policy_id = policy_id

        # TODO: initialise per-arm cost-weighted ridge matrices A_cost, b_cost
        self._A: dict[Action, np.ndarray] = {}
        self._b: dict[Action, np.ndarray] = {}

    def select_action(self, context: Context) -> tuple[Action, float]:
        # TODO: compute cost-weighted UCB score per arm; return argmax and propensity
        # UCB_cost(a) = -θ_a.T @ x + alpha * sqrt(x.T @ A_a^{-1} @ x)
        # Negative because we minimise cost (reward = -cost).
        raise NotImplementedError

    def update(self, context: Context, action: Action, reward: Reward) -> None:
        # TODO: update A[action], b[action] with cost-weighted reward
        # Feed cost scalar to drift detector; trigger adapt.py on detected drift
        raise NotImplementedError

    def reset(self) -> None:
        # TODO: reinitialise A and b matrices; called by drift/adapt.py on drift event
        raise NotImplementedError

    @property
    def policy_id(self) -> str:
        return self._policy_id
