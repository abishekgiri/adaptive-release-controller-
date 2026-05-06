"""Phase 22 ablation policy variants for the cost-sensitive bandit study.

Three variants let us isolate which components of the full model contribute
to cost reduction:
  BinaryRewardBandit  — same LinUCB structure but uses binary (-1/0) reward
                         instead of the asymmetric operational cost signal.
  ImmediateLinUCB     — marker class used by the ablation runner to trigger
                         the no-delay update path in run_ablations.py.
  NoDriftBandit       — CostSensitiveBandit with reset_on_drift=False;
                         constructed in run_ablations.py using config kwargs.
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np

from data.schemas import Action, Context, Outcome, Reward
from policies.base import FeatureEncoder
from policies.linucb import LinUCBConfig, LinUCBPolicy


class BinaryRewardBandit(LinUCBPolicy):
    """LinUCB bandit using binary reward (-1=failure, 0=success) instead of cost.

    Phase 22 ablation: replaces ``r = -cost`` with ``r = -1`` on failure and
    ``r = 0`` on success/blocked. Tests whether asymmetric cost weighting adds
    value over a simple binary failure signal.
    """

    def __init__(
        self,
        config: LinUCBConfig,
        feature_dim: int,
        rng: np.random.Generator,
        encoder: Optional[FeatureEncoder] = None,
        policy_id: str = "binary_reward_bandit",
    ) -> None:
        super().__init__(
            config=config,
            feature_dim=feature_dim,
            rng=rng,
            encoder=encoder,
            policy_id=policy_id,
        )

    def update(self, context: Context, action: Action, reward: Reward) -> None:
        """Update A and b using a binary failure signal, not the cost value."""
        if reward.censored or not math.isfinite(reward.cost):
            return
        x = self._encoder.encode(context)
        r = -1.0 if reward.outcome == Outcome.FAILURE else 0.0
        self._A[action] = self._A[action] + np.outer(x, x)
        self._b[action] = self._b[action] + r * x


class ImmediateLinUCB(LinUCBPolicy):
    """Marker class: standard LinUCB used in the no-delay ablation path.

    The ablation runner (run_ablations.py) detects this class by isinstance
    check and calls policy.update() immediately at each step instead of
    queueing rewards in the delayed buffer.  The model itself is identical
    to LinUCBPolicy.
    """

    def __init__(
        self,
        config: LinUCBConfig,
        feature_dim: int,
        rng: np.random.Generator,
        encoder: Optional[FeatureEncoder] = None,
        policy_id: str = "no_delay_bandit",
    ) -> None:
        super().__init__(
            config=config,
            feature_dim=feature_dim,
            rng=rng,
            encoder=encoder,
            policy_id=policy_id,
        )
