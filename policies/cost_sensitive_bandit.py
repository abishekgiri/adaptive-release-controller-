"""Cost-sensitive contextual bandit with delayed rewards and drift adaptation.

This is the first research-contribution policy. It keeps the LinUCB-style
disjoint linear model, but optimizes operational reward ``r = -cost`` and owns a
pending-reward buffer so learning only happens after feedback is observable.
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass
from typing import Optional

import numpy as np

from data.schemas import Action, Context, Outcome, Reward
from delayed.buffer import PendingReward, PendingRewardBuffer
from drift.detectors import DriftDetector
from policies.base import FeatureEncoder, Policy
from rewards.cost_model import CostConfig


@dataclass(frozen=True)
class CostSensitiveBanditConfig:
    """Hyperparameters for the cost-sensitive delayed contextual bandit."""

    alpha: float = 1.0
    lambda_reg: float = 1.0
    cost_config: CostConfig = CostConfig()
    reset_on_drift: bool = True
    min_delay: int = 1
    max_delay: int = 1


@dataclass(frozen=True)
class CostSensitiveBanditStats:
    """Lightweight runtime counters for experiment logging."""

    cumulative_cost: float
    cumulative_regret: float
    action_counts: dict[str, int]
    delayed_updates_applied: int
    drift_resets: int
    pending_rewards: int


class CostSensitiveBandit(Policy):
    """LinUCB-style bandit for delayed, asymmetric deployment costs.

    The model estimates expected reward per action, where reward is the negative
    operational cost. The public delayed-feedback path is:

    1. ``select_action(context)``
    2. ``record_pending_reward(...)`` when the environment/logged replay has a
       computed outcome and cost
    3. ``advance_to(step)`` to apply only matured, non-censored rewards

    ``update(...)`` remains available for the base ``Policy`` interface and is
    treated as the matured-reward primitive. Experiment loops should prefer the
    buffer path above when simulating delayed feedback.
    """

    def __init__(
        self,
        config: CostSensitiveBanditConfig,
        feature_dim: int,
        rng: np.random.Generator,
        buffer: Optional[PendingRewardBuffer] = None,
        detector: Optional[DriftDetector] = None,
        encoder: Optional[FeatureEncoder] = None,
        policy_id: str = "cost_sensitive_bandit",
    ) -> None:
        if config.alpha < 0:
            raise ValueError("alpha must be non-negative")
        if config.lambda_reg <= 0:
            raise ValueError("lambda_reg must be positive")

        self._config = config
        self._feature_dim = feature_dim
        self._rng = rng
        self._buffer = buffer or PendingRewardBuffer(
            rng=rng,
            min_delay=config.min_delay,
            max_delay=config.max_delay,
        )
        self._detector = detector
        self._encoder = encoder or FeatureEncoder()
        self._policy_id = policy_id
        self._A: dict[Action, np.ndarray] = {}
        self._b: dict[Action, np.ndarray] = {}
        self._action_counts: Counter[Action] = Counter()
        self._cumulative_cost = 0.0
        self._cumulative_regret = 0.0
        self._delayed_updates_applied = 0
        self._drift_resets = 0
        self._init_arms()

    def select_action(self, context: Context) -> tuple[Action, float]:
        """Select the action with the highest optimistic negative-cost reward."""

        x = self._encoder.encode(context)
        best_action = Action.DEPLOY
        best_score = -np.inf

        for action in Action:
            A_inv_x = np.linalg.solve(self._A[action], x)
            theta = np.linalg.solve(self._A[action], self._b[action])
            exploit = float(theta @ x)
            explore = self._config.alpha * float(np.sqrt(max(0.0, x @ A_inv_x)))
            score = exploit + explore
            if score > best_score:
                best_score = score
                best_action = action

        return best_action, 1.0

    def record_pending_reward(
        self,
        *,
        action_id: str,
        context: Context,
        action: Action,
        outcome: Outcome,
        cost: float,
        current_step: int,
        delay: int | None = None,
        censored: bool = False,
    ) -> PendingReward:
        """Schedule a computed reward; no model update happens here."""

        self._action_counts[action] += 1
        return self._buffer.add(
            action_id=action_id,
            context=context,
            action=action,
            outcome=outcome,
            current_step=current_step,
            cost=cost,
            delay=delay,
            censored=censored,
        )

    def advance_to(self, step: int) -> list[PendingReward]:
        """Apply matured rewards up to ``step`` and return the applied entries."""

        matured = self._buffer.pop_available(step)
        applied: list[PendingReward] = []
        for pending in matured:
            if pending.reward.censored or pending.reward.outcome == Outcome.CENSORED:
                continue
            self.update(pending.context, pending.action, pending.reward)
            applied.append(pending)
        return applied

    def update(self, context: Context, action: Action, reward: Reward) -> None:
        """Apply a matured delayed reward using reward = -cost."""

        if reward.censored or reward.outcome == Outcome.CENSORED:
            return
        if not math.isfinite(reward.cost):
            return

        x = self._encoder.encode(context)
        observed_reward = -float(reward.cost)
        self._A[action] = self._A[action] + np.outer(x, x)
        self._b[action] = self._b[action] + observed_reward * x
        self._cumulative_cost += float(reward.cost)
        self._delayed_updates_applied += 1

        if self._detector is not None and self._detector.update(float(reward.cost)):
            self._handle_drift()

    def reset(
        self,
        *,
        clear_pending: bool = False,
        reset_stats: bool = False,
    ) -> None:
        """Refresh learned model state.

        Drift-triggered resets call this with ``clear_pending=False`` so delayed
        rewards already in flight are still delivered later.
        """

        self._init_arms()
        if clear_pending:
            self._buffer = PendingRewardBuffer(
                rng=self._rng,
                min_delay=self._config.min_delay,
                max_delay=self._config.max_delay,
            )
        if reset_stats:
            self._action_counts.clear()
            self._cumulative_cost = 0.0
            self._cumulative_regret = 0.0
            self._delayed_updates_applied = 0
            self._drift_resets = 0

    @property
    def stats(self) -> CostSensitiveBanditStats:
        """Return current counters in JSON-friendly form."""

        return CostSensitiveBanditStats(
            cumulative_cost=self._cumulative_cost,
            cumulative_regret=self._cumulative_regret,
            action_counts={action.value: self._action_counts[action] for action in Action},
            delayed_updates_applied=self._delayed_updates_applied,
            drift_resets=self._drift_resets,
            pending_rewards=len(self._buffer),
        )

    @property
    def policy_id(self) -> str:
        return self._policy_id

    def _handle_drift(self) -> None:
        self._drift_resets += 1
        if self._config.reset_on_drift:
            self.reset(clear_pending=False, reset_stats=False)
        if self._detector is not None:
            self._detector.reset()

    def _init_arms(self) -> None:
        d = self._feature_dim
        lam = self._config.lambda_reg
        for action in Action:
            self._A[action] = lam * np.eye(d, dtype=np.float64)
            self._b[action] = np.zeros(d, dtype=np.float64)
