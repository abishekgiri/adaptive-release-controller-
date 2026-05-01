"""Heuristic risk-score policy with a fixed threshold; experimental baseline 2."""

from __future__ import annotations

from data.schemas import Action, Context, Reward
from policies.base import Policy


class HeuristicScorePolicy(Policy):
    """Applies a hand-crafted risk score with a fixed decision threshold; does not learn.

    Wraps the retired calculate_risk_score from features/extractor.py. Used to
    show that a static heuristic score is insufficient; the bandit must beat this.
    propensity is 1.0 for the deterministic action.
    """

    def __init__(self, threshold: float = 0.5, policy_id: str = "heuristic_score") -> None:
        # TODO: store threshold; the score function is ported from features/extractor.py
        self._threshold = threshold
        self._policy_id = policy_id

    def select_action(self, context: Context) -> tuple[Action, float]:
        # TODO: compute heuristic risk score from context fields; compare to threshold
        # score >= threshold → Action.BLOCK; score < threshold → Action.DEPLOY
        raise NotImplementedError

    def update(self, context: Context, action: Action, reward: Reward) -> None:
        # Fixed heuristic; no learning.
        pass

    def reset(self) -> None:
        pass

    @property
    def policy_id(self) -> str:
        return self._policy_id
