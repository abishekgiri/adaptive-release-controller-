"""Heuristic risk-score policy with a fixed threshold; experimental baseline 2.

Computes a deterministic risk score from Context fields and maps it to
{DEPLOY, CANARY, BLOCK} via two thresholds. Does not learn. Serves as the
"slightly better than static rules" baseline — the bandit must beat this.
"""

from __future__ import annotations

from data.schemas import Action, Context, Reward
from policies.base import Policy


class HeuristicScorePolicy(Policy):
    """Applies a hand-crafted risk score with fixed decision thresholds; does not learn.

    Score formula (normalised to [0, 1]):
        0.25 * min(files_changed / 50, 1.0)
        + 0.25 * min((lines_added + lines_deleted) / 1000, 1.0)
        + 0.20 * recent_failure_rate
        + 0.15 * has_risky_path_change
        + 0.10 * has_dependency_change
        + 0.05 * (1 - min(author_experience / 10, 1.0))

    Decision:
        score >= canary_threshold → BLOCK
        score >= deploy_threshold → CANARY
        otherwise                → DEPLOY
    """

    def __init__(
        self,
        deploy_threshold: float = 0.25,
        canary_threshold: float = 0.55,
        policy_id: str = "heuristic_score",
    ) -> None:
        self._deploy_threshold = deploy_threshold
        self._canary_threshold = canary_threshold
        self._policy_id = policy_id

    def select_action(self, context: Context) -> tuple[Action, float]:
        score = self._score(context)
        if score >= self._canary_threshold:
            return Action.BLOCK, 1.0
        if score >= self._deploy_threshold:
            return Action.CANARY, 1.0
        return Action.DEPLOY, 1.0

    def _score(self, context: Context) -> float:
        churn = context.lines_added + context.lines_deleted
        experience_penalty = 1.0 - min(context.author_experience / 10.0, 1.0)
        score = (
            0.25 * min(context.files_changed / 50.0, 1.0)
            + 0.25 * min(churn / 1000.0, 1.0)
            + 0.20 * context.recent_failure_rate
            + 0.15 * float(context.has_risky_path_change)
            + 0.10 * float(context.has_dependency_change)
            + 0.05 * experience_penalty
        )
        return min(score, 1.0)

    def update(self, context: Context, action: Action, reward: Reward) -> None:
        pass  # Fixed heuristic; no learning.

    def reset(self) -> None:
        pass

    @property
    def policy_id(self) -> str:
        return self._policy_id
