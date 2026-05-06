"""Heuristic risk-score policy with a fixed threshold; experimental baseline 2."""

from __future__ import annotations

from data.schemas import Action, Context, Reward
from policies.base import Policy


class HeuristicScorePolicy(Policy):
    """Applies a hand-crafted risk score with a fixed decision threshold; does not learn.

    The score uses only pre-action ``Context`` fields and maps the old binary
    score into the three-action space with a conservative block threshold and a
    lower canary threshold for ambiguous changes.
    """

    def __init__(self, threshold: float = 0.5, policy_id: str = "heuristic_score") -> None:
        if not 0.0 < threshold <= 1.0:
            raise ValueError("threshold must be in (0, 1]")
        self._threshold = threshold
        self._policy_id = policy_id

    def select_action(self, context: Context) -> tuple[Action, float]:
        score = self.risk_score(context)
        if score >= self._threshold:
            return Action.BLOCK, 1.0
        if score >= self._threshold * 0.6:
            return Action.CANARY, 1.0
        return Action.DEPLOY, 1.0

    def update(self, context: Context, action: Action, reward: Reward) -> None:
        # Fixed heuristic; no learning.
        pass

    def reset(self) -> None:
        pass

    @property
    def policy_id(self) -> str:
        return self._policy_id

    def risk_score(self, context: Context) -> float:
        """Return an explainable risk score in ``[0, 1]`` from pre-action signals."""

        file_count_score = _clamp(context.files_changed / 50.0)
        churn_score = _clamp(context.src_churn / 1500.0)
        past_failure_score = _clamp(context.recent_failure_rate)
        ci_time_score = _clamp(context.build_duration_s / 600.0)
        risky_path_score = 1.0 if context.has_risky_path_change else 0.0
        dependency_score = 1.0 if context.has_dependency_change else 0.0

        score = (
            0.25 * file_count_score
            + 0.25 * churn_score
            + 0.20 * past_failure_score
            + 0.15 * ci_time_score
            + 0.10 * risky_path_score
            + 0.05 * dependency_score
        )
        return round(_clamp(score), 4)


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, float(value)))
