"""Static rule-based policy wrapping the existing baseline_decision logic; experimental baseline 1."""

from __future__ import annotations

from data.schemas import Action, Context, Reward
from policies.base import Policy

# Risk thresholds ported from experiments/baseline.py::failure_probability
_HIGH_FAILURE_RATE: float = 0.30   # recent_failure_rate → block tier
_MED_FAILURE_RATE: float = 0.15    # recent_failure_rate → canary tier

_LARGE_FILES: int = 20             # files_changed → block tier (baseline: +0.20 risk)
_MEDIUM_FILES: int = 10            # files_changed → canary tier (baseline: +0.10 risk)

_LARGE_CHURN: int = 1000           # src_churn → block tier (baseline: +0.20 risk)
_MEDIUM_CHURN: int = 300           # src_churn → canary tier (baseline: +0.10 risk)


class StaticRulesPolicy(Policy):
    """Deterministic three-tier policy ported from experiments/baseline.py.

    Maps the original binary (deploy / block) logic to the three-action space
    by inserting a canary tier for medium-risk changes. Uses only pre-action
    Context fields; ignores all reward / outcome information.

    Propensity is 1.0 for the chosen action (deterministic policy), 0.0 for others.
    Used as the lowest bar any bandit must clear.
    """

    def __init__(self, policy_id: str = "static_rules") -> None:
        self._policy_id = policy_id

    def select_action(self, context: Context) -> tuple[Action, float]:
        """Return (action, propensity=1.0) based on static risk thresholds."""
        action = self._classify(context)
        return action, 1.0

    def update(self, context: Context, action: Action, reward: Reward) -> None:
        pass  # static policy; no learning

    def reset(self) -> None:
        pass  # no learned state

    @property
    def policy_id(self) -> str:
        return self._policy_id

    # ------------------------------------------------------------------
    # Internal classification — no Context field may be an outcome proxy
    # ------------------------------------------------------------------

    def _classify(self, context: Context) -> Action:
        """Three-tier risk classification derived from baseline failure_probability.

        Tier mapping:
            HIGH risk  → BLOCK   (would have been blocked by baseline + extra risk signals)
            MED  risk  → CANARY  (ambiguous; partial rollout limits blast radius)
            LOW  risk  → DEPLOY  (baseline would have deployed; no red flags)
        """
        if self._is_high_risk(context):
            return Action.BLOCK
        if self._is_medium_risk(context):
            return Action.CANARY
        return Action.DEPLOY

    def _is_high_risk(self, context: Context) -> bool:
        """True when any single high-severity signal fires."""
        return (
            context.recent_failure_rate > _HIGH_FAILURE_RATE
            or (
                context.has_risky_path_change
                and (
                    context.files_changed >= _LARGE_FILES
                    or context.src_churn >= _LARGE_CHURN
                )
            )
        )

    def _is_medium_risk(self, context: Context) -> bool:
        """True when at least one medium-severity signal fires (and no high signal)."""
        return (
            context.has_risky_path_change
            or context.files_changed >= _MEDIUM_FILES
            or context.src_churn >= _MEDIUM_CHURN
            or context.recent_failure_rate > _MED_FAILURE_RATE
        )
