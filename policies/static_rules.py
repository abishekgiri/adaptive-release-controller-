"""Static rule-based policy wrapping the existing baseline_decision logic; experimental baseline 1."""

from __future__ import annotations

from data.schemas import Action, Context, Reward
from policies.base import Policy


class StaticRulesPolicy(Policy):
    """Deterministic policy based on CI pass/fail thresholds; does not learn from feedback.

    Wraps experiments/baseline.py::baseline_decision. Used as the lowest bar any bandit
    must clear.  propensity is 1.0 for the deterministic action, 0.0 for all others.
    """

    def __init__(self, policy_id: str = "static_rules") -> None:
        # TODO: store any threshold parameters surfaced from experiments/baseline.py
        self._policy_id = policy_id

    def select_action(self, context: Context) -> tuple[Action, float]:
        # TODO: port baseline_decision logic; map binary deploy/block to three-action space
        # block → Action.BLOCK, deploy → Action.DEPLOY (no canary in static rules)
        raise NotImplementedError

    def update(self, context: Context, action: Action, reward: Reward) -> None:
        # Static policy; no learning. No-op.
        pass

    def reset(self) -> None:
        # No state to reset.
        pass

    @property
    def policy_id(self) -> str:
        return self._policy_id
