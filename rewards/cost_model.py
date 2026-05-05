"""Explicit cost matrix mapping (action, outcome) pairs to operational costs."""

from __future__ import annotations

import math
from dataclasses import dataclass

from data.schemas import Action, Outcome


@dataclass(frozen=True)
class CostConfig:
    """Configurable cost matrix; all values must be non-negative floats.

    Default values reflect the asymmetric operational costs of deployment decisions:
    shipping a bad change to production (deploy + failure) is far more expensive
    than unnecessarily holding back a safe one (block + safe).

    Field names use the semantic labels from docs/problem-formulation.md:
        deploy_success  — no outage, no opportunity cost             default: 0
        deploy_failure  — production incident, on-call, rollback     default: 10
        canary_success  — canary overhead + promotion latency        default: 1
        canary_failure  — partial-rollout incident, limited blast    default: 4
        block_safe      — safe change delayed; developer wait-time   default: 2
        block_bad       — risky change correctly held back; re-review overhead  default: 0.5
        block_unknown   — change blocked, counterfactual unobserved  default: 2.0
                          (replay mode; treated conservatively = block_safe)
    """

    deploy_success: float = 0.0
    deploy_failure: float = 10.0
    canary_success: float = 1.0
    canary_failure: float = 4.0
    block_safe: float = 2.0     # (BLOCK, SUCCESS)  — would have succeeded
    block_bad: float = 0.5      # (BLOCK, FAILURE)  — would have failed
    block_unknown: float = 2.0  # (BLOCK, BLOCKED)  — counterfactual unknown

    def __post_init__(self) -> None:
        for field, value in self.__dataclass_fields__.items():
            v = getattr(self, field)
            if not isinstance(v, (int, float)) or v < 0:
                raise ValueError(
                    f"CostConfig.{field} must be a non-negative number, got {v!r}"
                )


# Sentinel returned when the outcome is CENSORED (reward not yet observable).
CENSORED_COST: float = float("nan")

# All valid (Action, Outcome) pairs; used for exhaustiveness checks in tests.
VALID_PAIRS: frozenset[tuple[Action, Outcome]] = frozenset({
    (Action.DEPLOY, Outcome.SUCCESS),
    (Action.DEPLOY, Outcome.FAILURE),
    (Action.CANARY, Outcome.SUCCESS),
    (Action.CANARY, Outcome.FAILURE),
    (Action.BLOCK,  Outcome.SUCCESS),   # would have succeeded (block_safe)
    (Action.BLOCK,  Outcome.FAILURE),   # would have failed    (block_bad)
    (Action.BLOCK,  Outcome.BLOCKED),   # counterfactual unknown (block_unknown)
})


def compute_cost(action: Action, outcome: Outcome, config: CostConfig) -> float:
    """Return the operational cost for a given (action, outcome) pair under config.

    Args:
        action:  The deployment action taken by the policy.
        outcome: The observed outcome after delay k. Pass Outcome.CENSORED when
                 the reward window closed before the outcome resolved.
        config:  Cost configuration. Use CostConfig() for the paper's default values.

    Returns:
        Non-negative cost scalar, or float('nan') when outcome is CENSORED.

    Raises:
        ValueError: If (action, outcome) is not a valid pair (e.g. DEPLOY + BLOCKED).
    """
    if outcome == Outcome.CENSORED:
        return CENSORED_COST

    _matrix: dict[tuple[Action, Outcome], float] = {
        (Action.DEPLOY, Outcome.SUCCESS): config.deploy_success,
        (Action.DEPLOY, Outcome.FAILURE): config.deploy_failure,
        (Action.CANARY, Outcome.SUCCESS): config.canary_success,
        (Action.CANARY, Outcome.FAILURE): config.canary_failure,
        (Action.BLOCK,  Outcome.SUCCESS): config.block_safe,
        (Action.BLOCK,  Outcome.FAILURE): config.block_bad,
        (Action.BLOCK,  Outcome.BLOCKED): config.block_unknown,
    }

    key = (action, outcome)
    if key not in _matrix:
        raise ValueError(
            f"Invalid (action, outcome) pair: ({action.value!r}, {outcome.value!r}). "
            f"Valid pairs: {sorted((a.value, o.value) for a, o in _matrix)}"
        )

    return _matrix[key]


def oracle_cost(outcome: Outcome, config: CostConfig) -> float:
    """Return the minimum possible cost for a given outcome under perfect foresight.

    The oracle always picks the action that minimises cost knowing the outcome in advance.
    Used to compute cumulative regret vs. oracle in evaluation/metrics.py.

    Args:
        outcome: The true counterfactual outcome.
        config:  Cost configuration.

    Returns:
        The minimum achievable cost for this outcome.
    """
    if outcome == Outcome.CENSORED:
        return CENSORED_COST

    if outcome == Outcome.SUCCESS:
        # Oracle can deploy, canary, or block. deploy_success is cheapest by default.
        return min(config.deploy_success, config.canary_success, config.block_safe)

    if outcome == Outcome.FAILURE:
        # Oracle blocks to avoid the incident. block_bad is the minimum cost.
        return min(config.deploy_failure, config.canary_failure, config.block_bad)

    if outcome == Outcome.BLOCKED:
        # BLOCKED means the logged policy blocked and the counterfactual outcome
        # (would-have-succeeded vs would-have-failed) was never observed. The oracle
        # cost is therefore undefined — we cannot know which action would have been
        # cheapest. We return NaN (CENSORED_COST) so callers can drop this step,
        # consistent with how valid_costs() handles censored observations.
        return CENSORED_COST

    raise ValueError(f"Unhandled outcome: {outcome!r}")  # exhaustiveness guard
