"""Explicit cost matrix mapping (action, outcome) pairs to operational costs."""

from __future__ import annotations

from dataclasses import dataclass

from data.schemas import Action, Outcome


@dataclass(frozen=True)
class CostConfig:
    """Configurable cost matrix; all values are non-negative floats.

    Default values reflect asymmetric incident costs: a deploy-and-fail outage costs
    much more than unnecessarily blocking a safe change.
    """

    deploy_failure: float = 10.0   # cost(deploy, failure)  — production outage
    canary_failure: float = 4.0    # cost(canary, failure)   — partial-rollout incident
    block_success: float = 2.0     # cost(block, success)    — delayed safe change
    deploy_success: float = 0.0    # cost(deploy, success)   — no cost
    canary_success: float = 1.0    # cost(canary, success)   — canary overhead before promote
    block_failure: float = 0.0     # cost(block, failure)    — correctly avoided incident


# Sentinel for censored rewards where cost cannot yet be determined.
CENSORED_COST: float = float("nan")


def compute_cost(action: Action, outcome: Outcome, config: CostConfig) -> float:
    """Return the cost for a given (action, outcome) pair under config.

    Args:
        action: The deployment action taken by the policy.
        outcome: The observed deployment outcome (after delay k).
        config: Cost configuration; use default CostConfig() for paper baseline.

    Returns:
        Non-negative cost scalar. Returns CENSORED_COST if outcome is CENSORED.
    """
    # TODO: implement full lookup table
    if outcome == Outcome.CENSORED:
        return CENSORED_COST

    _matrix: dict[tuple[Action, Outcome], float] = {
        (Action.DEPLOY, Outcome.FAILURE): config.deploy_failure,
        (Action.CANARY, Outcome.FAILURE): config.canary_failure,
        (Action.BLOCK,  Outcome.SUCCESS): config.block_success,
        (Action.DEPLOY, Outcome.SUCCESS): config.deploy_success,
        (Action.CANARY, Outcome.SUCCESS): config.canary_success,
        (Action.BLOCK,  Outcome.FAILURE): config.block_failure,
    }
    # TODO: raise informative error for unexpected (action, outcome) combinations
    raise NotImplementedError
