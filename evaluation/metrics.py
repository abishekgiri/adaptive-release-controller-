"""Cost-first evaluation metrics for deployment policy experiments."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Iterable

import numpy as np

from data.schemas import Action


@dataclass
class EpisodeRecord:
    """Per-step cost record for one policy run over one trajectory."""

    policy_id: str
    seed: int
    costs: list[float] = field(default_factory=list)
    oracle_costs: list[float] = field(default_factory=list)
    drift_steps: list[int] = field(default_factory=list)


def valid_costs(costs: Iterable[float]) -> list[float]:
    """Return finite costs, dropping NaN censored observations."""

    return [float(cost) for cost in costs if math.isfinite(float(cost))]


def cumulative_cost(costs_or_record: Iterable[float] | EpisodeRecord) -> np.ndarray:
    """Return cumulative operational cost, ignoring NaN censored values."""

    costs = (
        costs_or_record.costs
        if isinstance(costs_or_record, EpisodeRecord)
        else costs_or_record
    )
    return np.cumsum(np.array(valid_costs(costs), dtype=np.float64))


def total_operational_cost(costs_or_record: Iterable[float] | EpisodeRecord) -> float:
    """Return total finite operational cost."""

    cumulative = cumulative_cost(costs_or_record)
    if cumulative.size == 0:
        return 0.0
    return float(cumulative[-1])


def mean_operational_cost(costs_or_record: Iterable[float] | EpisodeRecord) -> float:
    """Return mean finite operational cost."""

    costs = (
        costs_or_record.costs
        if isinstance(costs_or_record, EpisodeRecord)
        else costs_or_record
    )
    finite_costs = valid_costs(costs)
    if not finite_costs:
        return 0.0
    return float(np.mean(finite_costs))


def cumulative_regret(record: EpisodeRecord) -> np.ndarray:
    """Return cumulative regret against oracle costs."""

    costs = valid_costs(record.costs)
    oracle = valid_costs(record.oracle_costs)
    length = min(len(costs), len(oracle))
    if length == 0:
        return np.array([], dtype=np.float64)
    return np.cumsum(np.array(costs[:length]) - np.array(oracle[:length]))


def best_in_hindsight_regret(
    record: EpisodeRecord,
    policy_costs: dict[str, list[float]],
) -> np.ndarray:
    """Return cumulative regret vs. the best constant policy in hindsight.

    The best constant policy is whichever policy in ``policy_costs`` achieves
    the lowest total finite cost over the trajectory. Regret at step t is the
    gap between this policy and ``record``'s cumulative cost.

    Args:
        record:       Episode record for the policy under evaluation.
        policy_costs: Dict mapping policy_id → list of per-step costs for each
                      policy that ran on the same trajectory.

    Returns:
        Array of cumulative regret values (same length as record's finite costs).
    """
    if not policy_costs:
        return np.array([], dtype=np.float64)

    best_total = math.inf
    best_costs: list[float] = []
    for costs in policy_costs.values():
        finite = valid_costs(costs)
        total = float(np.sum(finite)) if finite else math.inf
        if total < best_total:
            best_total = total
            best_costs = finite

    eval_costs = valid_costs(record.costs)
    length = min(len(eval_costs), len(best_costs))
    if length == 0:
        return np.array([], dtype=np.float64)
    return np.cumsum(
        np.array(eval_costs[:length]) - np.array(best_costs[:length])
    )


def cost_cdf(
    costs: Iterable[float],
    thresholds: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the empirical CDF of per-step operational costs.

    Args:
        costs:      Per-step cost sequence (NaN / censored values are dropped).
        thresholds: Evaluation points. Defaults to 100 linearly spaced values
                    from 0 to max(costs).

    Returns:
        (thresholds, cdf_values) — fraction of steps with cost ≤ threshold.
    """
    finite = np.array(valid_costs(costs), dtype=np.float64)
    if finite.size == 0:
        empty = np.array([], dtype=np.float64)
        return empty, empty

    if thresholds is None:
        thresholds = np.linspace(0.0, float(finite.max()), 100)

    cdf_values = np.array(
        [float(np.mean(finite <= t)) for t in thresholds], dtype=np.float64
    )
    return thresholds, cdf_values


def action_distribution(actions: list[Action]) -> dict[Action, float]:
    """Return the fraction of each action taken over a trajectory."""

    if not actions:
        return {action: 0.0 for action in Action}
    return {
        action: actions.count(action) / len(actions)
        for action in Action
    }
