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


def action_distribution(actions: list[Action]) -> dict[Action, float]:
    """Return the fraction of each action taken over a trajectory."""

    if not actions:
        return {action: 0.0 for action in Action}
    return {
        action: actions.count(action) / len(actions)
        for action in Action
    }
