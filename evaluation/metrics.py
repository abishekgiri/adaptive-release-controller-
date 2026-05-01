"""Cumulative cost, regret, cost-CDF, and drift-recovery metrics; primary paper metrics."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from data.schemas import Action, Outcome, Reward


@dataclass
class EpisodeRecord:
    """Accumulates per-step results for one policy run over one trajectory."""

    policy_id: str
    seed: int
    costs: list[float] = field(default_factory=list)         # cost at each step
    oracle_costs: list[float] = field(default_factory=list)  # oracle cost at each step
    drift_steps: list[int] = field(default_factory=list)     # steps where drift was detected


def cumulative_cost(record: EpisodeRecord) -> np.ndarray:
    """Return the cumulative sum of per-step costs over the trajectory."""
    # TODO: np.cumsum(record.costs)
    raise NotImplementedError


def cumulative_regret(record: EpisodeRecord) -> np.ndarray:
    """Return cumulative regret vs. oracle: Σ cost(a_t) - cost(π*(x_t))."""
    # TODO: cumulative_cost(record) - cumulative oracle cost
    raise NotImplementedError


def best_in_hindsight_regret(record: EpisodeRecord) -> float:
    """Return total regret vs. the best constant policy in hindsight."""
    # TODO: sum(costs) - min over actions of sum of per-step cost if that action had been taken always
    # Requires per-step counterfactual costs for all actions — supply via oracle_costs by action
    raise NotImplementedError


def cost_cdf(record: EpisodeRecord, bins: int = 100) -> tuple[np.ndarray, np.ndarray]:
    """Return (bin_edges, cdf) of the per-step cost distribution."""
    # TODO: np.histogram then cumsum normalised
    raise NotImplementedError


def time_to_recover(
    record: EpisodeRecord,
    drift_step: int,
    epsilon: float,
    pre_drift_window: int = 50,
) -> int:
    """Number of steps after drift_step before instantaneous regret returns within epsilon.

    Args:
        record: Episode results containing per-step costs.
        drift_step: Step index at which drift was detected.
        epsilon: Tolerance; recovery declared when regret drops within epsilon of pre-drift mean.
        pre_drift_window: Number of steps before drift_step used to estimate baseline regret.

    Returns:
        Number of recovery steps, or -1 if recovery was not achieved within the trajectory.
    """
    # TODO: compute pre-drift mean regret; scan post-drift steps for recovery
    raise NotImplementedError


def action_distribution(actions: list[Action]) -> dict[Action, float]:
    """Return the fraction of each action taken over a trajectory."""
    # TODO: count per action; normalise
    raise NotImplementedError
