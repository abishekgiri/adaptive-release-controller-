"""Offline policy evaluation via IPS, SNIPS, and doubly-robust estimation over logged CI/CD data."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from data.schemas import Action, Context, Reward, Trajectory, TrajectoryStep
from policies.base import Policy


@dataclass(frozen=True)
class IPSConfig:
    """Controls IPS estimator behaviour."""

    propensity_clip: float = 20.0   # max importance weight; prevents high-variance estimates
    # Documented in docs/problem-formulation.md as a stated threat to validity


def ips_estimate(
    policy: Policy,
    trajectory: Trajectory,
    config: IPSConfig = IPSConfig(),
) -> float:
    """Inverse propensity scoring (IPS) estimate of policy value over a logged trajectory.

    V_IPS(π) = (1/T) Σ_t [ I(a_t = π(x_t)) / p_t ] * r_t

    Args:
        policy: The evaluation policy (not the logging policy).
        trajectory: Logged trajectory with known logging propensities.
        config: IPS configuration.

    Returns:
        Estimated mean per-step cost under the evaluation policy.
    """
    # TODO: iterate steps; compute importance weights; clip at config.propensity_clip; average
    raise NotImplementedError


def snips_estimate(
    policy: Policy,
    trajectory: Trajectory,
    config: IPSConfig = IPSConfig(),
) -> float:
    """Self-normalised IPS (SNIPS) estimate; lower variance than IPS at cost of slight bias."""
    # TODO: same as IPS but divide by sum of importance weights instead of T
    raise NotImplementedError


def dr_estimate(
    policy: Policy,
    trajectory: Trajectory,
    direct_model: Callable[[Context, Action], float],
    config: IPSConfig = IPSConfig(),
) -> float:
    """Doubly-robust (DR) estimate; consistent if either IPS or direct model is correct.

    V_DR(π) = V_DM(π) + (1/T) Σ_t [ w_t * (r_t - dm(x_t, a_t)) ]

    where V_DM is the direct model estimate and w_t is the clipped importance weight.

    Args:
        policy: The evaluation policy.
        trajectory: Logged trajectory with known logging propensities.
        direct_model: Callable (context, action) → predicted cost; trained separately.
        config: IPS configuration (propensity clipping applies here too).

    Returns:
        Doubly-robust estimate of mean per-step cost.
    """
    # TODO: compute DM baseline; add IPS residual correction
    raise NotImplementedError


def evaluate_all(
    policies: list[Policy],
    trajectories: list[Trajectory],
    config: IPSConfig = IPSConfig(),
) -> dict[str, list[float]]:
    """Run IPS evaluation for each policy over all trajectories.

    Returns:
        Dict mapping policy_id → list of per-trajectory IPS estimates (one per trajectory).
        Use evaluation/statistical.py to compute CIs over this list.
    """
    # TODO: iterate policies × trajectories; collect IPS estimates
    raise NotImplementedError
