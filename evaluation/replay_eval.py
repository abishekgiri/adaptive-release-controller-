"""Offline policy evaluation via inverse propensity scoring over logged CI/CD data."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from data.schemas import Action, Context, Outcome, Reward, Trajectory, TrajectoryStep
from policies.base import Policy


@dataclass(frozen=True)
class IPSConfig:
    """Controls IPS estimator behaviour."""

    propensity_clip: float = 20.0
    allow_missing_propensity_for_deterministic: bool = False
    deterministic_default_propensity: float = 1.0


@dataclass(frozen=True)
class IPSResult:
    """Inverse propensity scoring evaluation result."""

    estimated_policy_value: float
    estimated_cumulative_cost: float
    estimated_mean_cost: float
    matched_actions: int
    effective_sample_size: float
    evaluated_steps: int
    skipped_censored: int
    skipped_missing_reward: int
    used_default_propensity: int
    weight_sum: float


def ips_estimate(
    policy: Policy,
    trajectory: Trajectory,
    config: IPSConfig = IPSConfig(),
) -> float:
    """Return the IPS estimated policy value for a logged trajectory.

    Policy value is represented as negative mean cost.
    """

    return evaluate_ips(policy=policy, trajectory=trajectory, config=config).estimated_policy_value


def evaluate_ips(
    policy: Policy,
    trajectory: Trajectory,
    config: IPSConfig = IPSConfig(),
) -> IPSResult:
    """Evaluate a candidate policy with inverse propensity scoring.

    For cost minimisation, the IPS cumulative cost is:

    ``Σ_t I(a_logged = a_candidate) / p_logged * cost_t``

    Censored rewards and missing rewards are skipped consistently because their
    operational cost is not observable.

    Args:
        policy: Candidate policy being evaluated. The policy is not trained here.
        trajectory: Logged trajectory with action, propensity, and reward fields.
        config: IPS configuration including clipping and missing propensity rules.

    Returns:
        IPSResult containing policy value, cost, matched-action count, and ESS.
    """

    if config.propensity_clip <= 0:
        raise ValueError("propensity_clip must be positive")
    if config.deterministic_default_propensity <= 0:
        raise ValueError("deterministic_default_propensity must be positive")

    weighted_cost = 0.0
    matched_actions = 0
    evaluated_steps = 0
    skipped_censored = 0
    skipped_missing_reward = 0
    used_default_propensity = 0
    weights: list[float] = []

    for step in trajectory.steps:
        if step.reward is None:
            skipped_missing_reward += 1
            continue
        if step.reward.censored or step.reward.outcome == Outcome.CENSORED:
            skipped_censored += 1
            continue

        evaluated_steps += 1
        candidate_action, candidate_propensity = policy.select_action(step.context)
        if candidate_action != step.action:
            weights.append(0.0)
            continue

        matched_actions += 1
        logged_propensity = _logged_propensity(
            step=step,
            candidate_propensity=candidate_propensity,
            config=config,
        )
        if logged_propensity.used_default:
            used_default_propensity += 1

        weight = min(1.0 / logged_propensity.value, config.propensity_clip)
        weights.append(weight)
        weighted_cost += weight * step.reward.cost

    estimated_mean_cost = _safe_divide(weighted_cost, evaluated_steps)
    return IPSResult(
        estimated_policy_value=-estimated_mean_cost,
        estimated_cumulative_cost=weighted_cost,
        estimated_mean_cost=estimated_mean_cost,
        matched_actions=matched_actions,
        effective_sample_size=_effective_sample_size(weights),
        evaluated_steps=evaluated_steps,
        skipped_censored=skipped_censored,
        skipped_missing_reward=skipped_missing_reward,
        used_default_propensity=used_default_propensity,
        weight_sum=sum(weights),
    )


@dataclass(frozen=True)
class _LoggedPropensity:
    value: float
    used_default: bool


def snips_estimate(
    policy: Policy,
    trajectory: Trajectory,
    config: IPSConfig = IPSConfig(),
) -> float:
    """Self-normalised IPS (SNIPS) estimate; lower variance than IPS at cost of slight bias."""
    # TODO: implement SNIPS after the first IPS-only evaluator is validated.
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
    # TODO: implement doubly robust estimation after IPS is validated.
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
    return {
        policy.policy_id: [
            ips_estimate(policy=policy, trajectory=trajectory, config=config)
            for trajectory in trajectories
        ]
        for policy in policies
    }


def _logged_propensity(
    step: TrajectoryStep,
    candidate_propensity: float,
    config: IPSConfig,
) -> _LoggedPropensity:
    propensity = step.propensity
    if propensity is not None and propensity > 0:
        return _LoggedPropensity(value=float(propensity), used_default=False)

    if (
        config.allow_missing_propensity_for_deterministic
        and candidate_propensity == 1.0
    ):
        return _LoggedPropensity(
            value=config.deterministic_default_propensity,
            used_default=True,
        )

    raise ValueError(
        "Logged propensity is missing or non-positive. Set "
        "allow_missing_propensity_for_deterministic=True only for deterministic "
        "baseline policies where a default propensity of 1.0 is justified."
    )


def _effective_sample_size(weights: list[float]) -> float:
    squared_sum = sum(weight * weight for weight in weights)
    if squared_sum == 0:
        return 0.0
    return round((sum(weights) ** 2) / squared_sum, 4)


def _safe_divide(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return 0.0
    return numerator / denominator
