"""Drift-mode evaluation using the synthetic deployment environment.

Evaluates all policies under three non-stationarity conditions (Phase G):

  none     — stationary hidden state; single segment for the full horizon.
  abrupt   — single discontinuity at the trajectory midpoint (low-risk →
             high-risk). Primary drift result.
  gradual  — hidden-state parameters linearly interpolated across the full
             trajectory in N_GRADUAL_SEGMENTS small steps. Stress test.

Uses SyntheticEnvironment (environment/synthetic.py) rather than replay data
so that the ground-truth failure probability is known and oracle cost can be
computed per step.

WARNING — SIMULATION ONLY.
Results reflect synthetic environment assumptions, not real deployment data.
Regret is relative to the oracle policy that observes the hidden state at each
step. Do NOT report as estimates of real-world cost savings.

Usage:
    python -m experiments.run_drift_eval --seeds 0 1 2 ... 29 --horizon 500
    python -m experiments.run_drift_eval --seeds 0 1 2 3 4  # smoke test
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from data.schemas import Action, Context, Outcome, Reward
from drift.detectors import PageHinkleyConfig, PageHinkleyDetector
from environment.synthetic import (
    DriftSchedule,
    SegmentParams,
    SyntheticEnvironment,
    default_drift_schedule,
)
from policies.base import FeatureEncoder
from policies.cost_sensitive_bandit import CostSensitiveBandit, CostSensitiveBanditConfig
from policies.heuristic_score import HeuristicScorePolicy
from policies.linucb import LinUCBConfig, LinUCBPolicy
from policies.static_rules import StaticRulesPolicy
from policies.thompson import ThompsonConfig, ThompsonSamplingPolicy
from rewards.cost_model import CostConfig, compute_cost, oracle_cost as _oracle_cost_by_outcome

DEFAULT_RESULTS_ROOT = Path("experiments/results/drift_eval")
N_BOOTSTRAP = 10_000
BOOTSTRAP_SEED = 42
N_GRADUAL_SEGMENTS = 25  # approximation granularity for gradual drift


# ---------------------------------------------------------------------------
# Drift schedule factories
# ---------------------------------------------------------------------------

_LOW_RISK = SegmentParams(
    base_failure_prob=0.10,
    infra_load_mean=0.20,
    change_complexity_mean=0.25,
    team_fatigue_mean=0.15,
)
_HIGH_RISK = SegmentParams(
    base_failure_prob=0.55,
    infra_load_mean=0.70,
    change_complexity_mean=0.70,
    team_fatigue_mean=0.60,
)


def none_drift_schedule(horizon: int) -> DriftSchedule:
    """Single stationary segment covering the entire trajectory."""
    return DriftSchedule(
        segment_length=horizon + 1,  # never crosses a boundary within horizon
        segments=(_LOW_RISK,),
    )


def abrupt_drift_schedule(horizon: int) -> DriftSchedule:
    """Single discontinuity at the trajectory midpoint (low-risk → high-risk)."""
    half = max(1, horizon // 2)
    return DriftSchedule(
        segment_length=half,
        segments=(_LOW_RISK, _HIGH_RISK),
    )


def gradual_drift_schedule(horizon: int) -> DriftSchedule:
    """Linearly interpolate hidden-state parameters across N_GRADUAL_SEGMENTS steps.

    Each segment's parameters are a convex combination of _LOW_RISK and _HIGH_RISK,
    so the environment smoothly shifts from low-risk at t=0 to high-risk at t=horizon.
    """
    n = N_GRADUAL_SEGMENTS
    seg_len = max(1, math.ceil(horizon / n))
    segments: list[SegmentParams] = []
    for i in range(n):
        alpha = i / max(1, n - 1)  # 0.0 at i=0, 1.0 at i=n-1
        segments.append(SegmentParams(
            base_failure_prob=(1 - alpha) * _LOW_RISK.base_failure_prob + alpha * _HIGH_RISK.base_failure_prob,
            infra_load_mean=(1 - alpha) * _LOW_RISK.infra_load_mean + alpha * _HIGH_RISK.infra_load_mean,
            change_complexity_mean=(1 - alpha) * _LOW_RISK.change_complexity_mean + alpha * _HIGH_RISK.change_complexity_mean,
            team_fatigue_mean=(1 - alpha) * _LOW_RISK.team_fatigue_mean + alpha * _HIGH_RISK.team_fatigue_mean,
        ))
    return DriftSchedule(segment_length=seg_len, segments=tuple(segments))


DRIFT_SCHEDULES: dict[str, Any] = {
    "none": none_drift_schedule,
    "abrupt": abrupt_drift_schedule,
    "gradual": gradual_drift_schedule,
}


# ---------------------------------------------------------------------------
# Policy factory
# ---------------------------------------------------------------------------

def build_policies(seed: int, cost_config: CostConfig) -> list:
    """Instantiate all policies for one seed."""
    lc = LinUCBConfig(alpha=1.0, lambda_reg=1.0)
    cc = CostSensitiveBanditConfig(
        alpha=1.0,
        lambda_reg=1.0,
        cost_config=cost_config,
        reset_on_drift=True,
    )
    cc_no_drift = CostSensitiveBanditConfig(
        alpha=1.0,
        lambda_reg=1.0,
        cost_config=cost_config,
        reset_on_drift=False,
    )
    dim = FeatureEncoder.DIM
    return [
        StaticRulesPolicy(policy_id="static_rules"),
        HeuristicScorePolicy(policy_id="heuristic_score"),
        LinUCBPolicy(config=lc, feature_dim=dim, rng=np.random.default_rng(seed), policy_id="linucb"),
        ThompsonSamplingPolicy(config=ThompsonConfig(), feature_dim=dim, rng=np.random.default_rng(seed), policy_id="thompson"),
        CostSensitiveBandit(
            config=cc,
            feature_dim=dim,
            rng=np.random.default_rng(seed),
            detector=PageHinkleyDetector(PageHinkleyConfig(lambda_=50.0)),
            policy_id="linucb_with_drift_full",
        ),
        CostSensitiveBandit(
            config=cc_no_drift,
            feature_dim=dim,
            rng=np.random.default_rng(seed),
            detector=PageHinkleyDetector(PageHinkleyConfig(lambda_=50.0)),
            policy_id="linucb_with_drift_no_reset",
        ),
    ]


# ---------------------------------------------------------------------------
# Single-trajectory runner (synthetic environment loop)
# ---------------------------------------------------------------------------

@dataclass
class DriftTrajectoryResult:
    policy_id: str
    drift_mode: str
    seed: int
    cumulative_cost: float
    cumulative_regret: float
    total_steps: int
    total_updates: int
    total_censored: int
    action_counts: dict[str, int]
    drift_resets: int  # only non-zero for LinUCBWithDrift with reset_on_drift=True


def run_drift_trajectory(
    policy,
    drift_mode: str,
    seed: int,
    horizon: int,
    cost_config: CostConfig,
    delay_p: float = 0.3,
    max_delay: int = 20,
) -> DriftTrajectoryResult:
    """Run one policy × drift_mode × seed trajectory.

    Regret uses the realized-empirical approach (§7): after each delayed reward
    matures, regret += compute_cost(policy_action, outcome) - oracle_cost(outcome)
    where oracle_cost(outcome) is the minimum cost achievable given the outcome.
    This is post-hoc and unbiased once outcomes are revealed.
    """
    schedule_factory = DRIFT_SCHEDULES[drift_mode]
    schedule = schedule_factory(horizon)

    env = SyntheticEnvironment(
        rng=np.random.default_rng(seed),
        horizon=horizon,
        drift_schedule=schedule,
        delay_p=delay_p,
        max_delay=max_delay,
    )
    policy.reset()

    cumulative_cost = 0.0
    cumulative_regret = 0.0
    total_updates = 0
    total_censored = 0
    action_counts: dict[str, int] = {a.value: 0 for a in Action}

    # Pending rewards: list of (reveal_at_step, context, action, reward)
    pending: list[tuple[int, Context, Action, Reward]] = []

    ctx = env.reset()

    while not env.done:
        env.advance_time()
        step = env.current_step

        # Deliver matured pending rewards to policy and accumulate cost/regret
        still_pending = []
        for (reveal_at, pctx, paction, preward) in pending:
            if step >= reveal_at:
                if not preward.censored:
                    cost = compute_cost(paction, preward.outcome, cost_config)
                    if math.isfinite(cost):
                        final_reward = Reward(
                            action_id=preward.action_id,
                            outcome=preward.outcome,
                            cost=cost,
                            delay_steps=preward.delay_steps,
                            censored=False,
                            observed_at_step=step,
                        )
                        policy.update(pctx, paction, final_reward)
                        cumulative_cost += cost
                        # Realized regret: actual cost minus oracle minimum for this outcome
                        min_cost = _oracle_cost_by_outcome(preward.outcome, cost_config)
                        if math.isfinite(min_cost):
                            cumulative_regret += max(0.0, cost - min_cost)
                        total_updates += 1
                    else:
                        total_censored += 1
                else:
                    total_censored += 1
            else:
                still_pending.append((reveal_at, pctx, paction, preward))
        pending = still_pending

        # Select action and step environment
        action, _ = policy.select_action(ctx)
        action_counts[action.value] += 1
        env.step(action)

        # Record the scheduled reward for later delivery
        if env._pending:
            reveal_at, reward = env._pending[-1]
            pending.append((reveal_at, ctx, action, reward))

        ctx = env.observe()

    # Count remaining pending as censored (flush_at_end=False for clean evaluation)
    for (_, _, _, preward) in pending:
        total_censored += 1

    drift_resets = 0
    if hasattr(policy, "stats") and hasattr(policy.stats, "drift_resets"):
        drift_resets = policy.stats.drift_resets

    return DriftTrajectoryResult(
        policy_id=policy.policy_id,
        drift_mode=drift_mode,
        seed=seed,
        cumulative_cost=cumulative_cost,
        cumulative_regret=cumulative_regret,
        total_steps=horizon,
        total_updates=total_updates,
        total_censored=total_censored,
        action_counts=action_counts,
        drift_resets=drift_resets,
    )


# ---------------------------------------------------------------------------
# Bootstrap CI
# ---------------------------------------------------------------------------

def bootstrap_ci(
    values: list[float],
    n_boot: int = N_BOOTSTRAP,
    rng: np.random.Generator | None = None,
) -> tuple[float, float]:
    if rng is None:
        rng = np.random.default_rng(BOOTSTRAP_SEED)
    arr = np.asarray(values, dtype=float)
    if len(arr) <= 1:
        v = float(arr[0]) if len(arr) == 1 else float("nan")
        return v, v
    boot_means = np.array([rng.choice(arr, size=len(arr), replace=True).mean() for _ in range(n_boot)])
    return float(np.percentile(boot_means, 2.5)), float(np.percentile(boot_means, 97.5))


# ---------------------------------------------------------------------------
# Full experiment
# ---------------------------------------------------------------------------

def run_drift_study(
    seeds: list[int],
    horizon: int,
    cost_config: CostConfig,
    results_root: Path,
) -> dict[str, Any]:
    """Run all drift modes × policies × seeds; aggregate and return summary."""
    drift_modes = list(DRIFT_SCHEDULES.keys())
    per_mode_policy: dict[str, dict[str, list[float]]] = {
        dm: {} for dm in drift_modes
    }
    per_mode_policy_regret: dict[str, dict[str, list[float]]] = {
        dm: {} for dm in drift_modes
    }
    per_mode_policy_resets: dict[str, dict[str, list[int]]] = {
        dm: {} for dm in drift_modes
    }

    for seed in seeds:
        print(f"  seed {seed} …", flush=True)
        policies = build_policies(seed, cost_config)
        for drift_mode in drift_modes:
            for policy in policies:
                res = run_drift_trajectory(policy, drift_mode, seed, horizon, cost_config)
                pid = policy.policy_id
                per_mode_policy[drift_mode].setdefault(pid, []).append(res.cumulative_cost)
                per_mode_policy_regret[drift_mode].setdefault(pid, []).append(res.cumulative_regret)
                per_mode_policy_resets[drift_mode].setdefault(pid, []).append(res.drift_resets)

    rng = np.random.default_rng(BOOTSTRAP_SEED)
    conditions: dict[str, Any] = {}
    for drift_mode in drift_modes:
        policies_summary: dict[str, Any] = {}
        for pid, costs in per_mode_policy[drift_mode].items():
            ci_lo, ci_hi = bootstrap_ci(costs, rng=rng)
            policies_summary[pid] = {
                "mean_cost": float(np.mean(costs)),
                "std_cost": float(np.std(costs, ddof=1)) if len(costs) > 1 else 0.0,
                "ci_lo_95": ci_lo,
                "ci_hi_95": ci_hi,
                "per_seed_costs": costs,
                "mean_regret": float(np.mean(per_mode_policy_regret[drift_mode][pid])),
                "mean_drift_resets": float(np.mean(per_mode_policy_resets[drift_mode][pid])),
            }
        conditions[drift_mode] = {"policies": policies_summary}

    report = {
        "evaluation_mode": "drift_eval_synthetic",
        "warning": (
            "Synthetic environment only. Regret is relative to oracle with full "
            "hidden-state knowledge. Not a real-world cost estimate."
        ),
        "horizon": horizon,
        "seeds": seeds,
        "n_seeds": len(seeds),
        "n_bootstrap": N_BOOTSTRAP,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "n_gradual_segments": N_GRADUAL_SEGMENTS,
        "cost_config": {
            "deploy_failure": cost_config.deploy_failure,
            "block_bad": cost_config.block_bad,
        },
        "conditions": conditions,
    }

    results_root.mkdir(parents=True, exist_ok=True)
    (results_root / "drift_eval_summary.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Phase G drift evaluation on synthetic environment."
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=list(range(5)),
        help="Seeds to run (default: 0-4; use 0-29 for ≥30-seed paper runs).",
    )
    parser.add_argument(
        "--horizon",
        type=int,
        default=500,
        help="Steps per trajectory (default: 500).",
    )
    parser.add_argument(
        "--results-root",
        default=str(DEFAULT_RESULTS_ROOT),
        help="Output directory.",
    )
    return parser.parse_args()


def _print_report(report: dict[str, Any]) -> None:
    print(f"\n# Drift Evaluation — {report['n_seeds']} seeds × horizon {report['horizon']}\n")
    print("> " + report["warning"] + "\n")
    for mode, cond in report["conditions"].items():
        print(f"## Drift mode: `{mode}`\n")
        print("| Policy | Mean Cost | Std | 95% CI | Mean Regret | Mean Resets |")
        print("| --- | ---: | ---: | --- | ---: | ---: |")
        for pid, r in cond["policies"].items():
            print(
                f"| {pid} | {r['mean_cost']:.2f} | {r['std_cost']:.2f} "
                f"| [{r['ci_lo_95']:.2f}, {r['ci_hi_95']:.2f}] "
                f"| {r['mean_regret']:.2f} | {r['mean_drift_resets']:.1f} |"
            )
        print()


def main() -> None:
    args = parse_args()
    results_root = Path(args.results_root)
    cost_config = CostConfig()

    print(f"Running drift evaluation: {len(args.seeds)} seeds × {args.horizon} steps × 3 drift modes …")
    report = run_drift_study(
        seeds=args.seeds,
        horizon=args.horizon,
        cost_config=cost_config,
        results_root=results_root,
    )
    _print_report(report)
    print(f"Results written to {results_root / 'drift_eval_summary.json'}")


if __name__ == "__main__":
    main()
