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
from dataclasses import asdict, dataclass, field, replace
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
CALIBRATED_THRESHOLD = None
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

class MomentRatePolicy:
    """Scalar failure-rate control for this simulator's known canary multiplier.

    A method-of-moments estimate, not a conjugate Bayesian posterior. Assumes
    delayed labels are also available for blocked changes, as the simulator does.
    """
    policy_id = "moment_rate"

    def __init__(self, config, window=None):
        self.window = window
        if window is not None:
            self.policy_id = f"moment_rate_window_{window}"
        self.config = config
        self.reset()

    def reset(self):
        self.failures, self.exposure = 1., 2.
        self.history = []

    def select_action(self, context):
        p = min(1., self.failures / self.exposure)
        values = {a: (1-p*(.4 if a == Action.CANARY else 1))*compute_cost(a,Outcome.SUCCESS,self.config)
                  +p*(.4 if a == Action.CANARY else 1)*compute_cost(a,Outcome.FAILURE,self.config) for a in Action}
        return min(values,key=values.get), 1.

    def update(self, context, action, reward):
        if not reward.censored and math.isfinite(reward.cost):
            self.failures += reward.outcome == Outcome.FAILURE
            exposure = .4 if action == Action.CANARY else 1.
            self.exposure += exposure
            self.history.append((reward.outcome == Outcome.FAILURE, exposure))
            if self.window is not None and len(self.history) > self.window:
                failure, exposure = self.history.pop(0)
                self.failures -= failure
                self.exposure -= exposure


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
    from policies.cost_rules import ConstantActionPolicy
    dim = FeatureEncoder.DIM
    return [
        *[ConstantActionPolicy(a) for a in Action],
        MomentRatePolicy(cost_config),
        MomentRatePolicy(cost_config,50),
        MomentRatePolicy(cost_config,100),
        *([CostSensitiveBandit(config=cc,feature_dim=dim,rng=np.random.default_rng(seed),
            detector=PageHinkleyDetector(PageHinkleyConfig(lambda_=CALIBRATED_THRESHOLD)),
            policy_id="linucb_with_drift_calibrated")] if CALIBRATED_THRESHOLD is not None else []),
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
    step_costs: list[float] = field(default_factory=list)    # cost at each decision step
    step_regrets: list[float] = field(default_factory=list)  # expected regret at each decision step


def run_drift_trajectory(
    policy,
    drift_mode: str,
    seed: int,
    horizon: int,
    cost_config: CostConfig,
    delay_p: float = 0.3,
    max_delay: int = 20,
) -> DriftTrajectoryResult:
    """Evaluate decisions 0..T-1 using current context and a hidden-state oracle.

    Report expected pseudo-regret on the decision axis. Realized costs are
    attributed to their originating decision, including feedback after T.
    Terminal flushing affects evaluation only, not the policy or reset counts.
    """
    env = SyntheticEnvironment(np.random.default_rng(seed), horizon,
        DRIFT_SCHEDULES[drift_mode](horizon), delay_p=delay_p, max_delay=max_delay)
    policy.reset()
    if hasattr(policy, "_rng"):
        policy._rng = np.random.default_rng(seed)
    context = env.reset()
    pending = {}
    step_costs = [float("nan")] * horizon
    step_regrets = []
    counts = {a.value: 0 for a in Action}
    updates = 0
    observed_outcomes = []

    def receive(rewards, learn):
        nonlocal updates
        for reward in rewards:
            index, ctx, action = pending.pop(reward.action_id)
            if reward.censored:
                continue
            cost = compute_cost(action, reward.outcome, cost_config)
            step_costs[index] = cost
            if learn:
                policy.update(ctx, action, replace(reward, cost=cost))
                observed_outcomes.append(reward.outcome == Outcome.FAILURE)
                updates += 1

    for step in range(horizon):
        if step:
            receive(env.advance_time(), True)
            context = env.observe()
        # The original synthetic field was a direct noisy projection of hidden p.
        # Use observed history instead of supplying that oracle-like risk feature.
        context = replace(context, recent_failure_rate=(float(np.mean(observed_outcomes[-50:]))
                          if observed_outcomes else 0.0))
        action, _ = policy.select_action(context)
        counts[action.value] += 1
        expected = env.expected_action_costs(cost_config)
        step_regrets.append(expected[action] - min(expected.values()))
        env.step(action)
        pending[env.last_action_id] = (step, context, action)

    resets = policy.stats.drift_resets if hasattr(policy, "stats") else 0
    while pending:
        receive(env.advance_time(), False)
    censored = sum(not math.isfinite(c) for c in step_costs)
    return DriftTrajectoryResult(policy.policy_id, drift_mode, seed,
        float(np.nansum(step_costs)), float(sum(step_regrets)), horizon,
        updates, censored, counts, resets, step_costs, step_regrets)


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
    boot_means = rng.choice(arr, size=(n_boot, len(arr)), replace=True).mean(axis=1)
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
    # Per-step arrays: mode → policy → list of per-seed step arrays (for mean curve)
    per_mode_policy_step_costs: dict[str, dict[str, list[list[float]]]] = {
        dm: {} for dm in drift_modes
    }
    per_mode_policy_step_regrets: dict[str, dict[str, list[list[float]]]] = {
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
                per_mode_policy_step_costs[drift_mode].setdefault(pid, []).append(res.step_costs)
                per_mode_policy_step_regrets[drift_mode].setdefault(pid, []).append(res.step_regrets)

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
                "per_seed_regrets": per_mode_policy_regret[drift_mode][pid],
                "per_seed_resets": per_mode_policy_resets[drift_mode][pid],
            }
        conditions[drift_mode] = {"policies": policies_summary}

    report = {
        "evaluation_mode": "drift_eval_synthetic",
        "warning": (
            "Synthetic environment only. Expected pseudo-regret is relative to oracle with full "
            "hidden-state knowledge. Not a real-world cost estimate."
        ),
        "horizon": horizon,
        "seeds": seeds,
        "n_seeds": len(seeds),
        "n_bootstrap": N_BOOTSTRAP,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "n_gradual_segments": N_GRADUAL_SEGMENTS,
        "cost_config": asdict(cost_config),
        "conditions": conditions,
    }

    results_root.mkdir(parents=True, exist_ok=True)
    (results_root / "drift_eval_summary.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    # Write mean step-cost and step-regret arrays per drift mode × policy.
    # Each array shape: (n_steps_with_reward,) — mean across seeds.
    # No RNG calls; arrays derived from already-computed trajectory results.
    for drift_mode in drift_modes:
        for pid, seed_arrays in per_mode_policy_step_costs[drift_mode].items():
            if seed_arrays and seed_arrays[0]:
                min_len = min(len(a) for a in seed_arrays)
                arr = np.array([a[:min_len] for a in seed_arrays], dtype=np.float32)
                np.save(results_root / f"step_costs_{drift_mode}_{pid}.npy", np.nanmean(arr, axis=0))
                np.save(results_root / f"per_seed_costs_{drift_mode}_{pid}.npy", arr)
        for pid, seed_arrays in per_mode_policy_step_regrets[drift_mode].items():
            if seed_arrays and seed_arrays[0]:
                min_len = min(len(a) for a in seed_arrays)
                arr = np.array([a[:min_len] for a in seed_arrays], dtype=np.float32)
                np.save(results_root / f"step_regrets_{drift_mode}_{pid}.npy", arr.mean(axis=0))

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
