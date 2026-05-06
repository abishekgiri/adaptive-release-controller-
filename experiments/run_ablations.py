"""Phase 22 ablation study for the cost-sensitive delayed bandit.

Compares four variants of the learning policy to isolate which components
contribute to cost reduction:

  full          — CostSensitiveBandit with delayed updates + cost reward +
                   PageHinkley drift detector + model reset on drift
  no_delay      — ImmediateLinUCB: same model, but updated at each step
                   without a delay buffer (look-ahead-free online learning)
  no_cost       — BinaryRewardBandit: delayed updates, but reward is -1 on
                   failure and 0 on success instead of -operational_cost
  no_drift      — CostSensitiveBandit with reset_on_drift=False; drift is
                   detected but the model is NOT reset on detection

WARNING — SIMULATION, NOT CAUSAL INFERENCE.
All costs use compute_cost(policy_action, logged_CI_outcome) as a proxy.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from data.loaders import TravisTorrentLoader, TravisTorrentRecord
from data.schemas import Action, Outcome, Reward
from drift.detectors import PageHinkleyConfig, PageHinkleyDetector
from evaluation.online_replay import (
    OnlineTrajectoryResult,
    _effective_outcome,
    _delay_from_record,
)
from policies.ablation_variants import BinaryRewardBandit, ImmediateLinUCB
from policies.base import FeatureEncoder
from policies.cost_sensitive_bandit import CostSensitiveBandit, CostSensitiveBanditConfig
from policies.linucb import LinUCBConfig
from rewards.cost_model import CostConfig, compute_cost


DEFAULT_DATASET_PATH = Path("data/raw/travistorrent_smoke.csv")
DEFAULT_RESULTS_ROOT = Path("experiments/results")
_ALPHA = 1.0
_LAMBDA_REG = 1.0


@dataclass(frozen=True)
class AblationConfig:
    """Configuration for the ablation experiment."""

    config_name: str = "ablation_smoke"
    dataset_path: Path = DEFAULT_DATASET_PATH
    results_root: Path = DEFAULT_RESULTS_ROOT
    min_builds: int = 500
    min_history_days: int = 365
    delay_step_seconds: int = 60
    cost_config: CostConfig = field(default_factory=CostConfig)
    linucb_alpha: float = _ALPHA
    linucb_lambda_reg: float = _LAMBDA_REG
    flush_at_end: bool = True


def build_ablation_policies(seed: int) -> list:
    """Instantiate all four ablation variants."""
    rng = np.random.default_rng(seed)
    lc = LinUCBConfig(alpha=_ALPHA, lambda_reg=_LAMBDA_REG)
    cc = CostSensitiveBanditConfig(
        alpha=_ALPHA,
        lambda_reg=_LAMBDA_REG,
        cost_config=CostConfig(),
        reset_on_drift=True,
    )
    cc_no_drift = CostSensitiveBanditConfig(
        alpha=_ALPHA,
        lambda_reg=_LAMBDA_REG,
        cost_config=CostConfig(),
        reset_on_drift=False,
    )
    return [
        # full: delayed + cost reward + drift reset
        CostSensitiveBandit(
            config=cc,
            feature_dim=FeatureEncoder.DIM,
            rng=np.random.default_rng(seed),
            detector=PageHinkleyDetector(PageHinkleyConfig(lambda_=50.0)),
            policy_id="full",
        ),
        # no_delay: immediate updates (ImmediateLinUCB triggers different runner path)
        ImmediateLinUCB(
            config=lc,
            feature_dim=FeatureEncoder.DIM,
            rng=np.random.default_rng(seed),
            policy_id="no_delay",
        ),
        # no_cost: binary reward instead of -cost
        BinaryRewardBandit(
            config=lc,
            feature_dim=FeatureEncoder.DIM,
            rng=np.random.default_rng(seed),
            policy_id="no_cost",
        ),
        # no_drift: delayed + cost reward, NO model reset on drift detection
        CostSensitiveBandit(
            config=cc_no_drift,
            feature_dim=FeatureEncoder.DIM,
            rng=np.random.default_rng(seed),
            detector=PageHinkleyDetector(PageHinkleyConfig(lambda_=50.0)),
            policy_id="no_drift",
        ),
    ]


def run_immediate_trajectory(
    policy,
    records: list[TravisTorrentRecord],
    *,
    cost_config: CostConfig,
    trajectory_id: str = "",
) -> OnlineTrajectoryResult:
    """Ablation runner: update policy immediately at each step (no buffer delay).

    The policy receives the reward in the same step the action was taken.
    This is NOT valid for real deployment (outcomes aren't instantly observable)
    but serves as an ablation baseline to quantify the value of delay handling.
    """
    project_slug = records[0].context.project_slug if records else ""
    if not records:
        return OnlineTrajectoryResult(
            policy_id=policy.policy_id,
            trajectory_id=trajectory_id,
            project_slug=project_slug,
            total_steps=0,
            total_updates=0,
            total_censored_skipped=0,
            cumulative_cost=0.0,
            action_counts={a.value: 0 for a in Action},
        )

    cumulative_cost = 0.0
    total_updates = 0
    total_censored = 0
    action_counts: dict[str, int] = {a.value: 0 for a in Action}

    for step, record in enumerate(records):
        policy_action, _ = policy.select_action(record.context)
        action_counts[policy_action.value] += 1

        effective = _effective_outcome(policy_action, record.outcome)
        cost = compute_cost(policy_action, effective, cost_config)
        is_censored = not math.isfinite(cost)

        if not is_censored:
            cumulative_cost += cost
            reward = Reward(
                action_id=f"{trajectory_id}:{step}",
                outcome=effective,
                cost=cost,
                delay_steps=0,
                censored=False,
                observed_at_step=step,
            )
            policy.update(record.context, policy_action, reward)
            total_updates += 1
        else:
            total_censored += 1

    return OnlineTrajectoryResult(
        policy_id=policy.policy_id,
        trajectory_id=trajectory_id,
        project_slug=project_slug,
        total_steps=len(records),
        total_updates=total_updates,
        total_censored_skipped=total_censored,
        cumulative_cost=cumulative_cost,
        action_counts=action_counts,
    )


def run_ablation_experiment(
    config: AblationConfig,
    seed: int,
) -> dict[str, list[OnlineTrajectoryResult]]:
    """Run all four ablation variants over every project trajectory."""
    from evaluation.online_replay import run_online_trajectory

    loader = TravisTorrentLoader(
        config.dataset_path,
        min_builds=config.min_builds,
        min_history_days=config.min_history_days,
    )
    records_by_project: dict[str, list[TravisTorrentRecord]] = {}
    for record in loader.iter_records():
        records_by_project.setdefault(record.context.project_slug, []).append(record)

    policies = build_ablation_policies(seed)
    rng = np.random.default_rng(seed)
    project_keys = sorted(records_by_project)
    project_seeds = rng.integers(0, 2**31, size=len(project_keys))

    results: dict[str, list[OnlineTrajectoryResult]] = {
        p.policy_id: [] for p in policies
    }

    for project_key, project_seed in zip(project_keys, project_seeds):
        records = records_by_project[project_key]
        for policy in policies:
            policy.reset()
            traj_id = f"ablation:{project_key}"

            if isinstance(policy, ImmediateLinUCB):
                result = run_immediate_trajectory(
                    policy=policy,
                    records=records,
                    cost_config=config.cost_config,
                    trajectory_id=traj_id,
                )
            else:
                result = run_online_trajectory(
                    policy=policy,
                    records=records,
                    cost_config=config.cost_config,
                    rng=np.random.default_rng(int(project_seed)),
                    delay_step_seconds=config.delay_step_seconds,
                    trajectory_id=traj_id,
                    flush_at_end=config.flush_at_end,
                )
            results[policy.policy_id].append(result)

    return results


def build_summary(
    config: AblationConfig,
    seed: int,
    results: dict[str, list[OnlineTrajectoryResult]],
) -> dict[str, Any]:
    """Build JSON-serializable ablation summary."""
    policies_summary: dict[str, Any] = {}
    for policy_id, traj_list in results.items():
        total_steps = sum(r.total_steps for r in traj_list)
        total_updates = sum(r.total_updates for r in traj_list)
        total_censored = sum(r.total_censored_skipped for r in traj_list)
        total_cost = sum(r.cumulative_cost for r in traj_list)
        action_counts: dict[str, int] = {}
        for r in traj_list:
            for a, cnt in r.action_counts.items():
                action_counts[a] = action_counts.get(a, 0) + cnt

        finite_steps = total_steps - total_censored
        policies_summary[policy_id] = {
            "total_steps": total_steps,
            "total_updates": total_updates,
            "total_censored_skipped": total_censored,
            "cumulative_cost": total_cost,
            "mean_cost_per_step": total_cost / finite_steps if finite_steps > 0 else None,
            "action_counts": action_counts,
            "action_fractions": {
                a: cnt / total_steps if total_steps > 0 else 0.0
                for a, cnt in action_counts.items()
            },
        }

    return {
        "config_name": config.config_name,
        "evaluation_mode": "ablation_online_replay_simulation",
        "warning": (
            "Ablation costs are simulation artefacts. "
            "CI outcome used as counterfactual proxy."
        ),
        "seed": seed,
        "dataset_path": str(config.dataset_path),
        "policies": policies_summary,
    }


def write_results(
    summary: dict[str, Any],
    config: AblationConfig,
    seed: int,
) -> Path:
    """Write JSON and Markdown under experiments/results/<config>/<seed>/."""
    output_dir = config.results_root / config.config_name / str(seed)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "ablation_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return output_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase 22 ablation study.")
    parser.add_argument(
        "--dataset",
        default=str(DEFAULT_DATASET_PATH),
        help="Path to TravisTorrent-format CSV.",
    )
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--results-root", default=str(DEFAULT_RESULTS_ROOT))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = AblationConfig(
        dataset_path=Path(args.dataset),
        results_root=Path(args.results_root),
    )
    results = run_ablation_experiment(config, seed=args.seed)
    summary = build_summary(config, seed=args.seed, results=results)
    output_dir = write_results(summary, config=config, seed=args.seed)

    # Print markdown table
    print(f"\n# Ablation Study — seed {args.seed}\n")
    print("| Variant | Steps | Updates | Censored | Cumul. Cost | Mean/Step | Deploy% | Canary% | Block% |")
    print("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for pid, result in summary["policies"].items():
        fracs = result["action_fractions"]
        mean = result["mean_cost_per_step"]
        print(
            f"| {pid} | {result['total_steps']} | {result['total_updates']} "
            f"| {result['total_censored_skipped']} | {result['cumulative_cost']:.2f} "
            f"| {mean:.4f} | {fracs.get('deploy', 0):.1%} "
            f"| {fracs.get('canary', 0):.1%} | {fracs.get('block', 0):.1%} |"
        )
    print(f"\nResults written to {output_dir}")


if __name__ == "__main__":
    main()
