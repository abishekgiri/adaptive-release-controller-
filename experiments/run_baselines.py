"""Run the first real TravisTorrent replay experiment.

This script is deliberately evaluator-only: it loads local data, constructs
logged trajectories with observed costs, evaluates existing policies with IPS,
and writes cost-first summaries. It does not download data.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from data.loaders import TravisTorrentLoader, TravisTorrentRecord
from data.schemas import Action, Outcome, Reward, Trajectory, TrajectoryStep
from evaluation.replay_eval import IPSConfig, IPSResult, evaluate_ips
from evaluation.statistical import BootstrapConfig, bootstrap_ci
from policies.base import FeatureEncoder, Policy
from policies.cost_sensitive_bandit import (
    CostSensitiveBandit,
    CostSensitiveBanditConfig,
)
from policies.linucb import LinUCBConfig, LinUCBPolicy
from policies.static_rules import StaticRulesPolicy
from rewards.cost_model import CostConfig, compute_cost


DEFAULT_CONFIG_PATH = Path("experiments/configs/first_real_result.json")
DEFAULT_RESULTS_ROOT = Path("experiments/results")

RUNNABLE_POLICIES = {"static-rules", "linucb", "cost-sensitive-bandit"}
TODO_POLICIES = {
    "heuristic-score": "TODO: HeuristicScorePolicy exists but select_action is not implemented.",
    "offline-classifier": "TODO: OfflineClassifierPolicy exists but fit/select_action are not implemented.",
    "thompson": "TODO: ThompsonSamplingPolicy exists but posterior updates are not implemented.",
}


@dataclass(frozen=True)
class ExperimentConfig:
    """Configuration for the first real replay experiment."""

    config_name: str
    dataset_path: Path
    results_root: Path = DEFAULT_RESULTS_ROOT
    min_builds: int = 500
    min_history_days: int = 365
    max_trajectories: int | None = None
    logged_propensity: float = 1.0
    delay_step_seconds: int = 60
    propensity_clip: float = 20.0
    bootstrap_resamples: int = 1000
    confidence: float = 0.95
    policies: tuple[str, ...] = (
        "static-rules",
        "heuristic-score",
        "offline-classifier",
        "linucb",
        "cost-sensitive-bandit",
        "thompson",
    )
    cost_config: CostConfig = field(default_factory=CostConfig)
    linucb_alpha: float = 1.0
    linucb_lambda_reg: float = 1.0
    cost_sensitive_alpha: float = 1.0
    cost_sensitive_lambda_reg: float = 1.0


def load_config(path: str | Path) -> ExperimentConfig:
    """Load experiment configuration from JSON."""

    config_path = Path(path)
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    cost_payload = payload.get("cost_config", {})
    return ExperimentConfig(
        config_name=str(payload["config_name"]),
        dataset_path=Path(payload["dataset_path"]),
        results_root=Path(payload.get("results_root", DEFAULT_RESULTS_ROOT)),
        min_builds=int(payload.get("min_builds", 500)),
        min_history_days=int(payload.get("min_history_days", 365)),
        max_trajectories=_optional_int(payload.get("max_trajectories")),
        logged_propensity=float(payload.get("logged_propensity", 1.0)),
        delay_step_seconds=int(payload.get("delay_step_seconds", 60)),
        propensity_clip=float(payload.get("propensity_clip", 20.0)),
        bootstrap_resamples=int(payload.get("bootstrap_resamples", 1000)),
        confidence=float(payload.get("confidence", 0.95)),
        policies=tuple(payload.get("policies", ExperimentConfig.policies)),
        cost_config=CostConfig(**cost_payload),
        linucb_alpha=float(payload.get("linucb_alpha", 1.0)),
        linucb_lambda_reg=float(payload.get("linucb_lambda_reg", 1.0)),
        cost_sensitive_alpha=float(payload.get("cost_sensitive_alpha", 1.0)),
        cost_sensitive_lambda_reg=float(payload.get("cost_sensitive_lambda_reg", 1.0)),
    )


def run_experiment(config: ExperimentConfig, seed: int) -> dict[str, Any]:
    """Run IPS evaluation for all configured policies."""

    trajectories = load_logged_trajectories(config)
    runnable_results: dict[str, list[IPSResult]] = {}
    placeholders: dict[str, str] = {}

    for policy_name in config.policies:
        if policy_name in TODO_POLICIES:
            placeholders[policy_name] = TODO_POLICIES[policy_name]
            continue
        if policy_name not in RUNNABLE_POLICIES:
            placeholders[policy_name] = f"TODO: unknown or unsupported policy {policy_name!r}."
            continue

        policy = build_policy(policy_name, config=config, seed=seed)
        runnable_results[policy.policy_id] = [
            evaluate_ips(
                policy=policy,
                trajectory=trajectory,
                config=IPSConfig(propensity_clip=config.propensity_clip),
            )
            for trajectory in trajectories
        ]

    summary = build_summary(
        config=config,
        seed=seed,
        trajectories=trajectories,
        results=runnable_results,
        placeholders=placeholders,
    )
    write_results(summary, config=config, seed=seed)
    return summary


def load_logged_trajectories(config: ExperimentConfig) -> list[Trajectory]:
    """Load local TravisTorrent records and attach observed cost rewards."""

    loader = TravisTorrentLoader(
        config.dataset_path,
        min_builds=config.min_builds,
        min_history_days=config.min_history_days,
    )
    records_by_project: dict[str, list[TravisTorrentRecord]] = {}
    for record in loader.iter_records():
        records_by_project.setdefault(record.context.project_slug, []).append(record)

    trajectories = []
    for project_slug in sorted(records_by_project):
        records = records_by_project[project_slug]
        steps = tuple(
            logged_step(
                record=record,
                index=index,
                config=config,
            )
            for index, record in enumerate(records)
        )
        trajectories.append(
            Trajectory(
                trajectory_id=f"travistorrent:{project_slug}",
                project_slug=project_slug,
                policy_id="logged_travistorrent",
                drift_segment_id=None,
                steps=steps,
            )
        )
        if config.max_trajectories is not None and len(trajectories) >= config.max_trajectories:
            break
    return trajectories


def logged_step(
    record: TravisTorrentRecord,
    index: int,
    config: ExperimentConfig,
) -> TrajectoryStep:
    """Convert a loaded record into an IPS-ready logged trajectory step."""

    censored = record.outcome == Outcome.CENSORED
    cost = compute_cost(record.action, record.outcome, config.cost_config)
    reward = Reward(
        action_id=f"{record.context.project_slug}:{record.context.commit_sha}:{index}",
        outcome=record.outcome,
        cost=cost,
        delay_steps=delay_steps(record, config.delay_step_seconds),
        censored=censored or not math.isfinite(cost),
        observed_at_step=record.context.step + delay_steps(record, config.delay_step_seconds),
    )
    return TrajectoryStep(
        context=record.context,
        action=record.action,
        propensity=config.logged_propensity,
        reward=reward,
    )


def build_policy(policy_name: str, config: ExperimentConfig, seed: int) -> Policy:
    """Instantiate a runnable policy by config name."""

    if policy_name == "static-rules":
        return StaticRulesPolicy(policy_id="static_rules")
    if policy_name == "linucb":
        return LinUCBPolicy(
            config=LinUCBConfig(
                alpha=config.linucb_alpha,
                lambda_reg=config.linucb_lambda_reg,
            ),
            feature_dim=FeatureEncoder.DIM,
            rng=np.random.default_rng(seed),
            policy_id="linucb",
        )
    if policy_name == "cost-sensitive-bandit":
        return CostSensitiveBandit(
            config=CostSensitiveBanditConfig(
                alpha=config.cost_sensitive_alpha,
                lambda_reg=config.cost_sensitive_lambda_reg,
                cost_config=config.cost_config,
            ),
            feature_dim=FeatureEncoder.DIM,
            rng=np.random.default_rng(seed),
            policy_id="cost_sensitive_bandit",
        )
    raise ValueError(f"Unsupported runnable policy: {policy_name}")


def build_summary(
    config: ExperimentConfig,
    seed: int,
    trajectories: list[Trajectory],
    results: dict[str, list[IPSResult]],
    placeholders: dict[str, str],
) -> dict[str, Any]:
    """Build JSON-serializable experiment summary."""

    bootstrap = BootstrapConfig(
        n_resamples=config.bootstrap_resamples,
        confidence=config.confidence,
        seed=seed,
    )
    policies: dict[str, Any] = {}
    for policy_id, policy_results in results.items():
        values = [result.estimated_policy_value for result in policy_results]
        costs = [result.estimated_cumulative_cost for result in policy_results]
        value_mean, value_lower, value_upper = bootstrap_ci(values, config=bootstrap)
        cost_mean, cost_lower, cost_upper = bootstrap_ci(costs, config=bootstrap)
        policies[policy_id] = {
            "status": "evaluated",
            "cumulative_operational_cost": sum(costs),
            "ips_estimated_policy_value": value_mean,
            "ips_policy_value_ci95": [value_lower, value_upper],
            "mean_trajectory_cost": cost_mean,
            "mean_trajectory_cost_ci95": [cost_lower, cost_upper],
            "matched_actions": sum(result.matched_actions for result in policy_results),
            "effective_sample_size": sum(result.effective_sample_size for result in policy_results),
            "evaluated_steps": sum(result.evaluated_steps for result in policy_results),
            "skipped_censored": sum(result.skipped_censored for result in policy_results),
        }

    for policy_name, reason in placeholders.items():
        policies[policy_name] = {
            "status": "todo_placeholder",
            "reason": reason,
        }

    return {
        "config_name": config.config_name,
        "seed": seed,
        "dataset_path": str(config.dataset_path),
        "trajectory_count": len(trajectories),
        "step_count": sum(len(trajectory.steps) for trajectory in trajectories),
        "metrics": [
            "cumulative_operational_cost",
            "ips_estimated_policy_value",
            "matched_actions",
            "effective_sample_size",
            "bootstrap_95_ci",
        ],
        "policies": policies,
    }


def write_results(summary: dict[str, Any], config: ExperimentConfig, seed: int) -> Path:
    """Write JSON and Markdown results under experiments/results/<config>/<seed>/."""

    output_dir = result_dir(config, seed)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output_dir / "summary.md").write_text(
        summary_markdown(summary),
        encoding="utf-8",
    )
    return output_dir


def result_dir(config: ExperimentConfig, seed: int) -> Path:
    """Return the required per-config, per-seed output directory."""

    return config.results_root / config.config_name / str(seed)


def summary_markdown(summary: dict[str, Any]) -> str:
    """Render a compact Markdown result table."""

    lines = [
        f"# First Real Result: {summary['config_name']}",
        "",
        f"Seed: `{summary['seed']}`",
        f"Trajectories: `{summary['trajectory_count']}`",
        f"Steps: `{summary['step_count']}`",
        "",
        "| Policy | Status | Cumulative Cost | IPS Policy Value | Matched Actions | ESS | Notes |",
        "| --- | --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for policy_id, result in summary["policies"].items():
        if result["status"] != "evaluated":
            lines.append(
                f"| {policy_id} | {result['status']} |  |  |  |  | {result['reason']} |"
            )
            continue
        lines.append(
            "| "
            + " | ".join(
                [
                    policy_id,
                    result["status"],
                    f"{result['cumulative_operational_cost']:.4f}",
                    f"{result['ips_estimated_policy_value']:.4f}",
                    str(result["matched_actions"]),
                    f"{result['effective_sample_size']:.4f}",
                    "cost-first IPS estimate",
                ]
            )
            + " |"
        )
    return "\n".join(lines) + "\n"


def delay_steps(record: TravisTorrentRecord, delay_step_seconds: int) -> int:
    """Convert build duration into at least one discrete delay step."""

    if delay_step_seconds <= 0:
        raise ValueError("delay_step_seconds must be positive")
    if record.context.build_duration_s <= 0:
        return 1
    return max(1, math.ceil(record.context.build_duration_s / delay_step_seconds))


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    return int(value)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run first real replay baseline experiment.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument("--seed", type=int, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    summary = run_experiment(config=config, seed=args.seed)
    print(summary_markdown(summary))
    print(f"Results written to {result_dir(config, args.seed)}")


if __name__ == "__main__":
    main()
