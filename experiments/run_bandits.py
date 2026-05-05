"""Online-replay experiment entrypoint for learning policies.

This script runs LinUCB and the cost-sensitive delayed bandit in online-replay
mode: policies learn during the trajectory (delayed rewards update the model
in chronological order) rather than being evaluated from a fixed prior.

IMPORTANT — SIMULATION LABEL
=============================
Online replay is NOT unbiased counterfactual evaluation. Costs are computed as
``compute_cost(policy_action, logged_outcome)`` using the CI outcome as a proxy
counterfactual. Results prove that learning occurs and allow learning curves to
be compared, but do not establish causal real-world cost estimates.

For offline IPS evaluation (static policies, no learning) see:
    experiments/run_baselines.py
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
from evaluation.online_replay import (
    OnlineTrajectoryResult,
    run_online_experiment,
)
from policies.base import FeatureEncoder
from policies.cost_sensitive_bandit import CostSensitiveBandit, CostSensitiveBanditConfig
from policies.heuristic_score import HeuristicScorePolicy
from policies.linucb import LinUCBConfig, LinUCBPolicy
from policies.static_rules import StaticRulesPolicy
from rewards.cost_model import CostConfig


DEFAULT_CONFIG_PATH = Path("experiments/configs/first_real_result.json")
DEFAULT_RESULTS_ROOT = Path("experiments/results")


@dataclass(frozen=True)
class OnlineExperimentConfig:
    """Configuration for the online-replay bandit experiment."""

    config_name: str
    dataset_path: Path
    results_root: Path = DEFAULT_RESULTS_ROOT
    min_builds: int = 500
    min_history_days: int = 365
    max_trajectories: int | None = None
    delay_step_seconds: int = 60
    cost_config: CostConfig = field(default_factory=CostConfig)
    linucb_alpha: float = 1.0
    linucb_lambda_reg: float = 1.0
    cost_sensitive_alpha: float = 1.0
    cost_sensitive_lambda_reg: float = 1.0
    flush_at_end: bool = True


def load_config(path: str | Path) -> OnlineExperimentConfig:
    """Load experiment configuration from JSON (shares schema with run_baselines)."""
    config_path = Path(path)
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    cost_payload = payload.get("cost_config", {})
    return OnlineExperimentConfig(
        config_name=str(payload["config_name"]),
        dataset_path=Path(payload["dataset_path"]),
        results_root=Path(payload.get("results_root", DEFAULT_RESULTS_ROOT)),
        min_builds=int(payload.get("min_builds", 500)),
        min_history_days=int(payload.get("min_history_days", 365)),
        max_trajectories=_optional_int(payload.get("max_trajectories")),
        delay_step_seconds=int(payload.get("delay_step_seconds", 60)),
        cost_config=CostConfig(**cost_payload),
        linucb_alpha=float(payload.get("linucb_alpha", 1.0)),
        linucb_lambda_reg=float(payload.get("linucb_lambda_reg", 1.0)),
        cost_sensitive_alpha=float(payload.get("cost_sensitive_alpha", 1.0)),
        cost_sensitive_lambda_reg=float(payload.get("cost_sensitive_lambda_reg", 1.0)),
    )


def load_records_by_project(
    config: OnlineExperimentConfig,
) -> dict[str, list[TravisTorrentRecord]]:
    """Load TravisTorrent records grouped by project slug."""
    loader = TravisTorrentLoader(
        config.dataset_path,
        min_builds=config.min_builds,
        min_history_days=config.min_history_days,
    )
    records_by_project: dict[str, list[TravisTorrentRecord]] = {}
    for record in loader.iter_records():
        records_by_project.setdefault(record.context.project_slug, []).append(record)

    if config.max_trajectories is not None:
        keys = sorted(records_by_project)[: config.max_trajectories]
        records_by_project = {k: records_by_project[k] for k in keys}

    return records_by_project


def build_policies(config: OnlineExperimentConfig, seed: int) -> list:
    """Instantiate all policies for an online-replay run."""
    rng = np.random.default_rng(seed)
    return [
        StaticRulesPolicy(policy_id="static_rules"),
        HeuristicScorePolicy(policy_id="heuristic_score"),
        LinUCBPolicy(
            config=LinUCBConfig(
                alpha=config.linucb_alpha,
                lambda_reg=config.linucb_lambda_reg,
            ),
            feature_dim=FeatureEncoder.DIM,
            rng=rng,
            policy_id="linucb",
        ),
        CostSensitiveBandit(
            config=CostSensitiveBanditConfig(
                alpha=config.cost_sensitive_alpha,
                lambda_reg=config.cost_sensitive_lambda_reg,
                cost_config=config.cost_config,
            ),
            feature_dim=FeatureEncoder.DIM,
            rng=np.random.default_rng(seed),
            policy_id="cost_sensitive_bandit",
        ),
    ]


def run_experiment(config: OnlineExperimentConfig, seed: int) -> dict[str, Any]:
    """Run online replay for all policies and write results."""
    records_by_project = load_records_by_project(config)
    policies = build_policies(config, seed)
    rng = np.random.default_rng(seed)

    all_results = run_online_experiment(
        policies=policies,
        records_by_project=records_by_project,
        cost_config=config.cost_config,
        rng=rng,
        delay_step_seconds=config.delay_step_seconds,
        flush_at_end=config.flush_at_end,
    )

    summary = build_summary(config=config, seed=seed, results=all_results)
    write_results(summary, config=config, seed=seed)
    return summary


def build_summary(
    config: OnlineExperimentConfig,
    seed: int,
    results: dict[str, list[OnlineTrajectoryResult]],
) -> dict[str, Any]:
    """Build JSON-serializable experiment summary."""
    policies_summary: dict[str, Any] = {}
    for policy_id, trajectory_results in results.items():
        total_steps = sum(r.total_steps for r in trajectory_results)
        total_updates = sum(r.total_updates for r in trajectory_results)
        total_censored = sum(r.total_censored_skipped for r in trajectory_results)
        total_cost = sum(r.cumulative_cost for r in trajectory_results)
        action_counts: dict[str, int] = {}
        for r in trajectory_results:
            for action, count in r.action_counts.items():
                action_counts[action] = action_counts.get(action, 0) + count

        finite_steps = total_steps - total_censored
        policies_summary[policy_id] = {
            "trajectory_count": len(trajectory_results),
            "total_steps": total_steps,
            "finite_steps": finite_steps,
            "total_updates": total_updates,
            "total_censored_skipped": total_censored,
            "cumulative_cost": total_cost,
            "mean_cost_per_step": total_cost / finite_steps if finite_steps > 0 else None,
            "action_counts": action_counts,
            "action_fractions": {
                a: count / total_steps if total_steps > 0 else 0.0
                for a, count in action_counts.items()
            },
            "note": (
                "SIMULATION: costs computed from logged CI outcome as counterfactual proxy. "
                "Not an unbiased causal estimate."
            ),
        }

    return {
        "config_name": config.config_name,
        "evaluation_mode": "online_replay_simulation",
        "warning": (
            "Online replay costs are simulation artefacts. "
            "CI outcome used as counterfactual proxy. "
            "Do not report as causal real-world cost estimates."
        ),
        "seed": seed,
        "dataset_path": str(config.dataset_path),
        "trajectory_count": len(records_by_project_count(results)),
        "policies": policies_summary,
    }


def records_by_project_count(results: dict[str, list[OnlineTrajectoryResult]]) -> set[str]:
    """Return the set of project slugs across all results."""
    projects: set[str] = set()
    for traj_list in results.values():
        for traj in traj_list:
            projects.add(traj.project_slug)
    return projects


def write_results(
    summary: dict[str, Any],
    config: OnlineExperimentConfig,
    seed: int,
) -> Path:
    """Write JSON and Markdown results under experiments/results/<config>/<seed>/."""
    output_dir = config.results_root / config.config_name / str(seed)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "online_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output_dir / "online_summary.md").write_text(
        _summary_markdown(summary),
        encoding="utf-8",
    )
    return output_dir


def _summary_markdown(summary: dict[str, Any]) -> str:
    lines = [
        f"# Online Replay Result: {summary['config_name']}",
        "",
        f"> **{summary['warning']}**",
        "",
        f"Seed: `{summary['seed']}`  |  Trajectories: `{summary['trajectory_count']}`",
        "",
        "| Policy | Steps | Updates | Censored | Cumul. Cost | Mean Cost/Step | Deploy% | Canary% | Block% |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for policy_id, result in summary["policies"].items():
        fracs = result.get("action_fractions", {})
        mean = result.get("mean_cost_per_step")
        mean_str = f"{mean:.4f}" if mean is not None else "—"
        lines.append(
            "| "
            + " | ".join([
                policy_id,
                str(result["total_steps"]),
                str(result["total_updates"]),
                str(result["total_censored_skipped"]),
                f"{result['cumulative_cost']:.4f}",
                mean_str,
                f"{fracs.get('deploy', 0):.1%}",
                f"{fracs.get('canary', 0):.1%}",
                f"{fracs.get('block', 0):.1%}",
            ])
            + " |"
        )
    return "\n".join(lines) + "\n"


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    return int(value)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Online-replay experiment for learning policies."
    )
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument("--seed", type=int, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    summary = run_experiment(config=config, seed=args.seed)
    print(_summary_markdown(summary))
    output_dir = config.results_root / config.config_name / str(args.seed)
    print(f"Results written to {output_dir}")


if __name__ == "__main__":
    main()
