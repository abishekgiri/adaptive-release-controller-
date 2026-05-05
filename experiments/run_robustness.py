"""Phase 21 robustness analysis for the online replay experiment.

Runs the same online replay experiment (static_rules, LinUCB, cost-sensitive-bandit)
across multiple experimental conditions (cost matrices, delay settings) and multiple
seeds, then aggregates with percentile bootstrap confidence intervals.

WARNING — SIMULATION, NOT CAUSAL INFERENCE.
Costs use compute_cost(policy_action, logged_CI_outcome) as a proxy counterfactual.

Usage:
    python -m experiments.run_robustness \\
        --configs experiments/configs/online_smoke.json \\
                  experiments/configs/robustness_high_failure.json \\
                  experiments/configs/robustness_low_block.json \\
                  experiments/configs/robustness_short_delay.json \\
                  experiments/configs/robustness_long_delay.json \\
        --seeds 0 1 2 3 4 \\
        --results-root experiments/results
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from experiments.run_bandits import (
    OnlineExperimentConfig,
    load_config,
    run_experiment,
)

DEFAULT_RESULTS_ROOT = Path("experiments/results/robustness")
N_BOOTSTRAP = 10_000
BOOTSTRAP_SEED = 42


def bootstrap_ci(
    values: list[float],
    n_boot: int = N_BOOTSTRAP,
    alpha: float = 0.05,
    rng: np.random.Generator | None = None,
) -> tuple[float, float]:
    """Percentile bootstrap 95% CI for the mean.

    Args:
        values: Observed values (one per seed).
        n_boot:  Number of bootstrap resamples.
        alpha:   Two-sided error rate; default 0.05 → 95% CI.
        rng:     Seeded RNG for reproducibility.

    Returns:
        (lower, upper) confidence interval bounds.
    """
    if rng is None:
        rng = np.random.default_rng(BOOTSTRAP_SEED)
    arr = np.asarray(values, dtype=float)
    if len(arr) == 0:
        return float("nan"), float("nan")
    if len(arr) == 1:
        return float(arr[0]), float(arr[0])
    boot_means = np.array([
        rng.choice(arr, size=len(arr), replace=True).mean()
        for _ in range(n_boot)
    ])
    lo = float(np.percentile(boot_means, 100 * alpha / 2))
    hi = float(np.percentile(boot_means, 100 * (1 - alpha / 2)))
    return lo, hi


def run_condition(
    config: OnlineExperimentConfig,
    seeds: list[int],
) -> dict[str, dict[str, Any]]:
    """Run one experimental condition over multiple seeds.

    Returns:
        Dict[policy_id → {mean_cost, ci_lo, ci_hi, per_seed_costs,
                           mean_deploy_frac, mean_canary_frac, mean_block_frac,
                           mean_updates}]
    """
    per_seed: dict[str, list[float]] = {}
    per_seed_fracs: dict[str, list[dict[str, float]]] = {}
    per_seed_updates: dict[str, list[int]] = {}

    for seed in seeds:
        summary = run_experiment(config, seed)
        for policy_id, result in summary["policies"].items():
            cost = result.get("cumulative_cost", float("nan"))
            per_seed.setdefault(policy_id, []).append(cost)
            fracs = result.get("action_fractions", {})
            per_seed_fracs.setdefault(policy_id, []).append(fracs)
            updates = result.get("total_updates", 0)
            per_seed_updates.setdefault(policy_id, []).append(updates)

    rng = np.random.default_rng(BOOTSTRAP_SEED)
    condition_results: dict[str, dict[str, Any]] = {}
    for policy_id in per_seed:
        costs = per_seed[policy_id]
        fracs_list = per_seed_fracs[policy_id]
        updates_list = per_seed_updates[policy_id]
        ci_lo, ci_hi = bootstrap_ci(costs, rng=rng)
        condition_results[policy_id] = {
            "mean_cost": float(np.mean(costs)),
            "std_cost": float(np.std(costs)),
            "ci_lo": ci_lo,
            "ci_hi": ci_hi,
            "per_seed_costs": costs,
            "mean_deploy_frac": float(np.mean([f.get("deploy", 0) for f in fracs_list])),
            "mean_canary_frac": float(np.mean([f.get("canary", 0) for f in fracs_list])),
            "mean_block_frac": float(np.mean([f.get("block", 0) for f in fracs_list])),
            "mean_updates": float(np.mean(updates_list)),
        }
    return condition_results


def run_robustness_study(
    config_paths: list[Path],
    seeds: list[int],
    results_root: Path,
) -> dict[str, Any]:
    """Run all conditions and collect aggregated results.

    Returns:
        Full robustness report as a JSON-serializable dict.
    """
    results_root.mkdir(parents=True, exist_ok=True)
    all_conditions: dict[str, Any] = {}

    for config_path in config_paths:
        config = load_config(config_path)
        print(f"Running condition: {config.config_name} × {len(seeds)} seeds …")
        condition_results = run_condition(config, seeds)
        all_conditions[config.config_name] = {
            "config_file": str(config_path),
            "dataset_path": str(config.dataset_path),
            "seeds": seeds,
            "cost_config": {
                "deploy_failure": config.cost_config.deploy_failure,
                "canary_failure": config.cost_config.canary_failure,
                "block_safe": config.cost_config.block_safe,
                "block_bad": config.cost_config.block_bad,
                "delay_step_seconds": config.delay_step_seconds,
            },
            "policies": condition_results,
        }

    report = {
        "evaluation_mode": "robustness_online_replay_simulation",
        "warning": (
            "Costs are simulation artefacts from online replay. "
            "CI outcome used as counterfactual proxy."
        ),
        "n_bootstrap": N_BOOTSTRAP,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "conditions": all_conditions,
    }
    (results_root / "robustness_summary.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase 21 robustness analysis.")
    parser.add_argument(
        "--configs",
        nargs="+",
        required=True,
        help="One or more config JSON paths.",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=list(range(5)),
        help="Seeds to run (default: 0-4).",
    )
    parser.add_argument(
        "--results-root",
        default=str(DEFAULT_RESULTS_ROOT),
    )
    return parser.parse_args()


def _print_report(report: dict[str, Any]) -> None:
    """Print a compact Markdown table for each condition."""
    print("\n# Phase 21 Robustness Results\n")
    print("> " + report["warning"] + "\n")
    for cond_name, cond in report["conditions"].items():
        df = cond["cost_config"]["deploy_failure"]
        delay = cond["cost_config"]["delay_step_seconds"]
        print(f"## Condition: `{cond_name}`  "
              f"(deploy_failure={df}, delay_step_seconds={delay})\n")
        print("| Policy | Mean Cost | 95% CI | Deploy% | Canary% | Block% | Updates |")
        print("| --- | ---: | --- | ---: | ---: | ---: | ---: |")
        for pid, r in cond["policies"].items():
            print(
                f"| {pid} | {r['mean_cost']:.2f} "
                f"| [{r['ci_lo']:.2f}, {r['ci_hi']:.2f}] "
                f"| {r['mean_deploy_frac']:.1%} "
                f"| {r['mean_canary_frac']:.1%} "
                f"| {r['mean_block_frac']:.1%} "
                f"| {r['mean_updates']:.0f} |"
            )
        print()


def main() -> None:
    args = parse_args()
    config_paths = [Path(p) for p in args.configs]
    seeds = list(args.seeds)
    results_root = Path(args.results_root)

    report = run_robustness_study(config_paths, seeds, results_root)
    _print_report(report)
    print(f"Results written to {results_root / 'robustness_summary.json'}")


if __name__ == "__main__":
    main()
