"""Cost-ratio sensitivity sweep for the online replay experiment.

Sweeps the deploy_failure / block_bad asymmetry ratio across ≥5 levels to verify
that LinUCB's advantages over static_rules hold across a reasonable range of
cost configurations. All other cost parameters scale proportionally to keep the
ratio invariant to units.

Sweep levels (deploy_failure : block_bad ratio):
  5:1   — mild asymmetry (deploy_failure=5.0,  block_bad=1.0)
  10:1  — moderate       (deploy_failure=5.0,  block_bad=0.5)
  20:1  — default paper  (deploy_failure=10.0, block_bad=0.5)  ← baseline
  40:1  — high penalty   (deploy_failure=20.0, block_bad=0.5)
  100:1 — extreme        (deploy_failure=50.0, block_bad=0.5)

Uses the same online-replay infrastructure as run_bandits.py / run_robustness.py.
Results allow claims about cost-matrix sensitivity (§8 Threats to Validity).

WARNING — SIMULATION, NOT CAUSAL INFERENCE.
All costs are simulation artefacts using CI outcome as a counterfactual proxy.

Usage:
    python -m experiments.run_cost_sweep \\
        --dataset data/raw/travistorrent_smoke.csv \\
        --seeds 0 1 2 3 4 \\
        --results-root experiments/results/cost_sweep
    # For ≥30-seed paper run: --seeds $(seq 0 29 | tr '\\n' ' ')
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from experiments.run_bandits import OnlineExperimentConfig, run_experiment
from rewards.cost_model import CostConfig

DEFAULT_DATASET = Path("data/raw/travistorrent_smoke.csv")
DEFAULT_RESULTS_ROOT = Path("experiments/results/cost_sweep")
N_BOOTSTRAP = 10_000
BOOTSTRAP_SEED = 42


# ---------------------------------------------------------------------------
# Sweep levels
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CostLevel:
    label: str              # short name for output tables and JSON keys
    deploy_failure: float
    canary_failure: float   # kept at 0.4 × deploy_failure by convention
    block_bad: float

    @property
    def ratio(self) -> float:
        return self.deploy_failure / self.block_bad

    def to_cost_config(self) -> CostConfig:
        return CostConfig(
            deploy_success=0.0,
            deploy_failure=self.deploy_failure,
            canary_success=1.0,
            canary_failure=self.canary_failure,
            block_safe=2.0,
            block_bad=self.block_bad,
            block_unknown=2.0,
        )


# Five sweep levels spanning 5:1 to 100:1.
COST_LEVELS: list[CostLevel] = [
    CostLevel(label="5:1",   deploy_failure=5.0,  canary_failure=2.0,  block_bad=1.0),
    CostLevel(label="10:1",  deploy_failure=5.0,  canary_failure=2.0,  block_bad=0.5),
    CostLevel(label="20:1",  deploy_failure=10.0, canary_failure=4.0,  block_bad=0.5),  # default
    CostLevel(label="40:1",  deploy_failure=20.0, canary_failure=8.0,  block_bad=0.5),
    CostLevel(label="100:1", deploy_failure=50.0, canary_failure=20.0, block_bad=0.5),
]


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
# Sweep runner
# ---------------------------------------------------------------------------

def run_one_level(
    level: CostLevel,
    dataset_path: Path,
    seeds: list[int],
) -> dict[str, Any]:
    """Run all seeds for one cost level; return per-policy aggregated stats."""
    per_seed: dict[str, list[float]] = {}

    for seed in seeds:
        config = OnlineExperimentConfig(
            config_name=f"cost_sweep_{level.label.replace(':', '_')}",
            dataset_path=dataset_path,
            cost_config=level.to_cost_config(),
            linucb_alpha=1.0,
            linucb_lambda_reg=1.0,
            cost_sensitive_alpha=1.0,
            cost_sensitive_lambda_reg=1.0,
            delay_step_seconds=60,
        )
        summary = run_experiment(config, seed)
        for pid, result in summary["policies"].items():
            cost = result.get("cumulative_cost", float("nan"))
            per_seed.setdefault(pid, []).append(cost)

    rng = np.random.default_rng(BOOTSTRAP_SEED)
    policies_summary: dict[str, Any] = {}
    for pid, costs in per_seed.items():
        ci_lo, ci_hi = bootstrap_ci(costs, rng=rng)
        policies_summary[pid] = {
            "mean_cost": float(np.mean(costs)),
            "std_cost": float(np.std(costs, ddof=1)) if len(costs) > 1 else 0.0,
            "ci_lo_95": ci_lo,
            "ci_hi_95": ci_hi,
            "per_seed_costs": costs,
        }
    return policies_summary


def run_cost_sweep(
    dataset_path: Path,
    seeds: list[int],
    results_root: Path,
) -> dict[str, Any]:
    """Sweep all cost levels; write summary JSON; return report dict."""
    levels_summary: dict[str, Any] = {}

    for level in COST_LEVELS:
        print(f"  ratio={level.label}  (deploy_failure={level.deploy_failure}, block_bad={level.block_bad}) …", flush=True)
        policies_summary = run_one_level(level, dataset_path, seeds)
        levels_summary[level.label] = {
            "deploy_failure": level.deploy_failure,
            "canary_failure": level.canary_failure,
            "block_bad": level.block_bad,
            "ratio": level.ratio,
            "policies": policies_summary,
        }

    report = {
        "evaluation_mode": "cost_sweep_online_replay_simulation",
        "warning": (
            "Costs are simulation artefacts from online replay. "
            "CI outcome used as counterfactual proxy. "
            "Do NOT report as causal real-world cost estimates."
        ),
        "dataset_path": str(dataset_path),
        "seeds": seeds,
        "n_seeds": len(seeds),
        "n_bootstrap": N_BOOTSTRAP,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "sweep_axis": "deploy_failure / block_bad ratio",
        "levels": levels_summary,
    }

    results_root.mkdir(parents=True, exist_ok=True)
    (results_root / "cost_sweep_summary.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Cost-ratio sensitivity sweep for the online replay experiment."
    )
    parser.add_argument(
        "--dataset",
        default=str(DEFAULT_DATASET),
        help="Path to TravisTorrent-format CSV (default: data/raw/travistorrent_smoke.csv).",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=list(range(5)),
        help="Seeds to run (default: 0-4; use 0-29 for ≥30-seed paper runs).",
    )
    parser.add_argument(
        "--results-root",
        default=str(DEFAULT_RESULTS_ROOT),
        help="Output directory (default: experiments/results/cost_sweep).",
    )
    return parser.parse_args()


def _print_report(report: dict[str, Any]) -> None:
    print(f"\n# Cost Sweep — {report['n_seeds']} seeds × {len(report['levels'])} levels\n")
    print("> " + report["warning"] + "\n")
    print("| Ratio | Policy | Mean Cost | Std | 95% CI |")
    print("| --- | --- | ---: | ---: | --- |")
    for ratio_label, level in report["levels"].items():
        for pid, r in level["policies"].items():
            print(
                f"| {ratio_label} | {pid} | {r['mean_cost']:.2f} "
                f"| {r['std_cost']:.2f} | [{r['ci_lo_95']:.2f}, {r['ci_hi_95']:.2f}] |"
            )


def main() -> None:
    args = parse_args()
    dataset_path = Path(args.dataset)
    seeds = list(args.seeds)
    results_root = Path(args.results_root)

    print(f"Running cost sweep: {len(COST_LEVELS)} levels × {len(seeds)} seeds …")
    report = run_cost_sweep(dataset_path, seeds, results_root)
    _print_report(report)
    print(f"Results written to {results_root / 'cost_sweep_summary.json'}")


if __name__ == "__main__":
    main()
