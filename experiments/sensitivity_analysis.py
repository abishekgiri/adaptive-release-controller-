"""Analyze how feedback sensitivity changes adaptive release behavior."""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/matplotlib-cache")
os.environ.setdefault("XDG_CACHE_HOME", "/private/tmp")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from experiments.evaluation import (
    DEFAULT_BLOCK_THRESHOLD,
    DEFAULT_DEPLOY_THRESHOLD,
    calculate_system_metrics,
    evaluate_threshold_policy,
    load_records,
    markdown_table,
    safe_divide,
)
from knowledge_base.db import DEFAULT_DB_PATH
from knowledge_base.learning import (
    FeedbackLoop,
    calculate_feedback_metrics,
    derive_decisions_from_risk_scores,
    load_deployment_history,
)


DEFAULT_SENSITIVITY_VALUES = (0.05, 0.10, 0.15, 0.20, 0.25, 0.30)
DEFAULT_RESULTS_DIR = Path("experiments/results")
DEFAULT_GRAPHS_DIR = DEFAULT_RESULTS_DIR / "graphs"
DEFAULT_MARKDOWN_PATH = DEFAULT_RESULTS_DIR / "sensitivity-results.md"
DEFAULT_JSON_PATH = DEFAULT_RESULTS_DIR / "sensitivity-results.json"


@dataclass(frozen=True)
class SensitivityResult:
    """Evaluation result for one sensitivity threshold."""

    sensitivity: float
    deploy_threshold: float
    block_threshold: float
    adjustment: str
    observed_false_negative_rate: float
    success_rate: float
    failure_rate: float
    false_positive_rate: float
    false_negative_rate: float
    deployment_velocity: float
    decision_accuracy: float
    deployed_or_canaried: int
    blocked: int
    total_records: int
    tradeoff_score: float


def run_sensitivity_analysis(
    db_path: str | Path = DEFAULT_DB_PATH,
    limit: int = 200,
    sensitivity_values: tuple[float, ...] = DEFAULT_SENSITIVITY_VALUES,
) -> list[SensitivityResult]:
    """Run sensitivity sweep and return comparable policy metrics."""

    history = load_deployment_history(db_path=db_path, limit=limit)
    baseline_policy_records = derive_decisions_from_risk_scores(
        history,
        deploy_threshold=DEFAULT_DEPLOY_THRESHOLD,
        block_threshold=DEFAULT_BLOCK_THRESHOLD,
    )
    feedback_metrics = calculate_feedback_metrics(baseline_policy_records)
    evaluation_records = load_records(db_path=db_path, limit=limit)

    results = []
    for sensitivity in sensitivity_values:
        policy = FeedbackLoop(sensitivity_threshold=sensitivity).run(
            baseline_policy_records
        )
        decisions = evaluate_threshold_policy(
            system="Adaptive",
            records=evaluation_records,
            deploy_threshold=policy.deploy_threshold,
            block_threshold=policy.block_threshold,
        )
        metrics = calculate_system_metrics("Adaptive", decisions)
        deployment_velocity = safe_divide(
            metrics.deployed_or_canaried,
            metrics.total_records,
        )
        results.append(
            SensitivityResult(
                sensitivity=sensitivity,
                deploy_threshold=policy.deploy_threshold,
                block_threshold=policy.block_threshold,
                adjustment=policy.adjustment,
                observed_false_negative_rate=feedback_metrics.false_negative_rate,
                success_rate=metrics.success_rate,
                failure_rate=metrics.failure_rate,
                false_positive_rate=metrics.false_positive_rate,
                false_negative_rate=metrics.false_negative_rate,
                deployment_velocity=deployment_velocity,
                decision_accuracy=metrics.decision_accuracy,
                deployed_or_canaried=metrics.deployed_or_canaried,
                blocked=metrics.blocked,
                total_records=metrics.total_records,
                tradeoff_score=tradeoff_score(
                    success_rate=metrics.success_rate,
                    failure_rate=metrics.failure_rate,
                    false_negative_rate=metrics.false_negative_rate,
                    deployment_velocity=deployment_velocity,
                    decision_accuracy=metrics.decision_accuracy,
                ),
            )
        )
    return results


def tradeoff_score(
    success_rate: float,
    failure_rate: float,
    false_negative_rate: float,
    deployment_velocity: float,
    decision_accuracy: float,
) -> float:
    """Score reliability and velocity on one interpretable 0-1 scale."""

    return round(
        (
            success_rate
            + (1 - failure_rate)
            + (1 - false_negative_rate)
            + deployment_velocity
            + decision_accuracy
        )
        / 5,
        4,
    )


def select_best_tradeoff(
    results: list[SensitivityResult],
) -> SensitivityResult | None:
    """Select the best reliability and velocity tradeoff from the sweep."""

    if not results:
        return None
    return max(
        results,
        key=lambda result: (
            result.tradeoff_score,
            -result.failure_rate,
            -result.false_negative_rate,
            result.deployment_velocity,
        ),
    )


def write_outputs(
    results: list[SensitivityResult],
    markdown_path: str | Path = DEFAULT_MARKDOWN_PATH,
    json_path: str | Path = DEFAULT_JSON_PATH,
    graphs_dir: str | Path = DEFAULT_GRAPHS_DIR,
) -> None:
    """Write Markdown, JSON, and graph outputs for sensitivity analysis."""

    Path(markdown_path).parent.mkdir(parents=True, exist_ok=True)
    Path(markdown_path).write_text(
        sensitivity_markdown(results),
        encoding="utf-8",
    )
    Path(json_path).parent.mkdir(parents=True, exist_ok=True)
    Path(json_path).write_text(
        json.dumps(
            {
                "results": [asdict(result) for result in results],
                "best_tradeoff": (
                    asdict(select_best_tradeoff(results))
                    if select_best_tradeoff(results)
                    else None
                ),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    generate_graphs(results=results, graphs_dir=graphs_dir)


def sensitivity_markdown(results: list[SensitivityResult]) -> str:
    """Build the sensitivity analysis Markdown report."""

    best = select_best_tradeoff(results)
    lines = [
        "# Sensitivity & Threshold Analysis",
        "",
        "## Research Question",
        "",
        (
            "How does changing the feedback sensitivity threshold affect deployment "
            "failure rate, false positives, false negatives, and delivery velocity?"
        ),
        "",
        "## Experiment Setup",
        "",
        markdown_table(
            ("Item", "Value"),
            [
                ("Dataset", "`knowledge_base/deployments.db`"),
                (
                    "Sensitivity values",
                    ", ".join(f"{result.sensitivity:.2f}" for result in results),
                ),
                ("Default deploy threshold", f"{DEFAULT_DEPLOY_THRESHOLD:.2f}"),
                ("Default block threshold", f"{DEFAULT_BLOCK_THRESHOLD:.2f}"),
                (
                    "Deployment velocity",
                    "Allowed releases divided by total deployment records",
                ),
            ],
        ),
        "",
        "## Sensitivity Sweep Results",
        "",
        sensitivity_table(results),
        "",
        "## Best Reliability and Velocity Tradeoff",
        "",
        best_tradeoff_markdown(best),
        "",
        "## Graph Outputs",
        "",
        markdown_table(
            ("Graph", "Path"),
            [
                (
                    "Sensitivity failure rate",
                    "`experiments/results/graphs/sensitivity_failure_rate.png`",
                ),
                (
                    "Sensitivity tradeoff",
                    "`experiments/results/graphs/sensitivity_tradeoff.png`",
                ),
            ],
        ),
        "",
        "## Research Interpretation",
        "",
        sensitivity_interpretation(results, best),
        "",
    ]
    return "\n".join(lines)


def sensitivity_table(results: list[SensitivityResult]) -> str:
    """Format the sensitivity sweep as a Markdown table."""

    return markdown_table(
        (
            "Sensitivity",
            "Deploy Threshold",
            "Block Threshold",
            "Adjustment",
            "Success Rate",
            "Failure Rate",
            "False Positive Rate",
            "False Negative Rate",
            "Deployment Velocity",
            "Decision Accuracy",
            "Tradeoff Score",
        ),
        [
            (
                f"{result.sensitivity:.2f}",
                f"{result.deploy_threshold:.2f}",
                f"{result.block_threshold:.2f}",
                result.adjustment,
                format_percent(result.success_rate),
                format_percent(result.failure_rate),
                format_percent(result.false_positive_rate),
                format_percent(result.false_negative_rate),
                format_percent(result.deployment_velocity),
                format_percent(result.decision_accuracy),
                f"{result.tradeoff_score:.4f}",
            )
            for result in results
        ],
    )


def best_tradeoff_markdown(best: SensitivityResult | None) -> str:
    """Explain the selected best tradeoff."""

    if best is None:
        return "No sensitivity results were generated."

    return (
        f"The best reliability/velocity tradeoff in this run is sensitivity "
        f"`{best.sensitivity:.2f}`. It uses thresholds `{best.deploy_threshold:.2f}` "
        f"and `{best.block_threshold:.2f}`, producing a "
        f"{format_percent(best.failure_rate)} failure rate, "
        f"{format_percent(best.false_negative_rate)} false negative rate, "
        f"{format_percent(best.deployment_velocity)} deployment velocity, and "
        f"{format_percent(best.decision_accuracy)} decision accuracy."
    )


def sensitivity_interpretation(
    results: list[SensitivityResult],
    best: SensitivityResult | None,
) -> str:
    """Interpret sensitivity effects in research language."""

    if not results or best is None:
        return "No sensitivity results were available for interpretation."

    adapted = [
        result
        for result in results
        if result.adjustment == "increase_risk_sensitivity"
    ]
    unchanged = [result for result in results if result.adjustment == "unchanged"]
    adapted_text = (
        f"Sensitivity values at or below {max(item.sensitivity for item in adapted):.2f} "
        "triggered conservative adaptation."
        if adapted
        else "No sensitivity value triggered conservative adaptation."
    )
    unchanged_text = (
        f"Sensitivity values at or above {min(item.sensitivity for item in unchanged):.2f} "
        "kept the default thresholds unchanged."
        if unchanged
        else "All sensitivity values triggered adaptation."
    )

    return (
        f"{adapted_text} {unchanged_text} Lower sensitivity made the controller "
        "more conservative by reducing the deploy and block thresholds, which lowered "
        "failure rate and false negatives but reduced deployment velocity. Higher "
        "sensitivity preserved delivery velocity but allowed more failed deployments. "
        f"The selected tradeoff is `{best.sensitivity:.2f}` because it gives the "
        "strongest combined reliability and velocity score for this dataset."
    )


def generate_graphs(
    results: list[SensitivityResult],
    graphs_dir: str | Path = DEFAULT_GRAPHS_DIR,
) -> None:
    """Generate sensitivity analysis graphs."""

    output_dir = Path(graphs_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    failure_rate_chart(
        results=results,
        output_path=output_dir / "sensitivity_failure_rate.png",
    )
    tradeoff_chart(
        results=results,
        output_path=output_dir / "sensitivity_tradeoff.png",
    )


def failure_rate_chart(
    results: list[SensitivityResult],
    output_path: Path,
) -> None:
    """Plot sensitivity against failure rate."""

    fig, ax = plt.subplots(figsize=(8, 5))
    x_values = [result.sensitivity for result in results]
    y_values = [result.failure_rate * 100 for result in results]
    ax.plot(x_values, y_values, marker="o", color="#e45756")
    ax.set_title("Sensitivity vs Failure Rate")
    ax.set_xlabel("Sensitivity Threshold")
    ax.set_ylabel("Failure Rate (%)")
    ax.set_xticks(x_values)
    ax.set_ylim(bottom=0)
    for x_value, y_value in zip(x_values, y_values):
        ax.text(x_value, y_value, f"{y_value:.2f}%", ha="center", va="bottom")
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def tradeoff_chart(
    results: list[SensitivityResult],
    output_path: Path,
) -> None:
    """Plot reliability and velocity tradeoffs by sensitivity."""

    fig, ax = plt.subplots(figsize=(9, 5))
    x_values = [result.sensitivity for result in results]
    ax.plot(
        x_values,
        [result.false_negative_rate * 100 for result in results],
        marker="o",
        label="False Negative Rate",
        color="#72b7b2",
    )
    ax.plot(
        x_values,
        [result.false_positive_rate * 100 for result in results],
        marker="o",
        label="False Positive Rate",
        color="#e45756",
    )
    ax.plot(
        x_values,
        [result.deployment_velocity * 100 for result in results],
        marker="o",
        label="Deployment Velocity",
        color="#4c78a8",
    )
    ax.plot(
        x_values,
        [result.decision_accuracy * 100 for result in results],
        marker="o",
        label="Decision Accuracy",
        color="#54a24b",
    )
    ax.set_title("Sensitivity Tradeoff")
    ax.set_xlabel("Sensitivity Threshold")
    ax.set_ylabel("Rate (%)")
    ax.set_xticks(x_values)
    ax.set_ylim(bottom=0)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def format_percent(value: float) -> str:
    """Format a ratio as a percentage."""

    return f"{value * 100:.2f}%"


def parse_sensitivity_values(raw_values: list[str]) -> tuple[float, ...]:
    """Parse sensitivity CLI values."""

    return tuple(float(value) for value in raw_values)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""

    parser = argparse.ArgumentParser(
        description="Run Phase 8 sensitivity and threshold analysis."
    )
    parser.add_argument(
        "--db",
        default=str(DEFAULT_DB_PATH),
        help="Path to the SQLite deployment database.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=200,
        help="Maximum deployment records to evaluate.",
    )
    parser.add_argument(
        "--sensitivity",
        nargs="*",
        default=[str(value) for value in DEFAULT_SENSITIVITY_VALUES],
        help="Sensitivity thresholds to evaluate.",
    )
    return parser.parse_args()


def main() -> None:
    """Run sensitivity analysis and persist artifacts."""

    args = parse_args()
    results = run_sensitivity_analysis(
        db_path=args.db,
        limit=args.limit,
        sensitivity_values=parse_sensitivity_values(args.sensitivity),
    )
    write_outputs(results)
    print("# Sensitivity & Threshold Analysis\n")
    print(sensitivity_table(results))
    best = select_best_tradeoff(results)
    if best is not None:
        print("\n## Best Tradeoff\n")
        print(best_tradeoff_markdown(best))
    print(f"\nMarkdown saved to {DEFAULT_MARKDOWN_PATH}")
    print(f"JSON saved to {DEFAULT_JSON_PATH}")
    print(f"Graphs saved to {DEFAULT_GRAPHS_DIR}")


if __name__ == "__main__":
    main()
