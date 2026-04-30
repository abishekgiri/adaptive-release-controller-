"""Evaluate the Phase 5 MAPE-K feedback loop."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from knowledge_base.db import DEFAULT_DB_PATH
from knowledge_base.learning import (
    DEFAULT_BLOCK_THRESHOLD,
    DEFAULT_DEPLOY_THRESHOLD,
    DEFAULT_POLICY_PATH,
    DeploymentOutcomeRecord,
    FeedbackLoop,
    FeedbackMetrics,
    LearnedPolicy,
    calculate_feedback_metrics,
    derive_decisions_from_risk_scores,
    load_deployment_history,
    save_policy,
)


DEFAULT_RESULTS_PATH = Path("experiments/results/feedback-loop-results.md")


def run_feedback_loop(
    db_path: str | Path = DEFAULT_DB_PATH,
    limit: int = 200,
    deploy_threshold: float = DEFAULT_DEPLOY_THRESHOLD,
    block_threshold: float = DEFAULT_BLOCK_THRESHOLD,
) -> tuple[list[DeploymentOutcomeRecord], FeedbackMetrics, LearnedPolicy]:
    """Load history, compute metrics, and adapt thresholds."""

    history = load_deployment_history(db_path=db_path, limit=limit)
    baseline_policy_records = derive_decisions_from_risk_scores(
        history,
        deploy_threshold=deploy_threshold,
        block_threshold=block_threshold,
    )
    baseline_metrics = calculate_feedback_metrics(baseline_policy_records)
    policy = FeedbackLoop(
        deploy_threshold=deploy_threshold,
        block_threshold=block_threshold,
    ).run(baseline_policy_records)
    return baseline_policy_records, baseline_metrics, policy


def validate_adaptation_examples() -> None:
    """Validate required adaptation scenarios."""

    scenarios = adaptation_examples()
    lower_policy = FeedbackLoop().run(scenarios["high_false_negative"])
    higher_policy = FeedbackLoop().run(scenarios["high_false_positive"])
    unchanged_policy = FeedbackLoop().run(scenarios["acceptable"])

    if lower_policy.deploy_threshold >= DEFAULT_DEPLOY_THRESHOLD:
        raise AssertionError("high false negative rate should lower deploy threshold")
    if lower_policy.block_threshold >= DEFAULT_BLOCK_THRESHOLD:
        raise AssertionError("high false negative rate should lower block threshold")

    if higher_policy.deploy_threshold <= DEFAULT_DEPLOY_THRESHOLD:
        raise AssertionError("high false positive rate should raise deploy threshold")
    if higher_policy.block_threshold <= DEFAULT_BLOCK_THRESHOLD:
        raise AssertionError("high false positive rate should raise block threshold")

    if unchanged_policy.deploy_threshold != DEFAULT_DEPLOY_THRESHOLD:
        raise AssertionError("acceptable rates should keep deploy threshold unchanged")
    if unchanged_policy.block_threshold != DEFAULT_BLOCK_THRESHOLD:
        raise AssertionError("acceptable rates should keep block threshold unchanged")


def adaptation_examples() -> dict[str, list[DeploymentOutcomeRecord]]:
    """Return deterministic validation scenarios for adaptation behavior."""

    high_false_negative = [
        DeploymentOutcomeRecord(1, "a", "DEPLOY", 0.25, "failure"),
        DeploymentOutcomeRecord(2, "b", "CANARY", 0.55, "failure"),
        DeploymentOutcomeRecord(3, "c", "DEPLOY", 0.30, "success"),
    ]
    high_false_positive = [
        DeploymentOutcomeRecord(1, "a", "BLOCK", 0.80, "success"),
        DeploymentOutcomeRecord(2, "b", "BLOCK", 0.75, "success"),
        DeploymentOutcomeRecord(3, "c", "DEPLOY", 0.20, "success"),
    ]
    acceptable = [
        DeploymentOutcomeRecord(1, "a", "DEPLOY", 0.20, "success"),
        DeploymentOutcomeRecord(2, "b", "BLOCK", 0.80, "failure"),
        DeploymentOutcomeRecord(3, "c", "CANARY", 0.55, "success"),
    ]

    return {
        "high_false_negative": high_false_negative,
        "high_false_positive": high_false_positive,
        "acceptable": acceptable,
    }


def adaptation_examples_markdown() -> str:
    """Format validation examples as a Markdown table."""

    rows = []
    labels = {
        "high_false_negative": "High false negative rate",
        "high_false_positive": "High false positive rate",
        "acceptable": "Acceptable rates",
    }
    for key, records in adaptation_examples().items():
        policy = FeedbackLoop().run(records)
        rows.append(
            (
                labels[key],
                f"{policy.metrics.false_negative_rate * 100:.2f}%",
                f"{policy.metrics.false_positive_rate * 100:.2f}%",
                f"{policy.previous_deploy_threshold:.2f}",
                f"{policy.deploy_threshold:.2f}",
                f"{policy.previous_block_threshold:.2f}",
                f"{policy.block_threshold:.2f}",
                policy.adjustment,
            )
        )
    return markdown_table(
        (
            "Scenario",
            "False Negative Rate",
            "False Positive Rate",
            "Deploy Before",
            "Deploy After",
            "Block Before",
            "Block After",
            "Adjustment",
        ),
        rows,
    )


def results_markdown(
    metrics: FeedbackMetrics,
    policy: LearnedPolicy,
    records: list[DeploymentOutcomeRecord],
    limit: int = 20,
) -> str:
    """Format feedback-loop results as Markdown."""

    lines = [
        "# MAPE-K Feedback Loop Results",
        "",
        "## Experiment Setup",
        "",
        markdown_table(
            ("Item", "Value"),
            [
                ("Phase", "Phase 5 - MAPE-K Feedback Loop"),
                ("Database", f"`{DEFAULT_DB_PATH}`"),
                ("Policy artifact", f"`{DEFAULT_POLICY_PATH}`"),
                ("Deploy threshold before", f"{policy.previous_deploy_threshold:.2f}"),
                ("Block threshold before", f"{policy.previous_block_threshold:.2f}"),
                ("Deploy threshold after", f"{policy.deploy_threshold:.2f}"),
                ("Block threshold after", f"{policy.block_threshold:.2f}"),
                ("Adjustment", policy.adjustment),
            ],
        ),
        "",
        "## Feedback Metrics",
        "",
        metrics_markdown(metrics),
        "",
        "## Learned Policy",
        "",
        markdown_table(
            ("Field", "Value"),
            [
                ("Previous deploy threshold", f"{policy.previous_deploy_threshold:.2f}"),
                ("Previous block threshold", f"{policy.previous_block_threshold:.2f}"),
                ("New deploy threshold", f"{policy.deploy_threshold:.2f}"),
                ("New block threshold", f"{policy.block_threshold:.2f}"),
                ("Reason", policy.reason),
            ],
        ),
        "",
        "## Decision Sample",
        "",
        records_markdown(records, limit=limit),
        "",
        "## Adaptation Validation Examples",
        "",
        adaptation_examples_markdown(),
        "",
        "## Interpretation",
        "",
        interpretation(policy),
        "",
    ]
    return "\n".join(lines)


def metrics_markdown(metrics: FeedbackMetrics) -> str:
    """Format feedback metrics as a Markdown table."""

    return markdown_table(
        ("Metric", "Value"),
        [
            ("Total records", metrics.total_records),
            ("Deployed or canaried", metrics.deployed_or_canaried),
            ("Blocked", metrics.blocked),
            ("Successes", metrics.successes),
            ("Failures", metrics.failures),
            ("Success rate", format_percent(metrics.success_rate)),
            ("Failure rate", format_percent(metrics.failure_rate)),
            ("False positives", metrics.false_positives),
            ("False negatives", metrics.false_negatives),
            ("False positive rate", format_percent(metrics.false_positive_rate)),
            ("False negative rate", format_percent(metrics.false_negative_rate)),
        ],
    )


def records_markdown(
    records: list[DeploymentOutcomeRecord],
    limit: int = 20,
) -> str:
    """Format a sample of policy decisions and outcomes."""

    return markdown_table(
        ("Commit", "Risk Score", "Decision", "Outcome"),
        [
            (
                record.commit_sha[:12],
                f"{record.risk_score:.4f}",
                record.decision,
                record.outcome,
            )
            for record in records[:limit]
        ],
    )


def interpretation(policy: LearnedPolicy) -> str:
    """Return a concise interpretation of the learned policy."""

    if policy.adjustment == "increase_risk_sensitivity":
        return (
            "The feedback loop detected a high false negative rate, so it lowered "
            "both thresholds. Future decisions become more conservative because "
            "more deployments move from DEPLOY to CANARY or from CANARY to BLOCK."
        )
    if policy.adjustment == "reduce_unnecessary_blocking":
        return (
            "The feedback loop detected a high false positive rate, so it raised "
            "both thresholds. Future decisions become less conservative because "
            "fewer deployments are unnecessarily blocked."
        )
    return (
        "The observed false positive and false negative rates are acceptable, so "
        "the feedback loop kept the policy unchanged."
    )


def save_results(markdown: str, path: str | Path = DEFAULT_RESULTS_PATH) -> None:
    """Save feedback-loop Markdown results."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(markdown, encoding="utf-8")


def markdown_table(headers: tuple[str, ...], rows: list[tuple[Any, ...]]) -> str:
    """Build a Markdown table."""

    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(value) for value in row) + " |")
    return "\n".join(lines)


def format_percent(value: float) -> str:
    """Format a ratio as a percentage."""

    return f"{value * 100:.2f}%"


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""

    parser = argparse.ArgumentParser(description="Run the MAPE-K feedback loop.")
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
        "--policy-output",
        default=str(DEFAULT_POLICY_PATH),
        help="Path for learned policy JSON output.",
    )
    parser.add_argument(
        "--results-output",
        default=str(DEFAULT_RESULTS_PATH),
        help="Path for feedback-loop Markdown output.",
    )
    return parser.parse_args()


def main() -> None:
    """Run the feedback loop and persist learned artifacts."""

    args = parse_args()
    validate_adaptation_examples()
    records, metrics, policy = run_feedback_loop(db_path=args.db, limit=args.limit)
    save_policy(policy, args.policy_output)
    markdown = results_markdown(metrics=metrics, policy=policy, records=records)
    save_results(markdown, args.results_output)

    print("# MAPE-K Feedback Loop Results\n")
    print(
        markdown_table(
            ("Threshold", "Before", "After"),
            [
                (
                    "Deploy",
                    f"{policy.previous_deploy_threshold:.2f}",
                    f"{policy.deploy_threshold:.2f}",
                ),
                (
                    "Block",
                    f"{policy.previous_block_threshold:.2f}",
                    f"{policy.block_threshold:.2f}",
                ),
            ],
        )
    )
    print("\n## Feedback Metrics\n")
    print(metrics_markdown(metrics))
    print(f"\nPolicy saved to {args.policy_output}")
    print(f"Results saved to {args.results_output}")


if __name__ == "__main__":
    main()
