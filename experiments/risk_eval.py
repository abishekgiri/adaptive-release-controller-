"""Evaluate the Phase 3 heuristic risk analysis engine."""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from knowledge_base.db import DEFAULT_DB_PATH, connect, initialize_database
from risk_engine.model import RiskInput, RiskPrediction, predict_risk


@dataclass(frozen=True)
class RiskEvaluation:
    """Risk prediction joined with the known deployment outcome."""

    deployment_id: int
    outcome: str
    prediction: RiskPrediction


@dataclass(frozen=True)
class RiskMetrics:
    """Aggregate risk evaluation metrics."""

    total_records: int
    success_count: int
    failure_count: int
    average_success_risk: float
    average_failure_risk: float
    risk_separation: float
    failure_correlation: float
    low_failure_rate: float
    medium_failure_rate: float
    high_failure_rate: float


def load_risk_inputs(
    db_path: str | Path = DEFAULT_DB_PATH,
    limit: int = 200,
) -> list[tuple[int, RiskInput, str]]:
    """Load deployment records and derive historical failure counts."""

    initialize_database(db_path)
    with connect(db_path) as connection:
        rows = list(
            connection.execute(
                """
                SELECT *
                FROM deployments
                ORDER BY id ASC
                LIMIT ?
                """,
                (limit,),
            )
        )

    total_history = len(rows)
    past_failures = 0
    loaded = []
    for row in rows:
        risk_input = RiskInput(
            commit_sha=str(row["commit_sha"]),
            files_changed=int(row["files_changed"]),
            lines_added=int(row["lines_added"]),
            lines_deleted=int(row["lines_deleted"]),
            ci_duration=float(row["ci_duration"]),
            past_failures=past_failures,
            historical_records=total_history,
        )
        outcome = str(row["outcome"])
        loaded.append((int(row["id"]), risk_input, outcome))
        if outcome == "failure":
            past_failures += 1

    return loaded


def evaluate_risk(
    records: list[tuple[int, RiskInput, str]]
) -> list[RiskEvaluation]:
    """Generate risk predictions for deployment records."""

    return [
        RiskEvaluation(
            deployment_id=deployment_id,
            outcome=outcome,
            prediction=predict_risk(risk_input),
        )
        for deployment_id, risk_input, outcome in records
    ]


def calculate_metrics(evaluations: list[RiskEvaluation]) -> RiskMetrics:
    """Calculate aggregate risk-analysis metrics."""

    successes = [item for item in evaluations if item.outcome == "success"]
    failures = [item for item in evaluations if item.outcome == "failure"]
    success_risk = average([item.prediction.risk_score for item in successes])
    failure_risk = average([item.prediction.risk_score for item in failures])

    return RiskMetrics(
        total_records=len(evaluations),
        success_count=len(successes),
        failure_count=len(failures),
        average_success_risk=success_risk,
        average_failure_risk=failure_risk,
        risk_separation=round(failure_risk - success_risk, 4),
        failure_correlation=correlation_with_failure(evaluations),
        low_failure_rate=failure_rate_for_level(evaluations, "low"),
        medium_failure_rate=failure_rate_for_level(evaluations, "medium"),
        high_failure_rate=failure_rate_for_level(evaluations, "high"),
    )


def average(values: list[float]) -> float:
    """Return the average for a list of values."""

    if not values:
        return 0.0
    return round(sum(values) / len(values), 4)


def failure_rate_for_level(
    evaluations: list[RiskEvaluation],
    level: str,
) -> float:
    """Return failure rate for one risk level."""

    matching = [item for item in evaluations if item.prediction.level == level]
    if not matching:
        return 0.0
    failures = [item for item in matching if item.outcome == "failure"]
    return round(len(failures) / len(matching), 4)


def correlation_with_failure(evaluations: list[RiskEvaluation]) -> float:
    """Return Pearson correlation between risk score and failure label."""

    if len(evaluations) < 2:
        return 0.0

    risk_scores = [item.prediction.risk_score for item in evaluations]
    failure_labels = [1.0 if item.outcome == "failure" else 0.0 for item in evaluations]
    return round(pearson_correlation(risk_scores, failure_labels), 4)


def pearson_correlation(left: list[float], right: list[float]) -> float:
    """Calculate Pearson correlation for two numeric lists."""

    if len(left) != len(right) or len(left) < 2:
        return 0.0

    left_mean = sum(left) / len(left)
    right_mean = sum(right) / len(right)
    numerator = sum(
        (left_value - left_mean) * (right_value - right_mean)
        for left_value, right_value in zip(left, right)
    )
    left_variance = sum((value - left_mean) ** 2 for value in left)
    right_variance = sum((value - right_mean) ** 2 for value in right)
    denominator = math.sqrt(left_variance * right_variance)

    if denominator == 0:
        return 0.0
    return numerator / denominator


def metrics_markdown(metrics: RiskMetrics) -> str:
    """Format aggregate metrics as a Markdown table."""

    rows = [
        ("Total records", metrics.total_records),
        ("Successful deployments", metrics.success_count),
        ("Failed deployments", metrics.failure_count),
        ("Average success risk", f"{metrics.average_success_risk:.4f}"),
        ("Average failure risk", f"{metrics.average_failure_risk:.4f}"),
        ("Risk separation", f"{metrics.risk_separation:.4f}"),
        ("Failure correlation", f"{metrics.failure_correlation:.4f}"),
        ("Low risk failure rate", format_percent(metrics.low_failure_rate)),
        ("Medium risk failure rate", format_percent(metrics.medium_failure_rate)),
        ("High risk failure rate", format_percent(metrics.high_failure_rate)),
    ]
    return markdown_table(("Metric", "Value"), rows)


def predictions_markdown(
    evaluations: list[RiskEvaluation],
    limit: int = 20,
) -> str:
    """Format per-record predictions as a Markdown table."""

    rows = [
        (
            item.prediction.commit_sha[:12],
            item.prediction.risk_score,
            item.prediction.confidence,
            item.prediction.level,
            item.prediction.decision,
            item.outcome,
        )
        for item in evaluations[:limit]
    ]
    return markdown_table(
        ("Commit", "Risk Score", "Confidence", "Level", "Decision", "Outcome"),
        rows,
    )


def level_distribution_markdown(evaluations: list[RiskEvaluation]) -> str:
    """Format failure rate by risk level."""

    rows: list[tuple[Any, ...]] = []
    for level in ("low", "medium", "high"):
        matching = [item for item in evaluations if item.prediction.level == level]
        failures = [item for item in matching if item.outcome == "failure"]
        rows.append(
            (
                level,
                len(matching),
                len(failures),
                format_percent(safe_divide(len(failures), len(matching))),
            )
        )
    return markdown_table(("Level", "Records", "Failures", "Failure Rate"), rows)


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


def safe_divide(numerator: float, denominator: float) -> float:
    """Divide while returning 0 for empty denominators."""

    if denominator == 0:
        return 0.0
    return numerator / denominator


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""

    parser = argparse.ArgumentParser(description="Evaluate the risk analysis engine.")
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
        "--table-limit",
        type=int,
        default=20,
        help="Number of prediction rows to print.",
    )
    return parser.parse_args()


def main() -> None:
    """Run risk evaluation and print result tables."""

    args = parse_args()
    records = load_risk_inputs(db_path=args.db, limit=args.limit)
    evaluations = evaluate_risk(records)
    metrics = calculate_metrics(evaluations)

    print("# Risk Analysis Engine Results\n")
    print(metrics_markdown(metrics))
    print("\n## Failure Rate By Risk Level\n")
    print(level_distribution_markdown(evaluations))
    print("\n## Prediction Sample\n")
    print(predictions_markdown(evaluations, limit=args.table_limit))

    if metrics.average_failure_risk > metrics.average_success_risk:
        print("\nResult: failed deployments have higher average risk.")
    else:
        print("\nResult: failed deployments do not have higher average risk yet.")


if __name__ == "__main__":
    main()

