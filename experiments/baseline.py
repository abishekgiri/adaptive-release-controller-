"""Static CI/CD baseline experiment.

The baseline deploys only when tests pass and coverage is above 80%.
It does not use adaptive feedback, risk score, rollback history, or outcomes.
"""

from __future__ import annotations

import argparse
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from knowledge_base.db import (
    DEFAULT_DB_PATH,
    connect,
    initialize_database,
    list_deployments,
)


COVERAGE_THRESHOLD = 80.0
RISKY_FOLDERS = (
    "auth/",
    "payments/",
    "database/",
    "deploy/",
    "infrastructure/",
    "config/",
)


@dataclass(frozen=True)
class BaselineInput:
    """Input fields needed by the static CI/CD baseline."""

    commit_sha: str
    files_changed: int
    lines_added: int
    lines_deleted: int
    tests_passed: bool
    coverage: float
    risky_folder_touched: bool
    known_outcome: str = "unknown"


@dataclass(frozen=True)
class BaselineEvaluation:
    """One baseline decision and its simulated deployment behavior."""

    commit_sha: str
    tests_passed: bool
    coverage: float
    decision: str
    simulated_outcome: str
    mttr: float
    false_positive: bool
    false_negative: bool


@dataclass(frozen=True)
class BaselineMetrics:
    """Aggregate metrics for the static CI/CD baseline."""

    total_records: int
    deployed: int
    blocked: int
    successful_deployments: int
    failed_deployments: int
    success_rate: float
    failure_rate: float
    average_mttr: float
    false_positives: int
    false_negatives: int
    false_positive_rate: float
    false_negative_rate: float


def baseline_decision(tests_passed: bool, coverage: float) -> str:
    """Return the static CI/CD decision."""

    if tests_passed and coverage > COVERAGE_THRESHOLD:
        return "deploy"
    return "block"


def evaluate_baseline(records: Iterable[BaselineInput]) -> list[BaselineEvaluation]:
    """Evaluate baseline decisions and simulated outcomes."""

    evaluations = []
    for record in records:
        decision = baseline_decision(record.tests_passed, record.coverage)
        simulated_outcome = deployment_outcome(record)
        mttr = simulated_mttr(record) if decision == "deploy" else 0.0

        evaluations.append(
            BaselineEvaluation(
                commit_sha=record.commit_sha,
                tests_passed=record.tests_passed,
                coverage=record.coverage,
                decision=decision,
                simulated_outcome=simulated_outcome,
                mttr=mttr if simulated_outcome == "failure" else 0.0,
                false_positive=decision == "block" and simulated_outcome == "success",
                false_negative=decision == "deploy" and simulated_outcome == "failure",
            )
        )
    return evaluations


def calculate_metrics(evaluations: list[BaselineEvaluation]) -> BaselineMetrics:
    """Calculate research metrics for baseline evaluations."""

    total_records = len(evaluations)
    deployed_records = [item for item in evaluations if item.decision == "deploy"]
    blocked = total_records - len(deployed_records)
    successful = [
        item for item in deployed_records if item.simulated_outcome == "success"
    ]
    failed = [item for item in deployed_records if item.simulated_outcome == "failure"]
    false_positives = [item for item in evaluations if item.false_positive]
    false_negatives = [item for item in evaluations if item.false_negative]

    deployed = len(deployed_records)
    return BaselineMetrics(
        total_records=total_records,
        deployed=deployed,
        blocked=blocked,
        successful_deployments=len(successful),
        failed_deployments=len(failed),
        success_rate=safe_divide(len(successful), deployed),
        failure_rate=safe_divide(len(failed), deployed),
        average_mttr=safe_divide(sum(item.mttr for item in failed), len(failed)),
        false_positives=len(false_positives),
        false_negatives=len(false_negatives),
        false_positive_rate=safe_divide(len(false_positives), total_records),
        false_negative_rate=safe_divide(len(false_negatives), total_records),
    )


def deployment_outcome(record: BaselineInput) -> str:
    """Return known outcome or deterministic simulated outcome."""

    if record.known_outcome in {"success", "failure"}:
        return record.known_outcome

    probability = failure_probability(record)
    random_value = stable_unit_interval(f"{record.commit_sha}:outcome")
    if random_value < probability:
        return "failure"
    return "success"


def failure_probability(record: BaselineInput) -> float:
    """Estimate deployment failure chance for the simulation."""

    changed_lines = record.lines_added + record.lines_deleted
    probability = 0.05

    if record.files_changed >= 20:
        probability += 0.20
    elif record.files_changed >= 10:
        probability += 0.10

    if changed_lines >= 1000:
        probability += 0.20
    elif changed_lines >= 300:
        probability += 0.10

    if record.risky_folder_touched:
        probability += 0.25
    if not record.tests_passed:
        probability += 0.35
    if record.coverage <= COVERAGE_THRESHOLD:
        probability += 0.15
    elif record.coverage < 85:
        probability += 0.05

    return min(probability, 0.95)


def simulated_mttr(record: BaselineInput) -> float:
    """Return deterministic simulated recovery time in minutes."""

    changed_lines = record.lines_added + record.lines_deleted
    jitter = stable_unit_interval(f"{record.commit_sha}:mttr") * 15
    mttr = 10 + (record.files_changed * 1.5) + (changed_lines / 75) + jitter

    if record.risky_folder_touched:
        mttr += 20

    return round(mttr, 2)


def from_mapping(row: Any) -> BaselineInput:
    """Convert a database row or mapping into a baseline input."""

    commit_sha = str(get_value(row, "commit_sha"))
    files_changed = int(get_value(row, "files_changed", default=0))
    lines_added = int(get_value(row, "lines_added", default=0))
    lines_deleted = int(get_value(row, "lines_deleted", default=0))
    tests_passed = parse_bool(
        get_value(row, "tests_passed", "test_passed", default=False)
    )
    known_outcome = str(get_value(row, "outcome", default="unknown"))

    coverage = get_value(row, "coverage", "coverage_percent", default=None)
    if coverage is None:
        coverage = simulated_coverage(
            commit_sha=commit_sha,
            tests_passed=tests_passed,
            files_changed=files_changed,
            lines_added=lines_added,
            lines_deleted=lines_deleted,
        )

    changed_files = get_value(row, "changed_files", "files", "file_paths", default=None)
    risky_folder = get_value(row, "risky_folder_touched", default=None)
    if risky_folder is None and changed_files is not None:
        risky_folder = path_touches_risky_folder(normalize_paths(changed_files))
    if risky_folder is None:
        risky_folder = simulated_risky_folder_touched(
            commit_sha=commit_sha,
            files_changed=files_changed,
        )

    return BaselineInput(
        commit_sha=commit_sha,
        files_changed=files_changed,
        lines_added=lines_added,
        lines_deleted=lines_deleted,
        tests_passed=tests_passed,
        coverage=float(coverage),
        risky_folder_touched=parse_bool(risky_folder),
        known_outcome=known_outcome,
    )


def get_value(row: Any, *names: str, default: Any = None) -> Any:
    """Read the first available field from a sqlite row or mapping."""

    keys = row.keys() if hasattr(row, "keys") else row
    for name in names:
        if name in keys:
            return row[name]
    return default


def simulated_coverage(
    commit_sha: str,
    tests_passed: bool,
    files_changed: int,
    lines_added: int,
    lines_deleted: int,
) -> float:
    """Generate deterministic coverage when Phase 1 data does not store it."""

    changed_lines = lines_added + lines_deleted
    base = 88.0 if tests_passed else 68.0
    size_penalty = min(files_changed * 0.35, 8.0) + min(changed_lines / 250, 8.0)
    jitter = (stable_unit_interval(f"{commit_sha}:coverage") * 10) - 5
    return round(clamp(base - size_penalty + jitter, 40.0, 98.0), 2)


def simulated_risky_folder_touched(commit_sha: str, files_changed: int) -> bool:
    """Generate deterministic risky-folder signal when paths are unavailable."""

    probability = 0.15 + min(files_changed / 100, 0.20)
    return stable_unit_interval(f"{commit_sha}:risky-folder") < probability


def normalize_paths(value: Any) -> list[str]:
    """Normalize path values from richer future datasets."""

    if isinstance(value, str):
        return [item.strip() for item in value.replace("\n", ",").split(",") if item]
    return [str(item) for item in value]


def parse_bool(value: Any) -> bool:
    """Parse common boolean representations."""

    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return value != 0
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "pass", "passed"}
    return bool(value)


def path_touches_risky_folder(paths: list[str]) -> bool:
    """Return whether any changed path is in a risky folder."""

    return any(
        path.startswith(RISKY_FOLDERS)
        or any(f"/{folder}" in path for folder in RISKY_FOLDERS)
        for path in paths
    )


def stable_unit_interval(seed: str) -> float:
    """Return a deterministic pseudo-random float in the range [0, 1)."""

    digest = hashlib.sha256(seed.encode("utf-8")).hexdigest()
    return int(digest[:16], 16) / 16**16


def safe_divide(numerator: float, denominator: float) -> float:
    """Divide while returning 0 for empty denominators."""

    if denominator == 0:
        return 0.0
    return numerator / denominator


def clamp(value: float, minimum: float, maximum: float) -> float:
    """Clamp a value between two bounds."""

    return max(minimum, min(maximum, value))


def load_records_from_database(
    db_path: str | Path = DEFAULT_DB_PATH,
    limit: int = 200,
) -> list[BaselineInput]:
    """Load Phase 1 deployment records as baseline inputs."""

    initialize_database(db_path)
    with connect(db_path) as connection:
        rows = list_deployments(connection, limit=limit)
    return [from_mapping(row) for row in rows]


def demo_records() -> list[BaselineInput]:
    """Return a small deterministic dataset for local smoke tests."""

    return [
        BaselineInput("demo-001", 2, 20, 3, True, 91.2, False, "success"),
        BaselineInput("demo-002", 12, 450, 90, True, 83.5, True, "failure"),
        BaselineInput("demo-003", 4, 40, 15, False, 72.0, False, "failure"),
        BaselineInput("demo-004", 7, 120, 50, True, 78.4, True, "success"),
        BaselineInput("demo-005", 1, 8, 2, True, 95.0, False, "success"),
    ]


def metrics_markdown(metrics: BaselineMetrics) -> str:
    """Format aggregate metrics as a Markdown table."""

    rows = [
        ("Total records", metrics.total_records),
        ("Deployed", metrics.deployed),
        ("Blocked", metrics.blocked),
        ("Successful deployments", metrics.successful_deployments),
        ("Failed deployments", metrics.failed_deployments),
        ("Success rate", format_percent(metrics.success_rate)),
        ("Failure rate", format_percent(metrics.failure_rate)),
        ("Average MTTR", f"{metrics.average_mttr:.2f} minutes"),
        ("False positives", metrics.false_positives),
        ("False negatives", metrics.false_negatives),
        ("False positive rate", format_percent(metrics.false_positive_rate)),
        ("False negative rate", format_percent(metrics.false_negative_rate)),
    ]
    return markdown_table(("Metric", "Value"), rows)


def decisions_markdown(evaluations: list[BaselineEvaluation], limit: int = 20) -> str:
    """Format individual decisions as a Markdown table."""

    rows = [
        (
            item.commit_sha[:12],
            "pass" if item.tests_passed else "fail",
            f"{item.coverage:.2f}",
            item.decision,
            item.simulated_outcome,
            f"{item.mttr:.2f}",
        )
        for item in evaluations[:limit]
    ]
    return markdown_table(
        ("Commit", "Tests", "Coverage", "Decision", "Outcome", "MTTR"),
        rows,
    )


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

    parser = argparse.ArgumentParser(description="Run the static CI/CD baseline.")
    parser.add_argument(
        "--db",
        default=str(DEFAULT_DB_PATH),
        help="Path to the Phase 1 SQLite deployment database.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=200,
        help="Maximum deployment records to evaluate.",
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Run against a small built-in demo dataset.",
    )
    return parser.parse_args()


def main() -> None:
    """Run the baseline experiment and print result tables."""

    args = parse_args()
    records = (
        demo_records()
        if args.demo
        else load_records_from_database(args.db, args.limit)
    )
    evaluations = evaluate_baseline(records)
    metrics = calculate_metrics(evaluations)

    print("# Static CI/CD Baseline Results\n")
    print(metrics_markdown(metrics))
    print("\n## Decisions\n")
    print(decisions_markdown(evaluations))


if __name__ == "__main__":
    main()
