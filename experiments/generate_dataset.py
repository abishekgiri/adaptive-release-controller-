"""Generate a deterministic simulated deployment dataset."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

from features.extractor import calculate_risk_score, decide_deployment
from knowledge_base.db import (
    DEFAULT_DB_PATH,
    DeploymentRecord,
    connect,
    initialize_database,
    insert_many,
)


DEFAULT_RECORD_COUNT = 100


def generate_records(count: int = DEFAULT_RECORD_COUNT) -> list[DeploymentRecord]:
    """Create simulated Phase 1 deployment records."""

    return [generate_record(index) for index in range(1, count + 1)]


def generate_record(index: int) -> DeploymentRecord:
    """Create one deterministic deployment record."""

    commit_sha = simulated_commit_sha(index)
    files_changed = simulated_int(commit_sha, "files", 1, 35)
    lines_added = simulated_int(commit_sha, "added", 5, 1200)
    lines_deleted = simulated_int(commit_sha, "deleted", 0, 600)
    ci_duration = simulated_float(commit_sha, "duration", 45.0, 3600.0)
    dependency_change = simulated_bool(commit_sha, "dependency", 0.18)
    risky_area_change = simulated_bool(commit_sha, "risky-area", 0.22)
    previous_failure_count = simulated_int(commit_sha, "previous-failures", 0, 4)

    test_passed = simulated_tests_passed(
        commit_sha=commit_sha,
        files_changed=files_changed,
        changed_lines=lines_added + lines_deleted,
        dependency_change=dependency_change,
        risky_area_change=risky_area_change,
    )
    risk_score = calculate_risk_score(
        files_changed=files_changed,
        lines_added=lines_added,
        lines_deleted=lines_deleted,
        test_passed=test_passed,
        ci_duration=ci_duration,
        dependency_change=dependency_change,
        risky_area_change=risky_area_change,
        previous_failure_count=previous_failure_count,
    )
    decision = decide_deployment(risk_score=risk_score, test_passed=test_passed)
    outcome = simulated_outcome(
        commit_sha=commit_sha,
        test_passed=test_passed,
        risk_score=risk_score,
        dependency_change=dependency_change,
        risky_area_change=risky_area_change,
    )

    return DeploymentRecord(
        commit_sha=commit_sha,
        files_changed=files_changed,
        lines_added=lines_added,
        lines_deleted=lines_deleted,
        test_passed=test_passed,
        ci_duration=round(ci_duration, 2),
        risk_score=risk_score,
        decision=decision,
        outcome=outcome,
    )


def simulated_tests_passed(
    commit_sha: str,
    files_changed: int,
    changed_lines: int,
    dependency_change: bool,
    risky_area_change: bool,
) -> bool:
    """Simulate CI test status from deployment risk factors."""

    failure_probability = 0.08
    failure_probability += min(files_changed / 200, 0.12)
    failure_probability += min(changed_lines / 6000, 0.18)
    if dependency_change:
        failure_probability += 0.08
    if risky_area_change:
        failure_probability += 0.06
    return stable_unit_interval(f"{commit_sha}:tests") >= min(
        failure_probability,
        0.65,
    )


def simulated_outcome(
    commit_sha: str,
    test_passed: bool,
    risk_score: float,
    dependency_change: bool,
    risky_area_change: bool,
) -> str:
    """Simulate post-deployment success or failure."""

    failure_probability = 0.04 + (risk_score * 0.55)
    if not test_passed:
        failure_probability += 0.20
    if dependency_change:
        failure_probability += 0.08
    if risky_area_change:
        failure_probability += 0.12

    if stable_unit_interval(f"{commit_sha}:outcome") < min(failure_probability, 0.9):
        return "failure"
    return "success"


def simulated_commit_sha(index: int) -> str:
    """Return a stable SHA-like identifier for a generated record."""

    return hashlib.sha1(f"deployment-{index}".encode("utf-8")).hexdigest()


def simulated_bool(commit_sha: str, field: str, probability: float) -> bool:
    """Return a deterministic boolean with the requested probability."""

    return stable_unit_interval(f"{commit_sha}:{field}") < probability


def simulated_int(commit_sha: str, field: str, minimum: int, maximum: int) -> int:
    """Return a deterministic integer in an inclusive range."""

    value = stable_unit_interval(f"{commit_sha}:{field}")
    return minimum + int(value * ((maximum - minimum) + 1))


def simulated_float(
    commit_sha: str,
    field: str,
    minimum: float,
    maximum: float,
) -> float:
    """Return a deterministic float in a range."""

    value = stable_unit_interval(f"{commit_sha}:{field}")
    return minimum + (value * (maximum - minimum))


def stable_unit_interval(seed: str) -> float:
    """Return a deterministic pseudo-random float in the range [0, 1)."""

    digest = hashlib.sha256(seed.encode("utf-8")).hexdigest()
    return int(digest[:16], 16) / 16**16


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""

    parser = argparse.ArgumentParser(
        description="Generate simulated deployment records for Phase 1."
    )
    parser.add_argument(
        "--db",
        default=str(DEFAULT_DB_PATH),
        help="Path to the SQLite deployment database.",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=DEFAULT_RECORD_COUNT,
        help="Number of deployment records to generate.",
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Delete existing deployment records before inserting generated data.",
    )
    return parser.parse_args()


def main() -> None:
    """Generate and store simulated deployment records."""

    args = parse_args()
    if args.count < 1:
        raise ValueError("--count must be at least 1")

    db_path = Path(args.db)
    initialize_database(db_path)
    records = generate_records(args.count)
    with connect(db_path) as connection:
        if args.reset:
            connection.execute("DELETE FROM deployments")
            connection.commit()
        row_ids = insert_many(connection, records)

    print(f"Inserted {len(row_ids)} deployment records into {db_path}")


if __name__ == "__main__":
    main()
