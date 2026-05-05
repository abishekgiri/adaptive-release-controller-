"""Generate a deterministic simulated deployment dataset (legacy fixture).

NOT for evaluation — features and outcomes are coupled by construction.
This script generates Phase 1-era synthetic records where the same risk score
that drives the simulated outcome is also used to make the deployment decision.
This circular coupling makes the data invalid for evaluating learned policies.

Kept here as a reproducibility fixture for the pre-pivot baseline system.
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path


DEFAULT_RECORD_COUNT = 100


def _calculate_risk_score(
    files_changed: int,
    lines_added: int,
    lines_deleted: int,
    test_passed: bool,
    ci_duration: float,
    dependency_change: bool = False,
    risky_area_change: bool = False,
    previous_failure_count: int = 0,
) -> float:
    """Local risk score (retired from features/extractor.py)."""
    changed_lines = lines_added + lines_deleted
    score = 0.0
    score += min(files_changed / 50, 0.25)
    score += min(changed_lines / 1000, 0.25)
    score += min(ci_duration / 3600, 0.15)
    score += min(previous_failure_count * 0.05, 0.15)
    if not test_passed:
        score += 0.30
    if dependency_change:
        score += 0.10
    if risky_area_change:
        score += 0.10
    return round(min(score, 1.0), 4)


def _decide_deployment(risk_score: float, test_passed: bool, threshold: float = 0.7) -> str:
    """Local decision (retired from features/extractor.py)."""
    if not test_passed:
        return "block"
    if risk_score >= threshold:
        return "block"
    return "deploy"


def generate_records(count: int = DEFAULT_RECORD_COUNT) -> list[dict]:
    """Create simulated Phase 1 deployment records."""
    return [generate_record(index) for index in range(1, count + 1)]


def generate_record(index: int) -> dict:
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
    risk_score = _calculate_risk_score(
        files_changed=files_changed,
        lines_added=lines_added,
        lines_deleted=lines_deleted,
        test_passed=test_passed,
        ci_duration=ci_duration,
        dependency_change=dependency_change,
        risky_area_change=risky_area_change,
        previous_failure_count=previous_failure_count,
    )
    decision = _decide_deployment(risk_score=risk_score, test_passed=test_passed)
    outcome = simulated_outcome(
        commit_sha=commit_sha,
        test_passed=test_passed,
        risk_score=risk_score,
        dependency_change=dependency_change,
        risky_area_change=risky_area_change,
    )

    return {
        "commit_sha": commit_sha,
        "files_changed": files_changed,
        "lines_added": lines_added,
        "lines_deleted": lines_deleted,
        "test_passed": test_passed,
        "ci_duration": round(ci_duration, 2),
        "risk_score": risk_score,
        "decision": decision,
        "outcome": outcome,
    }


def simulated_tests_passed(
    commit_sha: str,
    files_changed: int,
    changed_lines: int,
    dependency_change: bool,
    risky_area_change: bool,
) -> bool:
    failure_probability = 0.08
    failure_probability += min(files_changed / 200, 0.12)
    failure_probability += min(changed_lines / 6000, 0.18)
    if dependency_change:
        failure_probability += 0.08
    if risky_area_change:
        failure_probability += 0.06
    return stable_unit_interval(f"{commit_sha}:tests") >= min(failure_probability, 0.65)


def simulated_outcome(
    commit_sha: str,
    test_passed: bool,
    risk_score: float,
    dependency_change: bool,
    risky_area_change: bool,
) -> str:
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
    return hashlib.sha1(f"deployment-{index}".encode("utf-8")).hexdigest()


def simulated_bool(commit_sha: str, field: str, probability: float) -> bool:
    return stable_unit_interval(f"{commit_sha}:{field}") < probability


def simulated_int(commit_sha: str, field: str, minimum: int, maximum: int) -> int:
    value = stable_unit_interval(f"{commit_sha}:{field}")
    return minimum + int(value * ((maximum - minimum) + 1))


def simulated_float(commit_sha: str, field: str, minimum: float, maximum: float) -> float:
    value = stable_unit_interval(f"{commit_sha}:{field}")
    return minimum + (value * (maximum - minimum))


def stable_unit_interval(seed: str) -> float:
    digest = hashlib.sha256(seed.encode("utf-8")).hexdigest()
    return int(digest[:16], 16) / 16**16


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate simulated deployment records (legacy Phase 1 fixture)."
    )
    parser.add_argument("--count", type=int, default=DEFAULT_RECORD_COUNT)
    parser.add_argument("--output", type=str, default="-", help="Output path (- for stdout)")
    return parser.parse_args()


def main() -> None:
    import json
    args = parse_args()
    records = generate_records(args.count)
    payload = json.dumps(records, indent=2)
    if args.output == "-":
        print(payload)
    else:
        Path(args.output).write_text(payload, encoding="utf-8")
        print(f"Wrote {len(records)} records to {args.output}")


if __name__ == "__main__":
    main()
