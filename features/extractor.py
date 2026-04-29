"""Feature extraction for commit and CI deployment signals."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from knowledge_base.db import DeploymentRecord


RISKY_PATH_PREFIXES = (
    ".github/",
    "deploy/",
    "deployment/",
    "infra/",
    "infrastructure/",
    "k8s/",
    "migrations/",
    "terraform/",
)

DEPENDENCY_FILES = (
    "package-lock.json",
    "package.json",
    "poetry.lock",
    "pyproject.toml",
    "requirements.txt",
    "go.mod",
    "go.sum",
    "pom.xml",
    "build.gradle",
    "Cargo.lock",
    "Cargo.toml",
)


@dataclass(frozen=True)
class DeploymentFeatures:
    """Extracted features used by the deployment knowledge base."""

    commit_sha: str
    files_changed: int
    lines_added: int
    lines_deleted: int
    test_passed: bool
    ci_duration: float
    risk_score: float
    decision: str
    outcome: str


def extract_deployment_features(
    commit_payload: dict[str, Any],
    ci_run_payload: dict[str, Any] | None = None,
    outcome: str = "unknown",
    previous_failure_count: int = 0,
) -> DeploymentFeatures:
    """Create normalized deployment features from GitHub commit and CI payloads."""

    files = commit_payload.get("files") or []
    filenames = [file_data.get("filename", "") for file_data in files]
    stats = commit_payload.get("stats") or {}

    files_changed = len(files)
    lines_added = sum_int(files, "additions", fallback=stats.get("additions", 0))
    lines_deleted = sum_int(files, "deletions", fallback=stats.get("deletions", 0))
    test_passed = ci_tests_passed(ci_run_payload)
    ci_duration = ci_run_duration_seconds(ci_run_payload)
    dependency_change = has_dependency_change(filenames)
    risky_area_change = has_risky_path_change(filenames)

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

    return DeploymentFeatures(
        commit_sha=commit_payload["sha"],
        files_changed=files_changed,
        lines_added=lines_added,
        lines_deleted=lines_deleted,
        test_passed=test_passed,
        ci_duration=ci_duration,
        risk_score=risk_score,
        decision=decision,
        outcome=outcome,
    )


def to_deployment_record(features: DeploymentFeatures) -> DeploymentRecord:
    """Convert extracted features to a database record."""

    return DeploymentRecord(
        commit_sha=features.commit_sha,
        files_changed=features.files_changed,
        lines_added=features.lines_added,
        lines_deleted=features.lines_deleted,
        test_passed=features.test_passed,
        ci_duration=features.ci_duration,
        risk_score=features.risk_score,
        decision=features.decision,
        outcome=features.outcome,
    )


def sum_int(
    file_payloads: list[dict[str, Any]],
    key: str,
    fallback: int = 0,
) -> int:
    """Sum integer values from GitHub file payloads."""

    if not file_payloads:
        return int(fallback or 0)
    return sum(int(file_data.get(key) or 0) for file_data in file_payloads)


def ci_tests_passed(ci_run_payload: dict[str, Any] | None) -> bool:
    """Return whether the CI run ended successfully."""

    if ci_run_payload is None:
        return False
    return ci_run_payload.get("conclusion") == "success"


def ci_run_duration_seconds(ci_run_payload: dict[str, Any] | None) -> float:
    """Calculate CI runtime from GitHub Actions timestamps."""

    if ci_run_payload is None:
        return 0.0

    started_at = parse_github_timestamp(
        ci_run_payload.get("run_started_at") or ci_run_payload.get("created_at")
    )
    completed_at = parse_github_timestamp(
        ci_run_payload.get("updated_at") or ci_run_payload.get("completed_at")
    )
    if started_at is None or completed_at is None:
        return 0.0
    return max((completed_at - started_at).total_seconds(), 0.0)


def parse_github_timestamp(value: str | None) -> datetime | None:
    """Parse a GitHub timestamp into a timezone-aware datetime."""

    if not value:
        return None
    return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(
        timezone.utc
    )


def has_dependency_change(filenames: list[str]) -> bool:
    """Return whether the commit changes a known dependency manifest."""

    return any(filename.endswith(DEPENDENCY_FILES) for filename in filenames)


def has_risky_path_change(filenames: list[str]) -> bool:
    """Return whether the commit touches operationally risky paths."""

    return any(filename.startswith(RISKY_PATH_PREFIXES) for filename in filenames)


def calculate_risk_score(
    files_changed: int,
    lines_added: int,
    lines_deleted: int,
    test_passed: bool,
    ci_duration: float,
    dependency_change: bool = False,
    risky_area_change: bool = False,
    previous_failure_count: int = 0,
) -> float:
    """Calculate an initial deterministic deployment risk score."""

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


def decide_deployment(
    risk_score: float,
    test_passed: bool,
    threshold: float = 0.7,
) -> str:
    """Return the first deploy/block decision for a deployment record."""

    if not test_passed:
        return "block"
    if risk_score >= threshold:
        return "block"
    return "deploy"
