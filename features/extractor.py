"""Utility functions for extracting CI/CD signals from GitHub webhook payloads.

These helpers parse raw API payloads into normalised scalar fields consumed by
Context-building code in data/loaders.py. They contain no decision logic and no
risk scoring — the policy layer (policies/) handles all action selection.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any


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
