"""Feature extraction: convert raw GitHub commit and CI payloads to a pre-action Context."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import numpy as np

from data.schemas import Context


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

# Normalisation denominators matching FeatureEncoder.encode() in policies/base.py.
_NORM_FACTORS = np.array([
    50.0,    # files_changed
    1000.0,  # lines_added
    500.0,   # lines_deleted
    1500.0,  # src_churn
    1.0,     # is_pr
    200.0,   # tests_run
    10.0,    # tests_added
    180.0,   # build_duration_s
    10.0,    # author_experience
    1.0,     # recent_failure_rate
    1.0,     # has_dependency_change
    1.0,     # has_risky_path_change
], dtype=np.float64)


def extract_context(
    commit_payload: dict[str, Any],
    ci_run_payload: dict[str, Any] | None = None,
    project_slug: str = "",
    step: int = 0,
    author_experience: int = 0,
    recent_failure_rate: float = 0.0,
    is_pr: bool = False,
    tests_run: int = 0,
    tests_added: int = 0,
) -> Context:
    """Return a Context of pre-action observable features from GitHub API payloads.

    All parameters that cannot be inferred from the raw payloads (author_experience,
    recent_failure_rate, etc.) must be supplied by the caller from external history.
    """
    files = commit_payload.get("files") or []
    filenames = [fd.get("filename", "") for fd in files]
    stats = commit_payload.get("stats") or {}

    files_changed = len(files)
    lines_added = sum_int(files, "additions", fallback=stats.get("additions", 0))
    lines_deleted = sum_int(files, "deletions", fallback=stats.get("deletions", 0))
    src_churn = lines_added + lines_deleted
    build_duration_s = ci_run_duration_seconds(ci_run_payload)

    return Context(
        commit_sha=commit_payload.get("sha", ""),
        project_slug=project_slug,
        step=step,
        files_changed=files_changed,
        lines_added=lines_added,
        lines_deleted=lines_deleted,
        src_churn=src_churn,
        is_pr=is_pr,
        tests_run=tests_run,
        tests_added=tests_added,
        build_duration_s=build_duration_s,
        author_experience=author_experience,
        recent_failure_rate=recent_failure_rate,
        has_dependency_change=has_dependency_change(filenames),
        has_risky_path_change=has_risky_path_change(filenames),
    )


def feature_vector(context: Context) -> np.ndarray:
    """Return a 1-D float64 array of shape (13,) from a Context (12 normalised fields + bias).

    Encoding matches FeatureEncoder.encode() in policies/base.py; use either interchangeably.
    """
    raw = np.array([
        context.files_changed,
        context.lines_added,
        context.lines_deleted,
        context.src_churn,
        float(context.is_pr),
        context.tests_run,
        context.tests_added,
        context.build_duration_s,
        context.author_experience,
        context.recent_failure_rate,
        float(context.has_dependency_change),
        float(context.has_risky_path_change),
    ], dtype=np.float64)
    normalised = raw / _NORM_FACTORS
    return np.append(normalised, 1.0)  # bias term last


# ---------------------------------------------------------------------------
# Utility functions (keep per CLAUDE.md §What To Keep)
# ---------------------------------------------------------------------------

def sum_int(
    file_payloads: list[dict[str, Any]],
    key: str,
    fallback: int = 0,
) -> int:
    """Sum integer values from GitHub file payloads."""
    if not file_payloads:
        return int(fallback or 0)
    return sum(int(fd.get(key) or 0) for fd in file_payloads)


def ci_tests_passed(ci_run_payload: dict[str, Any] | None) -> bool:
    """Return whether the CI run ended successfully."""
    if ci_run_payload is None:
        return False
    return ci_run_payload.get("conclusion") == "success"


def ci_run_duration_seconds(ci_run_payload: dict[str, Any] | None) -> float:
    """Calculate CI runtime in seconds from GitHub Actions timestamps."""
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
    """Parse a GitHub ISO-8601 timestamp into a timezone-aware datetime."""
    if not value:
        return None
    return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(timezone.utc)


def has_dependency_change(filenames: list[str]) -> bool:
    """Return whether the commit changes a known dependency manifest."""
    return any(filename.endswith(DEPENDENCY_FILES) for filename in filenames)


def has_risky_path_change(filenames: list[str]) -> bool:
    """Return whether the commit touches operationally risky paths."""
    return any(filename.startswith(RISKY_PATH_PREFIXES) for filename in filenames)
