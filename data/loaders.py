"""TravisTorrent and Apachejit dataset loaders; converts raw CSVs to unified Trajectory format."""

from __future__ import annotations

from pathlib import Path
from typing import Iterator

from data.schemas import Trajectory


# Minimum project quality filters (from data/README.md).
_MIN_BUILDS = 500
_MIN_HISTORY_DAYS = 365


def load_travistorrent(
    csv_path: Path,
    min_builds: int = _MIN_BUILDS,
    min_history_days: int = _MIN_HISTORY_DAYS,
) -> Iterator[Trajectory]:
    """Load TravisTorrent CSV and yield one Trajectory per qualifying project.

    Args:
        csv_path: Path to the travistorrent_*.csv file in data/raw/.
        min_builds: Minimum number of builds for a project to be included.
        min_history_days: Minimum history span (days) for inclusion.

    Yields:
        One Trajectory per project, ordered by build start time.

    Raises:
        FileNotFoundError: If csv_path does not exist.
        ValueError: If required columns are missing from the CSV.
    """
    # TODO: read CSV with pandas/polars; validate columns against required field list in README;
    # group by gh_project_name; filter by build count and history span;
    # for each project, sort by tr_started_at; construct Context and Reward per row;
    # yield Trajectory(trajectory_id=slug, steps=(...))
    raise NotImplementedError


def load_apachejit(
    data_dir: Path,
) -> Iterator[Trajectory]:
    """Load ApacheJIT commit files and yield one Trajectory per project.

    Args:
        data_dir: Path to the ApacheJIT data directory in data/raw/.

    Yields:
        One Trajectory per project, ordered by commit timestamp.

    Notes:
        ApacheJIT does not include CI build outcomes. Context features are
        commit-metadata only. Reward (SZZ label) is the fix-inducing flag,
        with delay equal to the time until the fixing commit lands.
    """
    # TODO: iterate project CSV files; construct Context with CI fields set to sentinel values;
    # construct Reward from is_buggy flag and fix_commit_date - commit_date delay;
    # yield Trajectory per project
    raise NotImplementedError


def validate_travistorrent_schema(csv_path: Path) -> list[str]:
    """Return list of missing required columns; empty list means schema is valid."""
    # TODO: open CSV header only; compare against REQUIRED_TT_COLUMNS
    raise NotImplementedError


REQUIRED_TT_COLUMNS: frozenset[str] = frozenset({
    "git_trigger_commit",
    "gh_project_name",
    "git_committed_at",
    "gh_files_changed",
    "gh_lines_added",
    "gh_lines_deleted",
    "gh_src_churn",
    "gh_tests_run",
    "gh_tests_added",
    "tr_duration",
    "gh_is_pr",
    "tr_status",
    "tr_started_at",
    "tr_finished_at",
})
