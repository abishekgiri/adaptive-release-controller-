"""Dataset loaders that convert local CI/CD files to unified trajectory records."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterator

from data.schemas import Action, Context, Outcome, Trajectory, TrajectoryStep
from features.extractor import has_dependency_change, has_risky_path_change


REQUIRED_TT_COLUMNS: frozenset[str] = frozenset(
    {
        "git_trigger_commit",
        "gh_project_name",
        "tr_status",
    }
)

OPTIONAL_TT_COLUMNS: frozenset[str] = frozenset(
    {
        "git_committed_at",
        "gh_files_changed",
        "gh_lines_added",
        "gh_lines_deleted",
        "gh_src_churn",
        "gh_tests_run",
        "gh_tests_added",
        "tr_duration",
        "gh_is_pr",
        "tr_started_at",
        "tr_finished_at",
        "git_author_email",
        "git_author_name",
        "git_committer_email",
        "gh_changed_files",
        "gh_files",
        "changed_files",
    }
)

DEFAULT_MIN_BUILDS = 500
DEFAULT_MIN_HISTORY_DAYS = 365

_SUCCESS_STATUSES = {"passed", "pass", "success", "successful"}
_FAILURE_STATUSES = {"failed", "failure", "errored", "error", "broken"}
_CENSORED_STATUSES = {"canceled", "cancelled", "timeout", "timedout"}
_MISSING = {"", "na", "n/a", "null", "none", "nan", "?"}


@dataclass(frozen=True)
class TravisTorrentRecord:
    """One unified record produced from a TravisTorrent build row.

    ``context`` contains only pre-action observables. ``outcome`` is deliberately
    stored separately so downstream policies cannot accidentally consume labels
    at decision time.
    """

    context: Context
    action: Action
    outcome: Outcome
    started_at: datetime | None
    finished_at: datetime | None


@dataclass(frozen=True)
class _ParsedRow:
    source_order: int
    commit_sha: str
    project_slug: str
    outcome: Outcome
    committed_at: datetime | None
    started_at: datetime | None
    finished_at: datetime | None
    files_changed: int
    lines_added: int
    lines_deleted: int
    src_churn: int
    is_pr: bool
    tests_run: int
    tests_added: int
    build_duration_s: float
    author_id: str | None
    changed_paths: tuple[str, ...]


class TravisTorrentLoader:
    """Load TravisTorrent CSV files from a local path.

    The loader never downloads data. Pass either a CSV file path or a directory
    containing extracted CSV files.
    """

    def __init__(
        self,
        path: str | Path,
        *,
        min_builds: int = DEFAULT_MIN_BUILDS,
        min_history_days: int = DEFAULT_MIN_HISTORY_DAYS,
    ) -> None:
        self.path = Path(path)
        self.min_builds = min_builds
        self.min_history_days = min_history_days

    def __iter__(self) -> Iterator[TravisTorrentRecord]:
        return self.iter_records()

    def iter_records(self) -> Iterator[TravisTorrentRecord]:
        """Yield unified records ordered by project and decision time."""

        parsed_rows = self._load_valid_rows()
        rows_by_project: dict[str, list[_ParsedRow]] = {}
        for row in parsed_rows:
            rows_by_project.setdefault(row.project_slug, []).append(row)

        for project_slug in sorted(rows_by_project):
            project_rows = sorted(
                rows_by_project[project_slug],
                key=lambda item: (
                    item.started_at or item.committed_at or datetime.min.replace(
                        tzinfo=timezone.utc
                    ),
                    item.source_order,
                ),
            )
            if not self._project_passes_filters(project_rows):
                continue

            author_counts: dict[str, int] = {}
            prior_outcomes: list[tuple[datetime | None, Outcome]] = []
            for step, row in enumerate(project_rows):
                author_experience = (
                    author_counts.get(row.author_id, 0) if row.author_id else 0
                )
                context = Context(
                    commit_sha=row.commit_sha,
                    project_slug=row.project_slug,
                    step=step,
                    files_changed=row.files_changed,
                    lines_added=row.lines_added,
                    lines_deleted=row.lines_deleted,
                    src_churn=row.src_churn,
                    is_pr=row.is_pr,
                    tests_run=row.tests_run,
                    tests_added=row.tests_added,
                    build_duration_s=row.build_duration_s,
                    author_experience=author_experience,
                    recent_failure_rate=_recent_failure_rate(
                        prior_outcomes,
                        row.started_at or row.committed_at,
                    ),
                    has_dependency_change=has_dependency_change(
                        list(row.changed_paths)
                    ),
                    has_risky_path_change=has_risky_path_change(
                        list(row.changed_paths)
                    ),
                )
                yield TravisTorrentRecord(
                    context=context,
                    action=Action.DEPLOY,
                    outcome=row.outcome,
                    started_at=row.started_at,
                    finished_at=row.finished_at,
                )

                if row.author_id:
                    author_counts[row.author_id] = author_counts.get(row.author_id, 0) + 1
                prior_outcomes.append((row.started_at or row.committed_at, row.outcome))

    def iter_trajectories(self) -> Iterator[Trajectory]:
        """Yield one trajectory per project with outcome held outside Context.

        TrajectoryStep.reward is left as ``None`` because reward construction
        belongs to the delayed reward pipeline, not to the data loader.
        """

        records_by_project: dict[str, list[TravisTorrentRecord]] = {}
        for record in self.iter_records():
            records_by_project.setdefault(record.context.project_slug, []).append(record)

        for project_slug in sorted(records_by_project):
            records = records_by_project[project_slug]
            yield Trajectory(
                trajectory_id=f"travistorrent:{project_slug}",
                project_slug=project_slug,
                policy_id="logged_travistorrent",
                drift_segment_id=None,
                steps=tuple(
                    TrajectoryStep(
                        context=record.context,
                        action=record.action,
                        propensity=1.0,
                        reward=None,
                    )
                    for record in records
                ),
            )

    def _load_valid_rows(self) -> list[_ParsedRow]:
        csv_paths = _resolve_csv_paths(self.path)
        parsed_rows: list[_ParsedRow] = []
        source_order = 0
        for csv_path in csv_paths:
            missing = validate_travistorrent_schema(csv_path)
            if missing:
                raise ValueError(
                    f"{csv_path} is missing required columns: {', '.join(missing)}"
                )

            with csv_path.open(newline="", encoding="utf-8") as handle:
                reader = csv.DictReader(handle)
                for row in reader:
                    parsed = _parse_row(row, source_order)
                    source_order += 1
                    if parsed is not None:
                        parsed_rows.append(parsed)
        return parsed_rows

    def _project_passes_filters(self, rows: list[_ParsedRow]) -> bool:
        if len(rows) < self.min_builds:
            return False
        if self.min_history_days <= 0:
            return True

        times = [
            row.started_at or row.committed_at
            for row in rows
            if row.started_at is not None or row.committed_at is not None
        ]
        if len(times) < 2:
            return False
        return (max(times) - min(times)).days >= self.min_history_days


def load_travistorrent(
    csv_path: Path,
    min_builds: int = DEFAULT_MIN_BUILDS,
    min_history_days: int = DEFAULT_MIN_HISTORY_DAYS,
) -> Iterator[Trajectory]:
    """Load local TravisTorrent CSV data and yield one trajectory per project."""

    return TravisTorrentLoader(
        csv_path,
        min_builds=min_builds,
        min_history_days=min_history_days,
    ).iter_trajectories()


def load_apachejit(data_dir: Path) -> Iterator[Trajectory]:
    """Placeholder for the fallback ApacheJIT loader."""

    raise NotImplementedError("ApacheJIT loading is outside Immediate Next Task 12")


def validate_travistorrent_schema(csv_path: Path) -> list[str]:
    """Return missing required TravisTorrent columns."""

    if not csv_path.exists():
        raise FileNotFoundError(csv_path)
    with csv_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        try:
            header = next(reader)
        except StopIteration:
            return sorted(REQUIRED_TT_COLUMNS)
    columns = {column.strip() for column in header}
    return sorted(REQUIRED_TT_COLUMNS - columns)


def _resolve_csv_paths(path: Path) -> list[Path]:
    if not path.exists():
        raise FileNotFoundError(path)
    if path.is_file():
        if path.suffix.lower() != ".csv":
            raise ValueError(f"Expected a CSV file: {path}")
        return [path]

    csv_paths = sorted(item for item in path.iterdir() if item.suffix.lower() == ".csv")
    if not csv_paths:
        raise ValueError(f"No CSV files found in {path}")
    return csv_paths


def _parse_row(row: dict[str, str], source_order: int) -> _ParsedRow | None:
    commit_sha = _clean(row.get("git_trigger_commit"))
    project_slug = _clean(row.get("gh_project_name"))
    outcome = _parse_outcome(row.get("tr_status"))
    if not commit_sha or not project_slug or outcome is None:
        return None

    started_at = _parse_timestamp(row.get("tr_started_at"))
    finished_at = _parse_timestamp(row.get("tr_finished_at"))
    duration = _parse_float(row.get("tr_duration"), default=0.0)
    if duration == 0.0 and started_at is not None and finished_at is not None:
        duration = max((finished_at - started_at).total_seconds(), 0.0)

    lines_added = _parse_int(row.get("gh_lines_added"))
    lines_deleted = _parse_int(row.get("gh_lines_deleted"))
    src_churn = _parse_int(row.get("gh_src_churn"), default=lines_added + lines_deleted)

    return _ParsedRow(
        source_order=source_order,
        commit_sha=commit_sha,
        project_slug=project_slug,
        outcome=outcome,
        committed_at=_parse_timestamp(row.get("git_committed_at")),
        started_at=started_at,
        finished_at=finished_at,
        files_changed=_parse_int(row.get("gh_files_changed")),
        lines_added=lines_added,
        lines_deleted=lines_deleted,
        src_churn=src_churn,
        is_pr=_parse_bool(row.get("gh_is_pr")),
        tests_run=_parse_int(row.get("gh_tests_run")),
        tests_added=_parse_int(row.get("gh_tests_added")),
        build_duration_s=duration,
        author_id=_first_clean(
            row.get("git_author_email"),
            row.get("git_author_name"),
            row.get("git_committer_email"),
        ),
        changed_paths=_parse_changed_paths(
            _first_clean(
                row.get("gh_changed_files"),
                row.get("gh_files"),
                row.get("changed_files"),
            )
        ),
    )


def _parse_outcome(value: str | None) -> Outcome | None:
    status = _clean(value).lower()
    if status in _SUCCESS_STATUSES:
        return Outcome.SUCCESS
    if status in _FAILURE_STATUSES:
        return Outcome.FAILURE
    if status in _CENSORED_STATUSES:
        return Outcome.CENSORED
    return None


def _recent_failure_rate(
    prior_outcomes: list[tuple[datetime | None, Outcome]],
    current_time: datetime | None,
) -> float:
    if current_time is None:
        history = prior_outcomes
    else:
        window_start = current_time - timedelta(days=7)
        history = [
            (observed_at, outcome)
            for observed_at, outcome in prior_outcomes
            if observed_at is not None and window_start <= observed_at < current_time
        ]
    if not history:
        return 0.0
    failures = sum(1 for _, outcome in history if outcome == Outcome.FAILURE)
    return round(failures / len(history), 4)


def _parse_timestamp(value: str | None) -> datetime | None:
    cleaned = _clean(value)
    if not cleaned:
        return None

    normalized = cleaned.replace("Z", "+00:00")
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError:
        for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d"):
            try:
                parsed = datetime.strptime(cleaned, fmt)
                break
            except ValueError:
                continue
        else:
            return None

    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _parse_changed_paths(value: str | None) -> tuple[str, ...]:
    cleaned = _clean(value)
    if not cleaned:
        return ()
    normalized = cleaned.replace(";", ",").replace("|", ",")
    return tuple(
        part.strip().strip("\"'")
        for part in normalized.split(",")
        if part.strip().strip("\"'")
    )


def _parse_int(value: str | None, default: int = 0) -> int:
    cleaned = _clean(value)
    if not cleaned:
        return default
    try:
        return int(float(cleaned))
    except ValueError:
        return default


def _parse_float(value: str | None, default: float = 0.0) -> float:
    cleaned = _clean(value)
    if not cleaned:
        return default
    try:
        return float(cleaned)
    except ValueError:
        return default


def _parse_bool(value: str | None) -> bool:
    cleaned = _clean(value).lower()
    return cleaned in {"1", "true", "t", "yes", "y"}


def _first_clean(*values: str | None) -> str | None:
    for value in values:
        cleaned = _clean(value)
        if cleaned:
            return cleaned
    return None


def _clean(value: str | None) -> str:
    if value is None:
        return ""
    cleaned = str(value).strip()
    if cleaned.lower() in _MISSING:
        return ""
    return cleaned
