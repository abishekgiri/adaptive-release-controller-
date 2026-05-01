"""Tests for local TravisTorrent data loading."""

from __future__ import annotations

import csv

from data.loaders import TravisTorrentLoader, validate_travistorrent_schema
from data.schemas import Action, Context, Outcome, Trajectory


def _write_csv(path, rows):
    fieldnames = [
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
        "git_author_email",
        "gh_changed_files",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def test_loader_yields_valid_context_objects(tmp_path):
    csv_path = tmp_path / "travistorrent.csv"
    _write_csv(
        csv_path,
        [
            {
                "git_trigger_commit": "abc123",
                "gh_project_name": "owner/repo",
                "git_committed_at": "2020-01-01T00:00:00Z",
                "gh_files_changed": "2",
                "gh_lines_added": "10",
                "gh_lines_deleted": "3",
                "gh_src_churn": "13",
                "gh_tests_run": "42",
                "gh_tests_added": "1",
                "tr_duration": "120",
                "gh_is_pr": "true",
                "tr_status": "passed",
                "tr_started_at": "2020-01-01T00:01:00Z",
                "tr_finished_at": "2020-01-01T00:03:00Z",
                "git_author_email": "dev@example.com",
                "gh_changed_files": "requirements.txt,src/app.py",
            },
        ],
    )

    records = list(
        TravisTorrentLoader(csv_path, min_builds=1, min_history_days=0).iter_records()
    )

    assert len(records) == 1
    assert isinstance(records[0].context, Context)
    assert records[0].context.commit_sha == "abc123"
    assert records[0].context.project_slug == "owner/repo"
    assert records[0].context.files_changed == 2
    assert records[0].context.lines_added == 10
    assert records[0].context.lines_deleted == 3
    assert records[0].context.src_churn == 13
    assert records[0].context.tests_run == 42
    assert records[0].context.tests_added == 1
    assert records[0].context.build_duration_s == 120.0
    assert records[0].context.is_pr is True
    assert records[0].context.has_dependency_change is True
    assert records[0].action == Action.DEPLOY
    assert records[0].outcome == Outcome.SUCCESS


def test_loader_keeps_outcome_out_of_context(tmp_path):
    csv_path = tmp_path / "travistorrent.csv"
    _write_csv(
        csv_path,
        [
            {
                "git_trigger_commit": "bad123",
                "gh_project_name": "owner/repo",
                "git_committed_at": "2020-01-01T00:00:00Z",
                "tr_status": "failed",
                "tr_started_at": "2020-01-01T00:01:00Z",
            },
            {
                "git_trigger_commit": "good456",
                "gh_project_name": "owner/repo",
                "git_committed_at": "2020-01-02T00:00:00Z",
                "tr_status": "passed",
                "tr_started_at": "2020-01-02T00:01:00Z",
            },
        ],
    )

    records = list(
        TravisTorrentLoader(csv_path, min_builds=1, min_history_days=0).iter_records()
    )
    first = records[0]

    assert first.outcome == Outcome.FAILURE
    assert first.context.recent_failure_rate == 0.0
    assert "outcome" not in first.context.__dict__
    assert "tr_status" not in first.context.__dict__
    assert "status" not in first.context.__dict__


def test_loader_handles_missing_optional_fields(tmp_path):
    csv_path = tmp_path / "minimal.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["git_trigger_commit", "gh_project_name", "tr_status"],
        )
        writer.writeheader()
        writer.writerow(
            {
                "git_trigger_commit": "abc123",
                "gh_project_name": "owner/repo",
                "tr_status": "errored",
            }
        )
        writer.writerow(
            {
                "git_trigger_commit": "",
                "gh_project_name": "owner/repo",
                "tr_status": "passed",
            }
        )

    assert validate_travistorrent_schema(csv_path) == []

    records = list(
        TravisTorrentLoader(csv_path, min_builds=1, min_history_days=0).iter_records()
    )

    assert len(records) == 1
    assert records[0].outcome == Outcome.FAILURE
    assert records[0].context.files_changed == 0
    assert records[0].context.lines_added == 0
    assert records[0].context.lines_deleted == 0
    assert records[0].context.src_churn == 0
    assert records[0].context.tests_run == 0
    assert records[0].context.build_duration_s == 0.0


def test_load_travistorrent_records_can_be_grouped_as_trajectories(tmp_path):
    csv_path = tmp_path / "travistorrent.csv"
    _write_csv(
        csv_path,
        [
            {
                "git_trigger_commit": "a",
                "gh_project_name": "owner/repo",
                "tr_status": "passed",
                "tr_started_at": "2020-01-02T00:00:00Z",
            },
            {
                "git_trigger_commit": "b",
                "gh_project_name": "owner/repo",
                "tr_status": "canceled",
                "tr_started_at": "2020-01-03T00:00:00Z",
            },
        ],
    )

    trajectories = list(
        TravisTorrentLoader(
            csv_path,
            min_builds=1,
            min_history_days=0,
        ).iter_trajectories()
    )

    assert len(trajectories) == 1
    assert isinstance(trajectories[0], Trajectory)
    assert trajectories[0].project_slug == "owner/repo"
    assert len(trajectories[0].steps) == 2
    assert trajectories[0].steps[0].reward is None
