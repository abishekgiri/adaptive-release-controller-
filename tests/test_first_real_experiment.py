"""Tests for first real experiment config parsing and output paths."""

from __future__ import annotations

import csv
import json
from pathlib import Path

from experiments.run_baselines import load_config, result_dir, run_experiment


def _write_config(path: Path, dataset_path: Path, results_root: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "config_name": "unit_real_result",
                "dataset_path": str(dataset_path),
                "results_root": str(results_root),
                "min_builds": 1,
                "min_history_days": 0,
                "max_trajectories": 1,
                "logged_propensity": 1.0,
                "delay_step_seconds": 60,
                "propensity_clip": 20.0,
                "bootstrap_resamples": 25,
                "confidence": 0.95,
                "policies": [
                    "static-rules",
                    "linucb",
                    "cost-sensitive-bandit",
                    "heuristic-score",
                ],
                "cost_config": {
                    "deploy_success": 0.0,
                    "deploy_failure": 10.0,
                    "canary_success": 1.0,
                    "canary_failure": 4.0,
                    "block_safe": 2.0,
                    "block_bad": 0.5,
                    "block_unknown": 2.0,
                },
            }
        ),
        encoding="utf-8",
    )


def _write_travistorrent_csv(path: Path) -> None:
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
    ]
    rows = [
        {
            "git_trigger_commit": "a",
            "gh_project_name": "owner/repo",
            "git_committed_at": "2020-01-01T00:00:00Z",
            "gh_files_changed": "1",
            "gh_lines_added": "10",
            "gh_lines_deleted": "0",
            "gh_src_churn": "10",
            "gh_tests_run": "20",
            "gh_tests_added": "1",
            "tr_duration": "60",
            "gh_is_pr": "false",
            "tr_status": "passed",
            "tr_started_at": "2020-01-01T00:00:00Z",
            "tr_finished_at": "2020-01-01T00:01:00Z",
            "git_author_email": "dev@example.com",
        },
        {
            "git_trigger_commit": "b",
            "gh_project_name": "owner/repo",
            "git_committed_at": "2020-01-02T00:00:00Z",
            "gh_files_changed": "2",
            "gh_lines_added": "40",
            "gh_lines_deleted": "5",
            "gh_src_churn": "45",
            "gh_tests_run": "30",
            "gh_tests_added": "2",
            "tr_duration": "120",
            "gh_is_pr": "true",
            "tr_status": "failed",
            "tr_started_at": "2020-01-02T00:00:00Z",
            "tr_finished_at": "2020-01-02T00:02:00Z",
            "git_author_email": "dev@example.com",
        },
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def test_load_config_parses_required_fields(tmp_path) -> None:
    dataset_path = tmp_path / "travistorrent.csv"
    config_path = tmp_path / "config.json"
    results_root = tmp_path / "results"
    _write_config(config_path, dataset_path, results_root)

    config = load_config(config_path)

    assert config.config_name == "unit_real_result"
    assert config.dataset_path == dataset_path
    assert config.results_root == results_root
    assert config.policies == (
        "static-rules",
        "linucb",
        "cost-sensitive-bandit",
        "heuristic-score",
    )
    assert result_dir(config, seed=3) == results_root / "unit_real_result" / "3"


def test_run_experiment_writes_seeded_result_directory(tmp_path) -> None:
    dataset_path = tmp_path / "travistorrent.csv"
    config_path = tmp_path / "config.json"
    results_root = tmp_path / "results"
    _write_travistorrent_csv(dataset_path)
    _write_config(config_path, dataset_path, results_root)

    config = load_config(config_path)
    summary = run_experiment(config, seed=11)

    output_dir = results_root / "unit_real_result" / "11"
    assert (output_dir / "summary.json").exists()
    assert (output_dir / "summary.md").exists()
    assert summary["policies"]["static_rules"]["status"] == "evaluated"
    assert summary["policies"]["linucb"]["status"] == "evaluated"
    assert summary["policies"]["cost_sensitive_bandit"]["status"] == "evaluated"
    assert summary["policies"]["heuristic-score"]["status"] == "todo_placeholder"
